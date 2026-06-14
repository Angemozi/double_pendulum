// =============================================================================
// PPOAgent.hpp
// -----------------------------------------------------------------------------
// Proximal Policy Optimization agent (Schulman et al. 2017) with a Gaussian
// policy for continuous torque control.
//
// COMPONENTS (as requested by the spec):
//   * actor network  : state -> mean of a diagonal Gaussian over actions
//   * critic network : state -> scalar state-value V(s)
//   * log_std        : learnable per-dimension log standard deviation
//   * rollout storage : RolloutBuffer (on-policy)
//   * GAE advantages  : RolloutBuffer::computeGAE
//   * entropy bonus   : encourages exploration, decays as std shrinks
//   * clipped surrogate objective : the core PPO trust-region surrogate
//   * checkpoint save/load
//
// POLICY:
//   pi(a|s) = N(a; mu_theta(s), diag(sigma^2)), sigma = exp(log_std).
//   We do NOT squash inside the distribution; the environment clamps the action
//   to its valid range on decode. This keeps the Gaussian log-prob and its
//   gradient exact and is the standard "clipped-action" PPO formulation.
//
// CLIPPED SURROGATE (per sample, maximized):
//   ratio = exp( logpi_new(a|s) - logpi_old(a|s) )
//   L_clip = min( ratio * A,  clip(ratio, 1-eps, 1+eps) * A )
//   objective = L_clip + entropyCoef * H(pi)
//   The value function is trained separately with an MSE regression to the GAE
//   return targets (its own optimizer / learning rate).
// =============================================================================
#pragma once

#include <vector>
#include <string>
#include <memory>

#include "core/Config.hpp"
#include "rl/NeuralNet.hpp"
#include "rl/RolloutBuffer.hpp"
#include "util/Random.hpp"
#include "util/ThreadPool.hpp"
#include "util/Math.hpp"

namespace dp::rl {

// Result of querying the policy at a state.
//
// The policy is a tanh-SQUASHED diagonal Gaussian with STATE-DEPENDENT std:
//   u ~ N(mu(s), sigma(s)),   a = tanh(u),   torque = a * maxTorque
// `action` stores the PRE-squash sample u (what the PPO log-prob is computed on
// and what the buffer keeps), while `squashed` = tanh(u) in (-1,1) is what the
// environment actually receives. Keeping u lets us recompute the exact Gaussian
// log-prob under the updated policy; the tanh Jacobian cancels in the PPO ratio
// (same u for old and new), so the surrogate stays exact.
struct PolicyOutput {
    Vec    action;        // pre-squash sample u (stored for the update)
    Vec    squashed;      // tanh(u) in (-1,1), sent to the environment
    Vec    sigma;         // per-state std exp(logStd(s)) -- exploration magnitude
    double logProb = 0.0; // log N(u; mu, sigma) under the Gaussian
    double value   = 0.0; // V(state)
};

// Aggregated diagnostics returned by one update() call (for logging/UI).
struct UpdateStats {
    double policyLoss  = 0.0;
    double valueLoss   = 0.0;
    double entropy     = 0.0;
    double approxKL    = 0.0;   // mean(logp_old - logp_new), an early-stop signal
    double clipFraction= 0.0;   // fraction of samples whose ratio was clipped
    double meanStd     = 0.0;   // mean policy std (exploration magnitude)
    double entropyCoef = 0.0;   // current (possibly adaptive) entropy weight
};

class PPOAgent {
public:
    PPOAgent(int obsDim, int actDim, const PpoConfig& cfg, std::uint64_t seed);

    // Query the policy. `deterministic` returns the mean (used at eval/deploy);
    // otherwise samples from the Gaussian (used during training rollouts).
    PolicyOutput act(const Vec& obs, bool deterministic = false);

    // Critic-only evaluation (used to bootstrap the final GAE value).
    double value(const Vec& obs);

    // Recompute log pi(a|s) and entropy for a given (obs, action) under the
    // CURRENT policy (needed inside the PPO epochs).
    void evaluate(const Vec& obs, const Vec& action,
                  double& logProb, double& entropy);

    // Run the full PPO optimization over a collected rollout (multiple epochs of
    // minibatched, clipped updates). The buffer must already have GAE computed.
    UpdateStats update(RolloutBuffer& buffer);

    bool save(const std::string& path) const;
    bool load(const std::string& path);

    int obsDim() const { return obsDim_; }
    int actDim() const { return actDim_; }
    int numWorkers() const { return numWorkers_; }

    // Training progress in [0,1]; drives linear learning-rate annealing.
    void setProgress(double frac) { progress_ = math::clamp(frac, 0.0, 1.0); }

private:
    // Decode the actor's raw 2*actDim output into the policy mean and per-state
    // log_std. The std channels are passed through a tanh map that smoothly
    // bounds log_std to [minLogStd, kLogStdMax] WITH gradients (the SAC trick),
    // so the network can make exploration state-dependent -- large during
    // recovery, small near balance -- without ever collapsing or exploding.
    // `dStdRaw` (optional) receives d(log_std)/d(raw) for backprop.
    void decodePolicy(const Vec& raw, Vec& mean, Vec& logStd, Vec* dStdRaw) const;

    // Per-minibatch reduction results accumulated by one worker (or the serial
    // path). Parameter gradients live inside the passed networks; these are the
    // scalar diagnostics reduced across shards.
    struct SampleAccum {
        double policyLoss = 0.0;
        double valueLoss  = 0.0;
        double entropy    = 0.0;
        double approxKL   = 0.0;
        double clipFrac   = 0.0;
        double stdSum     = 0.0;   // sum over samples of mean-over-dims sigma(s)
        long long count   = 0;
        void reset(int /*actDim*/) {
            policyLoss = valueLoss = entropy = approxKL = clipFrac = stdSum = 0.0;
            count = 0;
        }
    };

    // Compute the clipped-surrogate + value gradient for ONE transition,
    // accumulating parameter gradients into the given actor/critic networks and
    // statistics/log_std gradient into `acc`. Shared by the serial and parallel
    // update paths so the math lives in exactly one place.
    void accumulateSampleGrad(MLP& actorNet, MLP& criticNet,
                              const Transition& t, double invBatch,
                              SampleAccum& acc) const;

    // The two implementations of update(); selected by numWorkers_.
    UpdateStats updateSerial(RolloutBuffer& buffer);
    UpdateStats updateParallel(RolloutBuffer& buffer);

    // Clip + Adam step on the master actor/critic (whose gradients are assumed
    // already accumulated). std is now a network output, so there is no separate
    // log_std parameter to optimize. Shared by both update paths.
    void applyOptimizerStep();

    // Average the accumulated reduction into the reported UpdateStats.
    UpdateStats finalizeStats(const SampleAccum& total) const;

    // True if the minibatch's mean approximate-KL exceeds 1.5x the target.
    bool klExceeded(const SampleAccum& acc) const {
        return cfg_.targetKL > 0.0 && acc.count > 0 &&
               (acc.approxKL / static_cast<double>(acc.count)) > 1.5 * cfg_.targetKL;
    }

    int       obsDim_, actDim_;
    PpoConfig cfg_;
    util::Rng rng_;

    MLP actor_;     // -> [mean (actDim), raw_logStd (actDim)]  (state-dependent)
    MLP critic_;    // -> value (1)
    int adamStep_ = 0;
    double progress_ = 0.0;     // 0..1 training progress for LR annealing
    double entropyCoef_ = 0.0;  // live entropy weight (adapted when enabled)

    // ---- Data-parallel update machinery -------------------------------------
    int numWorkers_ = 1;                       // resolved core budget (>=1)
    std::unique_ptr<util::ThreadPool> pool_;   // null when numWorkers_ == 1
    std::vector<MLP> actorWorkers_;            // per-worker network clones
    std::vector<MLP> criticWorkers_;
    std::vector<SampleAccum> workerAccum_;     // per-worker reduction scratch
};

} // namespace dp::rl
