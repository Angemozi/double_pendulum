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

#include "core/Config.hpp"
#include "rl/NeuralNet.hpp"
#include "rl/RolloutBuffer.hpp"
#include "util/Random.hpp"

namespace dp::rl {

// Result of querying the policy at a state.
struct PolicyOutput {
    Vec    action;        // sampled (stochastic) or mean (deterministic), raw
    double logProb = 0.0; // log pi(action|state) under the Gaussian
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
    Vec logStd() const { return logStd_; }

private:
    // Diagonal-Gaussian log-prob of `action` given `mean` and current logStd_.
    double gaussianLogProb(const Vec& mean, const Vec& action) const;
    double policyEntropy() const;

    int       obsDim_, actDim_;
    PpoConfig cfg_;
    util::Rng rng_;

    MLP actor_;     // -> mean (actDim)
    MLP critic_;    // -> value (1)
    Vec logStd_;    // learnable, length actDim
    // Adam moments for the standalone log_std parameter vector.
    Vec logStdM_, logStdV_;
    int adamStep_ = 0;
};

} // namespace dp::rl
