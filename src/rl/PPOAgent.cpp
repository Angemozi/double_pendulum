// =============================================================================
// PPOAgent.cpp
// -----------------------------------------------------------------------------
// PPO implementation. The numerically interesting part is the analytic gradient
// of the clipped surrogate through the diagonal-Gaussian policy; it is derived
// inline next to the code that uses it.
// =============================================================================
#include "rl/PPOAgent.hpp"
#include "util/Math.hpp"
#include "util/Logger.hpp"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>

namespace dp::rl {

namespace {
// Adam hyperparameters shared by actor, critic, and log_std.
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kEps   = 1e-8;
constexpr double kLog2Pi = 1.8378770664093454836; // log(2*pi)
} // namespace

PPOAgent::PPOAgent(int obsDim, int actDim, const PpoConfig& cfg, std::uint64_t seed)
    : obsDim_(obsDim), actDim_(actDim), cfg_(cfg), rng_(seed),
      actor_(obsDim, cfg.hidden1, cfg.hidden2, actDim, rng_),
      critic_(obsDim, cfg.hidden1, cfg.hidden2, 1, rng_),
      logStd_(actDim, cfg.initLogStd),
      logStdM_(actDim, 0.0), logStdV_(actDim, 0.0) {}

double PPOAgent::gaussianLogProb(const Vec& mean, const Vec& action) const {
    // Diagonal Gaussian: sum_i [ -0.5*((a-mu)/sigma)^2 - log(sigma) - 0.5*log(2pi) ]
    double lp = 0.0;
    for (int i = 0; i < actDim_; ++i) {
        const double sigma = std::exp(logStd_[i]);
        const double z = (action[i] - mean[i]) / sigma;
        lp += -0.5 * z * z - logStd_[i] - 0.5 * kLog2Pi;
    }
    return lp;
}

double PPOAgent::policyEntropy() const {
    // Differential entropy of a diagonal Gaussian: sum_i [ log(sigma) + 0.5*log(2*pi*e) ].
    double h = 0.0;
    for (int i = 0; i < actDim_; ++i)
        h += logStd_[i] + 0.5 * (kLog2Pi + 1.0);
    return h;
}

PolicyOutput PPOAgent::act(const Vec& obs, bool deterministic) {
    PolicyOutput out;
    const Vec mean = actor_.forward(obs);
    out.value = critic_.forward(obs)[0];

    out.action.resize(actDim_);
    if (deterministic) {
        out.action = mean;
    } else {
        for (int i = 0; i < actDim_; ++i) {
            const double sigma = std::exp(logStd_[i]);
            out.action[i] = mean[i] + sigma * rng_.gaussian();
        }
    }
    out.logProb = gaussianLogProb(mean, out.action);
    return out;
}

double PPOAgent::value(const Vec& obs) {
    return critic_.forward(obs)[0];
}

void PPOAgent::evaluate(const Vec& obs, const Vec& action,
                        double& logProb, double& entropy) {
    const Vec mean = actor_.forward(obs); // caches activations for backward()
    logProb = gaussianLogProb(mean, action);
    entropy = policyEntropy();
}

UpdateStats PPOAgent::update(RolloutBuffer& buffer) {
    UpdateStats stats;
    const std::size_t n = buffer.size();
    if (n == 0) return stats;

    buffer.normalizeAdvantages();

    // Index list shuffled each epoch with our deterministic RNG (Fisher-Yates).
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    long long sampleCount = 0;
    double sumPolicyLoss = 0.0, sumValueLoss = 0.0, sumEntropy = 0.0;
    double sumKL = 0.0, sumClip = 0.0;

    for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
        for (std::size_t i = n; i-- > 0;) {
            const std::size_t j = static_cast<std::size_t>(rng_.uniformInt(0, static_cast<int>(i)));
            std::swap(indices[i], indices[j]);
        }

        for (std::size_t start = 0; start < n; start += cfg_.miniBatch) {
            const std::size_t end = std::min(start + static_cast<std::size_t>(cfg_.miniBatch), n);
            const double invBatch = 1.0 / static_cast<double>(end - start);

            actor_.zeroGrad();
            critic_.zeroGrad();
            Vec gradLogStd(actDim_, 0.0);

            for (std::size_t k = start; k < end; ++k) {
                const Transition& t = buffer[indices[k]];

                // ---- Actor / policy gradient -------------------------------
                const Vec mean = actor_.forward(t.observation);
                const double logpNew = gaussianLogProb(mean, t.action);
                const double ratio = std::exp(logpNew - t.logProb);
                const double A = t.advantage;

                // Clipped surrogate selection. We MAXIMIZE
                //   L_clip = min(ratio*A, clip(ratio,1-eps,1+eps)*A).
                // d ratio / d logpNew = ratio. The gradient of L_clip w.r.t.
                // logpNew is ratio*A, UNLESS the clipped branch is the active
                // (smaller) term and ratio lies outside the clip band, in which
                // case the objective is flat and the gradient is zero.
                const double eps = cfg_.clipEps;
                const double unclipped = ratio * A;
                const double clippedRatio = math::clamp(ratio, 1.0 - eps, 1.0 + eps);
                const double clipped = clippedRatio * A;
                double dObj_dLogp;       // d L_clip / d logpNew
                bool   wasClipped = false;
                if (unclipped <= clipped) {
                    dObj_dLogp = ratio * A;          // unclipped branch active
                } else {
                    // clipped branch active; flat (grad 0) when truly clamped.
                    if (ratio > 1.0 - eps && ratio < 1.0 + eps) dObj_dLogp = ratio * A;
                    else { dObj_dLogp = 0.0; wasClipped = true; }
                }

                // We minimize the negative objective; scale by 1/batch to average.
                const double dLoss_dLogp = -dObj_dLogp * invBatch;

                // Backprop the policy-gradient signal into the actor network and
                // the log_std parameter via the Gaussian log-prob derivatives:
                //   d logp / d mu_i      = (a_i - mu_i) / sigma_i^2
                //   d logp / d logStd_i  = ((a_i - mu_i)^2 / sigma_i^2) - 1
                Vec gradMean(actDim_);
                for (int d = 0; d < actDim_; ++d) {
                    const double sigma2 = std::exp(2.0 * logStd_[d]);
                    const double diff = t.action[d] - mean[d];
                    gradMean[d] = dLoss_dLogp * (diff / sigma2);
                    const double dLogp_dLogStd = (diff * diff) / sigma2 - 1.0;
                    gradLogStd[d] += dLoss_dLogp * dLogp_dLogStd;
                }
                actor_.backward(gradMean);

                // Entropy bonus: maximize entropyCoef*H => minimize -coef*H.
                // dH/dlogStd_i = 1, so add -entropyCoef (averaged) to each dim.
                for (int d = 0; d < actDim_; ++d)
                    gradLogStd[d] += (-cfg_.entropyCoef) * 1.0 * invBatch;

                // ---- Critic / value regression -----------------------------
                const double v = critic_.forward(t.observation)[0];
                const double vErr = v - t.ret;            // d(0.5*err^2)/dv = err
                critic_.backward(Vec{ vErr * invBatch });

                // ---- Diagnostics -------------------------------------------
                sumPolicyLoss += -std::min(unclipped, clipped);
                sumValueLoss  += 0.5 * vErr * vErr;
                sumEntropy    += policyEntropy();
                sumKL         += (t.logProb - logpNew); // approx KL >= 0 in expectation
                sumClip       += wasClipped ? 1.0 : 0.0;
                ++sampleCount;
            }

            // ---- Optimizer step ----------------------------------------------
            ++adamStep_;
            actor_.clipGradGlobal(cfg_.maxGradNorm);
            critic_.clipGradGlobal(cfg_.maxGradNorm);
            actor_.applyAdam(cfg_.lrActor, kBeta1, kBeta2, kEps, adamStep_);
            critic_.applyAdam(cfg_.lrCritic, kBeta1, kBeta2, kEps, adamStep_);

            // Manual Adam step for the standalone log_std vector.
            const double bc1 = 1.0 - std::pow(kBeta1, adamStep_);
            const double bc2 = 1.0 - std::pow(kBeta2, adamStep_);
            for (int d = 0; d < actDim_; ++d) {
                const double g = gradLogStd[d];
                logStdM_[d] = kBeta1 * logStdM_[d] + (1.0 - kBeta1) * g;
                logStdV_[d] = kBeta2 * logStdV_[d] + (1.0 - kBeta2) * g * g;
                const double mHat = logStdM_[d] / bc1;
                const double vHat = logStdV_[d] / bc2;
                logStd_[d] -= cfg_.lrActor * mHat / (std::sqrt(vHat) + kEps);
                // Keep std in a sane range: avoid collapse-to-zero (no
                // exploration) and runaway growth (divergence).
                logStd_[d] = math::clamp(logStd_[d], -3.0, 1.0);
            }
        }
    }

    const double inv = sampleCount > 0 ? 1.0 / static_cast<double>(sampleCount) : 0.0;
    stats.policyLoss   = sumPolicyLoss * inv;
    stats.valueLoss    = sumValueLoss * inv;
    stats.entropy      = sumEntropy * inv;
    stats.approxKL     = sumKL * inv;
    stats.clipFraction = sumClip * inv;
    double meanStd = 0.0;
    for (double ls : logStd_) meanStd += std::exp(ls);
    stats.meanStd = meanStd / actDim_;
    return stats;
}

bool PPOAgent::save(const std::string& path) const {
    std::ofstream os(path, std::ios::out | std::ios::trunc);
    if (!os.is_open()) { DP_LOG_ERROR("PPOAgent::save failed: %s", path.c_str()); return false; }
    os.precision(17);
    os << "DPPPO 1\n";
    os << obsDim_ << ' ' << actDim_ << '\n';
    for (double ls : logStd_) os << ls << ' ';
    os << '\n';
    actor_.serialize(os);
    critic_.serialize(os);
    return true;
}

bool PPOAgent::load(const std::string& path) {
    std::ifstream is(path);
    if (!is.is_open()) { DP_LOG_ERROR("PPOAgent::load failed: %s", path.c_str()); return false; }
    std::string magic; int version = 0;
    is >> magic >> version;
    if (magic != "DPPPO") { DP_LOG_ERROR("PPOAgent::load bad magic in %s", path.c_str()); return false; }
    int obs, act;
    is >> obs >> act;
    if (obs != obsDim_ || act != actDim_) {
        DP_LOG_ERROR("PPOAgent::load dimension mismatch");
        return false;
    }
    for (double& ls : logStd_) is >> ls;
    actor_.deserialize(is);
    critic_.deserialize(is);
    return true;
}

} // namespace dp::rl
