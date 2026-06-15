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
#include <thread>

namespace dp::rl {

namespace {
// Adam hyperparameters shared by actor and critic.
constexpr double kBeta1 = 0.9;
constexpr double kBeta2 = 0.999;
constexpr double kEps   = 1e-8;
constexpr double kLog2Pi = 1.8378770664093454836; // log(2*pi)

// Diagonal-Gaussian log-prob with per-dimension (state-dependent) log_std.
double gaussLogProb(const Vec& mean, const Vec& logStd, const Vec& u, int n) {
    double lp = 0.0;
    for (int i = 0; i < n; ++i) {
        const double sigma = std::exp(logStd[i]);
        const double z = (u[i] - mean[i]) / sigma;
        lp += -0.5 * z * z - logStd[i] - 0.5 * kLog2Pi;
    }
    return lp;
}
// Differential entropy of the (pre-squash) diagonal Gaussian.
double gaussEntropy(const Vec& logStd, int n) {
    double h = 0.0;
    for (int i = 0; i < n; ++i) h += logStd[i] + 0.5 * (kLog2Pi + 1.0);
    return h;
}
} // namespace

void PPOAgent::decodePolicy(const Vec& raw, Vec& mean, Vec& logStd, Vec* dStdRaw) const {
    // raw = [ mean(0..actDim-1), rawLogStd(actDim..2actDim-1) ].
    // The tanh map bounds log_std to [minLogStd, maxLogStd]. maxLogStd is the
    // exploration CEILING: with it the policy can never diffuse past exp(maxLogStd),
    // so a transiently-bad advantage signal can't blow sigma up to a huge value.
    const double mn = cfg_.minLogStd, mx = cfg_.maxLogStd;
    mean.resize(actDim_);
    logStd.resize(actDim_);
    if (dStdRaw) dStdRaw->resize(actDim_);
    for (int d = 0; d < actDim_; ++d) {
        mean[d] = raw[d];
        const double th = std::tanh(raw[actDim_ + d]);
        // logStd = mn + 0.5*(mx-mn)*(tanh(raw)+1)  in [mn, mx], smooth & bounded.
        logStd[d] = mn + 0.5 * (mx - mn) * (th + 1.0);
        if (dStdRaw) (*dStdRaw)[d] = 0.5 * (mx - mn) * (1.0 - th * th); // dlogStd/draw
    }
}

PPOAgent::PPOAgent(int obsDim, int actDim, const PpoConfig& cfg, std::uint64_t seed)
    : obsDim_(obsDim), actDim_(actDim), cfg_(cfg), rng_(seed),
      // Actor emits BOTH the mean and the raw (pre-tanh) log_std => 2*actDim.
      actor_(obsDim, cfg.hidden1, cfg.hidden2, 2 * actDim, rng_),
      critic_(obsDim, cfg.hidden1, cfg.hidden2, 1, rng_),
      entropyCoef_(cfg.entropyCoef) {
    // Resolve the worker budget: 0 => auto (all hardware threads), else as set.
    int requested = cfg.numWorkers;
    if (requested <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        requested = (hc > 0) ? static_cast<int>(hc) : 1;
    }
    // Never use more workers than there are minibatch samples to split.
    numWorkers_ = std::max(1, std::min(requested, cfg.miniBatch));

    if (numWorkers_ > 1) {
        // One network clone + reduction scratch per worker. Clones start as
        // copies of the master and are re-synced with master weights each
        // minibatch before computing gradients.
        actorWorkers_.assign(static_cast<std::size_t>(numWorkers_), actor_);
        criticWorkers_.assign(static_cast<std::size_t>(numWorkers_), critic_);
        workerAccum_.resize(static_cast<std::size_t>(numWorkers_));
        pool_ = std::make_unique<util::ThreadPool>(numWorkers_);
    }
}

PolicyOutput PPOAgent::act(const Vec& obs, bool deterministic) {
    PolicyOutput out;
    Vec mean, logStd;
    decodePolicy(actor_.forward(obs), mean, logStd, nullptr);
    out.value = critic_.forward(obs)[0];

    out.action.resize(actDim_);   // pre-squash sample u
    out.squashed.resize(actDim_); // tanh(u) for the environment
    out.sigma.resize(actDim_);    // exposed for diagnostics / the UI
    for (int i = 0; i < actDim_; ++i) {
        const double sigma = std::exp(logStd[i]);
        // Deterministic policy (eval/deploy) uses the mean; training samples.
        const double u = deterministic ? mean[i] : mean[i] + sigma * rng_.gaussian();
        out.action[i]   = u;
        out.squashed[i] = std::tanh(u);
        out.sigma[i]    = sigma;
    }
    out.logProb = gaussLogProb(mean, logStd, out.action, actDim_);
    return out;
}

double PPOAgent::value(const Vec& obs) {
    return critic_.forward(obs)[0];
}

void PPOAgent::evaluate(const Vec& obs, const Vec& action,
                        double& logProb, double& entropy) {
    Vec mean, logStd;
    decodePolicy(actor_.forward(obs), mean, logStd, nullptr);
    logProb = gaussLogProb(mean, logStd, action, actDim_);
    entropy = gaussEntropy(logStd, actDim_);
}

// -----------------------------------------------------------------------------
// accumulateSampleGrad: the per-transition PPO gradient. This is the single
// source of truth for the policy/value math; both the serial and the parallel
// update paths call it (the parallel path simply passes per-worker network
// clones and a per-worker accumulator). Parameter gradients are accumulated
// inside `actorNet`/`criticNet`; statistics and the log_std gradient go into
// `acc`. Reads only shared, immutable-during-the-region state (logStd_, cfg_).
// -----------------------------------------------------------------------------
void PPOAgent::accumulateSampleGrad(MLP& actorNet, MLP& criticNet,
                                    const Transition& t, double invBatch,
                                    SampleAccum& acc) const {
    // ---- Actor / policy gradient --------------------------------------------
    Vec mean, logStd, dStdRaw;
    decodePolicy(actorNet.forward(t.observation), mean, logStd, &dStdRaw);
    const double logpNew = gaussLogProb(mean, logStd, t.action, actDim_);
    const double ratio = std::exp(logpNew - t.logProb);
    const double A = t.advantage;

    // Clipped surrogate selection. We MAXIMIZE
    //   L_clip = min(ratio*A, clip(ratio,1-eps,1+eps)*A).
    // d ratio / d logpNew = ratio. The gradient of L_clip w.r.t. logpNew is
    // ratio*A, UNLESS the clipped branch is the active (smaller) term and ratio
    // lies outside the clip band, in which case the objective is flat (grad 0).
    const double eps = cfg_.clipEps;
    const double unclipped = ratio * A;
    const double clippedRatio = math::clamp(ratio, 1.0 - eps, 1.0 + eps);
    const double clipped = clippedRatio * A;
    double dObj_dLogp;
    bool   wasClipped = false;
    if (unclipped <= clipped) {
        dObj_dLogp = ratio * A;
    } else {
        if (ratio > 1.0 - eps && ratio < 1.0 + eps) dObj_dLogp = ratio * A;
        else { dObj_dLogp = 0.0; wasClipped = true; }
    }

    // Minimize the negative objective; scale by 1/batch to average over samples.
    const double dLoss_dLogp = -dObj_dLogp * invBatch;

    // Build the gradient w.r.t. the actor's 2*actDim output. The first actDim
    // entries are d/d(mean), the second actDim are d/d(raw_logStd) (chained
    // through the tanh std-map). Gaussian log-prob derivatives:
    //   d logp / d mu_i     = (u_i - mu_i) / sigma_i^2
    //   d logp / d logStd_i = ((u_i - mu_i)^2 / sigma_i^2) - 1
    // The entropy bonus contributes dH/dlogStd_i = 1 (we minimize -coef*H).
    double sigmaMean = 0.0;
    Vec gradOut(2 * actDim_, 0.0);
    for (int d = 0; d < actDim_; ++d) {
        const double sigma2 = std::exp(2.0 * logStd[d]);
        const double diff = t.action[d] - mean[d];
        gradOut[d] = dLoss_dLogp * (diff / sigma2);             // -> mean head

        const double dLogp_dLogStd = (diff * diff) / sigma2 - 1.0;
        const double dLoss_dLogStd = dLoss_dLogp * dLogp_dLogStd
                                   + (-entropyCoef_) * invBatch; // entropy bonus
        gradOut[actDim_ + d] = dLoss_dLogStd * dStdRaw[d];        // chain via tanh map
        sigmaMean += std::exp(logStd[d]);
    }
    actorNet.backward(gradOut);
    acc.stdSum += sigmaMean / actDim_;

    // ---- Critic / value regression (with return-scaled value clipping) -------
    // The runaway here is a positive feedback loop: a high LR makes the critic
    // overshoot the (large, unbounded) GAE return target; because GAE bootstraps
    // off the critic, the targets follow the overshoot, valueLoss explodes, and
    // the resulting garbage advantages leave the policy with no gradient (so it
    // diffuses to max sigma -- the observed std=2.718 collapse). PPO value
    // clipping caps how far the critic may move from the value it had at
    // collection time (t.value), which breaks the loop. The clip band and the
    // reported loss are SCALED by the running return std so a band of ~0.2 is
    // meaningful whether returns are O(1) or O(1000) (raw critic, GAE-consistent;
    // no checkpoint reinterpretation, so warm-starts keep working).
    const double sigma  = (cfg_.normalizeReturns ? retStd_ : 1.0);
    const double invSig2 = 1.0 / (sigma * sigma);
    const double v     = criticNet.forward(t.observation)[0];
    const double vErrU = v - t.ret;                 // d(0.5*err^2)/dv = err
    double vErr = vErrU;
    if (cfg_.clipValueLoss) {
        const double band     = cfg_.valueClipEps * sigma;
        const double vClipped = t.value + math::clamp(v - t.value, -band, band);
        const double vErrC    = vClipped - t.ret;
        // Value loss = max(unclipped^2, clipped^2). The gradient is the unclipped
        // error UNLESS the clipped term dominates while v has already moved beyond
        // the band -- there vClipped is constant in v, so the gradient is 0 and the
        // critic is held back. That zero-gradient region is what bounds divergence.
        if (vErrC * vErrC > vErrU * vErrU && std::abs(v - t.value) >= band)
            vErr = 0.0;
    }
    criticNet.backward(Vec{ vErr * invBatch });

    // ---- Diagnostics ---------------------------------------------------------
    acc.policyLoss += -std::min(unclipped, clipped);
    acc.valueLoss  += 0.5 * vErrU * vErrU * invSig2; // normalized -> readable scale
    acc.entropy    += gaussEntropy(logStd, actDim_);
    acc.approxKL   += (t.logProb - logpNew);
    acc.clipFrac   += wasClipped ? 1.0 : 0.0;
    ++acc.count;
}

void PPOAgent::applyOptimizerStep() {
    ++adamStep_;
    // Linear learning-rate annealing: full LR at start -> (near) zero at the end
    // of training. Decaying the step size as the policy nears its optimum is the
    // single most effective guard against the "peak-then-drift" degradation,
    // because late, large updates on a low-variance policy overshoot. The std is
    // now produced by the actor network, so its bound is enforced inside
    // decodePolicy()'s tanh map -- there is no separate parameter to clamp here.
    const double lrScale = cfg_.annealLr ? std::max(0.05, 1.0 - progress_) : 1.0;

    actor_.clipGradGlobal(cfg_.maxGradNorm);
    critic_.clipGradGlobal(cfg_.maxGradNorm);
    actor_.applyAdam(cfg_.lrActor * lrScale, kBeta1, kBeta2, kEps, adamStep_);
    critic_.applyAdam(cfg_.lrCritic * lrScale, kBeta1, kBeta2, kEps, adamStep_);
}

UpdateStats PPOAgent::finalizeStats(const SampleAccum& total) const {
    UpdateStats stats;
    const double inv = total.count > 0 ? 1.0 / static_cast<double>(total.count) : 0.0;
    stats.policyLoss   = total.policyLoss * inv;
    stats.valueLoss    = total.valueLoss * inv;
    stats.entropy      = total.entropy * inv;
    stats.approxKL     = total.approxKL * inv;
    stats.clipFraction = total.clipFrac * inv;
    stats.meanStd      = total.stdSum * inv;  // mean per-state std over the batch
    return stats;
}

UpdateStats PPOAgent::update(RolloutBuffer& buffer) {
    if (buffer.size() == 0) return UpdateStats{};
    // Refresh the return-scale BEFORE the epochs (read by every value-clip below,
    // including the parallel workers -- it is not mutated inside the update region).
    if (cfg_.normalizeReturns) updateReturnStd(buffer);
    buffer.normalizeAdvantages();
    UpdateStats st = (numWorkers_ > 1) ? updateParallel(buffer) : updateSerial(buffer);

    // Adaptive entropy controller: nudge the entropy coefficient to hold the
    // policy entropy near the target. A simple multiplicative (exp) controller
    // keeps the coefficient positive and reacts proportionally to the gap. This
    // is what prevents the slow exploration collapse: as entropy drifts down,
    // the coefficient rises and pushes back, rather than losing to 1/sigma^2.
    if (cfg_.adaptiveEntropy) {
        const double target = cfg_.targetEntropyPerDim * actDim_;
        const double err = target - st.entropy;        // >0 => too little entropy
        entropyCoef_ *= std::exp(0.05 * err);
        entropyCoef_ = math::clamp(entropyCoef_, cfg_.minEntropyCoef, cfg_.maxEntropyCoef);
    }
    st.entropyCoef = entropyCoef_;
    return st;
}

// EMA-smoothed std of this rollout's GAE return targets. The first rollout snaps
// the estimate (so a warm-started run with large raw returns gets a correctly
// scaled value-clip band immediately rather than spending warmup at unit scale).
void PPOAgent::updateReturnStd(const RolloutBuffer& buffer) {
    const std::size_t n = buffer.size();
    if (n == 0) return;
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean += buffer[i].ret;
    mean /= static_cast<double>(n);
    double var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dd = buffer[i].ret - mean;
        var += dd * dd;
    }
    var /= static_cast<double>(n);
    const double s = std::max(std::sqrt(var), 1e-6); // guard against a degenerate 0
    retStd_ = retStdInit_ ? (0.95 * retStd_ + 0.05 * s) : s;
    retStdInit_ = true;
}

UpdateStats PPOAgent::updateSerial(RolloutBuffer& buffer) {
    const std::size_t n = buffer.size();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    SampleAccum total; total.reset(actDim_);

    for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
        // Fisher-Yates shuffle with our deterministic RNG.
        for (std::size_t i = n; i-- > 0;) {
            const std::size_t j = static_cast<std::size_t>(rng_.uniformInt(0, static_cast<int>(i)));
            std::swap(indices[i], indices[j]);
        }
        for (std::size_t start = 0; start < n; start += cfg_.miniBatch) {
            const std::size_t end = std::min(start + static_cast<std::size_t>(cfg_.miniBatch), n);
            const double invBatch = 1.0 / static_cast<double>(end - start);

            actor_.zeroGrad();
            critic_.zeroGrad();
            SampleAccum acc; acc.reset(actDim_);
            for (std::size_t k = start; k < end; ++k)
                accumulateSampleGrad(actor_, critic_, buffer[indices[k]], invBatch, acc);

            applyOptimizerStep();

            total.policyLoss += acc.policyLoss; total.valueLoss += acc.valueLoss;
            total.entropy += acc.entropy; total.approxKL += acc.approxKL;
            total.clipFrac += acc.clipFrac; total.count += acc.count;
            total.stdSum += acc.stdSum;

            if (klExceeded(acc)) return finalizeStats(total); // trust-region guard
        }
    }
    return finalizeStats(total);
}

// -----------------------------------------------------------------------------
// updateParallel: data-parallel minibatch gradient computation across CPU cores.
//
// For each minibatch we (1) sync the master weights into every worker clone,
// (2) split the minibatch into FIXED contiguous shards (shard w -> worker w),
// (3) have each worker compute partial gradients into its OWN clone + scratch,
// then (4) reduce the per-worker gradients into the master IN WORKER-ID ORDER
// and take a single optimizer step. Because the shard assignment and the
// reduction order are fixed, results are bit-reproducible for a given worker
// count (numWorkers=1 reproduces updateSerial exactly).
// -----------------------------------------------------------------------------
UpdateStats PPOAgent::updateParallel(RolloutBuffer& buffer) {
    const std::size_t n = buffer.size();
    const int W = numWorkers_;
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    SampleAccum total; total.reset(actDim_);

    for (int epoch = 0; epoch < cfg_.epochs; ++epoch) {
        for (std::size_t i = n; i-- > 0;) {
            const std::size_t j = static_cast<std::size_t>(rng_.uniformInt(0, static_cast<int>(i)));
            std::swap(indices[i], indices[j]);
        }
        for (std::size_t start = 0; start < n; start += cfg_.miniBatch) {
            const std::size_t end = std::min(start + static_cast<std::size_t>(cfg_.miniBatch), n);
            const std::size_t mb = end - start;
            const double invBatch = 1.0 / static_cast<double>(mb);

            // (1) Sync master weights into the worker clones and reset scratch.
            for (int w = 0; w < W; ++w) {
                actorWorkers_[static_cast<std::size_t>(w)].copyParamsFrom(actor_);
                criticWorkers_[static_cast<std::size_t>(w)].copyParamsFrom(critic_);
                actorWorkers_[static_cast<std::size_t>(w)].zeroGrad();
                criticWorkers_[static_cast<std::size_t>(w)].zeroGrad();
                workerAccum_[static_cast<std::size_t>(w)].reset(actDim_);
            }

            // (2)+(3) Each worker processes a fixed contiguous shard of [start,end).
            pool_->run([&](int w) {
                const std::size_t shardBegin = start + (mb * static_cast<std::size_t>(w)) / static_cast<std::size_t>(W);
                const std::size_t shardEnd   = start + (mb * static_cast<std::size_t>(w + 1)) / static_cast<std::size_t>(W);
                MLP& aNet = actorWorkers_[static_cast<std::size_t>(w)];
                MLP& cNet = criticWorkers_[static_cast<std::size_t>(w)];
                SampleAccum& acc = workerAccum_[static_cast<std::size_t>(w)];
                for (std::size_t k = shardBegin; k < shardEnd; ++k)
                    accumulateSampleGrad(aNet, cNet, buffer[indices[k]], invBatch, acc);
            });

            // (4) Deterministic reduction into the master, then one optimizer step.
            actor_.zeroGrad();
            critic_.zeroGrad();
            for (int w = 0; w < W; ++w) {
                const std::size_t wi = static_cast<std::size_t>(w);
                actor_.addGradsFrom(actorWorkers_[wi]);
                critic_.addGradsFrom(criticWorkers_[wi]);
                const SampleAccum& acc = workerAccum_[wi];
                total.policyLoss += acc.policyLoss; total.valueLoss += acc.valueLoss;
                total.entropy += acc.entropy; total.approxKL += acc.approxKL;
                total.clipFrac += acc.clipFrac; total.count += acc.count;
                total.stdSum += acc.stdSum;
            }
            applyOptimizerStep();

            // Trust-region guard: stop optimizing this rollout once the policy has
            // moved too far from the data-collection policy (mean KL over the
            // minibatch shards exceeds 1.5x target). Prevents the late-epoch
            // overfitting that compounds into reward drift.
            double mbKL = 0.0; long long mbCount = 0;
            for (int w = 0; w < W; ++w) {
                mbKL += workerAccum_[static_cast<std::size_t>(w)].approxKL;
                mbCount += workerAccum_[static_cast<std::size_t>(w)].count;
            }
            if (cfg_.targetKL > 0.0 && mbCount > 0 &&
                (mbKL / static_cast<double>(mbCount)) > 1.5 * cfg_.targetKL)
                return finalizeStats(total);
        }
    }
    return finalizeStats(total);
}

bool PPOAgent::save(const std::string& path) const {
    std::ofstream os(path, std::ios::out | std::ios::trunc);
    if (!os.is_open()) { DP_LOG_ERROR("PPOAgent::save failed: %s", path.c_str()); return false; }
    os.precision(17);
    // Version 2: std is a network output, so no separate log_std vector is saved.
    os << "DPPPO 2\n";
    os << obsDim_ << ' ' << actDim_ << '\n';
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
    if (version != 2) {
        DP_LOG_ERROR("PPOAgent::load unsupported checkpoint version %d (expected 2)", version);
        return false;
    }
    int obs, act;
    is >> obs >> act;
    if (obs != obsDim_ || act != actDim_) {
        DP_LOG_ERROR("PPOAgent::load dimension mismatch");
        return false;
    }
    actor_.deserialize(is);
    critic_.deserialize(is);
    return true;
}

} // namespace dp::rl
