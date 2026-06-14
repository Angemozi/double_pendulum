// =============================================================================
// SimWorld.cpp  -- see header for the design rationale.
// =============================================================================
#include "core/SimWorld.hpp"
#include "util/Math.hpp"
#include "util/Logger.hpp"

#include <cmath>
#include <filesystem>

namespace dp {

namespace {
// Entropy of a diagonal Gaussian from its per-dim std: sum(log(sigma)+0.5 ln 2pi e).
double entropyFromSigma(const std::vector<double>& sigma) {
    constexpr double kHalfLog2PiE = 0.5 * (1.8378770664093454836 + 1.0);
    double h = 0.0;
    for (double s : sigma) h += std::log(std::max(1e-12, s)) + kHalfLog2PiE;
    return h;
}
} // namespace

SimWorld::SimWorld(const Config& cfg) : cfg_(cfg), env_(cfg) {
    obs_ = env_.reset();
    last_.state = env_.getState();
    last_.kinematics = env_.physics().kinematics();
}

bool SimWorld::loadPolicy(const std::string& path) {
    // Construct an inference-only agent: force a single worker so no thread pool
    // is spun up (the simulator never trains).
    Config infCfg = cfg_;
    infCfg.ppo.numWorkers = 1;
    auto agent = std::make_unique<rl::PPOAgent>(
        env_.observationDim(), env_.actionDim(), infCfg.ppo, cfg_.train.seed);
    if (!agent->load(path)) {
        DP_LOG_WARN("SimWorld: failed to load policy '%s' (running policy-free)", path.c_str());
        return false;
    }
    agent_ = std::move(agent);
    hasPolicy_ = true;
    policyName_ = std::filesystem::path(path).filename().string();
    policyPath_ = path;
    std::error_code ec;
    policyMtime_ = std::filesystem::last_write_time(path, ec);
    DP_LOG_INFO("SimWorld: loaded policy '%s'", policyName_.c_str());
    return true;
}

bool SimWorld::reloadIfChanged() {
    if (policyPath_.empty()) return false;
    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(policyPath_, ec);
    if (ec) return false;                       // file vanished/locked: skip quietly
    if (mtime == policyMtime_) return false;    // unchanged
    // The trainer may be mid-write; loadPolicy() validates and fails safely, in
    // which case we keep the previous policy and retry on the next change.
    return loadPolicy(policyPath_);
}

void SimWorld::reset() {
    obs_ = env_.reset();
    episodeReturn_ = 0.0;
    last_ = SimFrame{};
    last_.state = env_.getState();
    last_.kinematics = env_.physics().kinematics();
}

void SimWorld::applyImpulse(double omega1Kick, double omega2Kick) {
    State s = env_.getState();
    s.omega1 += omega1Kick;
    s.omega2 += omega2Kick;
    env_.physics().setState(s);
}

SimFrame SimWorld::step(const Action& manualTorque) {
    rl::PolicyOutput po;
    const rl::PolicyOutput* poPtr = nullptr;
    Action torque = manualTorque;

    if (hasPolicy_) {
        po = agent_->act(obs_, /*deterministic=*/!stochastic_);
        torque = env_.decode(po.squashed);
        poPtr = &po;
    }

    const StepResult sr = env_.step(torque);
    episodeReturn_ += sr.reward;
    obs_ = sr.observation;

    last_ = buildFrame(torque, sr, poPtr);

    // Auto-restart a fresh episode on terminal/time-limit so the sandbox keeps
    // running continuously (the UI can still trigger manual resets).
    if (sr.terminal || sr.truncated) {
        obs_ = env_.reset();
        episodeReturn_ = 0.0;
    }
    return last_;
}

SimFrame SimWorld::buildFrame(const Action& torque, const StepResult& sr,
                              const rl::PolicyOutput* po) {
    SimFrame f;
    f.state         = env_.getState();
    f.kinematics    = env_.physics().kinematics();
    f.torque        = torque;
    f.reward        = sr.reward;
    f.episodeReturn = episodeReturn_;
    f.energy        = sr.energy;
    f.uprightScore  = sr.uprightScore;
    f.episodeStep   = env_.episodeStep();
    f.terminal      = sr.terminal;
    f.truncated     = sr.truncated;
    if (po) {
        f.value     = po->value;
        f.sigma     = po->sigma;
        double sum = 0.0;
        for (double s : po->sigma) sum += s;
        f.meanSigma = po->sigma.empty() ? 0.0 : sum / po->sigma.size();
        f.entropy   = entropyFromSigma(po->sigma);
    }
    return f;
}

SimWorld::Heatmap SimWorld::computeHeatmap(int res, HeatmapKind kind) {
    Heatmap hm;
    hm.res = res;
    hm.data.assign(static_cast<std::size_t>(res) * res, 0.0f);
    if (!hasPolicy_ || res <= 0) { hm.lo = 0.0f; hm.hi = 1.0f; return hm; }

    float lo = 1e30f, hi = -1e30f;
    for (int j = 0; j < res; ++j) {
        // y axis -> theta2 in [-pi, pi]
        const double th2 = -math::kPi + math::kTwoPi * (j + 0.5) / res;
        for (int i = 0; i < res; ++i) {
            const double th1 = -math::kPi + math::kTwoPi * (i + 0.5) / res;
            const Probe p = probe(State{th1, th2, 0.0, 0.0});
            float v = 0.0f;
            switch (kind) {
                case HeatmapKind::Torque: v = static_cast<float>(p.torque1); break;
                case HeatmapKind::Value:  v = static_cast<float>(p.value);   break;
                case HeatmapKind::Sigma:  v = static_cast<float>(p.meanSigma); break;
            }
            hm.data[static_cast<std::size_t>(j) * res + i] = v;
            lo = std::min(lo, v); hi = std::max(hi, v);
        }
    }
    hm.lo = lo; hm.hi = (hi - lo < 1e-6f) ? lo + 1.0f : hi;
    return hm;
}

SimWorld::Probe SimWorld::probe(const State& s) {
    if (!hasPolicy_) return Probe{0.0, 0.0, 0.0, 0.0};
    const Observation obs = env_.encodeState(s);
    const rl::PolicyOutput po = agent_->act(obs, /*deterministic=*/true);
    const Action a = env_.decode(po.squashed);
    double sum = 0.0;
    for (double sg : po.sigma) sum += sg;
    const double meanSigma = po.sigma.empty() ? 0.0 : sum / po.sigma.size();
    return Probe{a.torque1, a.torque2, po.value, meanSigma};
}

} // namespace dp
