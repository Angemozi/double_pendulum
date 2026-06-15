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
    stillStreak_ = 0;            // restart the streak; the once-only flag persists
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

    // ---- One-time static-equilibrium ("monk mode") detection ----------------
    // Upright is theta == pi in this codebase's convention, so we measure the
    // wrapped distance of each link from pi, and require both links AND both
    // angular velocities to stay within strict tolerances for staticHoldSteps
    // consecutive steps before declaring perfect static equilibrium -- exactly
    // once per SimWorld lifetime (the flag prevents log spam).
    {
        const State& es = last_.state;
        const EnvConfig& e = cfg_.env;
        const bool still =
            std::abs(math::wrapAngle(es.theta1 - math::kPi)) < e.staticAngleTol &&
            std::abs(math::wrapAngle(es.theta2 - math::kPi)) < e.staticAngleTol &&
            std::abs(es.omega1) < e.staticVelTol &&
            std::abs(es.omega2) < e.staticVelTol;
        stillStreak_ = still ? stillStreak_ + 1 : 0;
        last_.stillStreak   = stillStreak_;
        last_.atEquilibrium = (stillStreak_ >= e.staticHoldSteps);
        if (last_.atEquilibrium && !equilibriumLogged_) {
            equilibriumLogged_ = true;
            const double secs = stillStreak_ * cfg_.physics.dt;
            DP_LOG_INFO(" ");
            DP_LOG_INFO("  +=================================================+");
            DP_LOG_INFO("  |  [SUCCESS] Perfect Static Equilibrium Achieved! |");
            DP_LOG_INFO("  +=================================================+");
            DP_LOG_INFO("   Held upright & still for %d steps (%.2fs). Like a monk.",
                        stillStreak_, secs);
            DP_LOG_INFO(" ");
        }
    }

    // Auto-restart a fresh episode on terminal/time-limit so the sandbox keeps
    // running continuously (the UI can still trigger manual resets).
    if (sr.terminal || sr.truncated) {
        stillStreak_ = 0;        // new episode: restart streak (flag stays latched)
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

SimWorld::Heatmap SimWorld::computeHeatmap(int res, HeatmapKind kind,
                                           HeatmapSlice slice, State center) {
    Heatmap hm;
    hm.res = res;
    hm.data.assign(static_cast<std::size_t>(res) * res, 0.0f);
    if (!hasPolicy_ || res <= 0) { hm.lo = 0.0f; hm.hi = 1.0f; return hm; }

    constexpr double kOmegaRange = 12.0;  // sweep omega in [-12, 12]
    // Map the normalized grid coords (gx, gy in [-1,1]) onto a state, varying the
    // two axes the slice selects and holding the rest at `center`.
    auto stateAt = [&](double gx, double gy) {
        State s = center;
        switch (slice) {
            case HeatmapSlice::Theta1Theta2: s.theta1 = gx * math::kPi; s.theta2 = gy * math::kPi; break;
            case HeatmapSlice::Theta2Omega2: s.theta2 = gx * math::kPi; s.omega2 = gy * kOmegaRange; break;
            case HeatmapSlice::Theta1Omega1: s.theta1 = gx * math::kPi; s.omega1 = gy * kOmegaRange; break;
        }
        return s;
    };

    float lo = 1e30f, hi = -1e30f;
    for (int j = 0; j < res; ++j) {
        const double gy = 2.0 * (j + 0.5) / res - 1.0;
        for (int i = 0; i < res; ++i) {
            const double gx = 2.0 * (i + 0.5) / res - 1.0;
            const Probe p = probe(stateAt(gx, gy));
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
