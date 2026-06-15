// =============================================================================
// SimWorld.hpp
// -----------------------------------------------------------------------------
// A renderer-agnostic, real-time simulation core that decouples *running* a
// policy from *training* one. The training pipeline (Trainer/PPOAgent) and the
// interactive simulator both consume the same physics + env, but the simulator
// only ever does INFERENCE -- it never touches the optimizer, rollout buffer, or
// worker pool. This is the clean seam that lets dp_train and dp_simulator be
// separate executables that share dp_core.
//
// SimWorld owns:
//   * a DoublePendulumEnv (physics + reward + termination),
//   * an OPTIONAL policy (a deterministic/stochastic PPOAgent loaded from a
//     checkpoint); with no policy it runs free dynamics or a user torque.
//
// It exposes everything a renderer/HUD needs (kinematics, reward, torque, value,
// per-state sigma, entropy, energy) without any UI dependency, so it is fully
// unit-testable headless.
// =============================================================================
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <filesystem>

#include "core/Config.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "rl/PPOAgent.hpp"

namespace dp {

// One frame of diagnostics published after each simulation step.
struct SimFrame {
    State      state;
    Kinematics kinematics;
    Action     torque;            // torque actually applied this step
    double     reward    = 0.0;
    double     episodeReturn = 0.0;
    double     value     = 0.0;   // critic V(s) (0 if no policy)
    double     entropy   = 0.0;   // policy entropy at this state (0 if no policy)
    double     meanSigma = 0.0;   // mean per-state exploration std
    std::vector<double> sigma;    // per-action-dim std
    double     energy    = 0.0;
    double     uprightScore = 0.0;
    int        episodeStep  = 0;
    bool       terminal     = false;
    bool       truncated    = false;
    // Static-equilibrium ("monk mode") tracking.
    int        stillStreak  = 0;      // consecutive steps within the strict tol
    bool       atEquilibrium = false; // streak has reached staticHoldSteps
};

class SimWorld {
public:
    explicit SimWorld(const Config& cfg);

    // Load a CPU PPO checkpoint (best.ckpt / latest.ckpt / final.ckpt). Returns
    // false on dimension mismatch or IO error; the world keeps running policy-free.
    bool loadPolicy(const std::string& path);
    bool hasPolicy() const noexcept { return hasPolicy_; }
    const std::string& policyName() const noexcept { return policyName_; }

    // Live hot-reload: if the currently-loaded checkpoint file has changed on
    // disk since it was loaded, reload it. Returns true if a reload happened.
    // This lets the simulator watch a training run's latest.ckpt in real time.
    bool reloadIfChanged();

    // Start a fresh episode (uses the env's task reset distribution).
    void reset();

    // Advance exactly one fixed physics timestep. If a policy is loaded it acts;
    // otherwise `manualTorque` (default 0) is applied -- handy for free-swing
    // sandboxing or letting the UI drive torque directly.
    SimFrame step(const Action& manualTorque = Action{});

    // Inject an instantaneous angular-velocity impulse (keyboard disturbance).
    void applyImpulse(double omega1Kick, double omega2Kick);

    // Use the stochastic policy (sample) vs deterministic (mean). Stochastic
    // makes the learned exploration visible; deterministic is cleaner to watch.
    void setStochastic(bool s) noexcept { stochastic_ = s; }
    bool stochastic() const noexcept { return stochastic_; }

    const SimFrame& last() const noexcept { return last_; }
    const Config&   config() const noexcept { return cfg_; }
    DoublePendulumEnv& env() noexcept { return env_; }

    // Query the policy at an ARBITRARY state without disturbing the live sim.
    // Returns mean torque, value, and mean sigma -- the primitive used to build
    // policy/value/sigma heatmaps over state space. No-op (zeros) if no policy.
    struct Probe { double torque1, torque2, value, meanSigma; };
    Probe probe(const State& s);

    // A res x res scalar field, row-major, plus its min/max for color mapping --
    // the primitive behind the policy/value/sigma heatmap overlays.
    //   kind  = which scalar to sample (torque1 / value / mean sigma).
    //   slice = which two state axes sweep the grid; the OTHER two dims are held
    //           at `center` (typically the live state), so you can inspect the
    //           policy on a 2-D slice through the 4-D state space.
    enum class HeatmapKind  { Torque, Value, Sigma };
    enum class HeatmapSlice { Theta1Theta2, Theta2Omega2, Theta1Omega1 };
    struct Heatmap { int res = 0; std::vector<float> data; float lo = 0, hi = 1; };
    Heatmap computeHeatmap(int res, HeatmapKind kind,
                           HeatmapSlice slice = HeatmapSlice::Theta1Theta2,
                           State center = State{});

private:
    SimFrame buildFrame(const Action& torque, const StepResult& sr,
                        const rl::PolicyOutput* po);

    Config             cfg_;
    DoublePendulumEnv  env_;
    std::unique_ptr<rl::PPOAgent> agent_;
    bool               hasPolicy_  = false;
    bool               stochastic_ = false;
    std::string        policyName_;
    std::string        policyPath_;                     // for hot-reload watching
    std::filesystem::file_time_type policyMtime_{};     // last seen modification time
    Observation        obs_;
    double             episodeReturn_ = 0.0;
    SimFrame           last_;
    // One-time static-equilibrium detector state (persists across episodes for
    // the lifetime of this SimWorld, so the success banner prints exactly once).
    int                stillStreak_ = 0;
    bool               equilibriumLogged_ = false;
};

} // namespace dp
