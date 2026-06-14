// =============================================================================
// DoublePendulumEnv.hpp
// -----------------------------------------------------------------------------
// OpenAI-Gym-style reinforcement learning environment wrapping the physics core.
//
// API (as requested by the spec):
//     reset()        -> Observation        (start a fresh episode)
//     step(Action)   -> StepResult         (advance one timestep)
//     getState()     -> State              (raw generalized coordinates)
//     isTerminal()   -> bool               (episode-ending condition)
//     computeReward()-> double             (current shaped reward)
//
// OBSERVATION ENCODING:
//     The network never sees raw angles. Instead each angle is encoded as
//     (sin, cos), which is continuous across the +-pi wrap and gives the policy a
//     smooth, unambiguous representation of orientation. Angular velocities are
//     scaled into a roughly unit range. Observation layout (6 doubles):
//         [ sin th1, cos th1, sin th2, cos th2, w1/wscale, w2/wscale ]
//
// ACTION SPACE:
//     Single mode -> 1 continuous torque applied at joint 1.
//     Dual mode   -> 2 continuous torques (joint 1 and joint 2).
//     Network outputs are in [-1, 1] (squashed) and scaled to +-maxTorque here.
// =============================================================================
#pragma once

#include <vector>
#include <array>

#include "core/Config.hpp"
#include "physics/DoublePendulum.hpp"
#include "util/Random.hpp"

namespace dp {

using Observation = std::vector<double>;

struct StepResult {
    Observation observation;
    double reward    = 0.0;
    bool   terminal  = false;   // task failure / success terminal state
    bool   truncated = false;   // hit the time limit (not a true terminal)
    // Diagnostics surfaced for logging / the UI.
    double uprightScore = 0.0;  // in [0,1]; 1 == perfectly inverted
    double energy       = 0.0;
};

class DoublePendulumEnv {
public:
    explicit DoublePendulumEnv(const Config& cfg);

    // ---- Gym-style core API -------------------------------------------------
    Observation reset();                       // uses internal RNG (seeded)
    Observation reset(std::uint64_t seed);     // reseed then reset (determinism)
    StepResult  step(const Action& action);

    const State& getState() const noexcept { return pendulum_.state(); }
    bool   isTerminal() const noexcept;
    double computeReward(const Action& lastAction) const noexcept;

    // ---- Introspection / wiring --------------------------------------------
    int  observationDim() const noexcept { return 6; }
    int  actionDim() const noexcept {
        return cfg_.env.actuator == ActuatorMode::Dual ? 2 : 1;
    }
    Observation encode() const;                       // current state -> obs
    Observation encodeState(const State& s) const;     // arbitrary state -> obs
    Action      decode(const std::vector<double>& a) const; // policy out -> torque

    const DoublePendulum& physics() const noexcept { return pendulum_; }
    DoublePendulum&       physics()       noexcept { return pendulum_; }
    int  episodeStep() const noexcept { return stepCount_; }
    util::Rng& rng() noexcept { return rng_; }

    // ---- Curriculum hook ----------------------------------------------------
    // Scale gravity (0..1 of nominal), override the episode length limit, and
    // scale disturbance intensity (0..1). disturbScale defaults to 1 (full).
    void setDifficulty(double gravityScale, int maxEpisodeSteps,
                       double disturbScale = 1.0);

private:
    void  applyDomainRandomization();
    void  maybeInjectDisturbance();
    double uprightScore(const State& s) const noexcept;

    Config         cfg_;
    PhysicsConfig  basephysics_;   // pristine params (DR perturbs copies)
    DoublePendulum pendulum_;
    util::Rng      rng_;
    int            stepCount_ = 0;
    int            episodeStepLimit_ = 0;
    double         nominalGravity_ = 9.81;
    double         disturbScale_ = 1.0;   // curriculum-scaled disturbance intensity
    Action         lastAction_{};
};

} // namespace dp
