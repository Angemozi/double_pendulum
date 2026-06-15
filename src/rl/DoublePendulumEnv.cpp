// =============================================================================
// DoublePendulumEnv.cpp
// -----------------------------------------------------------------------------
// Environment logic: reset distributions, observation encoding, reward shaping,
// termination, domain randomization, disturbance injection, and the curriculum
// difficulty hook.
//
// REWARD DESIGN (the part that actually shapes behavior):
//   The goal is to drive the system to, and hold it at, the inverted (upright)
//   configuration while spending little control effort and not thrashing.
//
//     reward =  wUpright * uprightScore                 (+) be upright
//             - wTorque  * sum(torque_i^2) / maxTorque^2 (-) cheap control
//             - wOmega   * (omega1^2 + omega2^2)         (-) no violent spinning
//             + wSurvival                                (+) optional alive bonus
//
//   uprightScore in [0,1] is built from how close each bob's vertical projection
//   is to "up". With theta measured from the downward vertical, the upward unit
//   is -cos(theta) (which is +1 when theta = pi). We average both links and remap
//   to [0,1]. This gives a smooth, dense gradient even far from upright, which is
//   what lets PPO discover swing-up rather than only fine-tuning near balance.
// =============================================================================
#include "rl/DoublePendulumEnv.hpp"
#include "util/Math.hpp"

#include <cmath>

namespace dp {

using math::sq;

namespace {
// Scale used to bring angular velocities into ~[-1,1] for the observation.
constexpr double kOmegaObsScale = 10.0;
} // namespace

DoublePendulumEnv::DoublePendulumEnv(const Config& cfg)
    : cfg_(cfg),
      basephysics_(cfg.physics),
      pendulum_(cfg.physics),
      rng_(cfg.train.seed),
      episodeStepLimit_(cfg.env.maxEpisodeSteps),
      nominalGravity_(cfg.physics.g) {}

void DoublePendulumEnv::setDifficulty(double gravityScale, int maxEpisodeSteps,
                                      double disturbScale) {
    // Curriculum scales gravity (weaker = easier to balance), the horizon, and
    // the disturbance intensity (recovery curriculum).
    basephysics_.g   = nominalGravity_ * gravityScale;
    episodeStepLimit_ = maxEpisodeSteps;
    disturbScale_     = disturbScale;
}

Observation DoublePendulumEnv::reset(std::uint64_t seed) {
    rng_.reseed(seed);
    return reset();
}

Observation DoublePendulumEnv::reset() {
    stepCount_ = 0;
    lastAction_ = Action{};

    // Start from a copy of the (possibly curriculum-scaled) base parameters,
    // then optionally randomize the dynamics for robustness.
    pendulum_.params() = basephysics_;
    if (cfg_.env.domainRandomize) applyDomainRandomization();

    // Initial-condition distribution depends on the task.
    State s;
    switch (cfg_.env.task) {
        case TaskType::SwingUp:
            // Start hanging near the bottom with a little noise: agent must swing.
            s.theta1 = rng_.gaussian(0.0, 0.1);
            s.theta2 = rng_.gaussian(0.0, 0.1);
            s.omega1 = rng_.gaussian(0.0, 0.05);
            s.omega2 = rng_.gaussian(0.0, 0.05);
            break;
        case TaskType::Stabilize:
            // Start near upright (theta = pi): agent must hold balance.
            s.theta1 = math::wrapAngle(math::kPi + rng_.gaussian(0.0, 0.15));
            s.theta2 = math::wrapAngle(math::kPi + rng_.gaussian(0.0, 0.15));
            s.omega1 = rng_.gaussian(0.0, 0.1);
            s.omega2 = rng_.gaussian(0.0, 0.1);
            break;
        case TaskType::Recovery:
            // Start upright but with a sizeable velocity disturbance to recover.
            s.theta1 = math::wrapAngle(math::kPi + rng_.gaussian(0.0, 0.3));
            s.theta2 = math::wrapAngle(math::kPi + rng_.gaussian(0.0, 0.3));
            s.omega1 = rng_.gaussian(0.0, 1.5);
            s.omega2 = rng_.gaussian(0.0, 1.5);
            break;
    }
    pendulum_.setState(s);
    return encode();
}

void DoublePendulumEnv::applyDomainRandomization() {
    auto jitter = [&](double base, double pct) {
        return base * (1.0 + rng_.uniform(-pct, pct));
    };
    PhysicsConfig& p = pendulum_.params();
    p.m1 = jitter(basephysics_.m1, cfg_.env.drMassPct);
    p.m2 = jitter(basephysics_.m2, cfg_.env.drMassPct);
    p.l1 = jitter(basephysics_.l1, cfg_.env.drLengthPct);
    p.l2 = jitter(basephysics_.l2, cfg_.env.drLengthPct);
    p.g  = jitter(basephysics_.g,  cfg_.env.drGravityPct);
    p.b1 = jitter(basephysics_.b1, cfg_.env.drDampingPct);
    p.b2 = jitter(basephysics_.b2, cfg_.env.drDampingPct);
    if (cfg_.env.drTimestepPct > 0.0)
        p.dt = jitter(basephysics_.dt, cfg_.env.drTimestepPct);
}

void DoublePendulumEnv::maybeInjectDisturbance() {
    if (!cfg_.env.disturbances || disturbScale_ <= 0.0) return;
    if (rng_.uniform01() >= cfg_.env.disturbProb * disturbScale_) return;
    // Apply an instantaneous angular-velocity kick to model an external impulse,
    // scaled by the (curriculum-controlled) disturbance intensity.
    const double mag = cfg_.env.disturbMagnitude * disturbScale_;
    State s = pendulum_.state();
    s.omega1 += rng_.gaussian(0.0, mag);
    s.omega2 += rng_.gaussian(0.0, mag);
    pendulum_.setState(s);
}

Action DoublePendulumEnv::decode(const std::vector<double>& a) const {
    const double lim = cfg_.physics.maxTorque;
    // Network outputs are squashed to [-1,1] by the agent; scale to torque.
    Action act;
    if (cfg_.env.actuator == ActuatorMode::Dual) {
        act.torque1 = math::clamp(a[0], -1.0, 1.0) * lim;
        act.torque2 = math::clamp(a[1], -1.0, 1.0) * lim;
    } else {
        act.torque1 = math::clamp(a[0], -1.0, 1.0) * lim;
        act.torque2 = 0.0; // underactuated: joint 2 is passive
    }
    return act;
}

Observation DoublePendulumEnv::encodeState(const State& s) const {
    return Observation{
        std::sin(s.theta1), std::cos(s.theta1),
        std::sin(s.theta2), std::cos(s.theta2),
        s.omega1 / kOmegaObsScale, s.omega2 / kOmegaObsScale
    };
}

Observation DoublePendulumEnv::encode() const {
    return encodeState(pendulum_.state());
}

double DoublePendulumEnv::uprightScore(const State& s) const noexcept {
    // -cos(theta) is +1 when the link points up (theta=pi), -1 when down.
    const double up1 = -std::cos(s.theta1);
    const double up2 = -std::cos(s.theta2);
    // Average the two links and remap from [-1,1] to [0,1].
    return math::clamp(0.5 * ((up1 + up2) * 0.5 + 1.0), 0.0, 1.0);
}

double DoublePendulumEnv::computeReward(const Action& a) const noexcept {
    const State& s = pendulum_.state();
    const EnvConfig& e = cfg_.env;

    const double upright = uprightScore(s);

    const double torqueCost = (sq(a.torque1) + sq(a.torque2))
                            / std::max(1e-9, sq(cfg_.physics.maxTorque));

    // Saturating angular-velocity penalty. The raw sum-of-squares is unbounded
    // and at high spin overwhelms the upright reward, teaching a "don't move"
    // policy that cannot perform the corrective motion balancing requires. With
    // omegaRefSpeed>0 we bound the penalty to [0,1) via tanh so it shapes
    // behavior without ever dominating the objective.
    const double omegaSq = sq(s.omega1) + sq(s.omega2);
    const double omegaCost = (e.omegaRefSpeed > 0.0)
        ? std::tanh(omegaSq / sq(e.omegaRefSpeed))
        : omegaSq;

    // Stillness gate: credit "upright" ONLY when the links are also slow. Without
    // this, the dense uprightScore (which sees angles only) can be farmed by a
    // near-upright limit cycle -- the agent jitters violently yet scores high.
    // exp(-k*||omega||^2) multiplies the upright reward toward 0 as motion grows,
    // so "upright AND still" is the only way to earn full reward. k=0 disables it.
    const double stillness = (e.stillnessSharpness > 0.0)
        ? std::exp(-e.stillnessSharpness * omegaSq) : 1.0;

    // Dynamic torque penalty: cheap far from upright (so the agent can afford the
    // big torque needed for swing-up / kick recovery), ramping to the harsh
    // wTorque near the top to enforce stillness. wTorqueFar < 0 => constant wTorque.
    const double wTfar = (e.wTorqueFar >= 0.0) ? e.wTorqueFar : e.wTorque;
    const double wTeff = wTfar + (e.wTorque - wTfar) * std::pow(upright, e.torqueShaping);

    // Dynamic omega penalty (same shape): don't punish the unavoidable velocity of
    // a kick-recovery swing far from upright, but harshly damp motion near the top.
    const double wOfar = (e.wOmegaFar >= 0.0) ? e.wOmegaFar : e.wOmega;
    const double wOeff = wOfar + (e.wOmega - wOfar) * std::pow(upright, e.omegaShaping);

    double r = e.wUpright * upright * stillness
             - wTeff      * torqueCost
             - wOeff      * omegaCost
             + e.wSurvival;

    // Settling bonus: a dense reward for being inside the strict static pose
    // (upright AND still). Because it pays every settled step, the policy is
    // pushed to return to equilibrium as FAST as possible after a disturbance
    // and to actively damp residual oscillation rather than coast.
    if (e.wSettled > 0.0) {
        const bool settled =
            std::abs(math::wrapAngle(s.theta1 - math::kPi)) < e.staticAngleTol &&
            std::abs(math::wrapAngle(s.theta2 - math::kPi)) < e.staticAngleTol &&
            std::abs(s.omega1) < e.staticVelTol && std::abs(s.omega2) < e.staticVelTol;
        if (settled) r += e.wSettled;
    }

    // Energy-aware shaping: drive total mechanical energy toward E_top, the
    // energy of the (motionless) upright configuration. Normalized by |E_top| so
    // the term is scale-free. Helps the swing-up task pump energy efficiently.
    if (e.wEnergy > 0.0) {
        const PhysicsConfig& p = pendulum_.params();
        const double eTop = (p.m1 + p.m2) * p.g * p.l1 + p.m2 * p.g * p.l2;
        const double eNow = pendulum_.energy();
        const double eScale = std::abs(eTop) + 1e-6;
        r -= e.wEnergy * std::abs(eNow - eTop) / eScale;
    }
    return r;
}

bool DoublePendulumEnv::isTerminal() const noexcept {
    const State& s = pendulum_.state();
    // Terminate on physically implausible spin (also catches NaN via the
    // comparison returning false on NaN, handled explicitly below).
    if (std::isnan(s.theta1) || std::isnan(s.theta2) ||
        std::isnan(s.omega1) || std::isnan(s.omega2)) {
        return true;
    }
    if (std::abs(s.omega1) > cfg_.env.angularSpeedLimit ||
        std::abs(s.omega2) > cfg_.env.angularSpeedLimit) {
        return true;
    }
    return false;
}

StepResult DoublePendulumEnv::step(const Action& action) {
    lastAction_ = action;

    maybeInjectDisturbance();
    pendulum_.step(action);
    ++stepCount_;

    StepResult result;
    result.reward       = computeReward(action);
    result.terminal     = isTerminal();
    result.truncated    = (stepCount_ >= episodeStepLimit_);
    result.observation  = encode();
    result.uprightScore = uprightScore(pendulum_.state());
    result.energy       = pendulum_.energy();
    return result;
}

} // namespace dp
