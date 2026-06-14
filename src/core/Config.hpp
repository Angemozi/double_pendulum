// =============================================================================
// Config.hpp
// -----------------------------------------------------------------------------
// Central configuration for physics, environment, reward shaping, PPO, and
// training. Grouped into nested structs so each subsystem only sees what it
// needs, but the whole thing is trivially copyable (deterministic snapshots).
//
// A tiny flat-YAML loader (`key: value`, one per line, `#` comments) is provided
// so configs live in real files under /configs without pulling in a YAML lib.
// The format is a strict subset of YAML, so the files are also valid YAML if you
// later swap in yaml-cpp.
// =============================================================================
#pragma once

#include <cstdint>
#include <string>

namespace dp {

// Which numerical integrator the physics core uses.
enum class IntegratorType { RK4, SemiImplicitEuler };

// How many actuators the agent controls.
enum class ActuatorMode { Single, Dual };

// Task variants. Determines reset distribution + reward emphasis.
enum class TaskType { SwingUp, Stabilize, Recovery };

struct PhysicsConfig {
    double m1 = 1.0;     // mass of bob 1 (kg)
    double m2 = 1.0;     // mass of bob 2 (kg)
    double l1 = 1.0;     // rod 1 length (m)
    double l2 = 1.0;     // rod 2 length (m)
    double i1 = 0.0;     // extra rotational inertia at joint 1 (kg*m^2)
    double i2 = 0.0;     // extra rotational inertia at joint 2 (kg*m^2)
    double g  = 9.81;    // gravitational acceleration (m/s^2)
    double b1 = 0.02;    // viscous damping coefficient at joint 1 (N*m*s)
    double b2 = 0.02;    // viscous damping coefficient at joint 2 (N*m*s)
    double dt = 1.0 / 240.0;             // fixed integration timestep (s)
    IntegratorType integrator = IntegratorType::RK4;
    double maxTorque = 8.0;              // saturation limit per actuator (N*m)
};

struct EnvConfig {
    ActuatorMode actuator = ActuatorMode::Single;
    TaskType     task     = TaskType::Stabilize;
    int    maxEpisodeSteps = 2000;       // episode length cap (steps)
    double angularSpeedLimit = 50.0;     // |omega| above this terminates episode
    // Reward weights ---------------------------------------------------------
    double wUpright   = 1.0;             // reward for being inverted/upright
    double wTorque    = 0.01;            // penalty per unit torque^2
    double wOmega     = 0.005;           // penalty per unit angular velocity^2
    double wSurvival  = 0.0;             // flat per-step bonus for staying alive
    double uprightThreshold = 0.9;       // cos-based threshold counted as "balanced"
    // Reference angular speed for the SATURATING omega penalty. The raw
    // (omega1^2+omega2^2) term is unbounded and at high spin dwarfs the upright
    // reward (the diagnosed reward trap). With omegaRefSpeed>0 the penalty
    // becomes wOmega*tanh((w1^2+w2^2)/omegaRefSpeed^2), bounded by wOmega so it
    // can never dominate. Set <=0 to restore the old unbounded behavior.
    double omegaRefSpeed = 8.0;
    // Domain randomization (applied at reset when enabled) -------------------
    bool   domainRandomize = false;
    double drMassPct   = 0.2;            // +/- fraction on masses
    double drLengthPct = 0.2;            // +/- fraction on lengths
    double drGravityPct= 0.1;            // +/- fraction on gravity
    double drDampingPct= 0.5;            // +/- fraction on damping
    // Disturbance injection --------------------------------------------------
    bool   disturbances     = false;
    double disturbProb      = 0.001;     // per-step probability of an impulse
    double disturbMagnitude = 2.0;       // angular-velocity kick magnitude
};

struct PpoConfig {
    int    hidden1 = 64;                 // first hidden layer width
    int    hidden2 = 64;                 // second hidden layer width
    double gamma   = 0.99;               // discount factor
    double lambda  = 0.95;               // GAE smoothing factor
    double clipEps = 0.2;                // PPO clip range
    double lrActor = 3e-4;               // actor learning rate
    double lrCritic= 1e-3;               // critic learning rate
    double entropyCoef = 0.003;          // entropy bonus weight
    double valueCoef   = 0.5;            // value loss weight (unused: separate opt)
    double maxGradNorm = 0.5;            // global gradient clipping
    double initLogStd  = -0.5;           // initial log std of the Gaussian policy
    // --- stability / exploration controls (see PPOAgent) --------------------
    double minLogStd   = -2.0;           // floor on log std => sigma >= exp(min)
    double targetKL    = 0.02;           // KL early-stop threshold (0 disables)
    bool   annealLr    = true;           // linearly decay LR over training
    // Adaptive entropy: instead of a fixed entropyCoef, automatically raise/lower
    // it to hold the policy's entropy near targetEntropyPerDim * actionDim. This
    // is the robust cure for exploration collapse -- the coefficient grows when
    // entropy falls too far, directly counteracting the 1/sigma^2 overconfidence.
    bool   adaptiveEntropy   = true;
    double targetEntropyPerDim = 0.4;    // ~ std 0.37/dim; tune for exploration
    int    epochs      = 10;             // optimization epochs per rollout
    int    miniBatch   = 64;             // minibatch size
    int    rolloutSteps= 2048;           // environment steps per policy update
    // Data-parallel gradient computation. 0 => auto (use all CPU cores). 1 =>
    // single-threaded (bit-exact reproducible across machines). >1 => fixed
    // worker count (reproducible for that count). See PPOAgent::update.
    int    numWorkers  = 0;
};

struct CurriculumConfig {
    bool   enabled   = false;
    double startGravityScale = 0.3;      // begin with weaker gravity (easier)
    double endGravityScale   = 1.0;      // ramp to full gravity
    int    rampUpdates       = 200;      // number of updates to reach full difficulty
    int    startEpisodeSteps = 400;      // begin with shorter episodes
    int    endEpisodeSteps   = 2000;     // ramp to full episode length
};

struct TrainConfig {
    std::uint64_t seed = 42;
    long long totalSteps = 2'000'000;    // total environment steps to train
    int  logEvery        = 1;            // log every N policy updates
    int  checkpointEvery = 20;           // save a checkpoint every N updates
    std::string runName      = "ppo_dp";
    std::string modelDir     = "models";
    std::string logDir       = "logs";
};

struct Config {
    PhysicsConfig    physics;
    EnvConfig        env;
    PpoConfig        ppo;
    CurriculumConfig curriculum;
    TrainConfig      train;

    // Load overrides from a flat-YAML file. Unknown keys are ignored (with a
    // warning) so configs are forward compatible. Returns false on file error.
    bool loadFromFile(const std::string& path);
};

} // namespace dp
