// =============================================================================
// Trainer.hpp
// -----------------------------------------------------------------------------
// Orchestrates the PPO training loop: rollout collection, GAE, policy updates,
// curriculum scheduling, statistics/CSV logging, checkpointing, and publishing
// render snapshots for the optional UI.
//
// The trainer is deliberately renderer-agnostic. In headless mode `runHeadless`
// drives it to completion; with a UI, the same `collectAndUpdate` is called from
// the simulation thread while the render thread consumes SharedState snapshots.
// =============================================================================
#pragma once

#include <deque>
#include <memory>
#include <string>

#include "core/Config.hpp"
#include "core/SharedState.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "rl/PPOAgent.hpp"
#include "rl/RolloutBuffer.hpp"
#include "util/Logger.hpp"
#include "util/Profiler.hpp"

namespace dp {

class Trainer {
public:
    Trainer(const Config& cfg, SharedState* shared = nullptr);

    // Collect one rollout (cfg.ppo.rolloutSteps steps), compute GAE, and run one
    // PPO update. Returns the update diagnostics. Honors pause/reset/frame-step
    // controls when a SharedState is attached.
    rl::UpdateStats collectAndUpdate();

    // Headless driver: loop collectAndUpdate until totalSteps, logging and
    // checkpointing on schedule. Returns the final running-average return.
    double runHeadless();

    // Pure evaluation rollout (deterministic policy, no learning). Returns the
    // episode return. Used by the eval app and the UI's "inference" mode.
    double evaluateEpisode(bool publishSnapshots);

    long long globalStep() const { return globalStep_; }
    int       updateCount() const { return updateCount_; }
    rl::PPOAgent& agent() { return agent_; }
    DoublePendulumEnv& env() { return env_; }

private:
    void applyCurriculum();
    void publishSnapshot(const rl::PolicyOutput& po, const StepResult& sr,
                         const Action& torque);
    void pushEpisodeReturn(double ret);
    double avgReturn() const;

    Config            cfg_;
    DoublePendulumEnv env_;
    rl::PPOAgent      agent_;
    rl::RolloutBuffer buffer_;
    SharedState*      shared_ = nullptr;

    Observation       obs_;            // current observation (persists across rollouts)
    double            episodeReturn_ = 0.0;
    long long         globalStep_ = 0;
    int               updateCount_ = 0;
    int               episodeIndex_ = 0;
    std::deque<double> recentReturns_; // sliding window for avg-return-100
    double             bestAvgReturn_ = -1e30; // best avgReturn100 seen so far

    util::CsvWriter   csv_;
    util::RateCounter simRate_{0.95};
    rl::UpdateStats   lastStats_{};
};

} // namespace dp
