// =============================================================================
// SharedState.hpp
// -----------------------------------------------------------------------------
// Thread-safe hand-off between the simulation/training thread (producer) and the
// render/UI thread (consumer).
//
// THREADING DECISION:
//   The simulation must keep stepping at full speed while the UI stays
//   responsive. Rather than share the live physics object across threads (which
//   would force the renderer to take a lock on the hot path), the trainer
//   publishes immutable *snapshots* of just what the UI needs, under a short
//   critical section. The UI publishes control intents (pause, reset, speed) the
//   same way. This keeps the producer's lock hold-time to a few field copies and
//   avoids any blocking inside the physics integration step itself.
// =============================================================================
#pragma once

#include <mutex>
#include <atomic>
#include <vector>

#include "physics/DoublePendulum.hpp"

namespace dp {

// Snapshot the renderer draws. POD-ish, cheap to copy.
struct RenderSnapshot {
    State      state;
    Kinematics kinematics;
    double     reward       = 0.0;
    double     uprightScore = 0.0;
    double     energy       = 0.0;
    long long  globalStep   = 0;
    int        episode      = 0;
    int        episodeStep  = 0;
    std::vector<double> lastObservation;   // network input
    std::vector<double> lastActionRaw;     // network output (pre-scale)
    std::vector<double> lastTorque;        // applied torque
    // Most-recent training diagnostics.
    double episodeReturn = 0.0;
    double avgReturn100  = 0.0;
    double policyLoss    = 0.0;
    double valueLoss     = 0.0;
    double entropy       = 0.0;
    double meanStd       = 0.0;
    double simRate       = 0.0;  // steps/sec
};

// Control intents the UI sets and the trainer reads each loop.
struct ControlState {
    std::atomic<bool>   pause{false};
    std::atomic<bool>   requestReset{false};
    std::atomic<bool>   frameStep{false};      // advance exactly one step while paused
    std::atomic<bool>   training{true};         // false => pure inference/eval
    std::atomic<bool>   quit{false};
    std::atomic<bool>   injectDisturbance{false};
    std::atomic<double> speedMultiplier{1.0};   // sim steps per render frame scaler
    std::atomic<bool>   slowMotion{false};
};

class SharedState {
public:
    void publish(const RenderSnapshot& snap) {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot_ = snap;
    }
    RenderSnapshot read() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return snapshot_;
    }
    ControlState control;

private:
    mutable std::mutex mutex_;
    RenderSnapshot     snapshot_;
};

} // namespace dp
