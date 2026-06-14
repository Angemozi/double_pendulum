// =============================================================================
// Recorder.hpp
// -----------------------------------------------------------------------------
// Episode recording + playback. A recorded episode is a flat sequence of POD
// frames (no policy, no physics needed to replay) so it can be:
//   * saved/loaded as a compact binary blob (deterministic round-trip),
//   * exported to CSV for external analysis,
//   * replayed by the simulator by simply indexing frames over time.
//
// The simulator uses this to capture "best", "failure", and "recovery" episodes
// (a small library of ring-buffered recent episodes the user can promote/save).
// Keeping replay data fully decoupled from SimWorld means playback never risks
// diverging from what actually happened (no re-simulation drift).
// =============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/SimWorld.hpp"

namespace dp {

// One recorded frame -- fixed-size POD for trivial binary serialization.
struct RecordFrame {
    double theta1, theta2, omega1, omega2;   // generalized state
    double torque1, torque2;                 // applied torque
    double reward, value, meanSigma, energy; // diagnostics
};

class EpisodeRecorder {
public:
    void clear() { frames_.clear(); }
    bool empty() const { return frames_.empty(); }
    std::size_t size() const { return frames_.size(); }
    const RecordFrame& at(std::size_t i) const { return frames_[i]; }
    const std::vector<RecordFrame>& frames() const { return frames_; }

    // Append the latest simulation frame.
    void record(const SimFrame& f) {
        frames_.push_back(RecordFrame{
            f.state.theta1, f.state.theta2, f.state.omega1, f.state.omega2,
            f.torque.torque1, f.torque.torque2,
            f.reward, f.value, f.meanSigma, f.energy});
    }

    // Sum of per-frame rewards (used to rank "best" episodes).
    double totalReturn() const {
        double r = 0.0;
        for (const auto& f : frames_) r += f.reward;
        return r;
    }

    bool saveBinary(const std::string& path) const;
    bool loadBinary(const std::string& path);
    bool saveCsv(const std::string& path) const;

private:
    std::vector<RecordFrame> frames_;
};

} // namespace dp
