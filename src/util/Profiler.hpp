// =============================================================================
// Profiler.hpp
// -----------------------------------------------------------------------------
// Lightweight timing utilities for benchmarking the simulation/training loop.
//
//  * StopWatch    : wall-clock interval timer (steady_clock).
//  * ScopedTimer  : RAII accumulator that adds its lifetime to a double sink.
//  * RateCounter  : exponential moving average of "events per second" (FPS / SPS).
//
// These intentionally avoid heap allocation so they can be sprinkled into hot
// loops without perturbing the measurement.
// =============================================================================
#pragma once

#include <chrono>

namespace dp::util {

class StopWatch {
public:
    using Clock = std::chrono::steady_clock;

    StopWatch() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    // Elapsed seconds since construction/reset.
    double seconds() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    double millis() const { return seconds() * 1000.0; }

private:
    Clock::time_point start_;
};

// Adds the scope's lifetime (in seconds) to an external accumulator on destruct.
class ScopedTimer {
public:
    explicit ScopedTimer(double& sink) : sink_(sink) {}
    ~ScopedTimer() { sink_ += watch_.seconds(); }
private:
    StopWatch watch_;
    double&   sink_;
};

// Smoothed rate estimate. Feed it the number of events processed and the elapsed
// time; read back a low-pass-filtered rate (e.g. steps/sec or frames/sec).
class RateCounter {
public:
    explicit RateCounter(double smoothing = 0.9) : smoothing_(smoothing) {}

    void update(double events, double dtSeconds) {
        if (dtSeconds <= 0.0) return;
        const double instant = events / dtSeconds;
        rate_ = initialized_ ? (smoothing_ * rate_ + (1.0 - smoothing_) * instant)
                             : instant;
        initialized_ = true;
    }

    double rate() const { return rate_; }

private:
    double smoothing_;
    double rate_        = 0.0;
    bool   initialized_ = false;
};

} // namespace dp::util
