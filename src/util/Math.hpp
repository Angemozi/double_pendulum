// =============================================================================
// Math.hpp
// -----------------------------------------------------------------------------
// Small, dependency-free math utilities used across the project.
//
// Design notes:
//  * Everything here is `constexpr`/`inline` and header-only so it can be
//    inlined aggressively in the physics hot loop.
//  * Angle wrapping is centralized so that *every* angle the agent observes is
//    normalized to [-pi, pi]. A consistent angle convention is essential for a
//    learning agent: an un-wrapped angle drifts unboundedly and destroys the
//    stationarity the network relies on.
// =============================================================================
#pragma once

#include <cmath>
#include <algorithm>

namespace dp::math {

// High-precision constants. We avoid relying on non-standard M_PI which is not
// guaranteed by the C++ standard library.
inline constexpr double kPi      = 3.14159265358979323846;
inline constexpr double kTwoPi   = 2.0 * kPi;
inline constexpr double kHalfPi  = 0.5 * kPi;

// -----------------------------------------------------------------------------
// wrapAngle
// Map an arbitrary angle to the canonical range [-pi, pi].
//
// We use the std::remainder approach because it is branch-light and numerically
// well behaved for very large magnitudes (no accumulation of subtraction error
// that a naive while-loop would incur).
// -----------------------------------------------------------------------------
inline double wrapAngle(double angle) noexcept {
    // std::remainder(x, 2pi) returns a value in [-pi, pi] by definition.
    return std::remainder(angle, kTwoPi);
}

// Smallest signed angular difference a-b, wrapped to [-pi, pi].
inline double angleDifference(double a, double b) noexcept {
    return wrapAngle(a - b);
}

// Clamp helper (std::clamp requires <algorithm>; provided for readability).
template <typename T>
inline constexpr T clamp(T v, T lo, T hi) noexcept {
    return std::clamp(v, lo, hi);
}

// Square helper to make energy / norm expressions read cleanly.
template <typename T>
inline constexpr T sq(T v) noexcept { return v * v; }

// Linear interpolation, used by the curriculum scheduler.
inline constexpr double lerp(double a, double b, double t) noexcept {
    return a + (b - a) * t;
}

} // namespace dp::math
