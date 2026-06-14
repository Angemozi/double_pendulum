// =============================================================================
// Random.hpp
// -----------------------------------------------------------------------------
// Deterministic, cross-platform random number generation.
//
// WHY A CUSTOM WRAPPER:
//   The standard distribution objects (std::uniform_real_distribution,
//   std::normal_distribution, ...) are NOT specified to produce identical
//   sequences across compilers/standard libraries. For a project whose hard
//   requirement is "same seed => same results" on Windows AND Linux, we must
//   not depend on them. We therefore:
//      * use std::mt19937_64 (its sequence *is* standardized for a given seed),
//      * derive uniform doubles via an explicit bit construction, and
//      * implement Gaussian sampling with Box-Muller ourselves.
//
//   The result is bit-for-bit reproducible physics, rollouts, and noise across
//   platforms, which makes deterministic replay and regression testing real.
// =============================================================================
#pragma once

#include <cstdint>
#include <random>
#include <cmath>

#include "Math.hpp"

namespace dp::util {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 0xDEADBEEFCAFEULL) : engine_(seed), seed_(seed) {}

    void reseed(std::uint64_t seed) {
        seed_ = seed;
        engine_.seed(seed);
        hasSpareGaussian_ = false;
    }

    std::uint64_t seed() const noexcept { return seed_; }

    // Uniform double in [0, 1). Constructed from the top 53 bits so every
    // representable double in the range is reachable and the mapping is
    // identical on every platform.
    double uniform01() noexcept {
        const std::uint64_t x = engine_();
        return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0); // 2^53
    }

    // Uniform double in [lo, hi).
    double uniform(double lo, double hi) noexcept {
        return lo + (hi - lo) * uniform01();
    }

    // Uniform integer in [lo, hi] inclusive.
    int uniformInt(int lo, int hi) noexcept {
        const int span = hi - lo + 1;
        return lo + static_cast<int>(uniform01() * span);
    }

    // Standard normal N(0,1) via the polar Box-Muller transform. We cache the
    // spare deviate that the transform produces, so two calls cost one transform.
    double gaussian() noexcept {
        if (hasSpareGaussian_) {
            hasSpareGaussian_ = false;
            return spareGaussian_;
        }
        double u, v, s;
        do {
            u = 2.0 * uniform01() - 1.0;
            v = 2.0 * uniform01() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double mul = std::sqrt(-2.0 * std::log(s) / s);
        spareGaussian_ = v * mul;
        hasSpareGaussian_ = true;
        return u * mul;
    }

    // N(mean, std).
    double gaussian(double mean, double stddev) noexcept {
        return mean + stddev * gaussian();
    }

private:
    std::mt19937_64 engine_;
    std::uint64_t   seed_;
    bool   hasSpareGaussian_ = false;
    double spareGaussian_    = 0.0;
};

} // namespace dp::util
