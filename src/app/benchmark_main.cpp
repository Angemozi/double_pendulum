// =============================================================================
// benchmark_main.cpp
// -----------------------------------------------------------------------------
// Render-disabled performance + numerics benchmark.
//
//   1. Physics throughput: raw steps/sec for RK4 and semi-implicit Euler.
//   2. Energy conservation: free swing (no damping/torque), report energy drift
//      for both integrators -- the validation that RK4 is the more faithful one.
//   3. Environment throughput: steps/sec including reward + observation encoding.
//
// Usage: dp_benchmark [--steps N]
// =============================================================================
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "core/Config.hpp"
#include "physics/DoublePendulum.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "util/Profiler.hpp"
#include "util/Logger.hpp"

using namespace dp;

namespace {
const char* argValue(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

// Run a frictionless, unforced swing and measure max relative energy deviation.
double energyDrift(IntegratorType integ, long long steps) {
    PhysicsConfig p;
    p.b1 = p.b2 = 0.0;          // no damping -> energy must be conserved
    p.integrator = integ;
    DoublePendulum pend(p);
    pend.setState(State{2.0, 1.0, 0.0, 0.0}); // released from a tilted pose
    const double e0 = pend.energy();
    double maxAbsRel = 0.0;
    for (long long i = 0; i < steps; ++i) {
        pend.step(Action{0.0, 0.0});
        const double rel = std::abs((pend.energy() - e0) / (std::abs(e0) + 1e-9));
        maxAbsRel = std::max(maxAbsRel, rel);
    }
    return maxAbsRel;
}

double throughput(IntegratorType integ, long long steps) {
    PhysicsConfig p;
    p.integrator = integ;
    DoublePendulum pend(p);
    pend.setState(State{2.0, 1.0, 0.5, -0.5});
    util::StopWatch w;
    double sink = 0.0;
    for (long long i = 0; i < steps; ++i) {
        pend.step(Action{0.1, 0.0});
        sink += pend.state().theta1; // accumulate so the loop can't be elided
    }
    // Consume `sink` in a way the optimizer cannot prove is dead.
    if (sink == 1234567.89) DP_LOG_INFO("unreachable %.1f", sink);
    return static_cast<double>(steps) / w.seconds();
}
} // namespace

int main(int argc, char** argv) {
    long long steps = 2'000'000;
    if (const char* s = argValue(argc, argv, "--steps")) steps = std::atoll(s);

    DP_LOG_INFO("=== Double Pendulum Benchmark (%lld steps) ===", steps);

    DP_LOG_INFO("Energy drift (frictionless, max |dE/E|):");
    DP_LOG_INFO("  RK4               : %.3e", energyDrift(IntegratorType::RK4, steps / 4));
    DP_LOG_INFO("  SemiImplicitEuler : %.3e", energyDrift(IntegratorType::SemiImplicitEuler, steps / 4));

    DP_LOG_INFO("Physics throughput (steps/sec):");
    DP_LOG_INFO("  RK4               : %.0f", throughput(IntegratorType::RK4, steps));
    DP_LOG_INFO("  SemiImplicitEuler : %.0f", throughput(IntegratorType::SemiImplicitEuler, steps));

    // Full environment throughput (reward + observation encoding included).
    {
        Config cfg;
        DoublePendulumEnv env(cfg);
        env.reset(123);
        std::vector<double> action(env.actionDim(), 0.1);
        util::StopWatch w;
        long long n = steps / 2;
        for (long long i = 0; i < n; ++i) {
            Action a = env.decode(action);
            StepResult sr = env.step(a);
            if (sr.terminal || sr.truncated) env.reset(123 + i);
        }
        DP_LOG_INFO("Environment throughput: %.0f steps/sec", static_cast<double>(n) / w.seconds());
    }
    return 0;
}
