// =============================================================================
// test_main.cpp
// -----------------------------------------------------------------------------
// Dependency-free unit/validation tests. Exits non-zero on any failure so it can
// gate CI. Covers: angle math, RNG determinism, equations of motion, integrator
// energy behavior, environment determinism, and GAE.
// =============================================================================
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

#include "util/Math.hpp"
#include "util/Random.hpp"
#include "physics/DoublePendulum.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "rl/RolloutBuffer.hpp"
#include "rl/NeuralNet.hpp"
#include "core/Config.hpp"
#include "core/SimWorld.hpp"
#include "core/Recorder.hpp"

#include <cstdio>

using namespace dp;

namespace {
int g_failures = 0;
int g_checks   = 0;

void check(bool cond, const std::string& name) {
    ++g_checks;
    if (!cond) { ++g_failures; std::printf("  [FAIL] %s\n", name.c_str()); }
    else        std::printf("  [ ok ] %s\n", name.c_str());
}
void checkNear(double a, double b, double tol, const std::string& name) {
    check(std::abs(a - b) <= tol, name +
          " (got " + std::to_string(a) + " vs " + std::to_string(b) + ")");
}
} // namespace

// ---- 1. Angle math ----------------------------------------------------------
static void testAngleMath() {
    std::printf("[angle math]\n");
    checkNear(math::wrapAngle(math::kPi + 0.1), -(math::kPi - 0.1), 1e-9, "wrap above +pi");
    checkNear(math::wrapAngle(-math::kPi - 0.1), (math::kPi - 0.1), 1e-9, "wrap below -pi");
    checkNear(math::wrapAngle(0.0), 0.0, 1e-12, "wrap zero");
    checkNear(math::wrapAngle(4.0 * math::kTwoPi + 0.3), 0.3, 1e-9, "wrap many turns");
    checkNear(math::angleDifference(0.1, -0.1), 0.2, 1e-12, "angle difference");
}

// ---- 2. RNG determinism -----------------------------------------------------
static void testRngDeterminism() {
    std::printf("[rng determinism]\n");
    util::Rng a(12345), b(12345);
    bool same = true;
    for (int i = 0; i < 1000; ++i) {
        if (a.uniform01() != b.uniform01()) same = false;
        if (a.gaussian()  != b.gaussian())  same = false;
    }
    check(same, "same seed -> identical sequence");

    util::Rng c(999);
    double mean = 0.0, m2 = 0.0;
    const int N = 200000;
    for (int i = 0; i < N; ++i) { double x = c.gaussian(); mean += x; m2 += x * x; }
    mean /= N; m2 = m2 / N - mean * mean;
    checkNear(mean, 0.0, 0.02, "gaussian mean ~ 0");
    checkNear(m2,   1.0, 0.05, "gaussian variance ~ 1");
}

// ---- 3. Equations of motion -------------------------------------------------
static void testEquationsOfMotion() {
    std::printf("[equations of motion]\n");
    PhysicsConfig p; p.b1 = p.b2 = 0.0;
    DoublePendulum pend(p);

    // At the hanging equilibrium (theta=0, omega=0) with no torque, the angular
    // accelerations must be exactly zero.
    double a1, a2;
    pend.acceleration(State{0,0,0,0}, Action{0,0}, a1, a2);
    checkNear(a1, 0.0, 1e-12, "no accel at stable equilibrium (link1)");
    checkNear(a2, 0.0, 1e-12, "no accel at stable equilibrium (link2)");

    // Displaced to the right (small positive theta1), gravity must pull back
    // (negative angular acceleration on link 1).
    pend.acceleration(State{0.2, 0.0, 0, 0}, Action{0,0}, a1, a2);
    check(a1 < 0.0, "gravity restores displaced link1");

    // Positive torque at joint 1 must produce positive angular acceleration there.
    double t1, t2;
    pend.acceleration(State{0,0,0,0}, Action{1.0, 0.0}, t1, t2);
    check(t1 > 0.0, "positive torque -> positive accel");
}

// ---- 4. Integrator energy validation ---------------------------------------
static void testIntegratorEnergy() {
    std::printf("[integrator energy]\n");
    auto drift = [](IntegratorType integ, long long steps) {
        PhysicsConfig p; p.b1 = p.b2 = 0.0; p.integrator = integ;
        DoublePendulum pend(p);
        pend.setState(State{1.0, 0.5, 0, 0});
        const double e0 = pend.energy();
        double maxRel = 0.0;
        for (long long i = 0; i < steps; ++i) {
            pend.step(Action{0,0});
            maxRel = std::max(maxRel, std::abs((pend.energy() - e0) / (std::abs(e0)+1e-9)));
        }
        return maxRel;
    };
    const double rk4  = drift(IntegratorType::RK4, 240 * 20);   // 20 simulated sec
    const double eul  = drift(IntegratorType::SemiImplicitEuler, 240 * 20);
    std::printf("    RK4 drift=%.3e  Euler drift=%.3e\n", rk4, eul);
    check(rk4 < 1e-3, "RK4 conserves energy tightly over 20s");
    check(eul < 5e-2, "semi-implicit Euler stays bounded (symplectic)");
    check(rk4 < eul,  "RK4 drifts less than Euler");
}

// ---- 5. Environment / physics determinism -----------------------------------
static void testEnvDeterminism() {
    std::printf("[env determinism]\n");
    Config cfg; cfg.env.actuator = ActuatorMode::Single;
    DoublePendulumEnv e1(cfg), e2(cfg);
    e1.reset(7); e2.reset(7);
    util::Rng actionRng(55);
    bool same = true;
    for (int i = 0; i < 500; ++i) {
        std::vector<double> a{ actionRng.uniform(-1.0, 1.0) };
        auto r1 = e1.step(e1.decode(a));
        auto r2 = e2.step(e2.decode(a));
        if (r1.reward != r2.reward) same = false;
        for (std::size_t k = 0; k < r1.observation.size(); ++k)
            if (r1.observation[k] != r2.observation[k]) same = false;
    }
    check(same, "same seed + actions -> identical trajectory");
}

// ---- 6. GAE -----------------------------------------------------------------
static void testGAE() {
    std::printf("[gae]\n");
    rl::RolloutBuffer buf;
    // Three steps, all V=0, rewards 1,1,1, not done -> with lastValue 0 the
    // return at step t equals sum of discounted future rewards.
    for (int i = 0; i < 3; ++i) {
        rl::Transition t; t.reward = 1.0; t.value = 0.0; t.done = false;
        buf.add(t);
    }
    const double gamma = 0.9, lambda = 1.0;
    buf.computeGAE(gamma, lambda, 0.0);
    // return_0 = 1 + 0.9*1 + 0.81*1 = 2.71
    checkNear(buf[0].ret, 1.0 + 0.9 + 0.81, 1e-9, "GAE return discounting (t=0)");
    checkNear(buf[2].ret, 1.0, 1e-9, "GAE return last step");
    // With V=0, advantage == return when lambda=1.
    checkNear(buf[0].advantage, buf[0].ret, 1e-9, "advantage==return when V=0,lambda=1");

    // A terminal in the middle must cut the bootstrap.
    rl::RolloutBuffer buf2;
    rl::Transition a; a.reward = 1.0; a.value = 0.0; a.done = true; buf2.add(a);
    rl::Transition b; b.reward = 1.0; b.value = 0.0; b.done = false; buf2.add(b);
    buf2.computeGAE(0.9, 1.0, 5.0);
    checkNear(buf2[0].ret, 1.0, 1e-9, "terminal cuts bootstrap");
}

// ---- 7. Neural net determinism + backprop sanity ---------------------------
static void testNeuralNet() {
    std::printf("[neural net]\n");
    util::Rng r1(3), r2(3);
    rl::MLP n1(4, 8, 8, 2, r1), n2(4, 8, 8, 2, r2);
    std::vector<double> x{0.1, -0.2, 0.3, 0.4};
    auto y1 = n1.forward(x);
    auto y2 = n2.forward(x);
    check(y1.size() == 2 && y1[0] == y2[0] && y1[1] == y2[1], "identical init -> identical output");

    // Gradient check: reduce 0.5*||y||^2 with a few Adam steps; loss must drop.
    util::Rng r3(11);
    rl::MLP net(3, 16, 16, 1, r3);
    std::vector<double> in{0.5, -0.3, 0.8};
    auto lossOf = [&]{ auto y = net.forward(in); return 0.5 * y[0] * y[0]; };
    const double before = lossOf();
    for (int step = 1; step <= 50; ++step) {
        net.zeroGrad();
        auto y = net.forward(in);
        net.backward(std::vector<double>{ y[0] }); // d(0.5 y^2)/dy = y
        net.applyAdam(1e-2, 0.9, 0.999, 1e-8, step);
    }
    const double after = lossOf();
    std::printf("    loss %.5f -> %.5f\n", before, after);
    check(after < before, "Adam reduces a simple quadratic loss");
}

// ---- 8. SimWorld (decoupled simulation core) -------------------------------
static void testSimWorld() {
    std::printf("[sim world]\n");
    Config cfg;
    SimWorld world(cfg);
    bool ok = true;
    for (int i = 0; i < 500; ++i) {
        const SimFrame& f = world.step();           // policy-free free dynamics
        if (std::isnan(f.state.theta1) || std::isnan(f.energy)) ok = false;
    }
    check(ok, "SimWorld steps without producing NaNs");
    // A missing checkpoint must fail gracefully and leave the world policy-free.
    check(!world.loadPolicy("does_not_exist.ckpt"), "missing policy load fails cleanly");
    check(!world.hasPolicy(), "world remains policy-free after failed load");
}

// ---- 9. Recorder round-trip ------------------------------------------------
static void testRecorder() {
    std::printf("[recorder]\n");
    EpisodeRecorder rec;
    for (int i = 0; i < 50; ++i) {
        SimFrame f;
        f.state = State{0.1 * i, -0.2 * i, 0.3, -0.4};
        f.torque = Action{0.5 * i, -0.1 * i};
        f.reward = 0.01 * i; f.value = i; f.meanSigma = 0.37; f.energy = -1.5;
        rec.record(f);
    }
    const std::string path = "test_replay.dprec";
    check(rec.saveBinary(path), "recorder saves binary");
    EpisodeRecorder loaded;
    check(loaded.loadBinary(path), "recorder loads binary");
    check(loaded.size() == rec.size(), "round-trip frame count matches");
    bool exact = loaded.size() == rec.size();
    for (std::size_t i = 0; i < loaded.size() && exact; ++i) {
        if (loaded.at(i).theta1 != rec.at(i).theta1) exact = false;
        if (loaded.at(i).torque1 != rec.at(i).torque1) exact = false;
        if (loaded.at(i).reward != rec.at(i).reward) exact = false;
    }
    check(exact, "round-trip frames are bit-identical");
    checkNear(loaded.totalReturn(), rec.totalReturn(), 1e-12, "totalReturn preserved");
    std::remove(path.c_str());
}

int main() {
    std::printf("==== Double Pendulum RL test suite ====\n");
    testAngleMath();
    testRngDeterminism();
    testEquationsOfMotion();
    testIntegratorEnergy();
    testEnvDeterminism();
    testGAE();
    testNeuralNet();
    testSimWorld();
    testRecorder();
    std::printf("---------------------------------------\n");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
