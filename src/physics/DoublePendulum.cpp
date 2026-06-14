// =============================================================================
// DoublePendulum.cpp
// -----------------------------------------------------------------------------
// Implementation of the equations of motion and the two integrators.
//
// Numerical integration overview:
//   The continuous dynamics are a first-order ODE in the 4-vector
//       y = [theta1, theta2, omega1, omega2]
//       dy/dt = f(y) = [omega1, omega2, alpha1, alpha2]
//   where (alpha1, alpha2) come from solving the mass-matrix system.
//
//   * RK4 (default): classic 4th-order Runge-Kutta. O(dt^4) local error, very
//     low energy drift for a chaotic system at dt = 1/240 s. This is the right
//     default for physical fidelity.
//
//   * Semi-implicit (symplectic) Euler: update velocity first, then advance
//     position with the NEW velocity. O(dt) accuracy but excellent long-term
//     energy behavior (bounded, oscillating error rather than secular drift) and
//     ~4x cheaper than RK4. Offered as a fast mode for bulk training throughput.
// =============================================================================
#include "physics/DoublePendulum.hpp"
#include "util/Math.hpp"

#include <cmath>

namespace dp {

using math::sq;

Action DoublePendulum::clampTorque(const Action& a) const noexcept {
    const double lim = cfg_.maxTorque;
    return Action{ math::clamp(a.torque1, -lim, lim),
                   math::clamp(a.torque2, -lim, lim) };
}

void DoublePendulum::acceleration(const State& s, const Action& a,
                                  double& alpha1, double& alpha2) const noexcept {
    const double m1 = cfg_.m1, m2 = cfg_.m2;
    const double l1 = cfg_.l1, l2 = cfg_.l2;
    const double g  = cfg_.g;
    const double delta = s.theta1 - s.theta2;
    const double cosD  = std::cos(delta);
    const double sinD  = std::sin(delta);

    // Mass matrix M (symmetric positive-definite for physical parameters).
    const double M11 = (m1 + m2) * sq(l1) + cfg_.i1;
    const double M22 =  m2       * sq(l2) + cfg_.i2;
    const double M12 =  m2 * l1 * l2 * cosD;

    // Generalized forces: gravity + coupling (Coriolis/centrifugal) + actuation
    // + viscous damping. See header derivation for term-by-term meaning.
    const double rhs1 = -(m1 + m2) * g * l1 * std::sin(s.theta1)
                        - m2 * l1 * l2 * sq(s.omega2) * sinD
                        + (a.torque1 - cfg_.b1 * s.omega1);

    const double rhs2 = -m2 * g * l2 * std::sin(s.theta2)
                        + m2 * l1 * l2 * sq(s.omega1) * sinD
                        + (a.torque2 - cfg_.b2 * s.omega2);

    // Solve M * [alpha1; alpha2] = [rhs1; rhs2] via Cramer's rule.
    const double det = M11 * M22 - M12 * M12;
    // det is strictly positive for valid masses/lengths; guard only against
    // pathological zero-mass configs to avoid producing NaNs.
    const double invDet = (std::abs(det) > 1e-12) ? (1.0 / det) : 0.0;

    alpha1 = (rhs1 * M22 - rhs2 * M12) * invDet;
    alpha2 = (rhs2 * M11 - rhs1 * M12) * invDet;
}

void DoublePendulum::step(const Action& action) noexcept {
    const Action a = clampTorque(action);
    switch (cfg_.integrator) {
        case IntegratorType::RK4:               integrateRK4(a); break;
        case IntegratorType::SemiImplicitEuler: integrateSemiImplicitEuler(a); break;
    }
    // Keep the stored angles in the canonical [-pi, pi] range so they never
    // accumulate unbounded magnitude across a long episode.
    state_.theta1 = math::wrapAngle(state_.theta1);
    state_.theta2 = math::wrapAngle(state_.theta2);
}

void DoublePendulum::integrateRK4(const Action& a) noexcept {
    const double dt = cfg_.dt;

    // f(state) -> derivative state. Torque is held constant over the step
    // (zero-order hold), which matches how the agent applies a discrete action.
    auto deriv = [&](const State& s) -> State {
        double al1, al2;
        acceleration(s, a, al1, al2);
        return State{ s.omega1, s.omega2, al1, al2 };
    };
    auto add = [](const State& s, const State& d, double h) -> State {
        return State{ s.theta1 + d.theta1 * h, s.theta2 + d.theta2 * h,
                      s.omega1 + d.omega1 * h, s.omega2 + d.omega2 * h };
    };

    const State k1 = deriv(state_);
    const State k2 = deriv(add(state_, k1, dt * 0.5));
    const State k3 = deriv(add(state_, k2, dt * 0.5));
    const State k4 = deriv(add(state_, k3, dt));

    const double sixth = dt / 6.0;
    state_.theta1 += sixth * (k1.theta1 + 2.0 * k2.theta1 + 2.0 * k3.theta1 + k4.theta1);
    state_.theta2 += sixth * (k1.theta2 + 2.0 * k2.theta2 + 2.0 * k3.theta2 + k4.theta2);
    state_.omega1 += sixth * (k1.omega1 + 2.0 * k2.omega1 + 2.0 * k3.omega1 + k4.omega1);
    state_.omega2 += sixth * (k1.omega2 + 2.0 * k2.omega2 + 2.0 * k3.omega2 + k4.omega2);
}

void DoublePendulum::integrateSemiImplicitEuler(const Action& a) noexcept {
    const double dt = cfg_.dt;
    double al1, al2;
    acceleration(state_, a, al1, al2);

    // Update velocities first (using accelerations at the current position)...
    state_.omega1 += al1 * dt;
    state_.omega2 += al2 * dt;
    // ...then update positions with the *new* velocities. This ordering is what
    // makes the scheme symplectic and energy-stable over long horizons.
    state_.theta1 += state_.omega1 * dt;
    state_.theta2 += state_.omega2 * dt;
}

double DoublePendulum::energy(const State& s) const noexcept {
    const double m1 = cfg_.m1, m2 = cfg_.m2;
    const double l1 = cfg_.l1, l2 = cfg_.l2;
    const double g  = cfg_.g;

    // Kinetic energy. v1^2 = (L1 w1)^2.
    // v2^2 = (L1 w1)^2 + (L2 w2)^2 + 2 L1 L2 w1 w2 cos(theta1 - theta2).
    const double v1sq = sq(l1 * s.omega1);
    const double v2sq = sq(l1 * s.omega1) + sq(l2 * s.omega2)
                      + 2.0 * l1 * l2 * s.omega1 * s.omega2
                        * std::cos(s.theta1 - s.theta2);
    const double T = 0.5 * m1 * v1sq + 0.5 * m2 * v2sq
                   + 0.5 * cfg_.i1 * sq(s.omega1) + 0.5 * cfg_.i2 * sq(s.omega2);

    // Potential energy with the pivot as the zero reference (y positive up).
    // y1 = -L1 cos(theta1), y2 = y1 - L2 cos(theta2).
    const double y1 = -l1 * std::cos(s.theta1);
    const double y2 = y1 - l2 * std::cos(s.theta2);
    const double V = m1 * g * y1 + m2 * g * y2;

    return T + V;
}

Kinematics DoublePendulum::kinematics(const State& s) const noexcept {
    const double l1 = cfg_.l1, l2 = cfg_.l2;
    Kinematics k;
    k.pivotX = 0.0; k.pivotY = 0.0;
    // x increases to the right, y increases downward in *world* terms; we keep
    // y positive-down here for direct screen mapping and let the renderer flip.
    k.jointX = l1 * std::sin(s.theta1);
    k.jointY = l1 * std::cos(s.theta1);
    k.bob1X  = k.jointX;
    k.bob1Y  = k.jointY;
    k.bob2X  = k.jointX + l2 * std::sin(s.theta2);
    k.bob2Y  = k.jointY + l2 * std::cos(s.theta2);

    const double mTotal = cfg_.m1 + cfg_.m2;
    k.comX = (cfg_.m1 * k.bob1X + cfg_.m2 * k.bob2X) / mTotal;
    k.comY = (cfg_.m1 * k.bob1Y + cfg_.m2 * k.bob2Y) / mTotal;
    return k;
}

} // namespace dp
