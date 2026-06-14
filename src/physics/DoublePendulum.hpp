// =============================================================================
// DoublePendulum.hpp
// -----------------------------------------------------------------------------
// Physically accurate double pendulum with configurable masses, lengths,
// inertia, gravity, viscous damping, and joint torques.
//
// STATE & ACTION (the RL-facing contract requested in the spec):
//
//     struct State  { theta1, theta2, omega1, omega2 }      // generalized coords
//     struct Action { torque1, torque2 }                    // generalized forces
//
// ANGLE CONVENTION:
//     theta is measured from the DOWNWARD vertical, increasing counter-clockwise.
//     theta = 0      -> bob hanging straight down (stable equilibrium)
//     theta = +-pi   -> bob pointing straight up   (unstable equilibrium, the
//                       configuration the agent must learn to balance at)
//
// EQUATIONS OF MOTION (derivation):
//     Using the Lagrangian L = T - V with generalized coordinates (theta1,theta2)
//     yields a coupled system that is linear in the angular accelerations. We
//     write it in mass-matrix form
//
//         M(theta) * [a1; a2] = rhs(theta, omega, torque)
//
//     where (point masses at the rod tips, plus optional extra inertia Ii):
//
//         M11 = (m1+m2) L1^2 + I1
//         M22 =  m2     L2^2 + I2
//         M12 = M21 = m2 L1 L2 cos(theta1 - theta2)
//
//         rhs1 = -(m1+m2) g L1 sin(theta1)
//                - m2 L1 L2 omega2^2 sin(theta1-theta2)
//                + (tau1 - b1*omega1)
//
//         rhs2 = -m2 g L2 sin(theta2)
//                + m2 L1 L2 omega1^2 sin(theta1-theta2)
//                + (tau2 - b2*omega2)
//
//     Solving the 2x2 system analytically (Cramer's rule) gives the angular
//     accelerations. This mass-matrix form (rather than the common explicit
//     "sin(2*theta)" closed form) is used precisely because it admits joint
//     TORQUES and DAMPING as generalized forces in a physically correct way.
// =============================================================================
#pragma once

#include "core/Config.hpp"

namespace dp {

// ------- RL-facing data contracts (exact names requested by the spec) --------
struct State {
    double theta1 = 0.0;
    double theta2 = 0.0;
    double omega1 = 0.0;
    double omega2 = 0.0;
};

struct Action {
    double torque1 = 0.0;
    double torque2 = 0.0;
};

// Cartesian positions of pivot, joint, and the two bobs (for rendering / COM).
struct Kinematics {
    double pivotX = 0.0, pivotY = 0.0;   // fixed origin
    double jointX = 0.0, jointY = 0.0;   // end of rod 1 / start of rod 2
    double bob1X  = 0.0, bob1Y  = 0.0;   // mass 1
    double bob2X  = 0.0, bob2Y  = 0.0;   // mass 2
    double comX   = 0.0, comY   = 0.0;   // system center of mass
};

class DoublePendulum {
public:
    explicit DoublePendulum(const PhysicsConfig& cfg) : cfg_(cfg) {}

    // Access / mutate physical parameters (used by domain randomization &
    // curriculum, which scale gravity at runtime).
    const PhysicsConfig& params() const noexcept { return cfg_; }
    PhysicsConfig&       params()       noexcept { return cfg_; }

    void  setState(const State& s) noexcept { state_ = s; }
    const State& state() const noexcept { return state_; }

    // Advance the simulation by exactly one fixed timestep dt using the
    // configured integrator. `action` torques are clamped to +-maxTorque.
    void step(const Action& action) noexcept;

    // Compute angular accelerations from the equations of motion. Exposed so the
    // integrators and unit tests can probe the continuous-time dynamics directly.
    // Returns {alpha1, alpha2}.
    void acceleration(const State& s, const Action& a,
                      double& alpha1, double& alpha2) const noexcept;

    // Total mechanical energy E = T + V. With damping=0 and torque=0 this is an
    // invariant; the integrator validation tests assert it stays bounded.
    double energy() const noexcept { return energy(state_); }
    double energy(const State& s) const noexcept;

    // Cartesian layout of the mechanism for the given state (defaults to current).
    Kinematics kinematics() const noexcept { return kinematics(state_); }
    Kinematics kinematics(const State& s) const noexcept;

private:
    // Clamp torques to the actuator saturation limit.
    Action clampTorque(const Action& a) const noexcept;

    // The two integrator implementations. Both consume a *pre-clamped* action.
    void integrateRK4(const Action& a) noexcept;
    void integrateSemiImplicitEuler(const Action& a) noexcept;

    PhysicsConfig cfg_;
    State         state_;
};

} // namespace dp
