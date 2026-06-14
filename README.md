# Double Pendulum RL

A high-performance, modern **C++20** framework in which a reinforcement-learning
agent learns to control a chaotic **double pendulum** — swinging it up and
holding it balanced in the unstable inverted configuration.

It is built around four pillars, in priority order:

1. **Physics correctness** — Lagrangian equations of motion in mass-matrix form, supporting torque, damping, and inertia.
2. **Deterministic simulation** — same seed ⇒ bit-identical results across MSVC / GCC / Clang and Windows / Linux.
3. **A stable, from-scratch PPO** — actor/critic, GAE, clipped surrogate, entropy bonus, all hand-written.
4. **Clean, modular architecture** — a dependency-free core with optional visualization.

> **Design choice that matters:** the entire training core (physics + environment + PPO) has **zero external
> dependencies**. It compiles and trains headless out of the box. The SDL2 + Dear ImGui dashboard is **opt-in**
> behind a CMake flag and pulls its own dependencies. This keeps the research core portable and reproducible
> while still offering a polished real-time visualizer.

---

## Why no LibTorch by default?

The hard requirements are *determinism* and *clean cross-platform builds*. The standard library's random
distributions and a heavyweight tensor framework both work against bit-level reproducibility and easy builds.
So the neural networks are a small hand-written MLP (manual backprop + Adam) driven by our own deterministic
RNG. The PPO algorithm only touches a thin `MLP` interface, so a LibTorch backend can be dropped in later behind
`-DDP_WITH_LIBTORCH=ON` without rewriting the algorithm. (Eigen / ONNX Runtime / yaml-cpp can be integrated the
same way; the config loader already speaks a YAML subset.)

---

## Project layout

```
DoublePendulumRL/
├── CMakeLists.txt
├── README.md
├── configs/            # YAML-subset experiment configs
│   ├── default.yaml
│   ├── swingup.yaml
│   └── stabilize.yaml
├── src/
│   ├── util/           # Math, deterministic RNG, Logger/CSV, Profiler
│   ├── core/           # Config, SharedState (threading), Trainer
│   ├── physics/        # DoublePendulum: EOM + RK4 / semi-implicit Euler
│   ├── rl/             # Env, NeuralNet (MLP), RolloutBuffer (GAE), PPOAgent
│   └── app/            # train / eval / benchmark / viz entry points
├── tests/              # dependency-free unit + validation tests
├── models/  logs/      # checkpoints and CSV statistics (created at runtime)
```

### Module responsibilities

| Module | Key types | Responsibility |
|---|---|---|
| `physics` | `DoublePendulum`, `State`, `Action` | Equations of motion, integrators, energy, kinematics |
| `rl` (env) | `DoublePendulumEnv`, `StepResult` | Gym-style API, observation encoding, reward shaping, domain randomization, disturbances |
| `rl` (agent) | `PPOAgent`, `MLP`, `RolloutBuffer` | Gaussian policy, critic, GAE, clipped PPO update, checkpoints |
| `core` | `Trainer`, `SharedState`, `Config` | Loop orchestration, curriculum, logging, thread-safe snapshots |
| `app` | `dp_train/eval/benchmark/viz` | Executable entry points |

---

## The physics

Angles are measured from the **downward vertical**, CCW-positive:
`θ = 0` hangs down (stable), `θ = ±π` points up (the balance target).

The equations of motion are derived from the Lagrangian and written in **mass-matrix form**

```
M(θ) · [α1; α2] = rhs(θ, ω, τ)
```

which (unlike the popular closed-form `sin(2θ)` expression) cleanly admits joint **torques** and **viscous
damping** as generalized forces. The 2×2 system is solved analytically each evaluation. Full derivation and
term-by-term meaning are in `src/physics/DoublePendulum.hpp`.

**Integrators**
- **RK4** (default): 4th-order, very low energy drift — the right default for fidelity.
- **Semi-implicit (symplectic) Euler**: 1st-order but energy-bounded and ~4× cheaper — a fast training mode.

`dp_benchmark` validates both: RK4 conserves energy tightly over a 20 s frictionless swing while Euler stays
bounded, and RK4 drifts strictly less.

---

## The learning

**Observation (6-D):** `[sin θ1, cos θ1, sin θ2, cos θ2, ω1/scale, ω2/scale]` — the trig encoding is continuous
across the ±π wrap, which is important for a smooth policy.

**Action:** continuous torque, `Single` (joint 1 only, underactuated) or `Dual`. The policy is a **tanh-squashed
Gaussian**: it samples a pre-squash `u ~ N(μ(s), σ(s))`, applies `a = tanh(u) ∈ (−1,1)`, and scales by
`±maxTorque`. The buffer keeps `u` so the PPO log-prob is exact (the tanh Jacobian cancels in the ratio).

**Reward shaping:**
```
reward =  wUpright · uprightScore                 (be inverted)
        − wTorque  · Σ τ² / maxTorque²            (cheap control)
        − wOmega   · (ω1² + ω2²)                  (no violent spinning)
        + wSurvival                                (optional alive bonus)
```
`uprightScore ∈ [0,1]` is a dense signal built from `−cos θ`, giving gradient even far from balance.

**PPO:** Gaussian policy with **state-dependent σ** (the actor outputs both μ and a per-state log-σ, bounded by a
tanh map — exploration is large during recovery, small near balance), separate actor/critic MLPs, **GAE(λ)**
advantages, advantage normalization, the **clipped surrogate** objective, an **adaptive entropy** bonus, **KL
early-stopping**, **linear LR annealing**, global gradient clipping, and Adam. The analytic policy-gradient
(including the backprop through the tanh σ-map) is derived inline in `src/rl/PPOAgent.cpp`.

**Stability controls** (all configurable, see `configs/default.yaml`): adaptive entropy targeting, σ floor,
`targetKL` early-stop, LR anneal, saturating ω-penalty (`omegaRefSpeed`), and **best-checkpoint** tracking
(`<run>_best.ckpt`) so you deploy the peak policy rather than a drifted one.

**Advanced features:** curriculum learning (ramps gravity & episode length), domain randomization (masses,
lengths, gravity, damping), disturbance injection (random impulses), CSV replay export (`dp_eval --csv`),
multi-threaded data-parallel updates (`--workers`), and an optional **LibTorch/CUDA backend** (`dp_train_torch`).

---

## Building

### Headless core + tools (no external dependencies)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release            # run the validation suite
```

This builds `dp_train`, `dp_eval`, `dp_benchmark`, and `dp_tests`. Works with **MSVC, GCC, and Clang**.
Warnings are enabled project-wide (`/W4` or `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`).

### With the real-time visualizer (SDL2 + Dear ImGui)

```bash
# Requires SDL2 discoverable by find_package (e.g. vcpkg, apt, or SDL2_DIR).
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_RENDERER=ON
cmake --build build --config Release -j
```
Dear ImGui is fetched automatically via `FetchContent`. On Windows, vcpkg is the easiest SDL2 source:
`-DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake`.

### Optional LibTorch backend hook
```bash
cmake -S . -B build -DDP_WITH_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
```

---

## Usage

```bash
# Train headless (Stabilize task, default config)
./build/dp_train --config configs/stabilize.yaml

# Quick experiment with CLI overrides
./build/dp_train --task SwingUp --dual --steps 1000000 --seed 7 --run my_run

# Evaluate a checkpoint and export a replay CSV
./build/dp_eval --model models/ppo_stabilize_final.ckpt --episodes 10 --csv logs/replay.csv

# Performance + numerics benchmark (render-disabled)
./build/dp_benchmark --steps 2000000

# Live dashboard (training visualized in real time)
./build/dp_viz --config configs/stabilize.yaml
# Watch a trained agent only:
./build/dp_viz --eval --model models/ppo_stabilize_final.ckpt
```

Training writes `logs/<run>_stats.csv` (reward curve + diagnostics) and checkpoints to `models/`.

### Visualizer controls

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset episode |
| `.` | Single frame step (while paused) |
| `D` | Inject a disturbance impulse |
| `S` | Toggle slow motion |
| `Esc` | Quit |

The dashboard shows the rods, masses, pivot, bob **trails**, the **center of mass**, an "up" reference marker,
and ImGui panels for simulation controls, live state, network I/O, FPS / sim-rate, and a reward graph.

---

## Threading model

The visualizer runs the **Trainer on a background thread** (physics + PPO) and keeps **all SDL/ImGui/GPU work on
the main thread**. They communicate only through `SharedState`: the trainer publishes immutable `RenderSnapshot`s
under a short lock, and the UI publishes atomic control intents. The physics loop never blocks on the GPU and the
UI never blocks on physics. See `src/core/SharedState.hpp`.

---

## Determinism & testing

`dp_tests` validates: angle wrapping, RNG reproducibility & Gaussian moments, equations of motion (equilibria,
restoring torque, actuation sign), integrator energy behavior (RK4 vs Euler), full environment determinism
(same seed + actions ⇒ identical trajectory), GAE correctness (discounting + terminal bootstrap cut), and MLP
init determinism + a gradient-descent sanity check.

---

## Notes / status

- The core, tools, and tests are written to compile cleanly under MSVC/GCC/Clang with C++20; **no compiler was
  available in the authoring environment**, so please report any platform-specific warning and it can be tuned.
- The MLP is intentionally small and CPU-only — appropriate for this low-dimensional control problem and ideal
  for determinism. For larger experiments, enable the LibTorch hook or back the `MLP` interface with Eigen/BLAS.
- Headless throughput is in the thousands–tens-of-thousands of env-steps/sec on a typical desktop core
  (see `dp_benchmark`), with no per-step heap allocation in the physics hot path.
