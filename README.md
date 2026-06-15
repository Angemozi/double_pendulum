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
> dependencies**. It compiles and trains headless out of the box. The visualizers (raylib simulator, SDL2 + Dear
> ImGui dashboard) are **opt-in** behind CMake flags and pull their own dependencies. This keeps the research core
> portable and reproducible while still offering polished real-time tooling.

### Build & test status

The headless core, tools, and tests build **warning-clean** with the project-wide strict flag set
(`-Wall -Wextra -Wpedantic -Wshadow -Wconversion` / `/W4 /permissive-`). Verified with **GCC 13.3** on Linux:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j   # 0 warnings, 0 errors
ctest --test-dir build                                                     # 31/31 checks pass
```

Representative numbers from `dp_benchmark` / a short `dp_train` on a desktop core:

| Metric | Value |
|---|---|
| RK4 energy drift (20 s frictionless) | ~2e-4 relative |
| Semi-implicit Euler drift | bounded (symplectic), strictly larger than RK4 |
| Physics throughput | ~7M (RK4) / ~17M (Euler) steps/s |
| Environment throughput | ~3M steps/s |
| PPO training (Stabilize) | learns to hold balance in well under 1M steps |

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
│   ├── stabilize.yaml
│   └── recovery.yaml
├── src/
│   ├── util/           # Math, deterministic RNG, Logger/CSV, ThreadPool, Profiler
│   ├── core/           # Config, SharedState (threading), Trainer, SimWorld, Recorder
│   ├── physics/        # DoublePendulum: EOM + RK4 / semi-implicit Euler
│   ├── rl/             # Env, NeuralNet (MLP), RolloutBuffer (GAE), PPOAgent
│   └── app/            # train / eval / benchmark / simulator / viz entry points
├── tests/              # dependency-free unit + validation tests
├── models/  logs/      # checkpoints and CSV statistics (created at runtime)
```

### Module responsibilities

| Module | Key types | Responsibility |
|---|---|---|
| `physics` | `DoublePendulum`, `State`, `Action` | Equations of motion, integrators, energy, kinematics |
| `rl` (env) | `DoublePendulumEnv`, `StepResult` | Gym-style API, observation encoding, reward shaping, domain randomization, disturbances |
| `rl` (agent) | `PPOAgent`, `MLP`, `RolloutBuffer` | Gaussian policy, critic, GAE, clipped PPO update, checkpoints |
| `core` | `Trainer`, `SharedState`, `Config`, `SimWorld`, `Recorder` | Loop orchestration, curriculum, logging, thread-safe snapshots, inference sandbox, replay |
| `app` | `dp_train/eval/benchmark/simulator/viz` | Executable entry points |

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
- **Semi-implicit (symplectic) Euler**: 1st-order but energy-bounded and ~2–4× cheaper — a fast training mode.

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
        − wOmega   · ω-penalty                     (no violent spinning; saturating)
        + wSurvival                                (optional alive bonus)
```
`uprightScore ∈ [0,1]` is a dense signal built from `−cos θ`, giving gradient even far from balance.

**PPO:** Gaussian policy with **state-dependent σ** (the actor outputs both μ and a per-state log-σ, bounded by a
tanh map — exploration is large during recovery, small near balance), separate actor/critic MLPs, **GAE(λ)**
advantages, advantage normalization, the **clipped surrogate** objective, an **adaptive entropy** bonus, **KL
early-stopping**, **linear LR annealing**, global gradient clipping, and Adam. The analytic policy-gradient
(including the backprop through the tanh σ-map) is derived inline in `src/rl/PPOAgent.cpp`.

**GAE with correct episode boundaries.** The buffer can hold several episodes per rollout, so the advantage
recursion must reset at every boundary. A **true terminal** (failure / NaN / over-speed) cuts the bootstrap to
zero. A **time-limit truncation** is *not* a true terminal: the episode would have continued, so GAE bootstraps
from the value of the real next state (`V(s_{t+1})`, captured before the env resets) and the next episode's
advantage is **not** allowed to leak back across the boundary. Both behaviors are unit-tested
(`tests/test_main.cpp` → `[gae]`).

**Stability controls** (all configurable, see `configs/default.yaml`): adaptive entropy targeting, σ floor,
`targetKL` early-stop, LR anneal, saturating ω-penalty (`omegaRefSpeed`), and **best-checkpoint** tracking
(`<run>_best.ckpt`) so you deploy the peak policy rather than a drifted one.

**Advanced features:** curriculum learning (ramps gravity, episode length, **and disturbance intensity**),
domain randomization (masses, lengths, gravity, damping/friction, **timestep**), disturbance injection,
**energy-aware reward shaping** (`wEnergy`, swing-up aid), CSV replay export, multi-threaded data-parallel
updates (`--workers`), **vectorized environments** (`numEnvs` — decorrelated per-env GAE), and an optional
**LibTorch/CUDA backend** (`dp_train_torch`).

**Platform tooling (`dp_simulator`):** self-describing checkpoints (auto-loads the run's sidecar config),
**live hot-reload** of a training checkpoint, **policy/value/σ heatmaps** over state space, and an automatic
**episode library** that captures and replays best / failure / recovery episodes.

---

## Building

### Headless core + tools (no external dependencies)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
ctest --test-dir build -C Release            # run the validation suite
```

This builds `dp_train`, `dp_eval`, `dp_benchmark`, and `dp_tests`. Works with **MSVC, GCC, and Clang**.
Warnings are enabled project-wide (`/W4` or `-Wall -Wextra -Wpedantic -Wshadow -Wconversion`) and the core
builds **clean** under them.

### Real-time simulator (raylib) — recommended

```bash
cmake -S . -B build-sim -DBUILD_SIMULATOR=ON
cmake --build build-sim --config Release
./build-sim/dp_simulator --config configs/stabilize.yaml --model models/<run>_best.ckpt
```
raylib is fetched and built automatically (statically linked — no system packages to ship). On Linux, raylib's
GLFW backend needs the usual X11 development headers present (e.g. `libxrandr-dev`, `libxinerama-dev`,
`libxcursor-dev`, `libxi-dev`, `libgl1-mesa-dev`); install them via your package manager if CMake reports a
missing `RandR`/X11 header. `dp_simulator` is a standalone consumer of `dp_core`: it loads a checkpoint and runs
**live inference** with real-time rendering, fully separate from training. **Simulator controls:**

| Key | Action | Key | Action |
|---|---|---|---|
| `Space` | pause/resume | `[` / `]` | slower / faster |
| `R` | reset episode | wheel | zoom (about cursor) |
| `.` | single frame step | right-drag | pan |
| `T` | stochastic ↔ deterministic | `Home` | reset camera |
| `←/→/↑/↓` | impulse disturbance | `1/2/3` | load best/latest/final |
| `D` | random kick | `O` / `P` | record / save replay |
| `F` | follow-latest (hot-reload) | `H` | cycle policy/value/σ heatmap |
| `C` | clear trail | `L` | cycle library playback (best/fail/recovery) |

The HUD shows reward, torque, value V(s), entropy, σ, upright score, and energy, with live graphs and a fading
trajectory trail of the tip. Checkpoints are **self-describing**, so `dp_simulator --model models/<run>_best.ckpt`
needs no `--config`. Point `F` (follow-latest) at `<run>_latest.ckpt` to watch a live training run improve in real
time; `H` overlays a policy/value/σ heatmap swept over (θ₁, θ₂); the simulator auto-captures best/failure/recovery
episodes for `L` playback.

### With the legacy visualizer (SDL2 + Dear ImGui)

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
```

CLI flags accepted by `dp_train`: `--config`, `--steps`, `--seed`, `--run`, `--workers`,
`--task {SwingUp|Stabilize|Recovery}`, `--dual`, `--verbose`. CLI flags override the file/defaults.

Training writes `logs/<run>_stats.csv` (reward curve + diagnostics) and checkpoints to `models/`
(`<run>_best.ckpt`, `<run>_latest.ckpt`, `<run>_final.ckpt`, plus a self-describing `<run>_config.yaml`).

If the SDL2 visualizer is built (`-DBUILD_RENDERER=ON`), `dp_viz` runs training visualized in real time:

```bash
./build/dp_viz --config configs/stabilize.yaml
./build/dp_viz --eval --model models/ppo_stabilize_final.ckpt   # watch a trained agent only
```

| Key | Action |
|---|---|
| `Space` | Pause / resume |
| `R` | Reset episode |
| `.` | Single frame step (while paused) |
| `D` | Inject a disturbance impulse |
| `S` | Toggle slow motion |
| `Esc` | Quit |

---

## Threading model

The visualizer runs the **Trainer on a background thread** (physics + PPO) and keeps **all rendering/UI work on
the main thread**. They communicate only through `SharedState`: the trainer publishes immutable `RenderSnapshot`s
under a short lock, and the UI publishes atomic control intents. The physics loop never blocks on the GPU and the
UI never blocks on physics. See `src/core/SharedState.hpp`.

The PPO update itself is **data-parallel** across CPU cores via a persistent `ThreadPool`: each worker computes
gradients on a fixed, contiguous shard of the minibatch and the master reduces them in worker-id order. Results
are bit-reproducible **for a fixed worker count** (`numWorkers: 1` reproduces the serial path exactly; the
default `0` = auto uses all cores and is reproducible per machine).

---

## Determinism & testing

`dp_tests` (31 checks) validates: angle wrapping, RNG reproducibility & Gaussian moments, equations of motion
(equilibria, restoring torque, actuation sign), integrator energy behavior (RK4 vs Euler), full environment
determinism (same seed + actions ⇒ identical trajectory), GAE correctness (discounting, **terminal bootstrap
cut, and truncation bootstrap without cross-episode leak**), MLP init determinism + a gradient-descent sanity
check, the `SimWorld` inference sandbox, and `Recorder` binary replay round-trips. The suite exits non-zero on
any failure so it can gate CI.

---

## Notes / status

- The core, tools, and tests build **warning-clean** under MSVC/GCC/Clang with C++20 and the strict flag set;
  verified with GCC 13.3 on Linux. Please report any platform-specific warning so it can be tuned.
- The MLP is intentionally small and CPU-only — appropriate for this low-dimensional control problem and ideal
  for determinism. For larger experiments, enable the LibTorch hook or back the `MLP` interface with Eigen/BLAS.
- A few config knobs are honored only by specific backends: `ppo.initLogStd` and `ppo.valueCoef` apply to the
  LibTorch backend (`dp_train_torch`); in the native core the policy std is a bounded network output, so the
  initial σ is set by the tanh-map midpoint of `[minLogStd, 1]` rather than by `initLogStd`.
- Headless throughput is in the millions of physics-steps/sec and thousands–tens-of-thousands of *training*
  env-steps/sec on a typical desktop core (see `dp_benchmark`), with no per-step heap allocation in the physics
  hot path.
</content>
