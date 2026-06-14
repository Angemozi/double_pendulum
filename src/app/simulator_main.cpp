// =============================================================================
// simulator_main.cpp  --  dp_simulator
// -----------------------------------------------------------------------------
// A polished, real-time double-pendulum sandbox built on raylib. It is a pure
// CONSUMER of dp_core: it loads a trained checkpoint, runs live inference via
// SimWorld, and renders. It shares no state with training -- dp_train and
// dp_simulator are fully separate executables.
//
// Loop design (single-threaded, frame-rate independent):
//   * raylib owns the window + 60 FPS render cadence.
//   * A fixed-timestep accumulator advances the physics deterministically: each
//     rendered frame consumes real elapsed time and runs floor(elapsed*speed/dt)
//     simulation steps, so the sim speed is decoupled from the render rate and a
//     speed multiplier / slow-motion just scales how many steps run per frame.
//   * All rendering happens after stepping, from the latest SimFrame.
// Physics is cheap (millions of steps/sec) so a single thread is plenty; the GPU
// is not involved (the CPU policy is tiny). See README for the threading notes.
//
// Build:  cmake -S . -B build-sim -DBUILD_SIMULATOR=ON && cmake --build build-sim
// Run:    dp_simulator --config configs/stabilize.yaml --model models/<run>_best.ckpt
// =============================================================================
#include <raylib.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "core/Config.hpp"
#include "core/SimWorld.hpp"
#include "core/Recorder.hpp"
#include "util/Logger.hpp"

using namespace dp;

namespace {

const char* argValue(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

constexpr float kPPM = 120.0f;     // pixels per meter at zoom 1
constexpr int   kHistory = 400;    // graph history length

// Derive the checkpoint base path (strip a known suffix) so keys 1/2/3 can load
// the best/latest/final variants of the same run.
std::string checkpointBase(const std::string& modelPath) {
    for (const char* suf : {"_best.ckpt", "_latest.ckpt", "_final.ckpt"}) {
        const std::size_t pos = modelPath.rfind(suf);
        if (pos != std::string::npos) return modelPath.substr(0, pos);
    }
    return modelPath;
}

// A scrolling line graph in screen-space rectangle (x,y,w,h).
void drawGraph(const std::deque<float>& data, float x, float y, float w, float h,
               Color color, const char* label) {
    DrawRectangleLines(static_cast<int>(x), static_cast<int>(y),
                       static_cast<int>(w), static_cast<int>(h), Fade(GRAY, 0.5f));
    DrawText(label, static_cast<int>(x) + 4, static_cast<int>(y) + 2, 10, color);
    if (data.size() < 2) return;
    float lo = data[0], hi = data[0];
    for (float v : data) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi - lo < 1e-6f) { hi = lo + 1.0f; }
    const float dx = w / static_cast<float>(kHistory - 1);
    for (std::size_t i = 1; i < data.size(); ++i) {
        auto map = [&](std::size_t idx, float v) {
            return Vector2{ x + dx * static_cast<float>(idx),
                            y + h - (v - lo) / (hi - lo) * h };
        };
        DrawLineV(map(i - 1, data[i - 1]), map(i, data[i]), color);
    }
    DrawText(TextFormat("%.2f", data.back()), static_cast<int>(x + w) - 44,
             static_cast<int>(y) + 2, 10, color);
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (const char* p = argValue(argc, argv, "--config")) cfg.loadFromFile(p);
    if (const char* s = argValue(argc, argv, "--task")) {
        if (std::strcmp(s, "SwingUp") == 0)   cfg.env.task = TaskType::SwingUp;
        if (std::strcmp(s, "Stabilize") == 0) cfg.env.task = TaskType::Stabilize;
        if (std::strcmp(s, "Recovery") == 0)  cfg.env.task = TaskType::Recovery;
    }
    const char* modelArg = argValue(argc, argv, "--model");
    const std::string base = modelArg ? checkpointBase(modelArg) : std::string();

    SimWorld world(cfg);
    if (modelArg) world.loadPolicy(modelArg);

    const int screenW = 1280, screenH = 800;
    InitWindow(screenW, screenH, "Double Pendulum RL -- Simulator");
    SetTargetFPS(60);

    Camera2D cam{};
    cam.offset = { screenW * 0.5f, screenH * 0.40f };
    cam.target = { 0.0f, 0.0f };
    cam.zoom   = 1.0f;

    bool   paused      = false;
    bool   frameStep   = false;
    double speed       = 1.0;      // simulation speed multiplier
    double accumulator = 0.0;      // fixed-timestep accumulator (seconds)
    const double dt    = cfg.physics.dt;

    std::deque<Vector2> trail;
    std::deque<float> hEntropy, hValue, hReward, hSigma;
    EpisodeRecorder recorder;
    bool recording = false;

    auto pushHistory = [&](const SimFrame& f) {
        auto push = [](std::deque<float>& d, float v) {
            d.push_back(v); if (d.size() > kHistory) d.pop_front();
        };
        push(hEntropy, static_cast<float>(f.entropy));
        push(hValue,   static_cast<float>(f.value));
        push(hReward,  static_cast<float>(f.reward));
        push(hSigma,   static_cast<float>(f.meanSigma));
    };

    while (!WindowShouldClose()) {
        // ---- Input ----------------------------------------------------------
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { world.reset(); trail.clear(); }
        if (IsKeyPressed(KEY_T)) world.setStochastic(!world.stochastic());
        if (IsKeyPressed(KEY_PERIOD)) frameStep = true;
        if (IsKeyPressed(KEY_C)) trail.clear();
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) speed = std::min(64.0, speed * 1.5);
        if (IsKeyPressed(KEY_LEFT_BRACKET))  speed = std::max(0.05, speed / 1.5);

        // Disturbance impulses.
        const double kick = 5.0;
        if (IsKeyPressed(KEY_LEFT))  world.applyImpulse(-kick, 0.0);
        if (IsKeyPressed(KEY_RIGHT)) world.applyImpulse(+kick, 0.0);
        if (IsKeyPressed(KEY_UP))    world.applyImpulse(0.0, +kick);
        if (IsKeyPressed(KEY_DOWN))  world.applyImpulse(0.0, -kick);
        if (IsKeyPressed(KEY_D)) world.applyImpulse(GetRandomValue(-60, 60) / 10.0,
                                                    GetRandomValue(-60, 60) / 10.0);

        // Checkpoint hot-loading.
        if (!base.empty()) {
            if (IsKeyPressed(KEY_ONE))   world.loadPolicy(base + "_best.ckpt");
            if (IsKeyPressed(KEY_TWO))   world.loadPolicy(base + "_latest.ckpt");
            if (IsKeyPressed(KEY_THREE)) world.loadPolicy(base + "_final.ckpt");
        }

        // Recording controls.
        if (IsKeyPressed(KEY_O)) { recording = !recording; if (recording) recorder.clear(); }
        if (IsKeyPressed(KEY_P) && !recorder.empty()) {
            recorder.saveBinary(cfg.train.modelDir + "/replay.dprec");
            recorder.saveCsv(cfg.train.logDir + "/replay.csv");
            DP_LOG_INFO("simulator: saved replay (%zu frames)", recorder.size());
        }

        // Camera: wheel zoom (about cursor), right-drag pan, HOME reset.
        const float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            const Vector2 mw = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.zoom = std::clamp(cam.zoom * (1.0f + wheel * 0.1f), 0.2f, 6.0f);
            const Vector2 mw2 = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.target.x += mw.x - mw2.x;
            cam.target.y += mw.y - mw2.y;
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const Vector2 d = GetMouseDelta();
            cam.target.x -= d.x / cam.zoom;
            cam.target.y -= d.y / cam.zoom;
        }
        if (IsKeyPressed(KEY_HOME)) { cam.target = {0, 0}; cam.zoom = 1.0f; }

        // ---- Fixed-timestep simulation -------------------------------------
        if (!paused) {
            accumulator += GetFrameTime() * speed;
            int budget = 4000; // cap to avoid a spiral of death after a hitch
            while (accumulator >= dt && budget-- > 0) {
                const SimFrame& f = world.step();
                accumulator -= dt;
                trail.push_back(Vector2{ static_cast<float>(f.kinematics.bob2X * kPPM),
                                         static_cast<float>(f.kinematics.bob2Y * kPPM) });
                if (trail.size() > 600) trail.pop_front();
                if (recording) recorder.record(f);
                pushHistory(f);
            }
        } else if (frameStep) {
            const SimFrame& f = world.step();
            trail.push_back(Vector2{ static_cast<float>(f.kinematics.bob2X * kPPM),
                                     static_cast<float>(f.kinematics.bob2Y * kPPM) });
            if (recording) recorder.record(f);
            pushHistory(f);
        }
        frameStep = false;

        const SimFrame& f = world.last();

        // ---- Render ---------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 16, 16, 22, 255 });

        BeginMode2D(cam);
        // "Up" target marker (balance goal) above the pivot.
        const float reach = static_cast<float>((cfg.physics.l1 + cfg.physics.l2) * kPPM);
        DrawLineEx({ -22, -reach }, { 22, -reach }, 2, Fade(GREEN, 0.4f));

        // Fading trajectory trail of the tip.
        for (std::size_t i = 1; i < trail.size(); ++i) {
            const float a = static_cast<float>(i) / static_cast<float>(trail.size());
            DrawLineEx(trail[i - 1], trail[i], 2.0f, Fade(SKYBLUE, a * 0.7f));
        }

        const Vector2 pivot { 0, 0 };
        const Vector2 joint { static_cast<float>(f.kinematics.jointX * kPPM),
                              static_cast<float>(f.kinematics.jointY * kPPM) };
        const Vector2 tip   { static_cast<float>(f.kinematics.bob2X * kPPM),
                              static_cast<float>(f.kinematics.bob2Y * kPPM) };
        const Vector2 com   { static_cast<float>(f.kinematics.comX * kPPM),
                              static_cast<float>(f.kinematics.comY * kPPM) };

        DrawLineEx(pivot, joint, 5.0f, Color{ 230, 230, 240, 255 });
        DrawLineEx(joint, tip,   5.0f, Color{ 230, 230, 240, 255 });
        DrawCircleV(pivot, 6, GRAY);
        DrawCircleV(joint, 13, Color{ 90, 200, 250, 255 });
        DrawCircleV(tip,   13, Color{ 250, 150, 90, 255 });
        DrawCircleV(com,   5,  Color{ 120, 250, 120, 255 });
        EndMode2D();

        // ---- HUD ------------------------------------------------------------
        const int x = 14; int y = 12; const int lh = 18;
        auto line = [&](const char* txt, Color c = RAYWHITE) {
            DrawText(txt, x, y, 16, c); y += lh;
        };
        DrawRectangle(0, 0, 320, 300, Fade(BLACK, 0.45f));
        line(TextFormat("FPS %d   speed %.2fx%s", GetFPS(), speed, paused ? "  [PAUSED]" : ""),
             paused ? YELLOW : RAYWHITE);
        line(TextFormat("policy: %s", world.hasPolicy() ? world.policyName().c_str() : "<none>"),
             world.hasPolicy() ? GREEN : ORANGE);
        line(TextFormat("mode: %s", world.stochastic() ? "stochastic" : "deterministic"));
        line(TextFormat("episode step %d   return %.1f", f.episodeStep, f.episodeReturn));
        line(TextFormat("reward     %+.3f", f.reward));
        line(TextFormat("torque     %+.2f  %+.2f", f.torque.torque1, f.torque.torque2));
        line(TextFormat("value V(s) %+.3f", f.value));
        line(TextFormat("entropy    %+.3f", f.entropy));
        line(TextFormat("sigma      %.3f", f.meanSigma));
        line(TextFormat("upright    %.3f", f.uprightScore));
        line(TextFormat("energy     %+.3f", f.energy), Fade(RAYWHITE, 0.8f));
        if (recording) line("RECORDING", RED);

        // Live graphs (bottom-left).
        const float gy = screenH - 96.0f;
        drawGraph(hReward,  14,  gy, 280, 80, GREEN,   "reward");
        drawGraph(hValue,   312, gy, 280, 80, SKYBLUE, "value");
        drawGraph(hEntropy, 610, gy, 280, 80, ORANGE,  "entropy");
        drawGraph(hSigma,   908, gy, 280, 80, VIOLET,  "sigma");

        // Controls cheatsheet (top-right).
        const int cx = screenW - 260; int cy = 12;
        DrawRectangle(cx - 10, 0, 270, 230, Fade(BLACK, 0.4f));
        auto ctl = [&](const char* t){ DrawText(t, cx, cy, 13, Fade(RAYWHITE, 0.8f)); cy += 15; };
        ctl("SPACE pause   R reset   . step");
        ctl("T stochastic/deterministic");
        ctl("ARROWS impulse   D random kick");
        ctl("[ / ] slower / faster");
        ctl("wheel zoom   R-drag pan   HOME");
        ctl("1/2/3 load best/latest/final");
        ctl("O record   P save replay   C clear");
        ctl("ESC quit");

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
