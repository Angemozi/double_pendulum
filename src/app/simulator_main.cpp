// =============================================================================
// simulator_main.cpp  --  dp_simulator
// -----------------------------------------------------------------------------
// A polished, real-time double-pendulum sandbox built on raylib. Pure CONSUMER
// of dp_core: it loads a trained checkpoint, runs live inference via SimWorld,
// and renders. Separate executable from training; they communicate only through
// checkpoint files on disk.
//
// Features:
//   * Self-describing checkpoints  -- auto-loads "<run>_config.yaml" next to the
//     checkpoint so geometry/network match the trained model with no --config.
//   * Live hot-reload (F)          -- watches the checkpoint file and reloads
//     when training overwrites it; turns the sandbox into a live training view.
//   * Policy/Value/Sigma heatmaps (H) -- sweeps state space via SimWorld::probe.
//   * Episode library              -- auto-records best/failure/recovery episodes
//     and plays them back (L) from disk; manual record (O) + save (P).
//
// Loop: fixed-timestep accumulator (deterministic physics, frame-rate
// independent), single thread, CPU-only (the policy is tiny). The GPU is not
// involved; raylib handles the 60 FPS render cadence.
// =============================================================================
#include <raylib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include "core/Config.hpp"
#include "core/SimWorld.hpp"
#include "core/Recorder.hpp"
#include "util/Logger.hpp"

using namespace dp;
namespace fs = std::filesystem;

namespace {

const char* argValue(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}
bool hasFlag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

constexpr float kPPM = 120.0f;     // pixels per meter at zoom 1
constexpr int   kHistory = 400;    // graph history length
constexpr int   kHeatRes = 48;     // heatmap grid resolution

std::string checkpointBase(const std::string& modelPath) {
    for (const char* suf : {"_best.ckpt", "_latest.ckpt", "_final.ckpt"}) {
        const std::size_t pos = modelPath.rfind(suf);
        if (pos != std::string::npos) return modelPath.substr(0, pos);
    }
    const std::size_t dot = modelPath.rfind(".ckpt");
    return dot != std::string::npos ? modelPath.substr(0, dot) : modelPath;
}

// Blue -> cyan -> green -> yellow -> red ramp for t in [0,1].
Color heatColor(float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    const std::array<Color, 5> stops{ Color{30,60,200,255}, Color{0,200,200,255},
        Color{40,200,40,255}, Color{240,220,40,255}, Color{230,60,40,255} };
    const float s = t * 4.0f;
    const int i = std::min(3, static_cast<int>(s));
    const float f = s - i;
    auto lerp = [&](unsigned char a, unsigned char b){
        return static_cast<unsigned char>(a + (b - a) * f); };
    return Color{ lerp(stops[i].r, stops[i+1].r), lerp(stops[i].g, stops[i+1].g),
                  lerp(stops[i].b, stops[i+1].b), 180 };
}

// World-space positions of the two bobs from raw angles (matches kinematics()).
void framePositions(double th1, double th2, double l1, double l2,
                    Vector2& joint, Vector2& tip) {
    const double jx = l1 * std::sin(th1), jy = l1 * std::cos(th1);
    joint = { static_cast<float>(jx * kPPM), static_cast<float>(jy * kPPM) };
    tip   = { static_cast<float>((jx + l2 * std::sin(th2)) * kPPM),
              static_cast<float>((jy + l2 * std::cos(th2)) * kPPM) };
}

void drawGraph(const std::deque<float>& data, float x, float y, float w, float h,
               Color color, const char* label) {
    DrawRectangleLines((int)x, (int)y, (int)w, (int)h, Fade(GRAY, 0.5f));
    DrawText(label, (int)x + 4, (int)y + 2, 10, color);
    if (data.size() < 2) return;
    float lo = data[0], hi = data[0];
    for (float v : data) { lo = std::min(lo, v); hi = std::max(hi, v); }
    if (hi - lo < 1e-6f) hi = lo + 1.0f;
    const float dx = w / static_cast<float>(kHistory - 1);
    for (std::size_t i = 1; i < data.size(); ++i) {
        auto map = [&](std::size_t idx, float v) {
            return Vector2{ x + dx * idx, y + h - (v - lo) / (hi - lo) * h }; };
        DrawLineV(map(i - 1, data[i - 1]), map(i, data[i]), color);
    }
    DrawText(TextFormat("%.2f", data.back()), (int)(x + w) - 44, (int)y + 2, 10, color);
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    const char* configArg = argValue(argc, argv, "--config");
    const char* modelArg  = argValue(argc, argv, "--model");
    const std::string base = modelArg ? checkpointBase(modelArg) : std::string();

    // Checkpoint self-description: prefer an explicit --config, else auto-load the
    // sidecar the trainer wrote next to the checkpoint.
    if (configArg) cfg.loadFromFile(configArg);
    else if (!base.empty() && fs::exists(base + "_config.yaml"))
        cfg.loadFromFile(base + "_config.yaml");

    SimWorld world(cfg);
    if (modelArg) world.loadPolicy(modelArg);

    const std::string replayDir = cfg.train.modelDir + "/replays";
    std::error_code ec; fs::create_directories(replayDir, ec);

    // Headless self-test: exercise every non-render runtime path (stepping,
    // heatmaps, recorder round-trip, hot-reload check) and exit. Lets a
    // windowless environment / CI verify the simulator's logic without a GPU.
    if (hasFlag(argc, argv, "--selftest")) {
        DP_LOG_INFO("dp_simulator selftest: policy=%s", world.hasPolicy() ? "yes" : "none");
        int maxStreak = 0; double minVel = 1e9;
        for (int i = 0; i < 4000; ++i) {
            const SimFrame& sf = world.step();
            maxStreak = std::max(maxStreak, sf.stillStreak);
            minVel = std::min(minVel, std::abs(sf.state.omega1) + std::abs(sf.state.omega2));
        }
        DP_LOG_INFO("dp_simulator selftest: max stillStreak=%d (need %d), min |w1|+|w2|=%.3f",
                    maxStreak, cfg.env.staticHoldSteps, minVel);
        const auto hmT = world.computeHeatmap(24, SimWorld::HeatmapKind::Torque);
        const auto hmV = world.computeHeatmap(24, SimWorld::HeatmapKind::Value);
        const auto hmS = world.computeHeatmap(24, SimWorld::HeatmapKind::Sigma);
        EpisodeRecorder rec;
        for (int i = 0; i < 100; ++i) rec.record(world.step());
        const std::string rp = replayDir + "/selftest.dprec";
        EpisodeRecorder back;
        const bool ok = rec.saveBinary(rp) && back.loadBinary(rp) && back.size() == rec.size();
        world.reloadIfChanged();
        DP_LOG_INFO("dp_simulator selftest: heatmaps %zu/%zu/%zu cells, replay roundtrip %s",
                    hmT.data.size(), hmV.data.size(), hmS.data.size(), ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    const int screenW = 1280, screenH = 800;
    InitWindow(screenW, screenH, "Double Pendulum RL -- Simulator");
    SetTargetFPS(60);

    Camera2D cam{};
    cam.offset = { screenW * 0.5f, screenH * 0.40f };
    cam.target = { 0.0f, 0.0f };
    cam.zoom   = 1.0f;

    bool   paused = false, frameStep = false, followLatest = false;
    double speed = 1.0, accumulator = 0.0;
    const double dt = cfg.physics.dt;

    std::deque<Vector2> trail;
    std::deque<float> hEntropy, hValue, hReward, hSigma;
    EpisodeRecorder manualRec; bool recording = false;

    // Episode library (auto-capture).
    EpisodeRecorder curEp; double bestReturn = -1e30, epMaxOmega = 0.0;
    // Playback: 0=live, 1=best, 2=failure, 3=recovery.
    int playback = 0; EpisodeRecorder playRec; std::size_t playIdx = 0;
    auto loadPlayback = [&](const char* name) {
        playRec.clear(); playIdx = 0;
        return playRec.loadBinary(replayDir + "/" + name + ".dprec") && !playRec.empty();
    };

    int heat = 0;                         // 0 off, 1 torque, 2 value, 3 sigma
    int heatSlice = 0;                    // 0 th1/th2, 1 th2/w2, 2 th1/w1
    SimWorld::Heatmap heatmap; double heatTimer = 0.0;
    double reloadTimer = 0.0;

    auto pushHistory = [&](const SimFrame& f) {
        auto push = [](std::deque<float>& d, float v){ d.push_back(v); if (d.size() > kHistory) d.pop_front(); };
        push(hEntropy, (float)f.entropy); push(hValue, (float)f.value);
        push(hReward, (float)f.reward);   push(hSigma, (float)f.meanSigma);
    };

    // Capture one stepped frame into trails/history/recorders + episode library.
    auto onFrame = [&](const SimFrame& f) {
        trail.push_back(Vector2{ (float)(f.kinematics.bob2X * kPPM), (float)(f.kinematics.bob2Y * kPPM) });
        if (trail.size() > 600) trail.pop_front();
        if (recording) manualRec.record(f);
        pushHistory(f);

        curEp.record(f);
        epMaxOmega = std::max(epMaxOmega, std::max(std::abs(f.state.omega1), std::abs(f.state.omega2)));
        if (f.terminal || f.truncated) {
            const double ret = curEp.totalReturn();
            const char* cls = nullptr;
            if (f.terminal)              cls = "failure";
            else if (epMaxOmega > 8.0)   cls = "recovery";
            else if (ret > bestReturn) { cls = "best"; bestReturn = ret; }
            if (cls) curEp.saveBinary(replayDir + "/" + std::string(cls) + ".dprec");
            curEp.clear(); epMaxOmega = 0.0;
        }
    };

    while (!WindowShouldClose()) {
        // ---- Input ----------------------------------------------------------
        if (IsKeyPressed(KEY_SPACE)) paused = !paused;
        if (IsKeyPressed(KEY_R)) { world.reset(); trail.clear(); curEp.clear(); }
        if (IsKeyPressed(KEY_T)) world.setStochastic(!world.stochastic());
        if (IsKeyPressed(KEY_PERIOD)) frameStep = true;
        if (IsKeyPressed(KEY_C)) trail.clear();
        if (IsKeyPressed(KEY_RIGHT_BRACKET)) speed = std::min(64.0, speed * 1.5);
        if (IsKeyPressed(KEY_LEFT_BRACKET))  speed = std::max(0.05, speed / 1.5);
        if (IsKeyPressed(KEY_F)) followLatest = !followLatest;
        if (IsKeyPressed(KEY_H)) { heat = (heat + 1) % 4; heatTimer = 0.0; }
        if (IsKeyPressed(KEY_G)) { heatSlice = (heatSlice + 1) % 3; heatTimer = 0.0; }

        const double kick = 5.0;
        if (IsKeyPressed(KEY_LEFT))  world.applyImpulse(-kick, 0.0);
        if (IsKeyPressed(KEY_RIGHT)) world.applyImpulse(+kick, 0.0);
        if (IsKeyPressed(KEY_UP))    world.applyImpulse(0.0, +kick);
        if (IsKeyPressed(KEY_DOWN))  world.applyImpulse(0.0, -kick);
        if (IsKeyPressed(KEY_D)) world.applyImpulse(GetRandomValue(-60, 60) / 10.0,
                                                    GetRandomValue(-60, 60) / 10.0);

        if (!base.empty()) {
            if (IsKeyPressed(KEY_ONE))   world.loadPolicy(base + "_best.ckpt");
            if (IsKeyPressed(KEY_TWO))   world.loadPolicy(base + "_latest.ckpt");
            if (IsKeyPressed(KEY_THREE)) world.loadPolicy(base + "_final.ckpt");
        }

        if (IsKeyPressed(KEY_O)) { recording = !recording; if (recording) manualRec.clear(); }
        if (IsKeyPressed(KEY_P) && !manualRec.empty()) {
            manualRec.saveBinary(cfg.train.modelDir + "/replay.dprec");
            manualRec.saveCsv(cfg.train.logDir + "/replay.csv");
        }
        if (IsKeyPressed(KEY_L)) {
            playback = (playback + 1) % 4; playIdx = 0;
            const char* src = playback == 1 ? "best" : playback == 2 ? "failure"
                            : playback == 3 ? "recovery" : nullptr;
            if (src && !loadPlayback(src)) playback = 0;  // nothing recorded yet
        }

        const float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            const Vector2 mw = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.zoom = std::clamp(cam.zoom * (1.0f + wheel * 0.1f), 0.2f, 6.0f);
            const Vector2 mw2 = GetScreenToWorld2D(GetMousePosition(), cam);
            cam.target.x += mw.x - mw2.x; cam.target.y += mw.y - mw2.y;
        }
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            const Vector2 d = GetMouseDelta();
            cam.target.x -= d.x / cam.zoom; cam.target.y -= d.y / cam.zoom;
        }
        if (IsKeyPressed(KEY_HOME)) { cam.target = {0, 0}; cam.zoom = 1.0f; }

        // ---- Hot-reload watch ----------------------------------------------
        reloadTimer += GetFrameTime();
        if (followLatest && reloadTimer > 0.5) {
            reloadTimer = 0.0;
            if (world.reloadIfChanged()) { heatTimer = 0.0; }  // force heatmap refresh
        }

        // ---- Step / playback ------------------------------------------------
        if (playback != 0 && !playRec.empty()) {
            // Replay mode: advance through recorded frames, no physics.
            if (!paused) {
                accumulator += GetFrameTime() * speed;
                while (accumulator >= dt) {
                    accumulator -= dt;
                    playIdx = (playIdx + 1) % playRec.size();
                }
            } else if (frameStep) {
                playIdx = (playIdx + 1) % playRec.size();   // step one recorded frame
            }
        } else if (!paused) {
            accumulator += GetFrameTime() * speed;
            int budget = 4000;
            while (accumulator >= dt && budget-- > 0) {
                onFrame(world.step());
                accumulator -= dt;
            }
        } else if (frameStep) {
            onFrame(world.step());
        }
        frameStep = false;

        // ---- Recompute heatmap occasionally --------------------------------
        heatTimer -= GetFrameTime();
        if (heat != 0 && heatTimer <= 0.0) {
            heatTimer = 0.5;
            const SimWorld::HeatmapKind k = heat == 1 ? SimWorld::HeatmapKind::Torque
                : heat == 2 ? SimWorld::HeatmapKind::Value : SimWorld::HeatmapKind::Sigma;
            const SimWorld::HeatmapSlice sl = heatSlice == 1 ? SimWorld::HeatmapSlice::Theta2Omega2
                : heatSlice == 2 ? SimWorld::HeatmapSlice::Theta1Omega1
                : SimWorld::HeatmapSlice::Theta1Theta2;
            // Slice through the LIVE state so off-axis dims reflect the current pose.
            heatmap = world.computeHeatmap(kHeatRes, k, sl, world.last().state);
        }

        const SimFrame& f = world.last();

        // ---- Render ---------------------------------------------------------
        BeginDrawing();
        ClearBackground(Color{ 16, 16, 22, 255 });

        // Determine what to draw (live state or a replay frame).
        Vector2 joint, tip, com{0,0};
        bool drawCom = false;
        if (playback != 0 && !playRec.empty()) {
            const RecordFrame& rf = playRec.at(playIdx);
            framePositions(rf.theta1, rf.theta2, cfg.physics.l1, cfg.physics.l2, joint, tip);
        } else {
            joint = { (float)(f.kinematics.jointX * kPPM), (float)(f.kinematics.jointY * kPPM) };
            tip   = { (float)(f.kinematics.bob2X * kPPM),  (float)(f.kinematics.bob2Y * kPPM) };
            com   = { (float)(f.kinematics.comX * kPPM),   (float)(f.kinematics.comY * kPPM) };
            drawCom = true;
        }

        BeginMode2D(cam);
        const float reach = (float)((cfg.physics.l1 + cfg.physics.l2) * kPPM);
        DrawLineEx({ -22, -reach }, { 22, -reach }, 2, Fade(GREEN, 0.4f));
        for (std::size_t i = 1; i < trail.size(); ++i) {
            const float a = (float)i / (float)trail.size();
            DrawLineEx(trail[i - 1], trail[i], 2.0f, Fade(SKYBLUE, a * 0.7f));
        }
        DrawLineEx({0,0}, joint, 5.0f, Color{ 230, 230, 240, 255 });
        DrawLineEx(joint, tip,  5.0f, Color{ 230, 230, 240, 255 });
        DrawCircleV({0,0}, 6, GRAY);
        DrawCircleV(joint, 13, Color{ 90, 200, 250, 255 });
        DrawCircleV(tip,   13, Color{ 250, 150, 90, 255 });
        if (drawCom) DrawCircleV(com, 5, Color{ 120, 250, 120, 255 });
        EndMode2D();

        // ---- Heatmap overlay (top-right) -----------------------------------
        if (heat != 0 && heatmap.res > 0) {
            const int cell = 5, dim = heatmap.res * cell;
            const int ox = screenW - dim - 16, oy = 250;
            DrawRectangle(ox - 2, oy - 18, dim + 4, dim + 22, Fade(BLACK, 0.5f));
            const char* kindN = heat == 1 ? "torque" : heat == 2 ? "value" : "sigma";
            const char* axes  = heatSlice == 1 ? "th2 x w2" : heatSlice == 2 ? "th1 x w1" : "th1 x th2";
            DrawText(TextFormat("%s [%s]  (G slice)", kindN, axes), ox, oy - 16, 12, RAYWHITE);
            const float span = heatmap.hi - heatmap.lo;
            for (int j = 0; j < heatmap.res; ++j)
                for (int i = 0; i < heatmap.res; ++i) {
                    const float v = heatmap.data[(std::size_t)j * heatmap.res + i];
                    DrawRectangle(ox + i * cell, oy + j * cell, cell, cell,
                                  heatColor((v - heatmap.lo) / span));
                }
        }

        // ---- Playback scrubber ---------------------------------------------
        if (playback != 0 && !playRec.empty()) {
            const float bx = 200, bw = screenW - 460, by = screenH - 118, bh = 12;
            const std::size_t last = playRec.size() - 1;
            const Vector2 m = GetMousePosition();
            const bool hover = m.x >= bx && m.x <= bx + bw && m.y >= by - 8 && m.y <= by + bh + 8;
            if (hover && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
                const float tpos = std::clamp((m.x - bx) / bw, 0.0f, 1.0f);
                playIdx = static_cast<std::size_t>(tpos * last);
            }
            DrawRectangle((int)bx, (int)by, (int)bw, (int)bh, Fade(GRAY, 0.45f));
            const float frac = last ? static_cast<float>(playIdx) / static_cast<float>(last) : 0.0f;
            DrawRectangle((int)bx, (int)by, (int)(bw * frac), (int)bh, VIOLET);
            DrawCircleV({ bx + bw * frac, by + bh * 0.5f }, 7, RAYWHITE);
            DrawText(TextFormat("frame %zu / %zu   (drag to scrub | . step | SPACE pause)",
                     playIdx, playRec.size()), (int)bx, (int)by - 16, 12, RAYWHITE);
        }

        // ---- HUD ------------------------------------------------------------
        int x = 14, y = 12; const int lh = 18;
        auto line = [&](const char* t, Color c = RAYWHITE){ DrawText(t, x, y, 16, c); y += lh; };
        DrawRectangle(0, 0, 330, 300, Fade(BLACK, 0.45f));
        line(TextFormat("FPS %d  speed %.2fx%s%s", GetFPS(), speed,
                        paused ? "  [PAUSED]" : "", followLatest ? "  [FOLLOW]" : ""),
             paused ? YELLOW : RAYWHITE);
        if (playback != 0)
            line(TextFormat("PLAYBACK: %s  frame %zu/%zu",
                 playback==1?"best":playback==2?"failure":"recovery", playIdx, playRec.size()), VIOLET);
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
        // Static-equilibrium ("monk mode") progress toward sustained stillness.
        if (f.atEquilibrium) line("MONK MODE: PERFECTLY STILL", GREEN);
        else line(TextFormat("stillness  %d/%d", f.stillStreak, cfg.env.staticHoldSteps),
                  f.stillStreak > 0 ? YELLOW : Fade(RAYWHITE, 0.55f));
        if (recording) line("RECORDING", RED);

        // Big banner the moment perfect static equilibrium is reached.
        if (f.atEquilibrium) {
            const char* msg = "PERFECT STATIC EQUILIBRIUM";
            const int fw = MeasureText(msg, 30);
            DrawRectangle(screenW / 2 - fw / 2 - 18, 58, fw + 36, 46, Fade(DARKGREEN, 0.75f));
            DrawText(msg, screenW / 2 - fw / 2, 66, 30, RAYWHITE);
        }

        const float gy = screenH - 96.0f;
        drawGraph(hReward,  14,  gy, 280, 80, GREEN,   "reward");
        drawGraph(hValue,   312, gy, 280, 80, SKYBLUE, "value");
        drawGraph(hEntropy, 610, gy, 280, 80, ORANGE,  "entropy");
        drawGraph(hSigma,   908, gy, 280, 80, VIOLET,  "sigma");

        int cx = screenW - 252, cy = 12;
        DrawRectangle(cx - 10, 0, 262, 230, Fade(BLACK, 0.4f));
        auto ctl = [&](const char* t){ DrawText(t, cx, cy, 13, Fade(RAYWHITE, 0.8f)); cy += 15; };
        ctl("SPACE pause  R reset  . step");
        ctl("T stochastic   F follow-latest");
        ctl("H heatmap  G slice  L playback");
        ctl("ARROWS impulse  D random kick");
        ctl("[ / ] slower / faster");
        ctl("wheel zoom  R-drag pan  HOME");
        ctl("1/2/3 best/latest/final");
        ctl("O record  P save  C clear  ESC");

        EndDrawing();
    }

    CloseWindow();
    return 0;
}
