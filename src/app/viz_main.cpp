// =============================================================================
// viz_main.cpp
// -----------------------------------------------------------------------------
// SDL2 + Dear ImGui real-time visualizer / training dashboard.
//
// THREADING MODEL:
//   * A background "simulation thread" runs the Trainer (PPO rollouts + updates),
//     publishing RenderSnapshots into a SharedState.
//   * The main thread owns the window, GL/SDL context, and ImGui, and only ever
//     READS snapshots and WRITES control intents. SDL and most GPU drivers are
//     not thread-safe, so all windowing/rendering stays on the main thread; the
//     physics never blocks on the GPU and the UI never blocks on physics.
//
// This whole translation unit is compiled only when DP_WITH_RENDERER is defined
// (see CMake option BUILD_RENDERER). The headless core has zero UI dependencies.
// =============================================================================
#include <thread>
#include <atomic>
#include <vector>
#include <deque>
#include <cmath>
#include <cstring>

#include <SDL.h>
#include "imgui.h"
#include "backends/imgui_impl_sdl2.h"
#include "backends/imgui_impl_sdlrenderer2.h"

#include "core/Config.hpp"
#include "core/Trainer.hpp"
#include "core/SharedState.hpp"
#include "util/Logger.hpp"

using namespace dp;

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

// Draw a filled circle with the midpoint algorithm (SDL has no native fill).
void fillCircle(SDL_Renderer* r, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
        const int dx = static_cast<int>(std::sqrt(static_cast<double>(radius*radius - dy*dy)));
        SDL_RenderDrawLine(r, cx - dx, cy + dy, cx + dx, cy + dy);
    }
}

} // namespace

int main(int argc, char** argv) {
    // ---- Config + control ----------------------------------------------------
    Config cfg;
    if (const char* path = argValue(argc, argv, "--config")) cfg.loadFromFile(path);
    if (const char* s = argValue(argc, argv, "--task")) {
        if (std::strcmp(s, "SwingUp") == 0)   cfg.env.task = TaskType::SwingUp;
        if (std::strcmp(s, "Stabilize") == 0) cfg.env.task = TaskType::Stabilize;
        if (std::strcmp(s, "Recovery") == 0)  cfg.env.task = TaskType::Recovery;
    }
    if (hasFlag(argc, argv, "--dual")) cfg.env.actuator = ActuatorMode::Dual;
    const bool evalOnly = hasFlag(argc, argv, "--eval");
    const char* loadModel = argValue(argc, argv, "--model");

    SharedState shared;
    shared.control.training.store(!evalOnly);

    // ---- Simulation thread ---------------------------------------------------
    Trainer trainer(cfg, &shared);
    if (loadModel) trainer.agent().load(loadModel);

    std::thread simThread([&] {
        if (evalOnly) {
            while (!shared.control.quit.load())
                trainer.evaluateEpisode(/*publishSnapshots=*/true);
        } else {
            while (!shared.control.quit.load() &&
                   trainer.globalStep() < cfg.train.totalSteps) {
                trainer.collectAndUpdate();
            }
        }
    });

    // ---- SDL + ImGui init (main thread) -------------------------------------
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        DP_LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
        shared.control.quit.store(true); simThread.join(); return 1;
    }
    const int W = 1280, H = 800;
    SDL_Window* window = SDL_CreateWindow("Double Pendulum RL",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H,
        SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
    ImGui_ImplSDLRenderer2_Init(renderer);

    std::deque<float>  rewardHistory;     // avg-return curve
    std::deque<ImVec2> trail1, trail2;    // bob trails (screen space)
    const std::size_t  kTrailMax = 240;

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            ImGui_ImplSDL2_ProcessEvent(&e);
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_SPACE: shared.control.pause.store(!shared.control.pause.load()); break;
                    case SDLK_r:     shared.control.requestReset.store(true); break;
                    case SDLK_PERIOD:shared.control.frameStep.store(true); break;
                    case SDLK_d:     shared.control.injectDisturbance.store(true); break;
                    case SDLK_s:     shared.control.slowMotion.store(!shared.control.slowMotion.load()); break;
                    case SDLK_ESCAPE:running = false; break;
                    default: break;
                }
            }
        }

        const RenderSnapshot snap = shared.read();

        ImGui_ImplSDLRenderer2_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        // ---- Control panel ---------------------------------------------------
        ImGui::Begin("Simulation");
        bool paused = shared.control.pause.load();
        if (ImGui::Checkbox("Pause (Space)", &paused)) shared.control.pause.store(paused);
        if (ImGui::Button("Reset (R)")) shared.control.requestReset.store(true);
        ImGui::SameLine();
        if (ImGui::Button("Frame Step (.)")) shared.control.frameStep.store(true);
        ImGui::SameLine();
        if (ImGui::Button("Disturb (D)")) shared.control.injectDisturbance.store(true);
        bool slow = shared.control.slowMotion.load();
        if (ImGui::Checkbox("Slow motion (S)", &slow)) shared.control.slowMotion.store(slow);
        ImGui::Text("Sim rate: %.0f steps/s", snap.simRate);
        ImGui::Text("Render FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("Global step: %lld", snap.globalStep);
        ImGui::Text("Episode %d  step %d", snap.episode, snap.episodeStep);
        ImGui::End();

        // ---- State / network panel ------------------------------------------
        ImGui::Begin("State & Policy");
        ImGui::Text("theta1=% .3f  omega1=% .3f", snap.state.theta1, snap.state.omega1);
        ImGui::Text("theta2=% .3f  omega2=% .3f", snap.state.theta2, snap.state.omega2);
        ImGui::Text("upright=%.3f  energy=%.3f", snap.uprightScore, snap.energy);
        ImGui::Separator();
        ImGui::Text("Observation:");
        for (std::size_t i = 0; i < snap.lastObservation.size(); ++i)
            ImGui::Text("  o[%zu]=% .3f", i, snap.lastObservation[i]);
        ImGui::Text("Policy output (raw):");
        for (std::size_t i = 0; i < snap.lastActionRaw.size(); ++i)
            ImGui::Text("  a[%zu]=% .3f", i, snap.lastActionRaw[i]);
        ImGui::Text("Applied torque: % .3f  % .3f",
                    snap.lastTorque.size() > 0 ? snap.lastTorque[0] : 0.0,
                    snap.lastTorque.size() > 1 ? snap.lastTorque[1] : 0.0);
        ImGui::End();

        // ---- Training panel --------------------------------------------------
        ImGui::Begin("Training");
        ImGui::Text("avgReturn100: %.2f", snap.avgReturn100);
        ImGui::Text("episodeReturn: %.2f", snap.episodeReturn);
        ImGui::Text("policyLoss: %.4f", snap.policyLoss);
        ImGui::Text("valueLoss:  %.4f", snap.valueLoss);
        ImGui::Text("entropy:    %.3f", snap.entropy);
        ImGui::Text("policy std: %.3f", snap.meanStd);
        if (rewardHistory.empty() || rewardHistory.back() != static_cast<float>(snap.avgReturn100)) {
            rewardHistory.push_back(static_cast<float>(snap.avgReturn100));
            if (rewardHistory.size() > 400) rewardHistory.pop_front();
        }
        if (!rewardHistory.empty()) {
            std::vector<float> tmp(rewardHistory.begin(), rewardHistory.end());
            ImGui::PlotLines("avgReturn100", tmp.data(), static_cast<int>(tmp.size()),
                             0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 80));
        }
        ImGui::End();

        // ---- Pendulum drawing ------------------------------------------------
        SDL_SetRenderDrawColor(renderer, 18, 18, 24, 255);
        SDL_RenderClear(renderer);

        int winW, winH; SDL_GetRendererOutputSize(renderer, &winW, &winH);
        const double originX = winW * 0.5;
        const double originY = winH * 0.40;
        const double scale = std::min(winW, winH) * 0.18; // meters -> pixels

        // World y is positive-down (matches kinematics()); screen y is also down.
        auto toScreenX = [&](double x){ return static_cast<int>(originX + x * scale); };
        auto toScreenY = [&](double y){ return static_cast<int>(originY + y * scale); };

        const Kinematics& k = snap.kinematics;
        const int px = toScreenX(0),         py = toScreenY(0);
        const int jx = toScreenX(k.jointX),  jy = toScreenY(k.jointY);
        const int b2x= toScreenX(k.bob2X),   b2y= toScreenY(k.bob2Y);
        const int cmx= toScreenX(k.comX),    cmy= toScreenY(k.comY);

        // Trails.
        trail1.push_back(ImVec2(static_cast<float>(jx), static_cast<float>(jy)));
        trail2.push_back(ImVec2(static_cast<float>(b2x), static_cast<float>(b2y)));
        if (trail1.size() > kTrailMax) trail1.pop_front();
        if (trail2.size() > kTrailMax) trail2.pop_front();
        SDL_SetRenderDrawColor(renderer, 70, 90, 140, 120);
        for (std::size_t i = 1; i < trail2.size(); ++i)
            SDL_RenderDrawLine(renderer, (int)trail2[i-1].x, (int)trail2[i-1].y,
                                         (int)trail2[i].x,   (int)trail2[i].y);

        // Rods.
        SDL_SetRenderDrawColor(renderer, 220, 220, 230, 255);
        SDL_RenderDrawLine(renderer, px, py, jx, jy);
        SDL_RenderDrawLine(renderer, jx, jy, b2x, b2y);

        // Pivot, bobs, center of mass.
        SDL_SetRenderDrawColor(renderer, 180, 180, 190, 255); fillCircle(renderer, px, py, 5);
        SDL_SetRenderDrawColor(renderer, 90, 200, 250, 255);  fillCircle(renderer, jx, jy, 12);
        SDL_SetRenderDrawColor(renderer, 250, 150, 90, 255);  fillCircle(renderer, b2x, b2y, 12);
        SDL_SetRenderDrawColor(renderer, 120, 250, 120, 255); fillCircle(renderer, cmx, cmy, 4);

        // "Up" reference marker (the balance target) above the pivot.
        SDL_SetRenderDrawColor(renderer, 80, 80, 90, 255);
        SDL_RenderDrawLine(renderer, px - 20, toScreenY(-(cfg.physics.l1 + cfg.physics.l2)),
                                     px + 20, toScreenY(-(cfg.physics.l1 + cfg.physics.l2)));

        ImGui::Render();
        ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData(), renderer);
        SDL_RenderPresent(renderer);
    }

    // ---- Shutdown ------------------------------------------------------------
    shared.control.quit.store(true);
    simThread.join();

    ImGui_ImplSDLRenderer2_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
