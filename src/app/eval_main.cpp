// =============================================================================
// eval_main.cpp
// -----------------------------------------------------------------------------
// Load a trained checkpoint and evaluate it deterministically. Optionally export
// a CSV replay of the trajectory (the Recording System's data path) so it can be
// plotted or played back.
//
// Usage:
//   dp_eval --model <path.ckpt> [--config <path>] [--episodes N]
//           [--csv <out.csv>] [--seed S]
// =============================================================================
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>

#include "core/Config.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "rl/PPOAgent.hpp"
#include "util/Logger.hpp"

using namespace dp;

namespace {
const char* argValue(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}
} // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (const char* path = argValue(argc, argv, "--config")) cfg.loadFromFile(path);

    const char* model = argValue(argc, argv, "--model");
    if (!model) { DP_LOG_ERROR("eval: --model <path.ckpt> is required"); return 1; }

    int episodes = 5;
    if (const char* s = argValue(argc, argv, "--episodes")) episodes = std::atoi(s);
    std::uint64_t seed = cfg.train.seed;
    if (const char* s = argValue(argc, argv, "--seed")) seed = std::strtoull(s, nullptr, 10);
    const char* csvPath = argValue(argc, argv, "--csv");

    DoublePendulumEnv env(cfg);
    rl::PPOAgent agent(env.observationDim(), env.actionDim(), cfg.ppo, seed);
    if (!agent.load(model)) return 1;
    DP_LOG_INFO("eval: loaded model '%s'", model);

    std::ofstream csv;
    if (csvPath) {
        csv.open(csvPath, std::ios::trunc);
        csv << "episode,step,theta1,theta2,omega1,omega2,torque1,torque2,reward,upright,energy\n";
    }

    double sumReturn = 0.0;
    for (int ep = 0; ep < episodes; ++ep) {
        Observation obs = env.reset(seed + ep);
        double ret = 0.0;
        int step = 0;
        for (; step < cfg.env.maxEpisodeSteps; ++step) {
            const rl::PolicyOutput po = agent.act(obs, /*deterministic=*/true);
            const Action torque = env.decode(po.action);
            const StepResult sr = env.step(torque);
            ret += sr.reward;
            if (csv.is_open()) {
                const State& s = env.getState();
                csv << ep << ',' << step << ','
                    << s.theta1 << ',' << s.theta2 << ',' << s.omega1 << ',' << s.omega2 << ','
                    << torque.torque1 << ',' << torque.torque2 << ','
                    << sr.reward << ',' << sr.uprightScore << ',' << sr.energy << '\n';
            }
            obs = sr.observation;
            if (sr.terminal || sr.truncated) break;
        }
        sumReturn += ret;
        DP_LOG_INFO("eval: episode %d return=%.2f steps=%d", ep, ret, step + 1);
    }
    DP_LOG_INFO("eval: mean return over %d episodes = %.2f", episodes, sumReturn / episodes);
    if (csv.is_open()) DP_LOG_INFO("eval: wrote replay CSV '%s'", csvPath);
    return 0;
}
