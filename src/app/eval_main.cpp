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
    double sigmaUpright = 0.0; long long nUpright = 0;   // mean sigma when balanced
    double sigmaRecover = 0.0; long long nRecover = 0;   // mean sigma when off-balance
    for (int ep = 0; ep < episodes; ++ep) {
        Observation obs = env.reset(seed + ep);
        double ret = 0.0;
        int step = 0;
        for (; step < cfg.env.maxEpisodeSteps; ++step) {
            const rl::PolicyOutput po = agent.act(obs, /*deterministic=*/true);
            const Action torque = env.decode(po.squashed);
            const StepResult sr = env.step(torque);
            ret += sr.reward;

            // Bucket the policy's exploration magnitude by how upright we are, to
            // confirm the state-dependent sigma behaves as intended.
            double meanSigma = 0.0;
            for (double s : po.sigma) meanSigma += s;
            meanSigma /= static_cast<double>(po.sigma.size());
            if (sr.uprightScore > 0.85) { sigmaUpright += meanSigma; ++nUpright; }
            else                        { sigmaRecover += meanSigma; ++nRecover; }
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
    if (nUpright > 0 && nRecover > 0)
        DP_LOG_INFO("eval: state-dependent sigma -> balanced=%.3f  recovery=%.3f  (recovery should be larger)",
                    sigmaUpright / static_cast<double>(nUpright),
                    sigmaRecover / static_cast<double>(nRecover));
    if (csv.is_open()) DP_LOG_INFO("eval: wrote replay CSV '%s'", csvPath);
    return 0;
}
