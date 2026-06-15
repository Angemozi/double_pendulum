// =============================================================================
// train_torch_main.cpp
// -----------------------------------------------------------------------------
// GPU-capable PPO backend built on LibTorch (PyTorch C++ API).
//
// It REUSES the dependency-free DoublePendulumEnv / Config / RolloutBuffer from
// dp_core and replaces ONLY the agent with a Torch actor-critic that runs on
// CUDA when available. The environment stays on the CPU (it is tiny and already
// runs at millions of steps/sec); the neural networks -- the actual compute
// bottleneck identified by profiling -- move to the GPU and use batched matmuls
// instead of the per-sample scalar MLP. This is the same change that both
// unlocks the GPU and gives a large CPU speedup via BLAS.
//
// BUILD (requires a LibTorch distribution; CUDA build for GPU):
//   cmake -S . -B build -DDP_WITH_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch
//   cmake --build build --config Release
//   ./build/dp_train_torch --config configs/stabilize.yaml
//
// NOTE: This translation unit is only added to the build when DP_WITH_LIBTORCH
// is ON. It could not be compiled in the authoring environment (no LibTorch
// present); validate it in a LibTorch-enabled toolchain. The math mirrors the
// verified CPU PPOAgent (clipped surrogate + GAE + entropy + std floor + KL
// early stop + LR anneal), so behavior should match up to floating point.
// =============================================================================
#include <torch/torch.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include "core/Config.hpp"
#include "rl/DoublePendulumEnv.hpp"
#include "rl/RolloutBuffer.hpp"
#include "util/Logger.hpp"

using namespace dp;

namespace {

const char* argValue(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return nullptr;
}

// Actor-critic with a shared-free design: separate trunks, a Gaussian policy
// head (state-independent log_std parameter, matching the CPU agent), and a
// scalar value head. tanh activations for parity with the CPU implementation.
struct ActorCriticImpl : torch::nn::Module {
    torch::nn::Linear a1{nullptr}, a2{nullptr}, aHead{nullptr};
    torch::nn::Linear c1{nullptr}, c2{nullptr}, cVal{nullptr};

    ActorCriticImpl(int obs, int act, int h1, int h2, double initLogStd) {
        a1 = register_module("a1", torch::nn::Linear(obs, h1));
        a2 = register_module("a2", torch::nn::Linear(h1, h2));
        // Actor head emits BOTH the mean and the raw (pre-tanh) log_std =>
        // STATE-DEPENDENT exploration (2*act outputs), matching the CPU backend.
        aHead = register_module("aHead", torch::nn::Linear(h2, 2 * act));
        c1 = register_module("c1", torch::nn::Linear(obs, h1));
        c2 = register_module("c2", torch::nn::Linear(h1, h2));
        cVal = register_module("cVal", torch::nn::Linear(h2, 1));
        torch::nn::init::uniform_(aHead->weight, -0.01, 0.01);
        actDim_ = act;
        (void)initLogStd; // initial logStd is set by the tanh-map midpoint
    }

    // Returns {mean [N,act], logStd [N,act]} with logStd bounded to [min,max] by
    // a tanh map (the SAC trick) so std is smooth, bounded, and state-dependent.
    std::pair<torch::Tensor, torch::Tensor> meanLogStd(torch::Tensor x,
                                                       double mn, double mx) {
        x = torch::tanh(a1->forward(x));
        x = torch::tanh(a2->forward(x));
        auto h = aHead->forward(x);
        auto mean = h.narrow(-1, 0, actDim_);
        auto raw  = h.narrow(-1, actDim_, actDim_);
        auto logStd = mn + 0.5 * (mx - mn) * (torch::tanh(raw) + 1.0);
        return {mean, logStd};
    }
    torch::Tensor value(torch::Tensor x) {
        x = torch::tanh(c1->forward(x));
        x = torch::tanh(c2->forward(x));
        return cVal->forward(x).squeeze(-1);
    }
    int actDim_ = 0;
};
TORCH_MODULE(ActorCritic);

// Diagonal-Gaussian log-prob summed over action dims (per-state logStd).
torch::Tensor gaussianLogProb(torch::Tensor mean, torch::Tensor logStd, torch::Tensor action) {
    const auto std = torch::exp(logStd);
    const auto z = (action - mean) / std;
    const auto lp = -0.5 * z * z - logStd - 0.5 * std::log(2.0 * M_PI);
    return lp.sum(-1);
}

// ---- Portable checkpoint bridge ---------------------------------------------
// The Torch ActorCritic is architecturally IDENTICAL to the CPU MLP
// (in->h1->tanh->h2->tanh->head). We therefore export the weights in the exact
// text format PPOAgent::save() produces ("DPPPO 2"), so a GPU-trained policy
// loads directly in dp_simulator / dp_eval with no LibTorch and no ONNX.
// PyTorch's Linear weight is [out, in] row-major, matching our W[o*in + i] and
// y[o] = sum_i W[o*in+i] x[i] + b[o].
void writeLinear(std::ostream& os, const torch::nn::Linear& lin) {
    const auto w = lin->weight.detach().to(torch::kCPU).to(torch::kDouble).contiguous();
    const auto b = lin->bias.detach().to(torch::kCPU).to(torch::kDouble).contiguous();
    const int out = static_cast<int>(w.size(0)), in = static_cast<int>(w.size(1));
    os << in << ' ' << out << '\n';
    const double* wp = w.data_ptr<double>();
    for (long k = 0; k < static_cast<long>(in) * out; ++k) os << wp[k] << ' ';
    os << '\n';
    const double* bp = b.data_ptr<double>();
    for (int o = 0; o < out; ++o) os << bp[o] << ' ';
    os << '\n';
}

void exportPortableCkpt(ActorCritic& net, int obsDim, int actDim, const std::string& path) {
    std::ofstream os(path, std::ios::trunc);
    if (!os.is_open()) { DP_LOG_ERROR("export ckpt failed: %s", path.c_str()); return; }
    os.precision(17);
    os << "DPPPO 2\n" << obsDim << ' ' << actDim << '\n';
    os << obsDim << ' ' << (2 * actDim) << '\n';          // actor MLP header
    writeLinear(os, net->a1); writeLinear(os, net->a2); writeLinear(os, net->aHead);
    os << obsDim << ' ' << 1 << '\n';                      // critic MLP header
    writeLinear(os, net->c1); writeLinear(os, net->c2); writeLinear(os, net->cVal);
    DP_LOG_INFO("dp_train_torch: exported portable checkpoint '%s'", path.c_str());
}

} // namespace

int main(int argc, char** argv) {
    Config cfg;
    if (const char* p = argValue(argc, argv, "--config")) cfg.loadFromFile(p);
    if (const char* s = argValue(argc, argv, "--steps")) cfg.train.totalSteps = std::atoll(s);

    torch::manual_seed(cfg.train.seed);
    const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    DP_LOG_INFO("dp_train_torch: device = %s", device.is_cuda() ? "CUDA" : "CPU");

    DoublePendulumEnv env(cfg);
    const int obsDim = env.observationDim();
    const int actDim = env.actionDim();

    ActorCritic net(obsDim, actDim, cfg.ppo.hidden1, cfg.ppo.hidden2, cfg.ppo.initLogStd);
    net->to(device);
    torch::optim::Adam opt(net->parameters(), torch::optim::AdamOptions(cfg.ppo.lrActor));

    const PpoConfig& p = cfg.ppo;
    Observation obs = env.reset();
    long long globalStep = 0;
    int update = 0;

    while (globalStep < cfg.train.totalSteps) {
        // ---- Rollout collection (CPU env, batched policy queries on device) --
        rl::RolloutBuffer buffer;
        buffer.reserve(static_cast<std::size_t>(p.rolloutSteps));
        for (int t = 0; t < p.rolloutSteps; ++t) {
            torch::NoGradGuard ng;
            auto obsT = torch::from_blob(obs.data(), {1, obsDim}, torch::kDouble)
                            .to(torch::kFloat).to(device);
            auto [mean, logStd] = net->meanLogStd(obsT, p.minLogStd, 1.0);
            auto std  = torch::exp(logStd);
            auto u = mean + std * torch::randn_like(mean);   // pre-squash sample
            auto logp = gaussianLogProb(mean, logStd, u);
            auto val  = net->value(obsT);

            // Store the pre-squash u; send tanh(u) to the environment.
            std::vector<double> uVec(actDim), aVec(actDim);
            auto uCpu = u.to(torch::kCPU).to(torch::kDouble).contiguous();
            std::memcpy(uVec.data(), uCpu.data_ptr<double>(), sizeof(double) * actDim);
            for (int k = 0; k < actDim; ++k) aVec[k] = std::tanh(uVec[k]);

            StepResult sr = env.step(env.decode(aVec));

            rl::Transition tr;
            tr.observation = obs;
            tr.action = uVec;
            tr.logProb = logp.item<double>();
            tr.value = val.item<double>();
            tr.reward = sr.reward;
            tr.done = sr.terminal;
            tr.truncated = sr.truncated && !sr.terminal;
            if (tr.truncated) {
                // Bootstrap a time-limit cut from the real next-state value.
                auto nObs = torch::from_blob(sr.observation.data(), {1, obsDim}, torch::kDouble)
                                .to(torch::kFloat).to(device);
                tr.nextValue = net->value(nObs).item<double>();
            }
            buffer.add(std::move(tr));

            obs = sr.observation;
            ++globalStep;
            if (sr.terminal || sr.truncated) obs = env.reset();
        }

        // Bootstrap + GAE (reuses the verified CPU implementation).
        double lastVal;
        {
            torch::NoGradGuard ng;
            auto obsT = torch::from_blob(obs.data(), {1, obsDim}, torch::kDouble)
                            .to(torch::kFloat).to(device);
            lastVal = net->value(obsT).item<double>();
        }
        buffer.computeGAE(p.gamma, p.lambda, lastVal);
        buffer.normalizeAdvantages();

        // ---- Pack the rollout into device tensors --------------------------
        const int N = static_cast<int>(buffer.size());
        auto obsBatch = torch::empty({N, obsDim}, torch::kFloat);
        auto actBatch = torch::empty({N, actDim}, torch::kFloat);
        auto oldLogp  = torch::empty({N}, torch::kFloat);
        auto adv      = torch::empty({N}, torch::kFloat);
        auto ret      = torch::empty({N}, torch::kFloat);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < obsDim; ++j) obsBatch[i][j] = static_cast<float>(buffer[i].observation[j]);
            for (int j = 0; j < actDim; ++j) actBatch[i][j] = static_cast<float>(buffer[i].action[j]);
            oldLogp[i] = static_cast<float>(buffer[i].logProb);
            adv[i] = static_cast<float>(buffer[i].advantage);
            ret[i] = static_cast<float>(buffer[i].ret);
        }
        obsBatch = obsBatch.to(device); actBatch = actBatch.to(device);
        oldLogp = oldLogp.to(device); adv = adv.to(device); ret = ret.to(device);

        // ---- PPO epochs over minibatches (batched, on device) --------------
        const double lrScale = p.annealLr
            ? std::max(0.05, 1.0 - static_cast<double>(globalStep) / cfg.train.totalSteps) : 1.0;
        for (auto& g : opt.param_groups())
            static_cast<torch::optim::AdamOptions&>(g.options()).lr(p.lrActor * lrScale);

        double lastEntropy = 0.0, lastKL = 0.0;
        bool stop = false;
        for (int epoch = 0; epoch < p.epochs && !stop; ++epoch) {
            auto perm = torch::randperm(N, torch::TensorOptions().device(device));
            for (int s = 0; s < N; s += p.miniBatch) {
                const int e = std::min(s + p.miniBatch, N);
                auto idx = perm.slice(0, s, e);
                auto mbObs = obsBatch.index_select(0, idx);
                auto mbAct = actBatch.index_select(0, idx);
                auto mbOldLp = oldLogp.index_select(0, idx);
                auto mbAdv = adv.index_select(0, idx);
                auto mbRet = ret.index_select(0, idx);

                auto [mean, logStd] = net->meanLogStd(mbObs, p.minLogStd, 1.0);
                auto newLp = gaussianLogProb(mean, logStd, mbAct);
                auto ratio = torch::exp(newLp - mbOldLp);

                // Clipped surrogate (negative => loss to minimize).
                auto surr1 = ratio * mbAdv;
                auto surr2 = torch::clamp(ratio, 1.0 - p.clipEps, 1.0 + p.clipEps) * mbAdv;
                auto policyLoss = -torch::min(surr1, surr2).mean();

                auto valuePred = net->value(mbObs);
                auto valueLoss = torch::mse_loss(valuePred, mbRet);

                // Per-state Gaussian entropy, averaged over the minibatch. The
                // tanh std-map keeps logStd bounded, so no explicit clamp needed.
                auto entropy = (logStd + 0.5 * std::log(2.0 * M_PI * std::exp(1.0)))
                                   .sum(-1).mean();

                auto loss = policyLoss + p.valueCoef * valueLoss - p.entropyCoef * entropy;

                opt.zero_grad();
                loss.backward();
                torch::nn::utils::clip_grad_norm_(net->parameters(), p.maxGradNorm);
                opt.step();

                // KL early stop (std bound is handled inside meanLogStd's tanh map).
                {
                    torch::NoGradGuard ng;
                    lastKL = (mbOldLp - newLp).mean().item<double>();
                    lastEntropy = entropy.item<double>();
                }
                if (p.targetKL > 0.0 && lastKL > 1.5 * p.targetKL) { stop = true; break; }
            }
        }

        ++update;
        if (update % std::max(1, cfg.train.logEvery) == 0)
            DP_LOG_INFO("upd %4d | step %9lld | entropy %6.3f | KL %7.4f | lrScale %.2f",
                        update, globalStep, lastEntropy, lastKL, lrScale);
    }

    const std::string stem = cfg.train.modelDir + "/" + cfg.train.runName;
    std::error_code mkec; std::filesystem::create_directories(cfg.train.modelDir, mkec);
    torch::save(net, stem + "_torch.pt");                         // native LibTorch
    cfg.saveToFile(stem + "_torch_config.yaml");                  // self-describing sidecar
    exportPortableCkpt(net, obsDim, actDim, stem + "_torch.ckpt"); // loads in dp_simulator
    DP_LOG_INFO("dp_train_torch: done. .pt + portable .ckpt written.");
    return 0;
}
