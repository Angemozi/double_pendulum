// =============================================================================
// train_main.cpp
// -----------------------------------------------------------------------------
// Headless PPO training entry point.
//
// Usage:
//   dp_train [--config <path>] [--steps N] [--seed S] [--run NAME]
//            [--task SwingUp|Stabilize|Recovery] [--dual]
//
// Everything is driven by a Config; CLI flags override file/defaults so quick
// experiments don't require editing YAML.
// =============================================================================
#include <cstdlib>
#include <cstring>
#include <string>

#include "core/Config.hpp"
#include "core/Trainer.hpp"
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
} // namespace

int main(int argc, char** argv) {
    Config cfg;

    if (const char* path = argValue(argc, argv, "--config"))
        cfg.loadFromFile(path);
    if (const char* s = argValue(argc, argv, "--steps"))
        cfg.train.totalSteps = std::atoll(s);
    if (const char* s = argValue(argc, argv, "--seed"))
        cfg.train.seed = static_cast<std::uint64_t>(std::strtoull(s, nullptr, 10));
    if (const char* s = argValue(argc, argv, "--run"))
        cfg.train.runName = s;
    if (const char* s = argValue(argc, argv, "--task")) {
        if (std::strcmp(s, "SwingUp") == 0)   cfg.env.task = TaskType::SwingUp;
        if (std::strcmp(s, "Stabilize") == 0) cfg.env.task = TaskType::Stabilize;
        if (std::strcmp(s, "Recovery") == 0)  cfg.env.task = TaskType::Recovery;
    }
    if (hasFlag(argc, argv, "--dual"))
        cfg.env.actuator = ActuatorMode::Dual;
    if (hasFlag(argc, argv, "--verbose"))
        util::Logger::instance().setLevel(util::LogLevel::Debug);

    Trainer trainer(cfg);
    trainer.runHeadless();
    return 0;
}
