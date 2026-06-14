// =============================================================================
// Config.cpp
// -----------------------------------------------------------------------------
// Flat-YAML loader implementation. Parses `key: value` lines, ignores blank
// lines and `#` comments, trims whitespace, and dispatches into the nested
// config structs. Enums accept human-readable string values.
// =============================================================================
#include "core/Config.hpp"
#include "util/Logger.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

namespace dp {
namespace {

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

bool parseBool(const std::string& v) {
    std::string lo = v;
    std::transform(lo.begin(), lo.end(), lo.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lo == "true" || lo == "1" || lo == "yes" || lo == "on";
}

} // namespace

bool Config::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        DP_LOG_ERROR("Config: could not open '%s'", path.c_str());
        return false;
    }

    auto setEnumInteg = [&](const std::string& v) {
        if (v == "RK4") physics.integrator = IntegratorType::RK4;
        else if (v == "SemiImplicitEuler" || v == "Euler") physics.integrator = IntegratorType::SemiImplicitEuler;
        else DP_LOG_WARN("Config: unknown integrator '%s'", v.c_str());
    };
    auto setEnumAct = [&](const std::string& v) {
        if (v == "Single") env.actuator = ActuatorMode::Single;
        else if (v == "Dual") env.actuator = ActuatorMode::Dual;
        else DP_LOG_WARN("Config: unknown actuator '%s'", v.c_str());
    };
    auto setEnumTask = [&](const std::string& v) {
        if (v == "SwingUp") env.task = TaskType::SwingUp;
        else if (v == "Stabilize") env.task = TaskType::Stabilize;
        else if (v == "Recovery") env.task = TaskType::Recovery;
        else DP_LOG_WARN("Config: unknown task '%s'", v.c_str());
    };

    std::string line;
    while (std::getline(file, line)) {
        // Strip inline comments and surrounding whitespace.
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if (val.empty()) continue;

        auto d  = [&] { return std::stod(val); };
        auto i  = [&] { return std::stoi(val); };
        auto ll = [&] { return static_cast<long long>(std::stoll(val)); };

        // -- physics ---------------------------------------------------------
        if      (key == "physics.m1") physics.m1 = d();
        else if (key == "physics.m2") physics.m2 = d();
        else if (key == "physics.l1") physics.l1 = d();
        else if (key == "physics.l2") physics.l2 = d();
        else if (key == "physics.i1") physics.i1 = d();
        else if (key == "physics.i2") physics.i2 = d();
        else if (key == "physics.g")  physics.g  = d();
        else if (key == "physics.b1") physics.b1 = d();
        else if (key == "physics.b2") physics.b2 = d();
        else if (key == "physics.dt") physics.dt = d();
        else if (key == "physics.maxTorque") physics.maxTorque = d();
        else if (key == "physics.integrator") setEnumInteg(val);
        // -- env -------------------------------------------------------------
        else if (key == "env.actuator") setEnumAct(val);
        else if (key == "env.task") setEnumTask(val);
        else if (key == "env.maxEpisodeSteps") env.maxEpisodeSteps = i();
        else if (key == "env.angularSpeedLimit") env.angularSpeedLimit = d();
        else if (key == "env.wUpright") env.wUpright = d();
        else if (key == "env.wTorque")  env.wTorque  = d();
        else if (key == "env.wOmega")   env.wOmega   = d();
        else if (key == "env.wSurvival")env.wSurvival= d();
        else if (key == "env.uprightThreshold") env.uprightThreshold = d();
        else if (key == "env.domainRandomize") env.domainRandomize = parseBool(val);
        else if (key == "env.drMassPct")    env.drMassPct = d();
        else if (key == "env.drLengthPct")  env.drLengthPct = d();
        else if (key == "env.drGravityPct") env.drGravityPct = d();
        else if (key == "env.drDampingPct") env.drDampingPct = d();
        else if (key == "env.disturbances") env.disturbances = parseBool(val);
        else if (key == "env.disturbProb")  env.disturbProb = d();
        else if (key == "env.disturbMagnitude") env.disturbMagnitude = d();
        // -- ppo -------------------------------------------------------------
        else if (key == "ppo.hidden1") ppo.hidden1 = i();
        else if (key == "ppo.hidden2") ppo.hidden2 = i();
        else if (key == "ppo.gamma")   ppo.gamma = d();
        else if (key == "ppo.lambda")  ppo.lambda = d();
        else if (key == "ppo.clipEps") ppo.clipEps = d();
        else if (key == "ppo.lrActor") ppo.lrActor = d();
        else if (key == "ppo.lrCritic")ppo.lrCritic = d();
        else if (key == "ppo.entropyCoef") ppo.entropyCoef = d();
        else if (key == "ppo.valueCoef")   ppo.valueCoef = d();
        else if (key == "ppo.maxGradNorm") ppo.maxGradNorm = d();
        else if (key == "ppo.initLogStd")  ppo.initLogStd = d();
        else if (key == "ppo.epochs")      ppo.epochs = i();
        else if (key == "ppo.miniBatch")   ppo.miniBatch = i();
        else if (key == "ppo.rolloutSteps")ppo.rolloutSteps = i();
        // -- curriculum ------------------------------------------------------
        else if (key == "curriculum.enabled") curriculum.enabled = parseBool(val);
        else if (key == "curriculum.startGravityScale") curriculum.startGravityScale = d();
        else if (key == "curriculum.endGravityScale")   curriculum.endGravityScale = d();
        else if (key == "curriculum.rampUpdates")       curriculum.rampUpdates = i();
        else if (key == "curriculum.startEpisodeSteps") curriculum.startEpisodeSteps = i();
        else if (key == "curriculum.endEpisodeSteps")   curriculum.endEpisodeSteps = i();
        // -- train -----------------------------------------------------------
        else if (key == "train.seed") train.seed = static_cast<std::uint64_t>(std::stoull(val));
        else if (key == "train.totalSteps")    train.totalSteps = ll();
        else if (key == "train.logEvery")      train.logEvery = i();
        else if (key == "train.checkpointEvery") train.checkpointEvery = i();
        else if (key == "train.runName")  train.runName = val;
        else if (key == "train.modelDir") train.modelDir = val;
        else if (key == "train.logDir")   train.logDir = val;
        else DP_LOG_WARN("Config: ignoring unknown key '%s'", key.c_str());
    }
    DP_LOG_INFO("Config: loaded '%s'", path.c_str());
    return true;
}

} // namespace dp
