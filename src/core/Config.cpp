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

namespace {
const char* integratorName(dp::IntegratorType t) {
    return t == dp::IntegratorType::RK4 ? "RK4" : "SemiImplicitEuler";
}
const char* actuatorName(dp::ActuatorMode m) {
    return m == dp::ActuatorMode::Dual ? "Dual" : "Single";
}
const char* taskName(dp::TaskType t) {
    switch (t) {
        case dp::TaskType::SwingUp:   return "SwingUp";
        case dp::TaskType::Stabilize: return "Stabilize";
        case dp::TaskType::Recovery:  return "Recovery";
    }
    return "Stabilize";
}
} // namespace

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
        else if (key == "env.wEnergy")  env.wEnergy  = d();
        else if (key == "env.wTorqueFar")    env.wTorqueFar = d();
        else if (key == "env.torqueShaping") env.torqueShaping = d();
        else if (key == "env.wOmegaFar")     env.wOmegaFar = d();
        else if (key == "env.omegaShaping")  env.omegaShaping = d();
        else if (key == "env.wSettled")      env.wSettled = d();
        else if (key == "env.wBottom")       env.wBottom = d();
        else if (key == "env.bottomShaping") env.bottomShaping = d();
        else if (key == "env.stillnessSharpness") env.stillnessSharpness = d();
        else if (key == "env.staticAngleTol")  env.staticAngleTol = d();
        else if (key == "env.staticVelTol")    env.staticVelTol = d();
        else if (key == "env.staticHoldSteps") env.staticHoldSteps = i();
        else if (key == "env.uprightThreshold") env.uprightThreshold = d();
        else if (key == "env.omegaRefSpeed") env.omegaRefSpeed = d();
        else if (key == "env.domainRandomize") env.domainRandomize = parseBool(val);
        else if (key == "env.drMassPct")    env.drMassPct = d();
        else if (key == "env.drLengthPct")  env.drLengthPct = d();
        else if (key == "env.drGravityPct") env.drGravityPct = d();
        else if (key == "env.drDampingPct") env.drDampingPct = d();
        else if (key == "env.drTimestepPct") env.drTimestepPct = d();
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
        else if (key == "ppo.minLogStd")   ppo.minLogStd = d();
        else if (key == "ppo.maxLogStd")   ppo.maxLogStd = d();
        else if (key == "ppo.targetKL")    ppo.targetKL = d();
        else if (key == "ppo.annealLr")    ppo.annealLr = parseBool(val);
        else if (key == "ppo.clipValueLoss")    ppo.clipValueLoss = parseBool(val);
        else if (key == "ppo.valueClipEps")     ppo.valueClipEps = d();
        else if (key == "ppo.normalizeReturns") ppo.normalizeReturns = parseBool(val);
        else if (key == "ppo.adaptiveEntropy") ppo.adaptiveEntropy = parseBool(val);
        else if (key == "ppo.targetEntropyPerDim") ppo.targetEntropyPerDim = d();
        else if (key == "ppo.minEntropyCoef")  ppo.minEntropyCoef = d();
        else if (key == "ppo.maxEntropyCoef")  ppo.maxEntropyCoef = d();
        else if (key == "ppo.epochs")      ppo.epochs = i();
        else if (key == "ppo.miniBatch")   ppo.miniBatch = i();
        else if (key == "ppo.rolloutSteps")ppo.rolloutSteps = i();
        else if (key == "ppo.numWorkers")  ppo.numWorkers = i();
        else if (key == "ppo.numEnvs")     ppo.numEnvs = i();
        // -- curriculum ------------------------------------------------------
        else if (key == "curriculum.enabled") curriculum.enabled = parseBool(val);
        else if (key == "curriculum.startGravityScale") curriculum.startGravityScale = d();
        else if (key == "curriculum.endGravityScale")   curriculum.endGravityScale = d();
        else if (key == "curriculum.rampUpdates")       curriculum.rampUpdates = i();
        else if (key == "curriculum.startEpisodeSteps") curriculum.startEpisodeSteps = i();
        else if (key == "curriculum.endEpisodeSteps")   curriculum.endEpisodeSteps = i();
        else if (key == "curriculum.rampDisturbances")  curriculum.rampDisturbances = parseBool(val);
        // -- train -----------------------------------------------------------
        else if (key == "train.seed") train.seed = static_cast<std::uint64_t>(std::stoull(val));
        else if (key == "train.totalSteps")    train.totalSteps = ll();
        else if (key == "train.logEvery")      train.logEvery = i();
        else if (key == "train.checkpointEvery") train.checkpointEvery = i();
        else if (key == "train.runName")  train.runName = val;
        else if (key == "train.modelDir") train.modelDir = val;
        else if (key == "train.logDir")   train.logDir = val;
        else if (key == "train.initFrom") train.initFrom = val;
        else DP_LOG_WARN("Config: ignoring unknown key '%s'", key.c_str());
    }
    DP_LOG_INFO("Config: loaded '%s'", path.c_str());
    return true;
}

bool Config::saveToFile(const std::string& path) const {
    std::ofstream os(path, std::ios::trunc);
    if (!os.is_open()) {
        DP_LOG_ERROR("Config: could not write '%s'", path.c_str());
        return false;
    }
    os.precision(10);
    os << "# Auto-generated self-describing config for this run.\n";
    os << "physics.m1: " << physics.m1 << "\nphysics.m2: " << physics.m2 << '\n';
    os << "physics.l1: " << physics.l1 << "\nphysics.l2: " << physics.l2 << '\n';
    os << "physics.i1: " << physics.i1 << "\nphysics.i2: " << physics.i2 << '\n';
    os << "physics.g: " << physics.g << "\nphysics.b1: " << physics.b1
       << "\nphysics.b2: " << physics.b2 << '\n';
    os << "physics.dt: " << physics.dt << "\nphysics.maxTorque: " << physics.maxTorque << '\n';
    os << "physics.integrator: " << integratorName(physics.integrator) << '\n';
    os << "env.actuator: " << actuatorName(env.actuator) << '\n';
    os << "env.task: " << taskName(env.task) << '\n';
    os << "env.maxEpisodeSteps: " << env.maxEpisodeSteps << '\n';
    os << "env.omegaRefSpeed: " << env.omegaRefSpeed << '\n';
    os << "env.stillnessSharpness: " << env.stillnessSharpness << '\n';
    os << "env.wTorqueFar: " << env.wTorqueFar << '\n';
    os << "env.torqueShaping: " << env.torqueShaping << '\n';
    os << "env.wOmegaFar: " << env.wOmegaFar << '\n';
    os << "env.omegaShaping: " << env.omegaShaping << '\n';
    os << "env.wSettled: " << env.wSettled << '\n';
    os << "env.wBottom: " << env.wBottom << '\n';
    os << "env.bottomShaping: " << env.bottomShaping << '\n';
    os << "env.staticAngleTol: " << env.staticAngleTol << '\n';
    os << "env.staticVelTol: " << env.staticVelTol << '\n';
    os << "env.staticHoldSteps: " << env.staticHoldSteps << '\n';
    os << "ppo.hidden1: " << ppo.hidden1 << "\nppo.hidden2: " << ppo.hidden2 << '\n';
    os << "ppo.minLogStd: " << ppo.minLogStd << '\n';
    os << "ppo.maxLogStd: " << ppo.maxLogStd << '\n';
    os << "ppo.clipValueLoss: " << (ppo.clipValueLoss ? "true" : "false") << '\n';
    os << "ppo.valueClipEps: " << ppo.valueClipEps << '\n';
    os << "ppo.normalizeReturns: " << (ppo.normalizeReturns ? "true" : "false") << '\n';
    os << "ppo.minEntropyCoef: " << ppo.minEntropyCoef << '\n';
    os << "ppo.maxEntropyCoef: " << ppo.maxEntropyCoef << '\n';
    return true;
}

} // namespace dp
