// =============================================================================
// Trainer.cpp
// -----------------------------------------------------------------------------
// Implementation of the PPO training loop. See header for the high-level design.
// =============================================================================
#include "core/Trainer.hpp"
#include "util/Math.hpp"

#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <filesystem>

namespace dp {

Trainer::Trainer(const Config& cfg, SharedState* shared)
    : cfg_(cfg),
      env_(cfg),
      agent_(env_.observationDim(), env_.actionDim(), cfg.ppo, cfg.train.seed ^ 0x9E3779B9ULL),
      shared_(shared) {
    buffer_.reserve(static_cast<std::size_t>(cfg_.ppo.rolloutSteps) + 1);

    // Ensure the output directories exist so checkpoint/CSV writes don't fail
    // when the program is launched from an arbitrary working directory.
    std::error_code ec;
    std::filesystem::create_directories(cfg_.train.modelDir, ec);
    std::filesystem::create_directories(cfg_.train.logDir, ec);

    // Open the statistics CSV (reward curve + diagnostics).
    const std::string csvPath = cfg_.train.logDir + "/" + cfg_.train.runName + "_stats.csv";
    csv_.open(csvPath, {"update", "globalStep", "avgReturn100", "policyLoss",
                        "valueLoss", "entropy", "approxKL", "clipFraction",
                        "meanStd", "simRate"});

    obs_ = env_.reset();
    applyCurriculum();
}

void Trainer::applyCurriculum() {
    if (!cfg_.curriculum.enabled) return;
    // Linearly ramp difficulty from "easy" to "full" over rampUpdates updates.
    const CurriculumConfig& c = cfg_.curriculum;
    const double t = std::min(1.0, static_cast<double>(updateCount_) /
                                   std::max(1, c.rampUpdates));
    const double gScale = math::lerp(c.startGravityScale, c.endGravityScale, t);
    const int    steps  = static_cast<int>(
        math::lerp(c.startEpisodeSteps, c.endEpisodeSteps, t));
    env_.setDifficulty(gScale, steps);
}

void Trainer::pushEpisodeReturn(double ret) {
    recentReturns_.push_back(ret);
    if (recentReturns_.size() > 100) recentReturns_.pop_front();
}

double Trainer::avgReturn() const {
    if (recentReturns_.empty()) return 0.0;
    return std::accumulate(recentReturns_.begin(), recentReturns_.end(), 0.0)
         / static_cast<double>(recentReturns_.size());
}

void Trainer::publishSnapshot(const rl::PolicyOutput& po, const StepResult& sr,
                              const Action& torque) {
    if (!shared_) return;
    RenderSnapshot snap;
    snap.state        = env_.getState();
    snap.kinematics   = env_.physics().kinematics();
    snap.reward       = sr.reward;
    snap.uprightScore = sr.uprightScore;
    snap.energy       = sr.energy;
    snap.globalStep   = globalStep_;
    snap.episode      = episodeIndex_;
    snap.episodeStep  = env_.episodeStep();
    snap.lastObservation = obs_;
    snap.lastActionRaw   = po.action;
    snap.lastTorque      = { torque.torque1, torque.torque2 };
    snap.episodeReturn   = episodeReturn_;
    snap.avgReturn100    = avgReturn();
    snap.policyLoss      = lastStats_.policyLoss;
    snap.valueLoss       = lastStats_.valueLoss;
    snap.entropy         = lastStats_.entropy;
    snap.meanStd         = lastStats_.meanStd;
    snap.simRate         = simRate_.rate();
    shared_->publish(snap);
}

rl::UpdateStats Trainer::collectAndUpdate() {
    buffer_.clear();
    util::StopWatch rolloutWatch;
    int stepsThisRollout = 0;

    for (int i = 0; i < cfg_.ppo.rolloutSteps; ++i) {
        // ---- Honor UI controls (pause / frame-step / reset / quit) ----------
        if (shared_) {
            if (shared_->control.quit.load()) break;
            if (shared_->control.requestReset.exchange(false)) {
                obs_ = env_.reset();
                episodeReturn_ = 0.0;
            }
            if (shared_->control.injectDisturbance.exchange(false)) {
                State s = env_.getState();
                s.omega1 += 3.0; s.omega2 -= 3.0;
                env_.physics().setState(s);
            }
            // When paused, busy-wait politely unless a single frame-step is asked.
            while (shared_->control.pause.load() && !shared_->control.frameStep.load()
                   && !shared_->control.quit.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            shared_->control.frameStep.store(false);
        }

        // ---- Policy step ----------------------------------------------------
        const rl::PolicyOutput po = agent_.act(obs_, /*deterministic=*/false);
        const Action torque = env_.decode(po.action);
        const StepResult sr = env_.step(torque);

        // ---- Store transition ----------------------------------------------
        rl::Transition tr;
        tr.observation = obs_;
        tr.action      = po.action;
        tr.logProb     = po.logProb;
        tr.value       = po.value;
        tr.reward      = sr.reward;
        // A transition is "done" for bootstrapping if the episode ended for ANY
        // reason at this step; we still bootstrap truncation via the next value
        // below, but a true terminal cuts the value to zero.
        tr.done        = sr.terminal;
        buffer_.add(std::move(tr));

        episodeReturn_ += sr.reward;
        obs_ = sr.observation;
        ++globalStep_;
        ++stepsThisRollout;

        publishSnapshot(po, sr, torque);

        // ---- Episode boundary ----------------------------------------------
        if (sr.terminal || sr.truncated) {
            pushEpisodeReturn(episodeReturn_);
            episodeReturn_ = 0.0;
            ++episodeIndex_;
            obs_ = env_.reset();
        }

        // Optional slow-motion / speed throttle for the UI (no effect headless).
        if (shared_) {
            if (shared_->control.slowMotion.load())
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
        }
    }

    simRate_.update(stepsThisRollout, rolloutWatch.seconds());

    // ---- Bootstrap value for GAE on the final (possibly non-terminal) state --
    const double lastValue = agent_.value(obs_);
    buffer_.computeGAE(cfg_.ppo.gamma, cfg_.ppo.lambda, lastValue);

    // ---- PPO update -----------------------------------------------------------
    lastStats_ = agent_.update(buffer_);
    ++updateCount_;
    applyCurriculum();

    // ---- Logging --------------------------------------------------------------
    if (updateCount_ % std::max(1, cfg_.train.logEvery) == 0) {
        DP_LOG_INFO("upd %4d | step %9lld | avgRet100 %8.2f | pLoss %7.4f | "
                    "vLoss %7.4f | ent %6.3f | KL %6.4f | clip %4.2f | std %5.3f | %6.0f sps",
                    updateCount_, globalStep_, avgReturn(), lastStats_.policyLoss,
                    lastStats_.valueLoss, lastStats_.entropy, lastStats_.approxKL,
                    lastStats_.clipFraction, lastStats_.meanStd, simRate_.rate());
        if (csv_.isOpen())
            csv_.writeRow(updateCount_, globalStep_, avgReturn(), lastStats_.policyLoss,
                          lastStats_.valueLoss, lastStats_.entropy, lastStats_.approxKL,
                          lastStats_.clipFraction, lastStats_.meanStd, simRate_.rate());
    }

    // ---- Checkpoint -----------------------------------------------------------
    if (updateCount_ % std::max(1, cfg_.train.checkpointEvery) == 0) {
        const std::string path = cfg_.train.modelDir + "/" + cfg_.train.runName + "_latest.ckpt";
        agent_.save(path);
    }
    return lastStats_;
}

double Trainer::runHeadless() {
    DP_LOG_INFO("Trainer: starting headless run '%s' for %lld steps",
                cfg_.train.runName.c_str(), cfg_.train.totalSteps);
    util::StopWatch wall;
    while (globalStep_ < cfg_.train.totalSteps) {
        if (shared_ && shared_->control.quit.load()) break;
        collectAndUpdate();
    }
    // Final checkpoint.
    const std::string path = cfg_.train.modelDir + "/" + cfg_.train.runName + "_final.ckpt";
    agent_.save(path);
    DP_LOG_INFO("Trainer: done. %lld steps in %.1fs (%.0f steps/s). final avgRet100=%.2f",
                globalStep_, wall.seconds(),
                globalStep_ / std::max(1e-9, wall.seconds()), avgReturn());
    return avgReturn();
}

double Trainer::evaluateEpisode(bool publishSnapshots) {
    Observation obs = env_.reset();
    double ret = 0.0;
    for (int i = 0; i < cfg_.env.maxEpisodeSteps; ++i) {
        const rl::PolicyOutput po = agent_.act(obs, /*deterministic=*/true);
        const Action torque = env_.decode(po.action);
        const StepResult sr = env_.step(torque);
        ret += sr.reward;
        obs = sr.observation;
        if (publishSnapshots && shared_) {
            obs_ = obs;
            publishSnapshot(po, sr, torque);
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        if (sr.terminal || sr.truncated) break;
    }
    return ret;
}

} // namespace dp
