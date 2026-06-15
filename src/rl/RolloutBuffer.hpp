// =============================================================================
// RolloutBuffer.hpp
// -----------------------------------------------------------------------------
// Fixed-capacity on-policy rollout storage for PPO, plus Generalized Advantage
// Estimation (GAE).
//
// PPO is on-policy: we collect a batch of transitions with the CURRENT policy,
// compute advantages, run a few optimization epochs, then discard the batch.
// This buffer stores exactly that batch and is reused (no per-step allocation)
// across updates.
//
// GAE (Schulman et al. 2016) trades bias for variance via lambda:
//     delta_t   = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
//     A_t       = delta_t + gamma * lambda * (1 - done) * A_{t+1}
//     return_t  = A_t + V(s_t)            (the critic's regression target)
// =============================================================================
#pragma once

#include <vector>
#include <cmath>
#include "rl/NeuralNet.hpp" // for Vec

namespace dp::rl {

struct Transition {
    Vec    observation;
    Vec    action;        // raw sampled action (pre-clamp), for log-prob recompute
    double logProb = 0.0; // log pi_old(a|s) at collection time
    double value   = 0.0; // V_old(s)
    double reward  = 0.0;
    bool   done    = false; // terminal OR truncated boundary (bootstrap cut)
    // --- episode-boundary handling for GAE ---
    bool   truncated = false; // time limit cut: bootstrap from nextValue, no carry
    double nextValue = 0.0; // V(s_{t+1}) of the SAME episode, used at a truncation

    // Filled in by computeGAE():
    double advantage = 0.0;
    double ret       = 0.0; // return target = advantage + value
};

class RolloutBuffer {
public:
    void reserve(std::size_t capacity) { data_.reserve(capacity); }
    void clear() { data_.clear(); }
    std::size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    void add(const Transition& t) { data_.push_back(t); }
    void add(Transition&& t) { data_.push_back(std::move(t)); }

    Transition&       operator[](std::size_t i)       { return data_[i]; }
    const Transition& operator[](std::size_t i) const { return data_[i]; }

    std::vector<Transition>&       data()       { return data_; }
    const std::vector<Transition>& data() const { return data_; }

    // Compute GAE advantages and return targets in place. `lastValue` is the
    // critic's estimate of the state AFTER the final stored transition, used to
    // bootstrap when the rollout was cut mid-episode (not terminal).
    void computeGAE(double gamma, double lambda, double lastValue) {
        const std::size_t n = data_.size();
        double nextValue = lastValue;
        double nextAdv   = 0.0;
        for (std::size_t idx = n; idx-- > 0;) {
            Transition& t = data_[idx];

            // Decide the bootstrap value and the advantage carry for THIS step,
            // resetting both at episode boundaries so advantage never leaks across
            // an episode (which the old single-mask recursion did at truncations).
            double vNext, advCarry;
            if (t.done) {
                vNext = 0.0;            // true terminal: no future value
                advCarry = 0.0;         // chain resets
            } else if (t.truncated) {
                vNext = t.nextValue;    // time-limit cut: bootstrap from the real next state
                advCarry = 0.0;         // but the next episode's advantage must NOT flow back
            } else {
                vNext = nextValue;      // mid-episode: chain normally
                advCarry = nextAdv;
            }

            // vNext / advCarry already encode the episode boundary, so no extra
            // mask is needed: a true terminal contributes V=0 and breaks the
            // advantage chain, while a time-limit truncation bootstraps from the
            // real next-state value but still breaks the chain (the next episode
            // must not leak advantage backward across the cut).
            const double delta = t.reward + gamma * vNext - t.value;
            t.advantage = delta + gamma * lambda * advCarry;
            t.ret       = t.advantage + t.value;
            nextValue = t.value;
            nextAdv   = t.advantage;
        }
    }

    // Normalize advantages to zero mean / unit variance. This is a standard PPO
    // trick that decouples the learning rate from the reward scale and markedly
    // stabilizes training.
    void normalizeAdvantages() {
        if (data_.empty()) return;
        double mean = 0.0;
        for (const auto& t : data_) mean += t.advantage;
        mean /= static_cast<double>(data_.size());
        double var = 0.0;
        for (const auto& t : data_) {
            const double d = t.advantage - mean;
            var += d * d;
        }
        var /= static_cast<double>(data_.size());
        const double std = std::sqrt(var) + 1e-8;
        for (auto& t : data_) t.advantage = (t.advantage - mean) / std;
    }

private:
    std::vector<Transition> data_;
};

} // namespace dp::rl
