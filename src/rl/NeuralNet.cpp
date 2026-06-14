// =============================================================================
// NeuralNet.cpp
// -----------------------------------------------------------------------------
// Linear layer + MLP implementation: forward, manual backprop, Adam, and a
// simple binary-ish text serialization for checkpointing.
// =============================================================================
#include "rl/NeuralNet.hpp"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dp::rl {

// ---------------------------------- Linear -----------------------------------
Linear::Linear(int inDim, int outDim, util::Rng& rng, double gain)
    : inDim_(inDim), outDim_(outDim),
      W_(static_cast<std::size_t>(inDim) * outDim, 0.0),
      b_(outDim, 0.0),
      gW_(W_.size(), 0.0), gb_(outDim, 0.0),
      mW_(W_.size(), 0.0), vW_(W_.size(), 0.0),
      mb_(outDim, 0.0),    vb_(outDim, 0.0) {
    // Xavier/Glorot-style initialization scaled by `gain`. Using a uniform draw
    // from our deterministic RNG keeps initialization reproducible.
    const double limit = gain * std::sqrt(6.0 / (inDim + outDim));
    for (double& w : W_) w = rng.uniform(-limit, limit);
}

Vec Linear::forward(const Vec& x) {
    lastInput_ = x;
    Vec y(outDim_, 0.0);
    for (int o = 0; o < outDim_; ++o) {
        double acc = b_[o];
        const double* wRow = &W_[static_cast<std::size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) acc += wRow[i] * x[i];
        y[o] = acc;
    }
    return y;
}

Vec Linear::backward(const Vec& gradOut) {
    Vec gradIn(inDim_, 0.0);
    for (int o = 0; o < outDim_; ++o) {
        const double go = gradOut[o];
        gb_[o] += go;
        double* gwRow = &gW_[static_cast<std::size_t>(o) * inDim_];
        const double* wRow = &W_[static_cast<std::size_t>(o) * inDim_];
        for (int i = 0; i < inDim_; ++i) {
            gwRow[i] += go * lastInput_[i]; // dL/dW = gradOut (outer) input
            gradIn[i] += go * wRow[i];      // dL/dx = W^T gradOut
        }
    }
    return gradIn;
}

void Linear::zeroGrad() {
    std::fill(gW_.begin(), gW_.end(), 0.0);
    std::fill(gb_.begin(), gb_.end(), 0.0);
}

void Linear::applyAdam(double lr, double beta1, double beta2, double eps, int step) {
    // Bias-correction factors for the Adam moment estimates.
    const double bc1 = 1.0 - std::pow(beta1, step);
    const double bc2 = 1.0 - std::pow(beta2, step);
    auto update = [&](Vec& p, Vec& g, Vec& m, Vec& v) {
        for (std::size_t k = 0; k < p.size(); ++k) {
            m[k] = beta1 * m[k] + (1.0 - beta1) * g[k];
            v[k] = beta2 * v[k] + (1.0 - beta2) * g[k] * g[k];
            const double mHat = m[k] / bc1;
            const double vHat = v[k] / bc2;
            p[k] -= lr * mHat / (std::sqrt(vHat) + eps);
        }
    };
    update(W_, gW_, mW_, vW_);
    update(b_, gb_, mb_, vb_);
}

double Linear::gradNormSquared() const {
    double s = 0.0;
    for (double g : gW_) s += g * g;
    for (double g : gb_) s += g * g;
    return s;
}

void Linear::scaleGrad(double factor) {
    for (double& g : gW_) g *= factor;
    for (double& g : gb_) g *= factor;
}

void Linear::serialize(std::ostream& os) const {
    os << inDim_ << ' ' << outDim_ << '\n';
    for (double w : W_) os << w << ' ';
    os << '\n';
    for (double b : b_) os << b << ' ';
    os << '\n';
}

void Linear::deserialize(std::istream& is) {
    int in, out;
    is >> in >> out;
    if (in != inDim_ || out != outDim_)
        throw std::runtime_error("Linear::deserialize dimension mismatch");
    for (double& w : W_) is >> w;
    for (double& b : b_) is >> b;
}

// ------------------------------------ MLP ------------------------------------
MLP::MLP(int inDim, int hidden1, int hidden2, int outDim, util::Rng& rng)
    : inDim_(inDim), outDim_(outDim),
      // Hidden layers use the standard tanh gain (~5/3); the output head uses a
      // small gain (0.01) so the initial policy mean / value is near zero, which
      // stabilizes the very first PPO updates.
      l1_(inDim, hidden1, rng, 5.0 / 3.0),
      l2_(hidden1, hidden2, rng, 5.0 / 3.0),
      l3_(hidden2, outDim, rng, 0.01) {}

Vec MLP::tanhVec(const Vec& v) {
    Vec out(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) out[k] = std::tanh(v[k]);
    return out;
}

Vec MLP::forward(const Vec& x) {
    preAct1_ = l1_.forward(x);
    act1_    = tanhVec(preAct1_);
    preAct2_ = l2_.forward(act1_);
    act2_    = tanhVec(preAct2_);
    return l3_.forward(act2_); // linear output head
}

Vec MLP::backward(const Vec& gradOutput) {
    // Backprop through output head.
    Vec g = l3_.backward(gradOutput);
    // Through tanh of layer 2: d/dx tanh = 1 - tanh^2.
    for (std::size_t k = 0; k < g.size(); ++k) g[k] *= (1.0 - act2_[k] * act2_[k]);
    g = l2_.backward(g);
    for (std::size_t k = 0; k < g.size(); ++k) g[k] *= (1.0 - act1_[k] * act1_[k]);
    g = l1_.backward(g);
    return g;
}

void MLP::zeroGrad() { l1_.zeroGrad(); l2_.zeroGrad(); l3_.zeroGrad(); }

void MLP::clipGradGlobal(double maxNorm) {
    const double total = std::sqrt(l1_.gradNormSquared()
                                 + l2_.gradNormSquared()
                                 + l3_.gradNormSquared());
    if (total > maxNorm && total > 1e-12) {
        const double scale = maxNorm / total;
        l1_.scaleGrad(scale);
        l2_.scaleGrad(scale);
        l3_.scaleGrad(scale);
    }
}

void MLP::applyAdam(double lr, double beta1, double beta2, double eps, int step) {
    l1_.applyAdam(lr, beta1, beta2, eps, step);
    l2_.applyAdam(lr, beta1, beta2, eps, step);
    l3_.applyAdam(lr, beta1, beta2, eps, step);
}

void MLP::serialize(std::ostream& os) const {
    os << inDim_ << ' ' << outDim_ << '\n';
    l1_.serialize(os);
    l2_.serialize(os);
    l3_.serialize(os);
}

void MLP::deserialize(std::istream& is) {
    int in, out;
    is >> in >> out;
    if (in != inDim_ || out != outDim_)
        throw std::runtime_error("MLP::deserialize dimension mismatch");
    l1_.deserialize(is);
    l2_.deserialize(is);
    l3_.deserialize(is);
}

} // namespace dp::rl
