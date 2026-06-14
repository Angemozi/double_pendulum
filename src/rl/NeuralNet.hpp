// =============================================================================
// NeuralNet.hpp
// -----------------------------------------------------------------------------
// A tiny, self-contained multilayer perceptron with manual backpropagation and
// the Adam optimizer. No external dependencies.
//
// WHY HAND-ROLLED (instead of LibTorch):
//   The hard requirements are determinism ("same seed => same results") and
//   "compiles cleanly everywhere". A hand-written MLP with our own deterministic
//   RNG gives bit-reproducible training across MSVC/GCC/Clang and Windows/Linux,
//   and keeps the core build dependency-free. A LibTorch backend can be dropped
//   in later behind a CMake flag (see DP_WITH_LIBTORCH) without touching the PPO
//   algorithm, because the agent only depends on this small interface.
//
// ARCHITECTURE:
//   input -> Linear -> tanh -> Linear -> tanh -> Linear(output, no activation)
//   tanh hidden activations are standard for continuous-control policies: bounded
//   and smooth, which keeps gradients well-scaled.
// =============================================================================
#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <ostream>
#include <istream>

#include "util/Random.hpp"

namespace dp::rl {

using Vec = std::vector<double>;

// A single fully-connected layer y = W x + b, with Adam optimizer state.
// Weights are stored row-major: W[o*inDim + i].
class Linear {
public:
    Linear() = default;
    Linear(int inDim, int outDim, util::Rng& rng, double gain);

    // Forward pass. Caches the input so backward() can compute weight grads.
    Vec forward(const Vec& x);

    // Backward pass. `gradOut` is dL/dy. Returns dL/dx and accumulates dL/dW,
    // dL/db internally (call zeroGrad() before a batch, applyAdam() after).
    Vec backward(const Vec& gradOut);

    void zeroGrad();
    void applyAdam(double lr, double beta1, double beta2, double eps, int step);

    // Global L2 norm of accumulated parameter gradients (for grad clipping).
    double gradNormSquared() const;
    void   scaleGrad(double factor);

    int inDim()  const { return inDim_; }
    int outDim() const { return outDim_; }

    void serialize(std::ostream& os) const;
    void deserialize(std::istream& is);

private:
    int inDim_ = 0, outDim_ = 0;
    Vec W_, b_;          // parameters
    Vec gW_, gb_;        // accumulated gradients
    Vec mW_, vW_, mb_, vb_; // Adam moments
    Vec lastInput_;      // cached for backward
};

// Multilayer perceptron: two tanh hidden layers + linear output head.
class MLP {
public:
    MLP() = default;
    MLP(int inDim, int hidden1, int hidden2, int outDim, util::Rng& rng);

    // Forward, caching all intermediate activations for a subsequent backward().
    Vec forward(const Vec& x);

    // Backward from output gradient; returns gradient w.r.t. the input. Param
    // gradients are accumulated in the layers.
    Vec backward(const Vec& gradOutput);

    void zeroGrad();
    void clipGradGlobal(double maxNorm);
    void applyAdam(double lr, double beta1, double beta2, double eps, int step);

    int inDim()  const { return inDim_; }
    int outDim() const { return outDim_; }

    void serialize(std::ostream& os) const;
    void deserialize(std::istream& is);

private:
    static Vec tanhVec(const Vec& v);

    int inDim_ = 0, outDim_ = 0;
    Linear l1_, l2_, l3_;
    // Cached activations for backprop.
    Vec preAct1_, act1_, preAct2_, act2_;
};

} // namespace dp::rl
