// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
/// \file NumericalDiff.hpp
/// \brief Central-difference Jacobians with per-component step scaling.
#pragma once

#include <cmath>
#include <functional>

#include "aether/core/Types.hpp"

namespace aether::math {

/// Vector-valued function used for numerical differentiation.
using VectorFunction = std::function<VecX(const VecX&)>;

/// \brief Central-difference Jacobian \f$J_{ij} = \partial f_i/\partial x_j\f$.
///
/// The perturbation applied to component \a j is
/// \f$h_j = \epsilon\,\max(|x_j|, x_{scale})\f$, which keeps the relative step sensible for
/// both small and large components. Central differences give \f$O(h^2)\f$ truncation error;
/// the default \f$\epsilon=10^{-6}\f$ balances that against round-off for double precision.
///
/// \param f       Function to differentiate.
/// \param x       Evaluation point.
/// \param epsilon Relative perturbation size [-].
/// \param x_scale Floor on the perturbation scale, in the units of \a x.
inline MatX numericalJacobian(const VectorFunction& f, const VecX& x, double epsilon = 1e-6,
                              double x_scale = 1e-3) {
  const VecX f0 = f(x);
  MatX J(f0.size(), x.size());
  VecX xp = x;
  for (Eigen::Index j = 0; j < x.size(); ++j) {
    const double h = epsilon * std::max(std::abs(x(j)), x_scale);
    const double orig = x(j);
    xp(j) = orig + h;
    const VecX fp = f(xp);
    xp(j) = orig - h;
    const VecX fm = f(xp);
    xp(j) = orig;
    J.col(j) = (fp - fm) / (2.0 * h);
  }
  return J;
}

}  // namespace aether::math
