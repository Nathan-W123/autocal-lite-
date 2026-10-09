/// \file LinearAlgebra.hpp
/// \brief Dense linear-algebra helpers: matrix exponential, Van Loan discretisation,
///        symmetrisation and numerical rank.
///
/// Reused from Aether6 (`include/aether/math/LinearAlgebra.hpp`). The Riccati/LQR routines,
/// which only the aircraft autopilot needs, were dropped; everything kept is unchanged.
#pragma once

#include <Eigen/Dense>
#include <stdexcept>
#include <string>

#include "aether/core/Types.hpp"

namespace aether::math {

/// \brief Matrix exponential via scaling-and-squaring with a Pade(6,6) approximant.
///
/// Implements Algorithm 11.3.1 of Golub & Van Loan, *Matrix Computations*. The input is
/// scaled by \f$2^{-s}\f$ until its infinity norm is below 1/2, the Pade approximant is
/// evaluated, and the result is squared \a s times.
///
/// \param A Square matrix [-].
/// \return \f$e^{A}\f$.
MatX expm(const MatX& A);

/// \brief Result of Van Loan's discretisation of a linear stochastic system.
struct DiscretizedSystem {
  MatX Ad;  ///< Discrete state-transition matrix \f$e^{A\Delta t}\f$.
  MatX Qd;  ///< Discrete process-noise covariance \f$\int_0^{\Delta t} e^{A\tau}Q_c e^{A^\top\tau}d\tau\f$.
};

/// \brief Exactly discretise \f$\dot x = Ax + w,\ \mathrm{cov}(w)=Q_c\delta(t)\f$ over \a dt.
///
/// Uses Van Loan's method: form \f$M = \begin{bmatrix}-A & Q_c\\ 0 & A^\top\end{bmatrix}\Delta t\f$,
/// exponentiate, then \f$A_d = (M_{22})^\top\f$ and \f$Q_d = A_d M_{12}\f$.
/// \param A  Continuous system matrix.
/// \param Qc Continuous process-noise power spectral density (symmetric PSD).
/// \param dt Time step [s].
DiscretizedSystem vanLoanDiscretize(const MatX& A, const MatX& Qc, double dt);

/// \brief Force a matrix to be exactly symmetric: \f$(M + M^\top)/2\f$.
inline MatX symmetrize(const MatX& M) { return 0.5 * (M + M.transpose()); }

/// \brief Numerical rank from a rank-revealing SVD with a relative tolerance.
int numericalRank(const MatX& M, double rel_tol = 1e-9);

}  // namespace aether::math
