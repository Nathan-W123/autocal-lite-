/// \file EskfCore.hpp
/// \brief State-agnostic error-state Kalman filter machinery.
///
/// Extracted from Aether6's `ErrorStateEkf` (`src/estimation/ErrorStateEkf.cpp`), where these
/// steps were private member functions bound to the 18-state navigation filter. Lifting them
/// out unchanged lets any error-state filter share the exact numerics that Aether6's test
/// suite already verifies:
///
/// - **Discretisation**: second-order series for \f$\Phi\f$ and the trapezoidal (first-order
///   Van Loan) form for \f$Q_d\f$, or the exact Van Loan form on request.
/// - **Update**: Mahalanobis gate, gain from an LDLT solve, Joseph-form covariance update and
///   explicit symmetrisation.
/// - **Reset**: the multiplicative-attitude reset Jacobian \f$G = I - \tfrac12[\delta\theta]_\times\f$
///   (Sola 2017, eq. 285-286), applied to every rotation block of the error state.
#pragma once

#include <vector>

#include "aether/core/Types.hpp"

namespace aether::estimation {

/// \brief Outcome of one linear-Gaussian measurement update.
struct UpdateResult {
  bool accepted = false;  ///< False when the innovation gate (or the LDLT) rejected it.
  double nis = 0.0;       ///< Normalised innovation squared \f$y^\top S^{-1} y\f$.
  VecX dx;                ///< Error-state correction (empty when rejected).
};

/// \brief Discretise the error dynamics \f$\delta\dot x = F\delta x + w\f$ over \a dt.
///
/// \param exact Use Van Loan's exact form instead of the second-order series. At 200 Hz the
///              neglected terms are \f$O(dt^3)\f$; Aether6's tests bound that difference.
/// \param[out] Phi State-transition matrix.
/// \param[out] Qd  Discrete process-noise covariance.
void discretize(const MatX& F, const MatX& Qc, double dt, bool exact, MatX* Phi, MatX* Qd);

/// \brief Kalman update with gating and a Joseph-form covariance update.
///
/// Computes \f$S = HPH^\top + R\f$, rejects the measurement when
/// \f$y^\top S^{-1} y > \texttt{gate\_per\_dof}\cdot\dim(y)\f$ (disabled for gate <= 0),
/// otherwise updates \a P in place with
/// \f$P \leftarrow (I-KH)P(I-KH)^\top + KRK^\top\f$, symmetrises it and returns
/// \f$\delta x = K y\f$. The caller injects \f$\delta x\f$ into its nominal state and then
/// applies the reset Jacobian.
UpdateResult josephUpdate(MatX& P, const VecX& innovation, const MatX& H, const MatX& R,
                          double gate_per_dof);

/// \brief Apply the multiplicative-attitude reset \f$P \leftarrow GPG^\top\f$.
///
/// \param rotation_blocks Start indices of every 3-element rotation-vector block of the error
///                        state, each perturbed on the right, \f$R = \hat R\,\mathrm{Exp}(\delta\theta)\f$.
/// \param dx              The correction that was just injected.
void applyAttitudeReset(MatX& P, const VecX& dx, const std::vector<int>& rotation_blocks);

}  // namespace aether::estimation
