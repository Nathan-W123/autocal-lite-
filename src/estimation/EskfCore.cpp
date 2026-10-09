#include "aether/estimation/EskfCore.hpp"

#include "aether/math/LinearAlgebra.hpp"
#include "aether/math/Rotation.hpp"

namespace aether::estimation {

void discretize(const MatX& F, const MatX& Qc, double dt, bool exact, MatX* Phi, MatX* Qd) {
  if (exact) {
    const auto d = math::vanLoanDiscretize(F, Qc, dt);
    *Phi = d.Ad;
    *Qd = d.Qd;
    return;
  }
  // Second-order series for the transition matrix and the trapezoidal (van-Loan first-order)
  // form for the discrete process noise (Aether6 ErrorStateEkf::predict).
  const MatX Fdt = F * dt;
  *Phi = MatX::Identity(F.rows(), F.cols()) + Fdt + 0.5 * Fdt * Fdt;
  *Qd = Qc * dt + 0.5 * dt * dt * (F * Qc + Qc * F.transpose());
}

UpdateResult josephUpdate(MatX& P, const VecX& innovation, const MatX& H, const MatX& R,
                          double gate_per_dof) {
  UpdateResult out;
  const MatX S = H * P * H.transpose() + R;
  const Eigen::LDLT<MatX> ldlt(S);
  if (ldlt.info() != Eigen::Success) return out;

  out.nis = innovation.dot(ldlt.solve(innovation));
  if (gate_per_dof > 0.0 && out.nis > gate_per_dof * static_cast<double>(innovation.size()))
    return out;

  const MatX K = P * H.transpose() * ldlt.solve(MatX::Identity(S.rows(), S.cols()));
  out.dx = K * innovation;

  // Joseph form keeps P symmetric and positive semi-definite even with a suboptimal gain.
  const MatX IKH = MatX::Identity(P.rows(), P.cols()) - K * H;
  P = IKH * P * IKH.transpose() + K * R * K.transpose();
  P = math::symmetrize(P);
  out.accepted = true;
  return out;
}

void applyAttitudeReset(MatX& P, const VecX& dx, const std::vector<int>& rotation_blocks) {
  // G is the identity except for I - 0.5[dtheta]x on each rotation block, so G P G^T only
  // touches the rows and columns of those blocks.
  MatX G = MatX::Identity(P.rows(), P.cols());
  for (int b : rotation_blocks)
    G.block<3, 3>(b, b) = Mat3::Identity() - 0.5 * math::skew(dx.segment<3>(b));
  P = math::symmetrize(G * P * G.transpose());
}

}  // namespace aether::estimation
