// Reused from Aether6 (src/math/LinearAlgebra.cpp): expm, Van Loan discretisation and
// numerical rank are unchanged; the Riccati/LQR solvers were dropped.
#include "aether/math/LinearAlgebra.hpp"

#include <algorithm>
#include <cmath>

namespace aether::math {

MatX expm(const MatX& A) {
  const Eigen::Index n = A.rows();
  if (A.cols() != n) throw std::invalid_argument("expm: matrix must be square");
  if (n == 0) return MatX(0, 0);

  const double norm = A.cwiseAbs().rowwise().sum().maxCoeff();  // infinity norm
  int s = 0;
  if (norm > 0.5) s = static_cast<int>(std::ceil(std::log2(norm / 0.5)));
  if (s < 0) s = 0;
  if (s > 60) s = 60;  // guard against pathological inputs
  const MatX As = A / std::pow(2.0, s);

  const int q = 6;
  const MatX I = MatX::Identity(n, n);
  MatX X = As;
  double c = 0.5;
  MatX E = I + c * As;
  MatX D = I - c * As;
  // D(x) = N(-x), so the k-th coefficient enters the denominator with sign (-1)^k:
  // k = 2 adds, k = 3 subtracts, and so on.
  bool plus = true;
  for (int k = 2; k <= q; ++k) {
    c = c * static_cast<double>(q - k + 1) / static_cast<double>(k * (2 * q - k + 1));
    X = As * X;
    const MatX cX = c * X;
    E += cX;
    D = plus ? (D + cX).eval() : (D - cX).eval();
    plus = !plus;
  }
  MatX F = D.partialPivLu().solve(E);
  for (int k = 0; k < s; ++k) F = (F * F).eval();
  return F;
}

DiscretizedSystem vanLoanDiscretize(const MatX& A, const MatX& Qc, double dt) {
  const Eigen::Index n = A.rows();
  MatX M = MatX::Zero(2 * n, 2 * n);
  M.topLeftCorner(n, n) = -A * dt;
  M.topRightCorner(n, n) = Qc * dt;
  M.bottomRightCorner(n, n) = A.transpose() * dt;
  const MatX E = expm(M);

  DiscretizedSystem out;
  out.Ad = E.bottomRightCorner(n, n).transpose();
  out.Qd = symmetrize(out.Ad * E.topRightCorner(n, n));
  return out;
}

int numericalRank(const MatX& M, double rel_tol) {
  if (M.size() == 0) return 0;
  Eigen::JacobiSVD<MatX> svd(M);
  const VecX sv = svd.singularValues();
  const double thresh = sv(0) * rel_tol;
  int r = 0;
  for (Eigen::Index i = 0; i < sv.size(); ++i)
    if (sv(i) > thresh) ++r;
  return r;
}

}  // namespace aether::math
