/// \file test_eskf_core.cpp
/// \brief State-agnostic error-state Kalman filter steps (`estimation/EskfCore.hpp`):
///        gated Joseph-form update, covariance discretisation and the multiplicative
///        attitude reset.
///
/// The discretisation-agreement case is ported from Aether6's
/// `tests/test_estimation.cpp` ("exact and approximate covariance discretisation agree at
/// typical IMU rates"), rewritten against the free functions that were lifted out of
/// Aether6's `ErrorStateEkf`.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "aether/core/Constants.hpp"
#include "aether/estimation/EskfCore.hpp"
#include "aether/math/LinearAlgebra.hpp"
#include "aether/math/NumericalDiff.hpp"
#include "aether/math/Rotation.hpp"

using namespace aether;
using Catch::Approx;

namespace {

MatX randomMatrix(Eigen::Index rows, Eigen::Index cols, std::mt19937_64& rng,
                  double sigma = 1.0) {
  std::normal_distribution<double> n(0.0, sigma);
  MatX M(rows, cols);
  for (Eigen::Index i = 0; i < M.size(); ++i) M(i) = n(rng);
  return M;
}

/// Symmetric positive-definite matrix L L^T + floor*I (condition number ~ 10^2 for n ~ 10),
/// symmetrised so it is bitwise symmetric whatever the GEMM summation order: the "untouched"
/// checks below compare entries exactly.
MatX randomSpd(Eigen::Index n, std::mt19937_64& rng, double floor = 0.5) {
  const MatX L = randomMatrix(n, n, rng);
  return math::symmetrize(L * L.transpose() + floor * MatX::Identity(n, n));
}

double minEigenvalue(const MatX& M) {
  Eigen::SelfAdjointEigenSolver<MatX> es(math::symmetrize(M));
  return es.eigenvalues().minCoeff();
}

/// Max-abs difference, used for "untouched" (exactly equal) checks.
double maxAbsDiff(const MatX& a, const MatX& b) { return (a - b).cwiseAbs().maxCoeff(); }

/// Reset Jacobian built element by element (independently of math::skew):
/// G = I - 0.5 [d]x on the 3x3 block starting at b.
MatX resetJacobian(Eigen::Index n, const VecX& dx, const std::vector<int>& blocks) {
  MatX G = MatX::Identity(n, n);
  for (int b : blocks) {
    const double x = dx(b), y = dx(b + 1), z = dx(b + 2);
    G(b + 0, b + 1) = 0.5 * z;
    G(b + 0, b + 2) = -0.5 * y;
    G(b + 1, b + 0) = -0.5 * z;
    G(b + 1, b + 2) = 0.5 * x;
    G(b + 2, b + 0) = 0.5 * y;
    G(b + 2, b + 1) = -0.5 * x;
  }
  return G;
}

}  // namespace

// ------------------------------------------------------------------------------------------
// josephUpdate
// ------------------------------------------------------------------------------------------

TEST_CASE("Joseph update equals the textbook Kalman update with the optimal gain",
          "[ekf][update]") {
  std::mt19937_64 rng(20261009);
  for (int trial = 0; trial < 20; ++trial) {
    const Eigen::Index n = 9, m = 4;
    const MatX P0 = randomSpd(n, rng);
    const MatX H = randomMatrix(m, n, rng);
    const MatX R = randomSpd(m, rng, 0.2);
    const VecX y = randomMatrix(m, 1, rng);

    // Textbook form: S = H P H^T + R, K = P H^T S^-1, P+ = (I - K H) P, dx = K y.
    const MatX S = H * P0 * H.transpose() + R;
    const MatX Sinv = S.inverse();
    const MatX K = P0 * H.transpose() * Sinv;
    const MatX P_textbook = (MatX::Identity(n, n) - K * H) * P0;
    const VecX dx_textbook = K * y;
    const double nis_textbook = y.dot(Sinv * y);

    MatX P = P0;
    const auto res = estimation::josephUpdate(P, y, H, R, 0.0);
    INFO("trial " << trial);
    REQUIRE(res.accepted);
    // With the optimal gain the Joseph form is algebraically identical to (I-KH)P; the two
    // only differ by round-off, O(eps * cond(S) * ||P||) with cond(S) ~ 1e2-1e3 here.
    REQUIRE(maxAbsDiff(P, P_textbook) / P0.norm() < 1e-11);
    REQUIRE((res.dx - dx_textbook).norm() / dx_textbook.norm() < 1e-11);
    REQUIRE(res.nis == Approx(nis_textbook).epsilon(1e-11));
    // Information can only be gained: P0 - P+ = K S K^T is positive semi-definite.
    REQUIRE(minEigenvalue(P0 - P) > -1e-12 * P0.norm());
  }
}

TEST_CASE("Joseph update leaves a covariance exactly symmetric and positive semi-definite",
          "[ekf][update][covariance]") {
  std::mt19937_64 rng(5);
  const Eigen::Index n = 12;
  MatX P = randomSpd(n, rng);
  // 300 sequential scalar-to-3D updates with random geometry, the last third with a nearly
  // noise-free sensor (R = 1e-10 I), which drives directions of P towards zero: the regime
  // where the plain (I-KH)P form loses symmetry and definiteness.
  for (int k = 0; k < 300; ++k) {
    const Eigen::Index m = 1 + k % 3;
    const MatX H = randomMatrix(m, n, rng);
    const MatX R = (k < 200 ? 0.3 : 1e-10) * MatX::Identity(m, m);
    const VecX y = randomMatrix(m, 1, rng, 0.1);
    const auto res = estimation::josephUpdate(P, y, H, R, 0.0);
    REQUIRE(res.accepted);
    REQUIRE((P - P.transpose()).cwiseAbs().maxCoeff() == 0.0);  // explicitly symmetrised
    // PSD up to the round-off of an n=12 eigen-decomposition relative to ||P||.
    INFO("update " << k << " min eigenvalue " << minEigenvalue(P));
    REQUIRE(minEigenvalue(P) > -1e-12 * std::max(1.0, P.norm()));
  }
}

TEST_CASE("Joseph update does not touch uncorrelated and unobserved states", "[ekf][update]") {
  // Block-diagonal P with H seeing only the first block: the gain rows of the second block
  // are exactly zero, so its covariance must come back unchanged.
  std::mt19937_64 rng(31);
  MatX P0 = MatX::Zero(10, 10);
  P0.topLeftCorner(6, 6) = randomSpd(6, rng);
  P0.bottomRightCorner(4, 4) = randomSpd(4, rng);
  MatX H = MatX::Zero(3, 10);
  H.leftCols(6) = randomMatrix(3, 6, rng);
  MatX P = P0;
  const auto res = estimation::josephUpdate(P, randomMatrix(3, 1, rng), H,
                                            0.5 * MatX::Identity(3, 3), 0.0);
  REQUIRE(res.accepted);
  REQUIRE(maxAbsDiff(P.bottomRightCorner(4, 4), P0.bottomRightCorner(4, 4)) < 1e-15);
  REQUIRE(P.topRightCorner(6, 4).cwiseAbs().maxCoeff() < 1e-15);
  REQUIRE(res.dx.tail(4).cwiseAbs().maxCoeff() < 1e-15);
  REQUIRE(res.dx.head(6).norm() > 0.0);
}

TEST_CASE("the innovation gate rejects when NIS exceeds gate times dimension",
          "[ekf][update][gate]") {
  // P = I, H selects two states, R = I: S = 2 I, so y = (2, 2) has NIS = 4 exactly and the
  // threshold gate*dim = 2*gate is crossed at gate = 2.
  MatX P0 = MatX::Identity(3, 3);
  P0(0, 2) = P0(2, 0) = 0.25;  // some correlation so an accepted update would change P
  MatX H = MatX::Zero(2, 3);
  H(0, 0) = 1.0;
  H(1, 1) = 1.0;
  const MatX R = MatX::Identity(2, 2);
  const VecX y = Vec2(2.0, 2.0);

  SECTION("above the gate: rejected with P untouched and NIS still reported") {
    MatX P = P0;
    const auto res = estimation::josephUpdate(P, y, H, R, 1.9);  // 4 > 3.8
    REQUIRE_FALSE(res.accepted);
    REQUIRE(res.nis == 4.0);
    REQUIRE(res.dx.size() == 0);
    REQUIRE(maxAbsDiff(P, P0) == 0.0);  // bitwise unchanged
  }
  SECTION("exactly on the gate: accepted (the test is strictly greater-than)") {
    MatX P = P0;
    const auto res = estimation::josephUpdate(P, y, H, R, 2.0);  // 4 > 4 is false
    REQUIRE(res.accepted);
    REQUIRE(res.nis == 4.0);
    REQUIRE(res.dx.size() == 3);
    REQUIRE(maxAbsDiff(P, P0) > 0.1);
  }
  SECTION("below the gate: accepted") {
    MatX P = P0;
    const auto res = estimation::josephUpdate(P, y, H, R, 2.1);
    REQUIRE(res.accepted);
    // dx = K y with K = P H^T S^-1 = 0.5 * P H^T: (1, 1, 0.25).
    REQUIRE(res.dx(0) == Approx(1.0).margin(1e-15));
    REQUIRE(res.dx(1) == Approx(1.0).margin(1e-15));
    REQUIRE(res.dx(2) == Approx(0.25).margin(1e-15));
  }
}

TEST_CASE("the innovation gate scales with the measurement dimension on a random system",
          "[ekf][update][gate]") {
  std::mt19937_64 rng(99);
  for (Eigen::Index m : {1, 2, 5}) {
    const MatX P0 = randomSpd(7, rng);
    const MatX H = randomMatrix(m, 7, rng);
    const MatX R = randomSpd(m, rng, 0.1);
    const VecX y = randomMatrix(m, 1, rng, 3.0);
    const MatX S = H * P0 * H.transpose() + R;
    const double nis = y.dot(S.ldlt().solve(y));
    const double per_dof = nis / static_cast<double>(m);
    INFO("dim " << m << " nis " << nis);

    MatX P = P0;
    const auto rejected = estimation::josephUpdate(P, y, H, R, per_dof * (1.0 - 1e-9));
    REQUIRE_FALSE(rejected.accepted);
    REQUIRE(rejected.nis == Approx(nis).epsilon(1e-12));
    REQUIRE(maxAbsDiff(P, P0) == 0.0);

    const auto accepted = estimation::josephUpdate(P, y, H, R, per_dof * (1.0 + 1e-9));
    REQUIRE(accepted.accepted);
    REQUIRE(accepted.dx.size() == 7);
  }
}

TEST_CASE("a non-positive gate disables gating", "[ekf][update][gate]") {
  std::mt19937_64 rng(1234);
  const MatX P0 = randomSpd(6, rng);
  const MatX H = randomMatrix(3, 6, rng);
  const MatX R = 0.01 * MatX::Identity(3, 3);
  const VecX y = Eigen::Vector3d(1e4, -2e4, 3e4);  // NIS ~ 1e9: any positive gate rejects

  MatX P_ref = P0;
  const auto ref = estimation::josephUpdate(P_ref, y, H, R, std::numeric_limits<double>::max());
  REQUIRE(ref.accepted);
  REQUIRE(ref.nis > 1e6);
  MatX P_gated = P0;
  REQUIRE_FALSE(estimation::josephUpdate(P_gated, y, H, R, 1000.0).accepted);

  for (double gate : {0.0, -1.0, -1e300}) {
    INFO("gate " << gate);
    MatX P = P0;
    const auto res = estimation::josephUpdate(P, y, H, R, gate);
    REQUIRE(res.accepted);
    REQUIRE(res.nis == ref.nis);
    REQUIRE(maxAbsDiff(P, P_ref) == 0.0);  // same arithmetic as the never-rejecting gate
    REQUIRE((res.dx - ref.dx).norm() == 0.0);
  }
}

// ------------------------------------------------------------------------------------------
// discretize
// ------------------------------------------------------------------------------------------

TEST_CASE("exact discretisation is exactly Van Loan's", "[ekf][numerics]") {
  std::mt19937_64 rng(42);
  const MatX F = randomMatrix(8, 8, rng);
  const MatX Qc = randomSpd(8, rng);
  MatX Phi, Qd;
  estimation::discretize(F, Qc, 0.005, true, &Phi, &Qd);
  const auto d = math::vanLoanDiscretize(F, Qc, 0.005);
  REQUIRE(maxAbsDiff(Phi, d.Ad) == 0.0);
  REQUIRE(maxAbsDiff(Qd, d.Qd) == 0.0);
}

TEST_CASE("series and exact discretisation agree to third order in dt", "[ekf][numerics]") {
  // Phi_series = I + F dt + (F dt)^2/2 truncates e^{F dt} after the second-order term, so
  // with a = ||F||_F dt (Frobenius norm is submultiplicative)
  //   ||Phi_series - Phi_exact|| <= sum_{k>=3} a^k/k! <= a^3/6 e^a.
  // Qd_series keeps the j+k <= 1 terms of int_0^dt e^{Fs} Qc e^{F^T s} ds; the remainder is
  //   ||Qd_series - Qd_exact|| <= ||Qc|| dt sum_{n>=2} (2a)^n/(n+1)! <= ||Qc|| dt (2a)^2/6 e^{2a},
  // i.e. O(dt^3). Halving dt must therefore shrink both errors by ~2^3 = 8.
  std::mt19937_64 rng(7);
  const MatX F = randomMatrix(6, 6, rng, 0.5);  // ||F||_F ~ 3: a "small" F
  const MatX Qc = randomSpd(6, rng);

  auto errors = [&](double dt) {
    MatX Phi_s, Qd_s, Phi_e, Qd_e;
    estimation::discretize(F, Qc, dt, false, &Phi_s, &Qd_s);
    estimation::discretize(F, Qc, dt, true, &Phi_e, &Qd_e);
    return std::make_pair((Phi_s - Phi_e).norm(), (Qd_s - Qd_e).norm());
  };

  const double dt = 0.005;
  const double a = F.norm() * dt;
  const auto [phi_err, qd_err] = errors(dt);
  INFO("a = " << a << ", Phi error " << phi_err << ", Qd error " << qd_err);
  REQUIRE(phi_err <= a * a * a / 6.0 * std::exp(a));
  REQUIRE(qd_err <= Qc.norm() * dt * (2.0 * a) * (2.0 * a) / 6.0 * std::exp(2.0 * a));
  REQUIRE(phi_err > 0.0);
  REQUIRE(qd_err > 0.0);

  // Convergence order: leading term ratio 8, perturbed by the next order, O(a) ~ 1 %.
  const auto [phi_half, qd_half] = errors(0.5 * dt);
  INFO("Phi ratio " << phi_err / phi_half << ", Qd ratio " << qd_err / qd_half);
  REQUIRE(phi_err / phi_half == Approx(8.0).epsilon(0.05));
  REQUIRE(qd_err / qd_half == Approx(8.0).epsilon(0.05));
}

TEST_CASE("series discretisation is exact for a nilpotent F and matches the closed form",
          "[ekf][numerics]") {
  // F with F^3 = 0 (position <- velocity <- acceleration chain): e^{F dt} = I + F dt +
  // (F dt)^2/2 exactly, so the series transition matrix must equal Van Loan's to round-off.
  MatX F = MatX::Zero(3, 3);
  F(0, 1) = 1.0;
  F(1, 2) = 1.0;
  const MatX Qc = MatX::Identity(3, 3);
  MatX Phi_s, Qd_s, Phi_e, Qd_e;
  estimation::discretize(F, Qc, 0.01, false, &Phi_s, &Qd_s);
  estimation::discretize(F, Qc, 0.01, true, &Phi_e, &Qd_e);
  REQUIRE(maxAbsDiff(Phi_s, Phi_e) < 1e-15);
  REQUIRE(Phi_s(0, 2) == Approx(0.5 * 0.01 * 0.01).epsilon(1e-14));
  // Qd_series = Qc dt + dt^2/2 (F Qc + Qc F^T): for Qc = I the (0,1) entry is dt^2/2.
  REQUIRE(Qd_s(0, 1) == Approx(0.5 * 0.01 * 0.01).epsilon(1e-14));
  REQUIRE(Qd_s(1, 1) == Approx(0.01).epsilon(1e-14));
}

TEST_CASE("exact and approximate covariance propagation agree at the IMU rate",
          "[ekf][numerics]") {
  // Ported from Aether6: propagate a 15-state navigation error covariance (position,
  // velocity, attitude, gyro bias, accel bias, the calibrator's nav block layout) for 400
  // steps at 200 Hz with both discretisations. Each step's Phi differs by O((F dt)^3) with
  // ||F dt|| ~ 0.07 (specific force ~ g), i.e. ~1e-6 relative, so after 400 steps the
  // covariances must agree to ~1e-4 relative, the bound Aether6 uses.
  constexpr int kP = 0, kV = 3, kA = 6, kBg = 9, kBa = 12;
  const Vec3 f(1.2, 0.4, -9.0), w(0.3, -0.2, 0.15);
  const Mat3 R = math::quatToRotation(math::eulerToQuat(0.2, 0.1, 0.4));
  MatX F = MatX::Zero(15, 15);
  F.block<3, 3>(kP, kV) = Mat3::Identity();
  F.block<3, 3>(kV, kA) = -R * math::skew(f);
  F.block<3, 3>(kV, kBa) = -R;
  F.block<3, 3>(kA, kA) = -math::skew(w);
  F.block<3, 3>(kA, kBg) = -Mat3::Identity();
  const double rate = 200.0, dt = 1.0 / rate;
  MatX Qc = MatX::Zero(15, 15);
  Qc.block<3, 3>(kV, kV) = Mat3::Identity() * 0.05 * 0.05 / rate;
  Qc.block<3, 3>(kA, kA) = Mat3::Identity() * 0.0035 * 0.0035 / rate;
  Qc.block<3, 3>(kBg, kBg) = Mat3::Identity() * 3e-4 * 3e-4;
  Qc.block<3, 3>(kBa, kBa) = Mat3::Identity() * 3e-3 * 3e-3;

  VecX sig(15);
  sig << 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.03, 0.03, 0.03, 0.002, 0.002, 0.002, 0.15,
      0.15, 0.15;
  MatX Pa = sig.array().square().matrix().asDiagonal();
  MatX Pe = Pa;
  MatX Phi_a, Qd_a, Phi_e, Qd_e;
  estimation::discretize(F, Qc, dt, false, &Phi_a, &Qd_a);
  estimation::discretize(F, Qc, dt, true, &Phi_e, &Qd_e);
  for (int i = 0; i < 400; ++i) {
    Pa = math::symmetrize(Phi_a * Pa * Phi_a.transpose() + Qd_a);
    Pe = math::symmetrize(Phi_e * Pe * Phi_e.transpose() + Qd_e);
  }
  const double rel = (Pa - Pe).norm() / Pe.norm();
  INFO("relative covariance difference " << rel);
  REQUIRE(rel < 1e-4);
  REQUIRE(minEigenvalue(Pa) > 0.0);
}

// ------------------------------------------------------------------------------------------
// applyAttitudeReset
// ------------------------------------------------------------------------------------------

TEST_CASE("attitude reset equals G P G^T and touches only the rotation blocks",
          "[ekf][reset]") {
  std::mt19937_64 rng(2718);
  const Eigen::Index n = 21;
  const std::vector<int> blocks = {6, 15};  // e.g. the attitude and extrinsic-rotation blocks
  const MatX P0 = randomSpd(n, rng);
  const VecX dx = randomMatrix(n, 1, rng, 0.05);

  MatX P = P0;
  estimation::applyAttitudeReset(P, dx, blocks);
  const MatX G = resetJacobian(n, dx, blocks);
  const MatX expected = G * P0 * G.transpose();
  // Same products, possibly different summation order: round-off ~ n eps ||P||.
  REQUIRE(maxAbsDiff(P, expected) / P0.norm() < 1e-14);
  REQUIRE((P - P.transpose()).cwiseAbs().maxCoeff() == 0.0);

  // Entries whose row AND column are both outside the rotation blocks are untouched: their
  // rows of G are unit vectors, so G P G^T reproduces them bit for bit.
  auto inBlock = [&](Eigen::Index i) {
    for (int b : blocks)
      if (i >= b && i < b + 3) return true;
    return false;
  };
  int changed_inside = 0;
  for (Eigen::Index i = 0; i < n; ++i)
    for (Eigen::Index j = 0; j < n; ++j) {
      if (!inBlock(i) && !inBlock(j)) {
        INFO("entry (" << i << ", " << j << ")");
        REQUIRE(P(i, j) == P0(i, j));
      } else if (P(i, j) != P0(i, j)) {
        ++changed_inside;
      }
    }
  REQUIRE(changed_inside > 0);  // the reset actually did something

  // The non-rotation components of dx (positions, biases, ...) do not enter G at all.
  VecX dx_rot_only = VecX::Zero(n);
  for (int b : blocks) dx_rot_only.segment<3>(b) = dx.segment<3>(b);
  MatX P_rot_only = P0;
  estimation::applyAttitudeReset(P_rot_only, dx_rot_only, blocks);
  REQUIRE(maxAbsDiff(P_rot_only, P) == 0.0);
}

TEST_CASE("attitude reset is the identity without blocks or without a rotation correction",
          "[ekf][reset]") {
  std::mt19937_64 rng(3);
  const MatX P0 = randomSpd(9, rng);
  const VecX dx = randomMatrix(9, 1, rng, 0.1);

  MatX P = P0;
  estimation::applyAttitudeReset(P, dx, {});
  REQUIRE(maxAbsDiff(P, P0) == 0.0);

  P = P0;
  estimation::applyAttitudeReset(P, VecX::Zero(9), {0, 3, 6});
  REQUIRE(maxAbsDiff(P, P0) == 0.0);
}

TEST_CASE("attitude reset Jacobian matches the right-perturbation error reset",
          "[ekf][reset]") {
  // With R = Rhat Exp(dtheta) (math::boxPlus), injecting the estimate dhat moves the nominal
  // to Rhat Exp(dhat) and the new error is
  //   dtheta+ = Log(Exp(-dhat) Exp(dtheta)) = boxMinus(boxPlus(q, dtheta), boxPlus(q, dhat)).
  // Its Jacobian at dtheta = dhat is the right Jacobian Jr(dhat) = I - 0.5[dhat]x + O(|dhat|^2),
  // so the reset P <- G P G^T must equal J P J^T up to O(|dhat|^2 ||P||). A sign error in G
  // (I + 0.5[dhat]x, the left-perturbation reset) would instead be off at O(|dhat| ||P||).
  std::mt19937_64 rng(1618);
  const Vec4 q = math::quatNormalize(Vec4(0.8, -0.3, 0.4, 0.2));
  for (int trial = 0; trial < 10; ++trial) {
    const Vec3 dhat = randomMatrix(3, 1, rng, 0.012);  // |dhat| ~ 0.02 rad ~ 1 deg
    const double th = dhat.norm();
    auto reset_map = [&](const VecX& x) -> VecX {
      return math::boxMinus(math::boxPlus(q, Vec3(x)), math::boxPlus(q, dhat));
    };
    // h = 1e-4 * |x_j| ~ 1e-6: truncation O(h^2) and round-off O(1e-16/h) both < 1e-9.
    const MatX J = math::numericalJacobian(reset_map, VecX(dhat), 1e-4);

    // A non-isotropic P so the first-order term -0.5([d]x P + P [d]x^T) is non-zero.
    MatX P0 = MatX::Zero(3, 3);
    P0.diagonal() << 1.0, 4.0, 9.0;
    P0(0, 1) = P0(1, 0) = 0.5;
    MatX P = P0;
    estimation::applyAttitudeReset(P, VecX(dhat), {0});

    const MatX expected = J * P0 * J.transpose();
    // G - Jr = ((1-cos th)/th - th/2)[u]x - ((th - sin th)/th)[u]x^2, and [u]x, [u]x^2 have
    // unit spectral norm, so ||G - Jr||_2 <= th^3/24 + th^2/6 < 0.25 th^2. Both G and Jr are
    // normal with ||.||_2 <= 1 + th, hence
    // ||G P G^T - J P J^T||_F <= 2 ||G - J||_2 ||P||_F (1 + th); 1e-8 covers the
    // finite-difference error of J.
    const double bound = 2.0 * 0.25 * th * th * P0.norm() * (1.0 + th) + 1e-8;
    INFO("trial " << trial << " |dhat| = " << th);
    REQUIRE((P - expected).norm() < bound);

    // The opposite-sign (left-perturbation) reset is clearly distinguishable at this size.
    const MatX G_wrong = MatX::Identity(3, 3) + 0.5 * math::skew(dhat);
    REQUIRE((G_wrong * P0 * G_wrong.transpose() - expected).norm() > 10.0 * bound);
  }
}
