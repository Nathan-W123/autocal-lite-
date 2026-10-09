/// \file test_math.cpp
/// \brief Quaternion algebra, rotation identities, matrix exponential, Van Loan
///        discretisation, numerical rank/Jacobian, seeded RNG and CSV output.
///
/// Ported from Aether6 (`tests/test_math.cpp`, plus the `[util]` cases of
/// `tests/test_config_io.cpp`) for the functions AutoCal-lite kept unchanged. The cases for the
/// removed Riccati/LQR/controllability/Levenberg-Marquardt routines were dropped; the
/// controllability-rank case is kept as a pure `numericalRank` test by building
/// \f$[B\;AB]\f$ by hand. Cases marked "AutoCal-lite" are new and pin down the conventions
/// the calibrator depends on (right-perturbed rotations, \f$\exp\f$ of a skew matrix, Van
/// Loan on a multi-state system, CSV round trip).
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "aether/core/Constants.hpp"
#include "aether/math/LinearAlgebra.hpp"
#include "aether/math/NumericalDiff.hpp"
#include "aether/math/Rotation.hpp"
#include "aether/util/Csv.hpp"
#include "aether/util/Random.hpp"

using namespace aether;
using Catch::Approx;

namespace {
Vec4 randomQuat(std::mt19937_64& rng) {
  std::normal_distribution<double> n(0.0, 1.0);
  return math::quatNormalize(Vec4(n(rng), n(rng), n(rng), n(rng)));
}

/// Split one CSV line into its comma-separated fields.
std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string field;
  while (std::getline(ss, field, ',')) out.push_back(field);
  return out;
}
}  // namespace

// ------------------------------------------------------------------------------------------
// Rotation.hpp (ported from Aether6)
// ------------------------------------------------------------------------------------------

TEST_CASE("quaternion normalisation and norm preservation", "[math][quaternion]") {
  std::mt19937_64 rng(12345);
  for (int i = 0; i < 200; ++i) {
    const Vec4 q = randomQuat(rng);
    REQUIRE(q.norm() == Approx(1.0).margin(1e-14));
    REQUIRE(q(0) >= 0.0);  // canonical sign convention
  }
  // A degenerate input must fall back to identity rather than produce NaNs.
  const Vec4 degenerate = math::quatNormalize(Vec4::Zero());
  REQUIRE(degenerate(0) == Approx(1.0));
  REQUIRE(degenerate.tail<3>().norm() == Approx(0.0));
}

TEST_CASE("quaternion product is associative and the conjugate inverts it",
          "[math][quaternion]") {
  std::mt19937_64 rng(999);
  for (int i = 0; i < 100; ++i) {
    const Vec4 a = randomQuat(rng), b = randomQuat(rng), c = randomQuat(rng);
    const Vec4 lhs = math::quatMultiply(math::quatMultiply(a, b), c);
    const Vec4 rhs = math::quatMultiply(a, math::quatMultiply(b, c));
    REQUIRE((lhs - rhs).norm() == Approx(0.0).margin(1e-13));

    const Vec4 identity = math::quatMultiply(a, math::quatConjugate(a));
    REQUIRE(identity(0) == Approx(1.0).margin(1e-13));
    REQUIRE(identity.tail<3>().norm() == Approx(0.0).margin(1e-13));
  }
}

TEST_CASE("rotation matrix from a quaternion is a proper orthogonal matrix",
          "[math][rotation]") {
  std::mt19937_64 rng(777);
  for (int i = 0; i < 200; ++i) {
    const Mat3 R = math::quatToRotation(randomQuat(rng));
    REQUIRE((R * R.transpose() - Mat3::Identity()).norm() == Approx(0.0).margin(1e-13));
    REQUIRE(R.determinant() == Approx(1.0).margin(1e-13));
  }
}

TEST_CASE("rotating with the sandwich product matches the rotation matrix",
          "[math][rotation]") {
  std::mt19937_64 rng(31337);
  std::normal_distribution<double> n(0.0, 5.0);
  for (int i = 0; i < 200; ++i) {
    const Vec4 q = randomQuat(rng);
    const Vec3 v(n(rng), n(rng), n(rng));
    const Vec3 by_quat = math::rotateBodyToNed(q, v);
    const Vec3 by_matrix = math::quatToRotation(q) * v;
    REQUIRE((by_quat - by_matrix).norm() == Approx(0.0).margin(1e-12));
    // Rotation preserves length, and the inverse rotation returns the original vector.
    REQUIRE(by_quat.norm() == Approx(v.norm()).margin(1e-12));
    REQUIRE((math::rotateNedToBody(q, by_quat) - v).norm() == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("quaternion product composes rotation matrices in the same order",
          "[math][rotation]") {
  // AutoCal-lite: q_AC = q_AB (x) q_BC must give R_AC = R_AB R_BC; the calibrator chains
  // q_WI (x) q_IC this way to get the camera attitude.
  std::mt19937_64 rng(8080);
  for (int i = 0; i < 100; ++i) {
    const Vec4 q_ab = randomQuat(rng), q_bc = randomQuat(rng);
    const Mat3 lhs = math::quatToRotation(math::quatMultiply(q_ab, q_bc));
    const Mat3 rhs = math::quatToRotation(q_ab) * math::quatToRotation(q_bc);
    REQUIRE((lhs - rhs).norm() == Approx(0.0).margin(1e-13));
  }
}

TEST_CASE("Euler <-> quaternion round trip", "[math][rotation]") {
  std::mt19937_64 rng(4242);
  std::uniform_real_distribution<double> roll(-constants::kPi, constants::kPi);
  std::uniform_real_distribution<double> pitch(-1.5, 1.5);  // avoid the gimbal singularity
  for (int i = 0; i < 300; ++i) {
    const double r = roll(rng), p = pitch(rng), y = roll(rng);
    const Vec3 e = math::quatToEuler(math::eulerToQuat(r, p, y));
    REQUIRE(math::wrapPi(e.x() - r) == Approx(0.0).margin(1e-11));
    REQUIRE(e.y() == Approx(p).margin(1e-11));
    REQUIRE(math::wrapPi(e.z() - y) == Approx(0.0).margin(1e-11));
  }
}

TEST_CASE("known rotations have the documented sign convention", "[math][rotation]") {
  // Yaw +90 deg: the body x-axis (nose) points East in NED.
  const Vec4 q_yaw = math::eulerToQuat(0.0, 0.0, constants::kPi / 2);
  const Vec3 nose = math::rotateBodyToNed(q_yaw, Vec3(1, 0, 0));
  REQUIRE(nose.x() == Approx(0.0).margin(1e-12));
  REQUIRE(nose.y() == Approx(1.0).margin(1e-12));

  // Pitch +90 deg (nose up): the body x-axis points straight up, i.e. -Down.
  const Vec4 q_pitch = math::eulerToQuat(0.0, constants::kPi / 2, 0.0);
  const Vec3 up = math::rotateBodyToNed(q_pitch, Vec3(1, 0, 0));
  REQUIRE(up.z() == Approx(-1.0).margin(1e-12));

  // Roll +90 deg (right wing down): the body y-axis points straight down.
  const Vec4 q_roll = math::eulerToQuat(constants::kPi / 2, 0.0, 0.0);
  const Vec3 wing = math::rotateBodyToNed(q_roll, Vec3(0, 1, 0));
  REQUIRE(wing.z() == Approx(1.0).margin(1e-12));
}

TEST_CASE("exp/log maps and boxplus/boxminus are consistent", "[math][quaternion]") {
  std::mt19937_64 rng(2024);
  std::normal_distribution<double> n(0.0, 0.7);
  for (int i = 0; i < 300; ++i) {
    const Vec3 dtheta(n(rng), n(rng), n(rng));
    if (dtheta.norm() > constants::kPi * 0.95) continue;  // log is only unique inside the ball
    const Vec3 recovered = math::logMapQuat(math::expMapQuat(dtheta));
    REQUIRE((recovered - dtheta).norm() == Approx(0.0).margin(1e-11));

    const Vec4 q = randomQuat(rng);
    const Vec4 q2 = math::boxPlus(q, dtheta);
    REQUIRE(q2.norm() == Approx(1.0).margin(1e-14));
    REQUIRE((math::boxMinus(q2, q) - dtheta).norm() == Approx(0.0).margin(1e-11));
  }
  // The small-angle branch must match the general branch.
  const Vec3 tiny(1e-10, -2e-10, 3e-11);
  REQUIRE((math::logMapQuat(math::expMapQuat(tiny)) - tiny).norm() == Approx(0.0).margin(1e-18));
}

TEST_CASE("boxplus perturbs the rotation on the right: R(q boxplus d) = R(q) Exp(d)",
          "[math][quaternion]") {
  // AutoCal-lite: the error-state convention R = Rhat * Exp(dtheta). Exp is computed
  // independently as the matrix exponential of the skew matrix (Rodrigues), so this checks
  // both the right-multiplication and the half-angle in expMapQuat.
  std::mt19937_64 rng(606);
  std::normal_distribution<double> n(0.0, 0.5);
  for (int i = 0; i < 100; ++i) {
    const Vec4 q = randomQuat(rng);
    const Vec3 d(n(rng), n(rng), n(rng));
    const Mat3 exp_d = math::expm(math::skew(d));
    REQUIRE((math::quatToRotation(math::expMapQuat(d)) - exp_d).norm() ==
            Approx(0.0).margin(1e-12));
    const Mat3 lhs = math::quatToRotation(math::boxPlus(q, d));
    REQUIRE((lhs - math::quatToRotation(q) * exp_d).norm() == Approx(0.0).margin(1e-12));
    // Left multiplication is a different rotation for a generic q (guards the side).
    REQUIRE((lhs - exp_d * math::quatToRotation(q)).norm() > 1e-3);
  }
  // A rotation of +90 deg about z maps x onto y (right-hand rule).
  const Mat3 Rz = math::quatToRotation(math::expMapQuat(Vec3(0.0, 0.0, constants::kPi / 2)));
  REQUIRE((Rz * Vec3::UnitX() - Vec3::UnitY()).norm() == Approx(0.0).margin(1e-15));
}

TEST_CASE("skew matrix reproduces the cross product", "[math]") {
  std::mt19937_64 rng(55);
  std::normal_distribution<double> n(0.0, 2.0);
  for (int i = 0; i < 100; ++i) {
    const Vec3 a(n(rng), n(rng), n(rng)), b(n(rng), n(rng), n(rng));
    REQUIRE((math::skew(a) * b - a.cross(b)).norm() == Approx(0.0).margin(1e-14));
    REQUIRE((math::skew(a) + math::skew(a).transpose()).norm() == Approx(0.0).margin(1e-15));
  }
}

TEST_CASE("wrapPi maps angles into the half-open interval -pi to pi", "[math]") {
  REQUIRE(math::wrapPi(0.0) == Approx(0.0));
  REQUIRE(math::wrapPi(constants::kPi) == Approx(constants::kPi));
  REQUIRE(math::wrapPi(-constants::kPi) == Approx(constants::kPi));
  REQUIRE(math::wrapPi(3.0 * constants::kPi) == Approx(constants::kPi));
  REQUIRE(math::wrapPi(2.0 * constants::kPi + 0.3) == Approx(0.3).margin(1e-12));
  REQUIRE(math::wrapPi(-2.0 * constants::kPi - 0.3) == Approx(-0.3).margin(1e-12));
}

// ------------------------------------------------------------------------------------------
// LinearAlgebra.hpp (ported from Aether6)
// ------------------------------------------------------------------------------------------

TEST_CASE("matrix exponential matches analytic results", "[math][linalg]") {
  // Diagonal case.
  MatX D(3, 3);
  D.setZero();
  D.diagonal() << 1.0, -2.0, 0.5;
  const MatX E = math::expm(D);
  REQUIRE(E(0, 0) == Approx(std::exp(1.0)).epsilon(1e-12));
  REQUIRE(E(1, 1) == Approx(std::exp(-2.0)).epsilon(1e-12));
  REQUIRE(E(2, 2) == Approx(std::exp(0.5)).epsilon(1e-12));

  // Rotation generator: expm([[0,-w],[w,0]]*t) is a planar rotation by w*t.
  const double w = 1.7, t = 0.9;
  MatX S(2, 2);
  S << 0.0, -w * t, w * t, 0.0;
  const MatX R = math::expm(S);
  REQUIRE(R(0, 0) == Approx(std::cos(w * t)).epsilon(1e-12));
  REQUIRE(R(0, 1) == Approx(-std::sin(w * t)).epsilon(1e-12));

  // Nilpotent case has an exactly truncating series.
  MatX N(3, 3);
  N.setZero();
  N(0, 1) = 2.0;
  N(1, 2) = 3.0;
  const MatX EN = math::expm(N);
  REQUIRE(EN(0, 1) == Approx(2.0).epsilon(1e-12));
  REQUIRE(EN(0, 2) == Approx(3.0).epsilon(1e-12));  // 0.5 * 2 * 3
  REQUIRE(EN(1, 2) == Approx(3.0).epsilon(1e-12));

  // Large-norm input still works thanks to scaling and squaring.
  MatX big(2, 2);
  big << 0.0, 30.0, -30.0, 0.0;
  const MatX Ebig = math::expm(big);
  REQUIRE(Ebig(0, 0) == Approx(std::cos(30.0)).margin(1e-10));
  REQUIRE((Ebig * Ebig.transpose() - MatX::Identity(2, 2)).norm() == Approx(0.0).margin(1e-10));
}

TEST_CASE("matrix exponential edge cases: zero and empty and non-square input", "[math][linalg]") {
  REQUIRE((math::expm(MatX::Zero(4, 4)) - MatX::Identity(4, 4)).norm() == 0.0);
  REQUIRE(math::expm(MatX(0, 0)).size() == 0);
  REQUIRE_THROWS_AS(math::expm(MatX::Zero(2, 3)), std::invalid_argument);
  // e^{A} e^{-A} = I for a generic (non-normal) matrix.
  std::mt19937_64 rng(17);
  std::normal_distribution<double> n(0.0, 1.0);
  MatX A(5, 5);
  for (Eigen::Index i = 0; i < A.size(); ++i) A(i) = n(rng);
  REQUIRE((math::expm(A) * math::expm(-A) - MatX::Identity(5, 5)).norm() ==
          Approx(0.0).margin(1e-11));
}

TEST_CASE("Van Loan discretisation reproduces a scalar Ornstein-Uhlenbeck process",
          "[math][linalg]") {
  // dx = -a x dt + dw with intensity q: stationary variance q/(2a), and the exact discrete
  // process noise is (q/(2a)) * (1 - exp(-2 a dt)).
  const double a = 0.8, q = 2.5, dt = 0.37;
  MatX A(1, 1), Qc(1, 1);
  A(0, 0) = -a;
  Qc(0, 0) = q;
  const auto d = math::vanLoanDiscretize(A, Qc, dt);
  REQUIRE(d.Ad(0, 0) == Approx(std::exp(-a * dt)).epsilon(1e-12));
  REQUIRE(d.Qd(0, 0) ==
          Approx(q / (2.0 * a) * (1.0 - std::exp(-2.0 * a * dt))).epsilon(1e-11));
}

TEST_CASE("Van Loan discretisation of a random-walk integrator chain", "[math][linalg]") {
  // AutoCal-lite: the bias -> velocity -> position chain of the calibrator is a double
  // integrator driven by white noise of intensity q on the last state. Closed form:
  // Ad = [[1, dt],[0, 1]], Qd = q [[dt^3/3, dt^2/2],[dt^2/2, dt]].
  const double q = 0.7, dt = 0.25;
  MatX A = MatX::Zero(2, 2);
  A(0, 1) = 1.0;
  MatX Qc = MatX::Zero(2, 2);
  Qc(1, 1) = q;
  const auto d = math::vanLoanDiscretize(A, Qc, dt);
  REQUIRE(d.Ad(0, 0) == Approx(1.0).epsilon(1e-13));
  REQUIRE(d.Ad(0, 1) == Approx(dt).epsilon(1e-13));
  REQUIRE(d.Ad(1, 0) == Approx(0.0).margin(1e-14));
  REQUIRE(d.Qd(0, 0) == Approx(q * dt * dt * dt / 3.0).epsilon(1e-12));
  REQUIRE(d.Qd(0, 1) == Approx(q * dt * dt / 2.0).epsilon(1e-12));
  REQUIRE(d.Qd(1, 1) == Approx(q * dt).epsilon(1e-12));
  REQUIRE(d.Qd(0, 1) == d.Qd(1, 0));  // explicitly symmetrised
}

TEST_CASE("Van Loan discretisation matches quadrature of the noise integral",
          "[math][linalg]") {
  // AutoCal-lite: generic 4-state system; Ad must be expm(A dt) and Qd the integral
  // int_0^dt e^{A s} Qc e^{A^T s} ds, evaluated here with composite Simpson (error O(h^4),
  // ~1e-12 relative with 200 intervals for ||A dt|| ~ 0.5).
  std::mt19937_64 rng(4711);
  std::normal_distribution<double> n(0.0, 1.0);
  const int nx = 4;
  MatX A(nx, nx), L(nx, nx);
  for (Eigen::Index i = 0; i < A.size(); ++i) {
    A(i) = n(rng);
    L(i) = n(rng);
  }
  const MatX Qc = L * L.transpose();
  const double dt = 0.2;
  const auto d = math::vanLoanDiscretize(A, Qc, dt);
  REQUIRE((d.Ad - math::expm(A * dt)).norm() == Approx(0.0).margin(1e-12));

  const int intervals = 200;
  const double h = dt / intervals;
  MatX integral = MatX::Zero(nx, nx);
  for (int k = 0; k <= intervals; ++k) {
    const MatX Es = math::expm(A * (k * h));
    const double w = (k == 0 || k == intervals) ? 1.0 : (k % 2 == 1 ? 4.0 : 2.0);
    integral += w * Es * Qc * Es.transpose();
  }
  integral *= h / 3.0;
  REQUIRE((d.Qd - integral).norm() / integral.norm() < 1e-9);
  REQUIRE((d.Qd - d.Qd.transpose()).norm() == 0.0);
  Eigen::SelfAdjointEigenSolver<MatX> es(d.Qd);
  REQUIRE(es.eigenvalues().minCoeff() > 0.0);
}

TEST_CASE("symmetrize returns the exactly symmetric part", "[math][linalg]") {
  std::mt19937_64 rng(88);
  std::normal_distribution<double> n(0.0, 3.0);
  MatX M(6, 6);
  for (Eigen::Index i = 0; i < M.size(); ++i) M(i) = n(rng);
  const MatX S = math::symmetrize(M);
  REQUIRE((S - S.transpose()).norm() == 0.0);  // bitwise: a+b == b+a in IEEE arithmetic
  REQUIRE((S - 0.5 * (M + M.transpose())).norm() == 0.0);
  // Idempotent, and a symmetric matrix is returned unchanged.
  REQUIRE((math::symmetrize(S) - S).norm() == 0.0);
}

TEST_CASE("numerical rank detects an uncontrollable mode", "[math][linalg]") {
  // Ported from Aether6's controllability test; [B, AB] is built by hand because
  // controllabilityMatrix() was dropped together with the LQR code.
  MatX A(2, 2);
  A << -1.0, 0.0, 0.0, -2.0;
  auto ctrb = [&](const MatX& B) {
    MatX C(2, 2);
    C << B, A * B;
    return C;
  };
  MatX B(2, 1);
  B << 1.0, 0.0;  // the second mode cannot be reached
  REQUIRE(math::numericalRank(ctrb(B)) == 1);
  MatX B2(2, 1);
  B2 << 1.0, 1.0;
  REQUIRE(math::numericalRank(ctrb(B2)) == 2);
}

TEST_CASE("numerical rank of constructed low-rank and degenerate matrices", "[math][linalg]") {
  // AutoCal-lite: U V^T with U, V of 3 columns has rank exactly 3; the relative tolerance
  // decides whether a tiny singular value counts.
  std::mt19937_64 rng(1312);
  std::normal_distribution<double> n(0.0, 1.0);
  MatX U(8, 3), V(6, 3);
  for (Eigen::Index i = 0; i < U.size(); ++i) U(i) = n(rng);
  for (Eigen::Index i = 0; i < V.size(); ++i) V(i) = n(rng);
  REQUIRE(math::numericalRank(U * V.transpose()) == 3);
  REQUIRE(math::numericalRank(MatX::Identity(5, 5)) == 5);
  REQUIRE(math::numericalRank(MatX(0, 0)) == 0);

  MatX D = MatX::Zero(3, 3);
  D.diagonal() << 1.0, 1e-3, 1e-12;
  REQUIRE(math::numericalRank(D) == 2);         // 1e-12 < 1e-9 * 1
  REQUIRE(math::numericalRank(D, 1e-13) == 3);  // looser threshold counts it
  REQUIRE(math::numericalRank(D, 1e-2) == 1);
}

// ------------------------------------------------------------------------------------------
// NumericalDiff.hpp (ported from Aether6)
// ------------------------------------------------------------------------------------------

TEST_CASE("numerical Jacobian matches an analytic one", "[math][diff]") {
  auto f = [](const VecX& x) {
    VecX y(3);
    y(0) = x(0) * x(1) + std::sin(x(2));
    y(1) = std::exp(0.3 * x(0)) - x(2) * x(2);
    y(2) = x(0) + 2.0 * x(1) - 0.5 * x(2);
    return y;
  };
  VecX x(3);
  x << 0.7, -1.3, 0.45;
  MatX J(3, 3);
  J << x(1), x(0), std::cos(x(2)),
       0.3 * std::exp(0.3 * x(0)), 0.0, -2.0 * x(2),
       1.0, 2.0, -0.5;
  const MatX Jn = math::numericalJacobian(f, x);
  REQUIRE((Jn - J).cwiseAbs().maxCoeff() < 1e-8);
}

TEST_CASE("numerical Jacobian of a non-square map and at a zero component", "[math][diff]") {
  // AutoCal-lite: a 2x4 map evaluated where one component is exactly zero, so the step falls
  // back to epsilon * x_scale = 1e-9; central differences are still exact for the linear
  // part and O(h^2) for the quadratic one.
  auto f = [](const VecX& x) {
    VecX y(2);
    y(0) = 3.0 * x(0) - x(1) * x(1) + 0.5 * x(3);
    y(1) = x(2) * x(0) + std::cos(x(1));
    return y;
  };
  VecX x(4);
  x << 0.0, 2.0, -1.5, 4.0;
  MatX J(2, 4);
  J << 3.0, -2.0 * x(1), 0.0, 0.5,
       x(2), -std::sin(x(1)), x(0), 0.0;
  const MatX Jn = math::numericalJacobian(f, x);
  REQUIRE(Jn.rows() == 2);
  REQUIRE(Jn.cols() == 4);
  // Round-off of a central difference is ~ |f| * eps_machine / h; with h = 1e-9 on the zero
  // component and |f| ~ 5 that is ~1e-6, hence the looser bound here.
  REQUIRE((Jn - J).cwiseAbs().maxCoeff() < 5e-6);
}

// ------------------------------------------------------------------------------------------
// Random.hpp (ported from Aether6 tests/test_config_io.cpp)
// ------------------------------------------------------------------------------------------

TEST_CASE("seed derivation is deterministic and well separated", "[util][random]") {
  REQUIRE(util::deriveSeed(1, 2, 3) == util::deriveSeed(1, 2, 3));
  REQUIRE(util::deriveSeed(1, 2, 3) != util::deriveSeed(1, 2, 4));
  REQUIRE(util::deriveSeed(1, 2, 3) != util::deriveSeed(1, 3, 3));
  REQUIRE(util::deriveSeed(1, 2, 3) != util::deriveSeed(2, 2, 3));

  // Consecutive indices must not produce correlated streams.
  double correlation = 0.0;
  const int n = 4000;
  for (int i = 0; i < n; ++i) {
    util::Rng a(util::deriveSeed(42, 0, static_cast<std::uint64_t>(i)));
    util::Rng b(util::deriveSeed(42, 0, static_cast<std::uint64_t>(i) + 1));
    correlation += a.gaussian() * b.gaussian();
  }
  // Under independence the mean product has sigma 1/sqrt(4000) = 0.016; 0.06 is ~3.8 sigma.
  REQUIRE(std::abs(correlation / n) < 0.06);

  util::Rng r(7);
  REQUIRE(r.seed() == 7);
  double sum = 0.0, sumsq = 0.0;
  const int m = 200000;
  for (int i = 0; i < m; ++i) {
    const double v = r.gaussian();
    sum += v;
    sumsq += v * v;
  }
  REQUIRE(sum / m == Approx(0.0).margin(0.01));
  REQUIRE(std::sqrt(sumsq / m) == Approx(1.0).epsilon(0.01));

  util::Rng u(9);
  double lo = 1e9, hi = -1e9;
  for (int i = 0; i < 10000; ++i) {
    const double v = u.uniform(-2.0, 5.0);
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  REQUIRE(lo >= -2.0);
  REQUIRE(hi <= 5.0);
  REQUIRE(lo < -1.9);
  REQUIRE(hi > 4.9);
}

TEST_CASE("SplitMix64 matches the published reference sequence", "[util][random]") {
  // AutoCal-lite: platform independence of every trial seed rests on this mixer. The first
  // two outputs of the reference SplitMix64 generator seeded with 0 (Vigna's splitmix64.c)
  // are 0xE220A8397B1DCDAF and 0x6E789E6AA1B965F4; splitMix64(x) is one step from state x.
  REQUIRE(util::splitMix64(0) == 0xE220A8397B1DCDAFULL);
  REQUIRE(util::splitMix64(0x9E3779B97F4A7C15ULL) == 0x6E789E6AA1B965F4ULL);
}

TEST_CASE("identically seeded Rng streams are bitwise identical", "[util][random]") {
  // AutoCal-lite: every draw type (gaussian, scaled gaussian, uniform, gaussian3) interleaved.
  util::Rng a(123456789), b(123456789), c(123456790);
  bool any_difference_from_c = false;
  for (int i = 0; i < 1000; ++i) {
    const double ga = a.gaussian(), gb = b.gaussian(), gc = c.gaussian();
    REQUIRE(ga == gb);
    any_difference_from_c = any_difference_from_c || (ga != gc);
    REQUIRE(a.gaussian(3.0, 0.5) == b.gaussian(3.0, 0.5));
    REQUIRE(a.uniform(-1.0, 1.0) == b.uniform(-1.0, 1.0));
    REQUIRE(a.gaussian3() == b.gaussian3());
    c.gaussian(3.0, 0.5);
    c.uniform(-1.0, 1.0);
    c.gaussian3();
  }
  REQUIRE(any_difference_from_c);
}

TEST_CASE("scaled gaussian draws have the requested mean and standard deviation",
          "[util][random]") {
  // AutoCal-lite: N = 100000 draws of N(3, 0.5^2). The sample mean has sigma
  // 0.5/sqrt(N) = 1.6e-3 and the sample std a relative sigma 1/sqrt(2N) = 2.2e-3; the bounds
  // below are ~5 sigma.
  util::Rng r(2718);
  const int n = 100000;
  double sum = 0.0, sumsq = 0.0;
  for (int i = 0; i < n; ++i) {
    const double v = r.gaussian(3.0, 0.5);
    sum += v;
    sumsq += v * v;
  }
  const double mean = sum / n;
  const double var = sumsq / n - mean * mean;
  REQUIRE(mean == Approx(3.0).margin(0.008));
  REQUIRE(std::sqrt(var) == Approx(0.5).epsilon(0.011));
}

// ------------------------------------------------------------------------------------------
// Csv.hpp (ported from Aether6 tests/test_config_io.cpp)
// ------------------------------------------------------------------------------------------

TEST_CASE("CSV writer produces a well-formed file", "[util][csv]") {
  const auto dir = std::filesystem::temp_directory_path() / "autocal_tests_math";
  std::filesystem::create_directories(dir);
  const auto path = (dir / "csv_test.csv").string();
  {
    util::CsvWriter w(path, {"a", "b", "c"});
    w.writeRow({1.0, 2.5, -3.25});
    w.writeRow({0.0, 1e-12, 1e12});
    REQUIRE(w.rows() == 2);
    REQUIRE(w.header().size() == 3);
    REQUIRE_THROWS_AS(w.writeRow({1.0}), std::invalid_argument);
    REQUIRE(w.rows() == 2);  // a rejected row is not counted
  }
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  REQUIRE(line == "a,b,c");
  std::getline(in, line);
  REQUIRE(line == "1,2.5,-3.25");
  std::getline(in, line);
  REQUIRE(line == "0,1e-12,1e+12");
  REQUIRE_FALSE(std::getline(in, line));  // nothing from the rejected row

  REQUIRE_THROWS_AS(util::CsvWriter("/nonexistent_dir_xyz/out.csv", {"a"}),
                    std::runtime_error);
}

TEST_CASE("CSV values round-trip to the configured number of significant digits",
          "[util][csv]") {
  // AutoCal-lite: what the eval script reads back. With 9 significant digits the relative
  // round-trip error is at most 0.5e-8; with 17 digits a double round-trips exactly.
  const auto dir = std::filesystem::temp_directory_path() / "autocal_tests_math";
  std::filesystem::create_directories(dir);
  const auto path9 = (dir / "csv_roundtrip9.csv").string();
  const auto path17 = (dir / "csv_roundtrip17.csv").string();

  std::mt19937_64 rng(31415);
  std::normal_distribution<double> mant(0.0, 1.0);
  std::uniform_int_distribution<int> expo(-12, 12);
  std::vector<std::vector<double>> rows;
  for (int i = 0; i < 50; ++i)
    rows.push_back({mant(rng) * std::pow(10.0, expo(rng)), mant(rng), -mant(rng) * 1e-5});
  rows.push_back({1.0 / 3.0, constants::kPi, -0.0});

  {
    util::CsvWriter w9(path9, {"x", "y", "z"});
    util::CsvWriter w17(path17, {"x", "y", "z"}, 17);
    for (const auto& r : rows) {
      w9.writeRow(r);
      w17.writeRow(r);
    }
    w9.close();
    w17.close();
    w9.close();  // closing twice is harmless
  }

  auto readBack = [](const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    REQUIRE(splitCsv(line) == std::vector<std::string>{"x", "y", "z"});
    std::vector<std::vector<double>> out;
    while (std::getline(in, line)) {
      std::vector<double> vals;
      for (const auto& f : splitCsv(line)) vals.push_back(std::stod(f));
      out.push_back(vals);
    }
    return out;
  };
  const auto back9 = readBack(path9);
  const auto back17 = readBack(path17);
  REQUIRE(back9.size() == rows.size());
  REQUIRE(back17.size() == rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    REQUIRE(back9[i].size() == 3);
    for (std::size_t j = 0; j < 3; ++j) {
      INFO("row " << i << " col " << j << " value " << rows[i][j]);
      REQUIRE(back17[i][j] == rows[i][j]);
      REQUIRE(std::abs(back9[i][j] - rows[i][j]) <= 5.0000001e-9 * std::abs(rows[i][j]));
    }
  }
}
