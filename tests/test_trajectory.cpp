/// \file test_trajectory.cpp
/// \brief Scripted figure-eight trajectory: analytic derivatives against finite differences,
///        the stationary start, the C2 quintic ramp-in and the yaw-only degenerate motion.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "aether/calib/Trajectory.hpp"
#include "aether/core/Constants.hpp"
#include "aether/math/Rotation.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::FigureEightTrajectory;
using calib::KinematicState;
using calib::TrajectoryConfig;

// Central-difference step [s]. The truncation error of a central difference is
// h^2/6 * |x'''|. Every excursion is bounded by ~1 m or ~0.5 rad, the highest frequency is
// 2*(2*pi/8) ~ 1.6 rad/s and the ramp adds at most |s'''| <= 60/T^3 ~ 2.2 1/s^3 (T = 3 s), so
// the third derivatives of position, velocity and attitude are O(10). With h = 1e-4 the
// truncation error is ~2e-8 and round-off (eps*|x|/h) is ~1e-12, so a 1e-6 tolerance is 50x
// above the numerical error while a wrong or missing chain-rule term would be O(0.1).
constexpr double kH = 1e-4;
constexpr double kDerivTol = 1e-6;

/// Rotation matrix from a [w,x,y,z] quaternion via Eigen (independent of math::quatToRotation).
Mat3 rotationOf(const Vec4& q) {
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).normalized().toRotationMatrix();
}

/// A deliberately asymmetric configuration: tilted mean attitude, offset centre, phases and
/// a shorter ramp, so that terms which vanish for the defaults (base_euler = 0) are exercised.
TrajectoryConfig tiltedConfig() {
  TrajectoryConfig c;
  c.static_duration = 1.5;
  c.ramp_duration = 2.0;
  c.center = Vec3(0.4, -0.3, -1.2);
  c.period = 6.5;
  c.phase_north = -0.4;
  c.base_euler = Vec3(0.12, -0.21, 0.6);
  c.euler_amplitude = Vec3(0.30, 0.45, 0.50);
  c.euler_frequency = Vec3(0.33, 0.27, 0.19);
  c.euler_phase = Vec3(0.5, -0.8, 1.7);
  return c;
}

/// Sample times covering the static period, the quintic ramp and the long-run motion, kept
/// more than one step h away from the two ramp ends (where higher derivatives jump).
std::vector<double> sampleTimes(const TrajectoryConfig& c) {
  const double t0 = c.static_duration;
  const double t1 = t0 + c.ramp_duration;
  std::vector<double> times;
  for (double f : {0.05, 0.3, 0.6, 0.97}) times.push_back(f * t0);              // static
  for (double f : {0.01, 0.1, 0.25, 0.5, 0.73, 0.9, 0.99}) {
    times.push_back(t0 + f * c.ramp_duration);                                 // ramp
  }
  for (double dt : {0.01, 0.4, 2.3, 7.77, 19.1, 41.3, 58.9, 233.3, 1000.7}) {
    times.push_back(t1 + dt);                                                  // after
  }
  return times;
}

/// Body angular rate from a central difference of the attitude: with right-multiplicative
/// boxminus, Log(q(t-h)^-1 q(t+h)) / 2h = omega^I(t) + O(h^2).
Vec3 numericalBodyRate(const FigureEightTrajectory& traj, double t, double h) {
  const Vec4 qp = traj.evaluate(t + h).quaternion;
  const Vec4 qm = traj.evaluate(t - h).quaternion;
  return math::boxMinus(qp, qm) / (2.0 * h);
}

}  // namespace

TEST_CASE("analytic velocity, acceleration and body rate match central differences",
          "[calib][trajectory]") {
  for (const TrajectoryConfig& cfg : {TrajectoryConfig{}, tiltedConfig()}) {
    const FigureEightTrajectory traj(cfg);
    for (double t : sampleTimes(cfg)) {
      INFO("t = " << t);
      const KinematicState k = traj.evaluate(t);
      const KinematicState kp = traj.evaluate(t + kH);
      const KinematicState km = traj.evaluate(t - kH);

      const Vec3 v_num = (kp.position - km.position) / (2.0 * kH);
      const Vec3 a_num = (kp.velocity - km.velocity) / (2.0 * kH);
      const Vec3 w_num = numericalBodyRate(traj, t, kH);

      CHECK((k.velocity - v_num).norm() == Approx(0.0).margin(kDerivTol));
      CHECK((k.acceleration - a_num).norm() == Approx(0.0).margin(kDerivTol));
      CHECK((k.angular_rate - w_num).norm() == Approx(0.0).margin(kDerivTol));
      CHECK(k.time == t);
    }
  }
}

TEST_CASE("the body rate is the body-frame rate, not the world-frame rate",
          "[calib][trajectory]") {
  // Guards against a world/body mix-up that the finite-difference test would not reveal if the
  // attitude stayed near identity: once the rig is tilted the two differ by R_WI.
  const FigureEightTrajectory traj(tiltedConfig());
  bool found_discriminating_sample = false;
  for (double t = 5.0; t < 25.0; t += 0.37) {
    const KinematicState k = traj.evaluate(t);
    const Vec3 w_world = rotationOf(k.quaternion) * k.angular_rate;
    if ((w_world - k.angular_rate).norm() > 0.05) found_discriminating_sample = true;
    // The left (world-frame) finite difference must reproduce R_WI * omega^I.
    const Vec4 qp = traj.evaluate(t + kH).quaternion;
    const Vec4 qm = traj.evaluate(t - kH).quaternion;
    const Vec3 w_world_num =
        math::logMapQuat(math::quatMultiply(qp, math::quatConjugate(qm))) / (2.0 * kH);
    INFO("t = " << t);
    CHECK((w_world - w_world_num).norm() == Approx(0.0).margin(kDerivTol));
  }
  REQUIRE(found_discriminating_sample);
}

TEST_CASE("the rig is exactly stationary during the static period", "[calib][trajectory]") {
  for (const TrajectoryConfig& cfg : {TrajectoryConfig{}, tiltedConfig()}) {
    const FigureEightTrajectory traj(cfg);
    const Vec4 q_rest = math::eulerToQuat(cfg.base_euler(0), cfg.base_euler(1), cfg.base_euler(2));
    // Includes t = 0 and t = static_duration itself (the ramp starts strictly after it).
    for (int i = 0; i <= 40; ++i) {
      const double t = cfg.static_duration * static_cast<double>(i) / 40.0;
      INFO("t = " << t);
      const KinematicState k = traj.evaluate(t);
      // Exact equalities: with s = s' = s'' = 0 every excursion term is multiplied by zero.
      CHECK((k.position - cfg.center).cwiseAbs().maxCoeff() == 0.0);
      CHECK(k.velocity.cwiseAbs().maxCoeff() == 0.0);
      CHECK(k.acceleration.cwiseAbs().maxCoeff() == 0.0);
      CHECK(k.angular_rate.cwiseAbs().maxCoeff() == 0.0);
      CHECK((k.quaternion - q_rest).cwiseAbs().maxCoeff() == 0.0);
      CHECK((k.euler - cfg.base_euler).cwiseAbs().maxCoeff() == 0.0);
    }
    // ...and it has started moving shortly after the ramp begins.
    const KinematicState moving = traj.evaluate(cfg.static_duration + 0.5 * cfg.ramp_duration);
    CHECK(moving.velocity.norm() > 1e-3);
    CHECK(moving.angular_rate.norm() > 1e-3);
  }
}

TEST_CASE("specific force at rest is gravity reaction resolved in the body frame",
          "[calib][trajectory]") {
  const double g = constants::kGravity;
  for (const TrajectoryConfig& cfg : {TrajectoryConfig{}, tiltedConfig()}) {
    const FigureEightTrajectory traj(cfg);
    for (double t : {0.0, 0.25 * cfg.static_duration, cfg.static_duration}) {
      const KinematicState k = traj.evaluate(t);
      const Vec3 expected = rotationOf(k.quaternion).transpose() * Vec3(0.0, 0.0, -g);
      const Vec3 f = k.specificForce(g);
      INFO("t = " << t);
      // Only rounding separates the two rotation paths: |error| ~ eps * g ~ 1e-15.
      CHECK((f - expected).norm() == Approx(0.0).margin(1e-12));
      CHECK(f.norm() == Approx(g).epsilon(1e-14));
    }
  }
  // Level rig: an NED accelerometer at rest reads -g on its down axis.
  const KinematicState level = FigureEightTrajectory().evaluate(0.0);
  const Vec3 f_level = level.specificForce(g);
  CHECK(f_level.x() == Approx(0.0).margin(1e-15));
  CHECK(f_level.y() == Approx(0.0).margin(1e-15));
  CHECK(f_level.z() == Approx(-g));
}

TEST_CASE("specific force in motion is R^T (a - g)", "[calib][trajectory]") {
  const double g = constants::kGravity;
  const FigureEightTrajectory traj(tiltedConfig());
  for (double t : {4.1, 7.3, 12.9, 33.0}) {
    const KinematicState k = traj.evaluate(t);
    const Vec3 expected =
        rotationOf(k.quaternion).transpose() * (k.acceleration - Vec3(0.0, 0.0, g));
    INFO("t = " << t);
    CHECK((k.specificForce(g) - expected).norm() == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("the quintic envelope is C2-continuous at both ends of the ramp",
          "[calib][trajectory]") {
  const TrajectoryConfig cfg = tiltedConfig();
  const FigureEightTrajectory traj(cfg);
  const double t0 = cfg.static_duration;
  const double T = cfg.ramp_duration;
  const double t1 = t0 + T;

  // Exact values at and outside the ramp.
  const Vec3 e_start = traj.envelope(t0);
  CHECK(e_start.cwiseAbs().maxCoeff() == 0.0);
  const Vec3 e_end = traj.envelope(t1);
  CHECK(e_end(0) == 1.0);
  CHECK(e_end(1) == 0.0);
  CHECK(e_end(2) == 0.0);
  CHECK(traj.envelope(t1 + 100.0) == Vec3(1.0, 0.0, 0.0));
  CHECK(traj.envelope(-1.0) == Vec3::Zero());

  // One-sided approach from inside the ramp. s = 10u^3 - 15u^4 + 6u^5 has a triple zero of s,
  // a double zero of s' and a simple zero of s'' at u = 0, and the mirror image at u = 1
  // (1 - s, s', s'' with the same multiplicities in 1 - u). The bounds below are the leading
  // Taylor coefficients (10, 30/T, 60/T^2) plus a 10 % allowance for the next term, so they
  // are exactly what C2 continuity with this polynomial predicts.
  for (double delta : {1e-2, 1e-4, 1e-6}) {
    INFO("delta = " << delta);
    const double u = delta / T;
    const Vec3 lo = traj.envelope(t0 + delta);
    CHECK(std::abs(lo(0)) <= 11.0 * u * u * u);
    CHECK(std::abs(lo(1)) <= 33.0 * u * u / T);
    CHECK(std::abs(lo(2)) <= 66.0 * u / (T * T));
    const Vec3 hi = traj.envelope(t1 - delta);
    CHECK(std::abs(1.0 - hi(0)) <= 11.0 * u * u * u);
    CHECK(std::abs(hi(1)) <= 33.0 * u * u / T);
    CHECK(std::abs(hi(2)) <= 66.0 * u / (T * T));
  }

  // Inside the ramp the returned derivatives are the derivatives of the returned value.
  // Fourth derivative of s is <= 360/T^4 ~ 23 here, so the h^2/6 truncation is ~4e-8.
  for (double f : {0.05, 0.2, 0.5, 0.8, 0.95}) {
    const double t = t0 + f * T;
    const Vec3 e = traj.envelope(t);
    const Vec3 ep = traj.envelope(t + kH);
    const Vec3 em = traj.envelope(t - kH);
    INFO("u = " << f);
    CHECK(e(1) == Approx((ep(0) - em(0)) / (2.0 * kH)).margin(kDerivTol));
    CHECK(e(2) == Approx((ep(1) - em(1)) / (2.0 * kH)).margin(kDerivTol));
    CHECK(e(0) >= 0.0);
    CHECK(e(0) <= 1.0);
    CHECK(e(1) >= 0.0);  // monotone ramp
  }
}

TEST_CASE("acceleration and body rate are continuous across both ramp ends",
          "[calib][trajectory]") {
  // C2 position and C1 attitude: the change of v, a, omega across a ramp end over a window of
  // 2*delta is at most 2*delta*max(|a|, |jerk|, |omega_dot|). For these trajectories those are
  // below ~2 (the third derivatives estimated above are O(1)), so 2*delta*10 leaves a 5x margin.
  // A discontinuity would instead give a delta-independent jump of O(0.1-1), which fails the
  // bound by orders of magnitude at delta = 1e-5.
  for (const TrajectoryConfig& cfg : {TrajectoryConfig{}, tiltedConfig()}) {
    const FigureEightTrajectory traj(cfg);
    for (double t_edge : {cfg.static_duration, cfg.static_duration + cfg.ramp_duration}) {
      for (double delta : {1e-3, 1e-5}) {
        const KinematicState a = traj.evaluate(t_edge - delta);
        const KinematicState b = traj.evaluate(t_edge + delta);
        INFO("edge = " << t_edge << ", delta = " << delta);
        CHECK((b.acceleration - a.acceleration).norm() <= 20.0 * delta);
        CHECK((b.angular_rate - a.angular_rate).norm() <= 20.0 * delta);
        CHECK((b.velocity - a.velocity).norm() <= 20.0 * delta);
      }
    }
  }
}

TEST_CASE("yaw-only motion rotates about the body axis aligned with world down",
          "[calib][trajectory][observability]") {
  TrajectoryConfig cfg = tiltedConfig();
  cfg.euler_amplitude = Vec3(0.0, 0.0, 0.45);  // roll and pitch amplitude zero
  const FigureEightTrajectory traj(cfg);

  double max_rate = 0.0;
  Mat3 info = Mat3::Zero();
  for (double t = cfg.static_duration + 0.01; t < 40.0; t += 0.093) {
    const KinematicState k = traj.evaluate(t);
    const Vec3 ez_body = rotationOf(k.quaternion).transpose() * Vec3::UnitZ();
    INFO("t = " << t);
    // Roll and pitch never leave their mean values.
    CHECK(k.euler(0) == cfg.base_euler(0));
    CHECK(k.euler(1) == cfg.base_euler(1));
    // omega x R^T e_z must vanish; only rounding of O(eps * |omega|) separates the paths.
    CHECK(k.angular_rate.cross(ez_body).norm() == Approx(0.0).margin(1e-12));
    // ...and the component along it is the yaw rate psi_dot (finite difference of the yaw).
    const double psi_dot =
        (traj.evaluate(t + kH).euler(2) - traj.evaluate(t - kH).euler(2)) / (2.0 * kH);
    CHECK(k.angular_rate.dot(ez_body) == Approx(psi_dot).margin(kDerivTol));
    max_rate = std::max(max_rate, k.angular_rate.norm());
    info += k.angular_rate * k.angular_rate.transpose();
  }
  REQUIRE(max_rate > 0.3);  // the check above was not vacuous
  // A single fixed body axis: the rate "information" matrix is rank one, which is exactly why
  // the lever-arm component along that axis is unobservable from yaw-only motion.
  Eigen::SelfAdjointEigenSolver<Mat3> es(info);
  CHECK(es.eigenvalues()(1) <= 1e-12 * es.eigenvalues()(2));
}

TEST_CASE("the default trajectory excites rotation about every body axis",
          "[calib][trajectory][observability]") {
  const FigureEightTrajectory traj;
  const TrajectoryConfig& cfg = traj.config();
  Mat3 info = Mat3::Zero();
  int n = 0;
  bool not_parallel_to_down = false;
  for (double t = cfg.static_duration + cfg.ramp_duration; t < 65.0; t += 0.05, ++n) {
    const KinematicState k = traj.evaluate(t);
    info += k.angular_rate * k.angular_rate.transpose();
    const Vec3 ez_body = rotationOf(k.quaternion).transpose() * Vec3::UnitZ();
    if (k.angular_rate.cross(ez_body).norm() > 0.1) not_parallel_to_down = true;
  }
  info /= static_cast<double>(n);
  // Per-axis Euler-rate variances are (a_i * 2 pi f_i)^2 / 2 ~ 0.27, 0.41, 0.21 rad^2/s^2.
  // Requiring the weakest body direction to have an RMS rate of at least 0.2 rad/s
  // (eigenvalue >= 0.04) is a factor ~5 below that and far above the rank-deficient
  // yaw-only case (~0).
  Eigen::SelfAdjointEigenSolver<Mat3> es(info);
  CHECK(es.eigenvalues()(0) >= 0.04);
  CHECK(not_parallel_to_down);
}

TEST_CASE("translation stays inside the configured figure-eight box", "[calib][trajectory]") {
  const FigureEightTrajectory traj;
  const TrajectoryConfig& c = traj.config();
  const Vec3 amp(c.amp_north, c.amp_east, c.amp_down);
  Vec3 max_dev = Vec3::Zero();
  for (double t = 0.0; t < 60.0; t += 0.01) {
    const Vec3 d = (traj.evaluate(t).position - c.center).cwiseAbs();
    max_dev = max_dev.cwiseMax(d);
  }
  for (int i = 0; i < 3; ++i) {
    INFO("axis " << i);
    CHECK(max_dev(i) <= amp(i) * (1.0 + 1e-12));
    CHECK(max_dev(i) >= 0.99 * amp(i));  // the full excursion is actually reached
  }
}

TEST_CASE("Euler angles reported by the trajectory match its quaternion", "[calib][trajectory]") {
  const FigureEightTrajectory traj(tiltedConfig());
  for (double t : {0.0, 2.2, 5.5, 13.1, 29.9}) {
    const KinematicState k = traj.evaluate(t);
    const Vec3 e = math::quatToEuler(k.quaternion);
    INFO("t = " << t);
    CHECK(e(0) == Approx(k.euler(0)).margin(1e-12));
    CHECK(e(1) == Approx(k.euler(1)).margin(1e-12));
    CHECK(math::wrapPi(e(2) - k.euler(2)) == Approx(0.0).margin(1e-12));
  }
}
