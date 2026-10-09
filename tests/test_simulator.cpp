/// \file test_simulator.cpp
/// \brief Stereo/IMU truth generator: reproducibility, frame timing, noise-free projection
///        through the hidden extrinsic, drift and knock scenarios, outliers and the IMU model.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "aether/calib/CalibSimulator.hpp"
#include "aether/core/Constants.hpp"
#include "aether/math/Rotation.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::CalibSimulator;
using calib::Extrinsic;
using calib::FigureEightTrajectory;
using calib::KinematicState;
using calib::SimConfig;
using calib::StereoFrame;
using calib::StereoObservation;

/// Rotation matrix from a [w,x,y,z] quaternion via Eigen (independent of math::quatToRotation).
Mat3 rotationOf(const Vec4& q) {
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).normalized().toRotationMatrix();
}

/// Rotation angle of R_a^T R_b via Eigen (independent of math::boxMinus).
double angleBetween(const Vec4& qa, const Vec4& qb) {
  return Eigen::AngleAxisd(rotationOf(qa).transpose() * rotationOf(qb)).angle();
}

bool sameExtrinsic(const Extrinsic& a, const Extrinsic& b) {
  return (a.q_ic - b.q_ic).cwiseAbs().maxCoeff() == 0.0 &&
         (a.p_ic - b.p_ic).cwiseAbs().maxCoeff() == 0.0;
}

/// Independently recomputed stereo view of one landmark.
struct ExpectedView {
  Vec3 P = Vec3::Zero();      ///< Point in the left camera frame [m]
  Vec2 left = Vec2::Zero();   ///< [px]
  Vec2 right = Vec2::Zero();  ///< [px]
  bool visible = false;       ///< Passes the depth and margin tests.
  bool ambiguous = false;     ///< Within rounding distance of a visibility threshold.
};

/// Pinhole view of world point \a f from IMU pose \a s through extrinsic \a e:
/// R_wc = R_wi R_ic, p_wc = p_wi + R_wi p_ic, P = R_wc^T (f - p_wc).
ExpectedView expectedView(const SimConfig& c, const KinematicState& s, const Extrinsic& e,
                          const Vec3& f) {
  const Mat3 R_wi = rotationOf(s.quaternion);
  const Mat3 R_wc = R_wi * rotationOf(e.q_ic);
  const Vec3 p_wc = s.position + R_wi * e.p_ic;
  ExpectedView v;
  v.P = R_wc.transpose() * (f - p_wc);
  const auto& k = c.camera;
  v.left = Vec2(k.fx * v.P.x() / v.P.z() + k.cx, k.fy * v.P.y() / v.P.z() + k.cy);
  v.right =
      Vec2(k.fx * (v.P.x() - k.baseline) / v.P.z() + k.cx, k.fy * v.P.y() / v.P.z() + k.cy);

  // Visibility with a guard band: our rotation path differs from the simulator's by rounding
  // (~1e-12 px), so a pixel within 1e-6 px of a threshold may legitimately go either way.
  constexpr double kBand = 1e-6;
  const double m = c.image_margin;
  const double xmax = k.width - 1.0 - m;
  const double ymax = k.height - 1.0 - m;
  auto margin_slack = [&](const Vec2& uv) {
    return std::min({uv.x() - m, xmax - uv.x(), uv.y() - m, ymax - uv.y()});
  };
  const double depth_slack = v.P.z() - c.min_depth;
  if (depth_slack < -kBand) {
    v.visible = false;
  } else {
    const double slack = std::min(margin_slack(v.left), margin_slack(v.right));
    v.visible = depth_slack >= 0.0 && slack >= 0.0;
    v.ambiguous = std::abs(depth_slack) < kBand || std::abs(slack) < kBand;
  }
  return v;
}

/// Short noise-free scenario with every truth effect switched on: a camera-IMU time offset,
/// an (exaggerated) rotation and lever-arm drift and a knock in the middle of the run.
SimConfig noiseFreeConfig() {
  SimConfig c;
  c.duration = 8.0;
  c.pixel_noise = 0.0;
  c.time_offset = 0.013;
  c.drift.start_time = 1.0;
  c.drift.rotation_rate = 3.0 * constants::kDegToRad;
  c.drift.rotation_axis = Vec3(0.3, -0.5, 0.8);
  c.drift.translation_rate = Vec3(0.002, -0.001, 0.003);
  c.drift.step_time = 5.0;
  c.drift.step_rotation = Vec3(0.02, -0.01, 0.03);
  c.drift.step_translation = Vec3(0.01, 0.0, -0.005);
  return c;
}

/// Largest pixel distance between an observation and its noise-free prediction.
double pixelError(const StereoObservation& o, const ExpectedView& v) {
  return std::max((o.left - v.left).norm(), (o.right - v.right).norm());
}

}  // namespace

TEST_CASE("the simulator is bit-for-bit reproducible for a fixed seed", "[calib][simulator]") {
  SimConfig c;
  c.duration = 10.0;
  c.time_offset = 0.004;
  c.outlier_fraction = 0.1;  // exercise every random stream
  c.drift.start_time = 2.0;
  CalibSimulator a(c, 1234), b(c, 1234);

  REQUIRE(a.landmarks().size() == b.landmarks().size());
  for (std::size_t j = 0; j < a.landmarks().size(); ++j) {
    REQUIRE((a.landmarks()[j] - b.landmarks()[j]).cwiseAbs().maxCoeff() == 0.0);
  }
  CHECK(sameExtrinsic(a.truthExtrinsic(0.0), b.truthExtrinsic(0.0)));
  CHECK(sameExtrinsic(a.truthExtrinsic(9.0), b.truthExtrinsic(9.0)));
  CHECK((a.driftAxis() - b.driftAxis()).cwiseAbs().maxCoeff() == 0.0);

  REQUIRE(a.imuCount() == b.imuCount());
  for (int i = 0; i < a.imuCount(); ++i) {
    const sensors::ImuSample sa = a.imu(i);
    const sensors::ImuSample sb = b.imu(i);
    INFO("imu " << i);
    REQUIRE(sa.time == sb.time);
    REQUIRE((sa.gyro - sb.gyro).cwiseAbs().maxCoeff() == 0.0);
    REQUIRE((sa.accel - sb.accel).cwiseAbs().maxCoeff() == 0.0);
    REQUIRE((a.truthGyroBias() - b.truthGyroBias()).cwiseAbs().maxCoeff() == 0.0);
    REQUIRE((a.truthAccelBias() - b.truthAccelBias()).cwiseAbs().maxCoeff() == 0.0);
  }
  REQUIRE(a.frameCount() == b.frameCount());
  for (int k = 0; k < a.frameCount(); ++k) {
    const StereoFrame fa = a.frame(k);
    const StereoFrame fb = b.frame(k);
    INFO("frame " << k);
    REQUIRE(fa.stamp == fb.stamp);
    REQUIRE(fa.observations.size() == fb.observations.size());
    for (std::size_t j = 0; j < fa.observations.size(); ++j) {
      REQUIRE(fa.observations[j].id == fb.observations[j].id);
      REQUIRE((fa.observations[j].left - fb.observations[j].left).cwiseAbs().maxCoeff() == 0.0);
      REQUIRE((fa.observations[j].right - fb.observations[j].right).cwiseAbs().maxCoeff() == 0.0);
    }
  }

  // A different seed changes the scene, the mount, the IMU errors and the pixel noise.
  CalibSimulator d(c, 1235);
  CHECK((d.landmarks()[0] - a.landmarks()[0]).norm() > 0.0);
  CHECK(angleBetween(d.truthExtrinsic(0.0).q_ic, a.truthExtrinsic(0.0).q_ic) > 1e-6);
  CHECK((d.imu(0).gyro - CalibSimulator(c, 1234).imu(0).gyro).norm() > 0.0);
}

TEST_CASE("sample counts and timestamps follow the configured rates and time offset",
          "[calib][simulator]") {
  SimConfig c;
  c.duration = 3.3;
  c.camera_rate = 15.0;
  c.imu.rate = 100.0;
  c.time_offset = 0.017;
  CalibSimulator sim(c, 5);
  // floor(duration * rate) + 1 samples, the first at t = 0.
  CHECK(sim.imuCount() == 331);
  CHECK(sim.frameCount() == 50);
  for (int i = 0; i < sim.imuCount(); ++i) {
    const double t = static_cast<double>(i) / 100.0;
    CHECK(sim.imuTime(i) == Approx(t).margin(1e-12));
    CHECK(sim.imu(i).time == sim.imuTime(i));
  }
  for (int k = 0; k < sim.frameCount(); ++k) {
    const double t_capture = static_cast<double>(k) / 15.0;
    INFO("frame " << k);
    CHECK(sim.captureTime(k) == Approx(t_capture).margin(1e-12));
    CHECK(sim.frameStamp(k) == Approx(t_capture - 0.017).margin(1e-12));
    // The camera clock lags the IMU clock: stamp = captureTime - t_d.
    CHECK(sim.frame(k).stamp == sim.captureTime(k) - c.time_offset);
  }
  CHECK(sim.captureTime(sim.frameCount() - 1) <= c.duration);
  CHECK(sim.imuTime(sim.imuCount() - 1) <= c.duration);
}

TEST_CASE("noise-free observations are the projections through the true extrinsic",
          "[calib][simulator]") {
  const SimConfig c = noiseFreeConfig();
  CalibSimulator sim(c, 77);
  const FigureEightTrajectory traj(c.trajectory);  // independent copy of the truth motion
  const double m = c.image_margin;
  const double xmax = c.camera.width - 1.0 - m;
  const double ymax = c.camera.height - 1.0 - m;

  std::size_t n_obs = 0;
  double max_err_at_stamp = 0.0;
  for (int k = 0; k < sim.frameCount(); ++k) {
    const StereoFrame fr = sim.frame(k);
    const double t = sim.captureTime(k);
    const KinematicState s = traj.evaluate(t);
    const Extrinsic e = sim.truthExtrinsic(t);
    // What a simulator that (wrongly) exposed at the camera-clock stamp would produce.
    const KinematicState s_stamp = traj.evaluate(fr.stamp);
    const Extrinsic e_stamp = sim.truthExtrinsic(fr.stamp);
    INFO("frame " << k << ", t = " << t);

    std::vector<bool> seen(sim.landmarks().size(), false);
    int prev_id = -1;
    for (const StereoObservation& o : fr.observations) {
      INFO("landmark " << o.id);
      REQUIRE(o.id > prev_id);  // strictly sorted, no duplicates
      prev_id = o.id;
      REQUIRE(o.id < static_cast<int>(sim.landmarks().size()));
      seen[static_cast<std::size_t>(o.id)] = true;

      const Vec3& f = sim.landmarks()[static_cast<std::size_t>(o.id)];
      const ExpectedView v = expectedView(c, s, e, f);
      // Two algebraically identical rotation paths (Eigen vs math::) differ by rounding only:
      // |dP| ~ 1e-15 m, i.e. ~1e-12 px after multiplying by fx/Z.
      CHECK(pixelError(o, v) == Approx(0.0).margin(1e-9));
      CHECK(v.P.z() >= c.min_depth - 1e-9);
      for (const Vec2& uv : {o.left, o.right}) {
        CHECK(uv.x() >= m);
        CHECK(uv.x() <= xmax);
        CHECK(uv.y() >= m);
        CHECK(uv.y() <= ymax);
      }
      CHECK(o.left.y() == Approx(o.right.y()).margin(1e-9));  // rectified: same row
      CHECK(o.left.x() > o.right.x());                          // positive disparity
      max_err_at_stamp = std::max(max_err_at_stamp,
                                  pixelError(o, expectedView(c, s_stamp, e_stamp, f)));
      ++n_obs;
    }
    // Completeness: every unambiguously visible landmark is reported, and nothing else.
    for (std::size_t j = 0; j < sim.landmarks().size(); ++j) {
      const ExpectedView v = expectedView(c, s, e, sim.landmarks()[j]);
      if (v.ambiguous) continue;
      INFO("landmark " << j);
      CHECK(seen[j] == v.visible);
    }
  }
  REQUIRE(n_obs > 1000);
  // Sensitivity of this test: exposing at the stamp instead of stamp + t_d (13 ms of motion at
  // up to ~1 rad/s) would move pixels by several px, far outside the 1e-9 px tolerance.
  CHECK(max_err_at_stamp > 1.0);
}

TEST_CASE("the default scene keeps plenty of landmarks in view", "[calib][simulator]") {
  SimConfig c;  // 60 s, 200 landmarks
  CalibSimulator sim(c, 2024);
  REQUIRE(sim.landmarks().size() == 200);
  for (const Vec3& p : sim.landmarks()) {
    CHECK((p.array() >= c.scene.box_min.array()).all());
    CHECK((p.array() <= c.scene.box_max.array()).all());
  }
  // Over the run roughly half the box is inside the 77 x 62 deg field of view. Ask for at least
  // 20 stereo observations in every frame: well above the 3 needed to constrain a pose and a
  // check that the rig never looks away from the scene.
  std::size_t min_obs = c.scene.landmarks;
  for (int k = 0; k < sim.frameCount(); ++k) {
    min_obs = std::min(min_obs, sim.frame(k).observations.size());
  }
  CHECK(min_obs >= 20);
}

TEST_CASE("a custom scene is honoured", "[calib][simulator]") {
  SimConfig c;
  c.duration = 1.0;
  c.scene.landmarks = 37;
  c.scene.box_min = Vec3(3.0, -1.0, -0.5);
  c.scene.box_max = Vec3(4.0, 1.0, 0.5);
  CalibSimulator sim(c, 9);
  REQUIRE(sim.landmarks().size() == 37);
  for (const Vec3& p : sim.landmarks()) {
    CHECK((p.array() >= c.scene.box_min.array()).all());
    CHECK((p.array() <= c.scene.box_max.array()).all());
  }
}

TEST_CASE("the truth mount is the nominal mount dispersed with the configured sigma",
          "[calib][simulator]") {
  SimConfig c;
  c.duration = 0.0;

  SECTION("zero dispersion gives exactly the nominal mount") {
    c.mount_rotation_dispersion = 0.0;
    c.mount_translation_dispersion = 0.0;
    const Extrinsic e = CalibSimulator(c, 31).truthExtrinsic(0.0);
    CHECK((e.q_ic - math::quatNormalize(c.mount.q_ic)).norm() == Approx(0.0).margin(1e-15));
    CHECK((e.p_ic - c.mount.p_ic).cwiseAbs().maxCoeff() == 0.0);
  }

  SECTION("sample statistics over many trials match the 1-sigma values") {
    // 400 trials x 3 axes = 1200 samples per quantity. The sample standard deviation has a
    // relative standard error of 1/sqrt(2N) = 2 %, so +-10 % is a 5-sigma band; the sample
    // mean has standard error sigma/sqrt(N) = 0.029 sigma, so +-0.15 sigma is 5-sigma.
    constexpr int kTrials = 400;
    double sum_r = 0.0, sum_r2 = 0.0, sum_p = 0.0, sum_p2 = 0.0;
    for (int i = 0; i < kTrials; ++i) {
      const Extrinsic e = CalibSimulator(c, static_cast<std::uint64_t>(1000 + i)).truthExtrinsic(0.0);
      // Right (camera-frame) perturbation of the nominal mount.
      const Vec3 dr = math::boxMinus(e.q_ic, c.mount.q_ic);
      const Vec3 dp = e.p_ic - c.mount.p_ic;
      sum_r += dr.sum();
      sum_r2 += dr.squaredNorm();
      sum_p += dp.sum();
      sum_p2 += dp.squaredNorm();
    }
    const double n = 3.0 * kTrials;
    const double sr = c.mount_rotation_dispersion;
    const double sp = c.mount_translation_dispersion;
    CHECK(std::abs(sum_r / n) <= 0.15 * sr);
    CHECK(std::abs(sum_p / n) <= 0.15 * sp);
    CHECK(std::sqrt(sum_r2 / n) == Approx(sr).epsilon(0.10));
    CHECK(std::sqrt(sum_p2 / n) == Approx(sp).epsilon(0.10));
  }
}

TEST_CASE("rotation drift is a constant-rate camera-frame rotation about the drift axis",
          "[calib][simulator][drift]") {
  SimConfig c;
  c.duration = 0.0;
  c.drift.start_time = 10.0;  // default rate: 0.1 deg/min
  c.drift.rotation_axis = Vec3(0.3, -0.5, 0.8);
  c.drift.translation_rate = Vec3(1e-4, -2e-4, 5e-5);
  const CalibSimulator sim(c, 4242);
  const double rate = c.drift.rotation_rate;
  REQUIRE(rate == Approx(0.1 * constants::kDegToRad / 60.0));

  // The configured axis is used, normalised.
  CHECK((sim.driftAxis() - c.drift.rotation_axis.normalized()).norm() ==
        Approx(0.0).margin(1e-15));

  // Constant up to and including the onset time.
  const Extrinsic e0 = sim.truthExtrinsic(0.0);
  for (double t : {0.0, 1.0, 5.0, 9.999, 10.0}) {
    INFO("t = " << t);
    CHECK(sameExtrinsic(sim.truthExtrinsic(t), e0));
  }

  const Extrinsic es = sim.truthExtrinsic(c.drift.start_time);
  const Mat3 Rs = rotationOf(es.q_ic);
  for (double dt : {0.5, 10.0, 60.0, 300.0, 600.0, 3600.0}) {
    const Extrinsic e = sim.truthExtrinsic(c.drift.start_time + dt);
    INFO("dt = " << dt);
    // Right perturbation: R(t) = R(t0) Exp(axis * rate * dt). Rounding in the quaternion
    // product and log is O(eps) ~ 1e-16 rad, so 1e-12 rad is a pure round-off tolerance (the
    // smallest drift checked is 1.5e-5 rad).
    const Vec3 dtheta = math::boxMinus(e.q_ic, es.q_ic);
    CHECK((dtheta - sim.driftAxis() * (rate * dt)).norm() == Approx(0.0).margin(1e-12));
    CHECK(angleBetween(es.q_ic, e.q_ic) == Approx(rate * dt).margin(1e-12));
    const Mat3 R_expected = Rs * Eigen::AngleAxisd(rate * dt, sim.driftAxis()).toRotationMatrix();
    CHECK((rotationOf(e.q_ic) - R_expected).norm() == Approx(0.0).margin(1e-12));
    // Linear lever-arm drift.
    CHECK((e.p_ic - es.p_ic - c.drift.translation_rate * dt).norm() ==
          Approx(0.0).margin(1e-12));
  }
  // 0.1 deg/min for one hour is 6 deg.
  CHECK(angleBetween(es.q_ic, sim.truthExtrinsic(c.drift.start_time + 3600.0).q_ic) *
            constants::kRadToDeg ==
        Approx(6.0).epsilon(1e-9));
}

TEST_CASE("a zero drift axis selects a random unit axis per seed", "[calib][simulator][drift]") {
  SimConfig c;
  c.duration = 0.0;
  c.drift.start_time = 0.0;
  c.drift.rotation_rate = 0.01;
  REQUIRE(c.drift.rotation_axis.norm() == 0.0);
  const CalibSimulator a(c, 1), a2(c, 1), b(c, 2);
  CHECK(a.driftAxis().norm() == Approx(1.0).margin(1e-14));
  CHECK(b.driftAxis().norm() == Approx(1.0).margin(1e-14));
  CHECK((a.driftAxis() - a2.driftAxis()).cwiseAbs().maxCoeff() == 0.0);
  CHECK((a.driftAxis() - b.driftAxis()).norm() > 1e-3);
  // ...and that axis is the one the truth rotates about.
  const Vec3 dtheta = math::boxMinus(a.truthExtrinsic(20.0).q_ic, a.truthExtrinsic(0.0).q_ic);
  CHECK((dtheta - a.driftAxis() * 0.2).norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("disabled drift leaves the truth extrinsic constant", "[calib][simulator][drift]") {
  SimConfig c;
  c.duration = 0.0;
  c.drift.start_time = -1.0;
  c.drift.rotation_rate = 1.0;               // would be enormous if it were applied
  c.drift.translation_rate = Vec3(1, 1, 1);  // likewise
  const CalibSimulator sim(c, 3);
  const Extrinsic e0 = sim.truthExtrinsic(0.0);
  for (double t : {1.0, 60.0, 1e4}) CHECK(sameExtrinsic(sim.truthExtrinsic(t), e0));
}

TEST_CASE("a knock is an instantaneous step at step_time", "[calib][simulator][drift]") {
  SimConfig c;
  c.duration = 0.0;
  c.drift.step_time = 3.0;
  c.drift.step_rotation = Vec3(0.01, -0.02, 0.005);
  c.drift.step_translation = Vec3(0.003, 0.0, -0.002);

  SECTION("without drift") {
    const CalibSimulator sim(c, 11);
    const Extrinsic e0 = sim.truthExtrinsic(0.0);
    CHECK(sameExtrinsic(sim.truthExtrinsic(2.0), e0));
    CHECK(sameExtrinsic(sim.truthExtrinsic(3.0 - 1e-9), e0));
    const Extrinsic e1 = sim.truthExtrinsic(3.0);  // applied at t >= step_time
    // Round-off tolerances as in the drift test.
    CHECK((math::boxMinus(e1.q_ic, e0.q_ic) - c.drift.step_rotation).norm() ==
          Approx(0.0).margin(1e-12));
    CHECK((e1.p_ic - e0.p_ic - c.drift.step_translation).norm() == Approx(0.0).margin(1e-15));
    // The step persists unchanged afterwards.
    CHECK(sameExtrinsic(sim.truthExtrinsic(3.5), e1));
    CHECK(sameExtrinsic(sim.truthExtrinsic(1000.0), e1));
  }

  SECTION("on top of drift") {
    c.drift.start_time = 1.0;
    c.drift.rotation_rate = 2.0 * constants::kDegToRad;  // exaggerated
    c.drift.rotation_axis = Vec3(0.0, 1.0, 0.0);
    const CalibSimulator sim(c, 11);
    // Across the knock the change is the knock itself, expressed in the (drifted) camera
    // frame; the drift over 1e-9 s adds only rate * 1e-9 ~ 3.5e-11 rad.
    const double eps = 1e-9;
    const Extrinsic before = sim.truthExtrinsic(3.0 - eps);
    const Extrinsic after = sim.truthExtrinsic(3.0);
    CHECK((math::boxMinus(after.q_ic, before.q_ic) - c.drift.step_rotation).norm() ==
          Approx(0.0).margin(1e-9));
    CHECK((after.p_ic - before.p_ic - c.drift.step_translation).norm() ==
          Approx(0.0).margin(1e-9));
    // ...and the drift carries on at the same rate afterwards.
    const Extrinsic later = sim.truthExtrinsic(13.0);
    CHECK(angleBetween(after.q_ic, later.q_ic) == Approx(c.drift.rotation_rate * 10.0).margin(1e-12));
  }
}

TEST_CASE("outliers are correspondences inconsistent with the projection",
          "[calib][simulator][outliers]") {
  SimConfig c;
  c.duration = 4.0;
  c.pixel_noise = 0.0;
  const FigureEightTrajectory traj(c.trajectory);

  SECTION("outlier_fraction = 1 corrupts every correspondence") {
    SimConfig co = c;
    co.outlier_fraction = 1.0;
    CalibSimulator clean(c, 606), dirty(co, 606);
    std::size_t n = 0, inconsistent = 0;
    for (int k = 0; k < dirty.frameCount(); ++k) {
      const StereoFrame fc = clean.frame(k);
      const StereoFrame fd = dirty.frame(k);
      const double t = dirty.captureTime(k);
      const KinematicState s = traj.evaluate(t);
      const Extrinsic e = dirty.truthExtrinsic(t);
      INFO("frame " << k);
      // Visibility is decided on the true geometry, so the same landmarks are reported.
      REQUIRE(fd.observations.size() == fc.observations.size());
      for (std::size_t j = 0; j < fd.observations.size(); ++j) {
        const StereoObservation& o = fd.observations[j];
        REQUIRE(o.id == fc.observations[j].id);
        // A wrong correspondence is still a plausible one: in the image, same row, positive
        // disparity in [2, 40) px.
        CHECK(o.left.x() >= 0.0);
        CHECK(o.left.x() <= c.camera.width - 1.0);
        CHECK(o.left.y() >= 0.0);
        CHECK(o.left.y() <= c.camera.height - 1.0);
        CHECK(o.right.y() == o.left.y());
        CHECK(o.left.x() - o.right.x() >= 2.0);
        CHECK(o.left.x() - o.right.x() < 40.0);
        const ExpectedView v =
            expectedView(c, s, e, dirty.landmarks()[static_cast<std::size_t>(o.id)]);
        if (pixelError(o, v) > 1.0) ++inconsistent;
        ++n;
      }
    }
    REQUIRE(n > 1000);
    // A uniform random left pixel lands within 1 px of the true one with probability
    // pi / (640 * 480) ~ 1e-5, so essentially every observation must be inconsistent.
    CHECK(static_cast<double>(inconsistent) >= 0.99 * static_cast<double>(n));
  }

  SECTION("a partial outlier fraction is honoured statistically") {
    SimConfig co = c;
    co.outlier_fraction = 0.3;
    CalibSimulator sim(co, 607);
    std::size_t n = 0, outliers = 0;
    for (int k = 0; k < sim.frameCount(); ++k) {
      const StereoFrame fr = sim.frame(k);
      const double t = sim.captureTime(k);
      const KinematicState s = traj.evaluate(t);
      const Extrinsic e = sim.truthExtrinsic(t);
      for (const StereoObservation& o : fr.observations) {
        const ExpectedView v =
            expectedView(co, s, e, sim.landmarks()[static_cast<std::size_t>(o.id)]);
        // Inliers are exact to ~1e-12 px with zero pixel noise.
        if (pixelError(o, v) > 1e-6) ++outliers;
        ++n;
      }
    }
    REQUIRE(n > 1000);
    // Binomial: 5-sigma band around the configured fraction.
    const double p = 0.3;
    const double frac = static_cast<double>(outliers) / static_cast<double>(n);
    CHECK(std::abs(frac - p) <= 5.0 * std::sqrt(p * (1.0 - p) / static_cast<double>(n)));
  }
}

TEST_CASE("a noise-free IMU reports the trajectory's body rate and specific force",
          "[calib][simulator][imu]") {
  SimConfig c;
  c.duration = 10.0;
  c.imu.gyro_noise_density = 0.0;
  c.imu.accel_noise_density = 0.0;
  c.imu.gyro_bias_initial = 0.0;
  c.imu.accel_bias_initial = 0.0;
  c.imu.gyro_bias_walk = 0.0;
  c.imu.accel_bias_walk = 0.0;
  CalibSimulator sim(c, 99);
  const FigureEightTrajectory traj(c.trajectory);
  const double g = c.gravity;
  for (int i = 0; i < sim.imuCount(); ++i) {
    const sensors::ImuSample m = sim.imu(i);
    const KinematicState k = traj.evaluate(sim.imuTime(i));
    INFO("t = " << m.time);
    // Every error term is an exact zero, so the rate must be reproduced exactly.
    REQUIRE((m.gyro - k.angular_rate).cwiseAbs().maxCoeff() == 0.0);
    // Specific force recomputed through Eigen's rotation: rounding only (|f| ~ 10 m/s^2).
    const Vec3 f = rotationOf(k.quaternion).transpose() * (k.acceleration - Vec3(0.0, 0.0, g));
    REQUIRE((m.accel - f).norm() == Approx(0.0).margin(1e-12));
    REQUIRE(sim.truthGyroBias().norm() == 0.0);
    REQUIRE(sim.truthAccelBias().norm() == 0.0);
  }
  // At rest (the static period) the level rig reads exactly -g on its down axis.
  CalibSimulator rest(c, 99);
  const sensors::ImuSample m0 = rest.imu(0);
  CHECK(m0.gyro.norm() == 0.0);
  CHECK((m0.accel - Vec3(0.0, 0.0, -g)).norm() == Approx(0.0).margin(1e-14));
}

TEST_CASE("IMU errors equal the reported truth biases when white noise is off",
          "[calib][simulator][imu]") {
  SimConfig c;
  c.duration = 10.0;
  c.imu.gyro_noise_density = 0.0;
  c.imu.accel_noise_density = 0.0;
  CalibSimulator sim(c, 123);
  const FigureEightTrajectory traj(c.trajectory);
  const double dt = 1.0 / c.imu.rate;
  Vec3 bg_prev = Vec3::Zero(), ba_prev = Vec3::Zero();
  double sum_dg2 = 0.0, sum_da2 = 0.0;
  int n = 0;
  for (int i = 0; i < sim.imuCount(); ++i) {
    const sensors::ImuSample m = sim.imu(i);
    const KinematicState k = traj.evaluate(m.time);
    INFO("t = " << m.time);
    // measurement - truth == bias "after the most recent imu() call"; rounding of
    // (x + b) - x is <= eps * |x| ~ 2e-15.
    REQUIRE((m.gyro - k.angular_rate - sim.truthGyroBias()).norm() == Approx(0.0).margin(1e-12));
    REQUIRE((m.accel - k.specificForce(c.gravity) - sim.truthAccelBias()).norm() ==
            Approx(0.0).margin(1e-12));
    if (i > 0) {
      sum_dg2 += (sim.truthGyroBias() - bg_prev).squaredNorm();
      sum_da2 += (sim.truthAccelBias() - ba_prev).squaredNorm();
      n += 3;
    }
    bg_prev = sim.truthGyroBias();
    ba_prev = sim.truthAccelBias();
  }
  // Random-walk increments have sigma = walk * sqrt(dt). With n = 6000 increments the sample
  // RMS has a relative standard error of 1/sqrt(2n) = 0.9 %; +-5 % is a 5-sigma band.
  CHECK(std::sqrt(sum_dg2 / n) == Approx(c.imu.gyro_bias_walk * std::sqrt(dt)).epsilon(0.05));
  CHECK(std::sqrt(sum_da2 / n) == Approx(c.imu.accel_bias_walk * std::sqrt(dt)).epsilon(0.05));
}
