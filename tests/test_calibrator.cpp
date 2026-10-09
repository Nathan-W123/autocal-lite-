/// \file test_calibrator.cpp
/// \brief End-to-end behaviour of the extrinsic calibrator on simulated data: IMU dead
///        reckoning against the truth, fixed point at the truth, convergence, covariance
///        health, static initialisation, observability (yaw-only ablation), time offset,
///        change detection after a knock and robustness to wrong correspondences.
///
/// These cases drive the whole pipeline (CalibSimulator -> ExtrinsicCalibrator -> scoring),
/// either through `calib::runTrial` or by streaming the simulator by hand exactly the way
/// `runTrial` does, when a test needs per-frame access to the filter. Every run uses a fixed
/// seed, so each case is deterministic.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <utility>
#include <vector>

#include "aether/calib/CalibSimulator.hpp"
#include "aether/calib/Campaign.hpp"
#include "aether/calib/ExtrinsicCalibrator.hpp"
#include "aether/core/Constants.hpp"
#include "aether/math/Rotation.hpp"
#include "aether/sensors/Imu.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::CalibSimulator;
using calib::Extrinsic;
using calib::ExtrinsicCalibrator;
using calib::FrameReport;
using calib::KinematicState;
using calib::TrialConfig;
using calib::TrialResult;

constexpr double kDeg = constants::kDegToRad;

/// Callback that does nothing (default hooks of streamRun).
struct NoOp {
  template <typename... Args>
  void operator()(Args&&...) const {}
};

/// Stream a simulated run through a calibrator in time order, exactly as `calib::runTrial`
/// does: every frame whose camera stamp is at most `frame_lookahead` ahead of the next IMU
/// sample is queued first, then the sample is added. `on_frame(k, frame)` sees frame \a k
/// before it is queued; `after_imu(i)` runs after IMU sample \a i.
template <typename AfterImu = NoOp, typename OnFrame = NoOp>
void streamRun(CalibSimulator& sim, ExtrinsicCalibrator& cal, AfterImu&& after_imu = AfterImu{},
               OnFrame&& on_frame = OnFrame{}) {
  const double lookahead = TrialConfig{}.frame_lookahead;
  int k = 0;
  for (int i = 0; i < sim.imuCount(); ++i) {
    const double t = sim.imuTime(i);
    while (k < sim.frameCount() && sim.frameStamp(k) <= t + lookahead) {
      calib::StereoFrame frame = sim.frame(k);
      on_frame(k, static_cast<const calib::StereoFrame&>(frame));
      cal.addFrame(std::move(frame));
      ++k;
    }
    cal.addImu(sim.imu(i));
    after_imu(i);
  }
}

/// The truth extrinsic offset by fixed rotation [rad] and lever-arm [m] errors of the given
/// magnitudes. (`runTrial` draws the direction at random; here it is pinned so the manual
/// runs are reproducible independently of `runTrial`'s internal seed streams.)
Extrinsic offsetPrior(const Extrinsic& truth, double rotation, double translation) {
  Extrinsic prior;
  prior.q_ic = math::boxPlus(truth.q_ic, rotation * Vec3(1.0, -1.0, 1.0).normalized());
  prior.p_ic = truth.p_ic + translation * Vec3(1.0, 1.0, -1.0).normalized();
  return prior;
}

/// Errors of IMU-only dead reckoning against the simulator truth.
struct DeadReckoning {
  double end_time = 0.0;
  double final_attitude = 0.0;  ///< \f$\|\mathrm{Log}(R_{est}^\top R_{true})\|\f$ at the end [rad]
  double max_attitude = 0.0;    ///< Worst attitude error over the run [rad]
  double final_velocity = 0.0;  ///< \f$\|v_{true} - v_{est}\|\f$ at the end [m/s]
  int frames = 0;               ///< Frames the filter processed.
  int updates = 0;              ///< Frames that produced an EKF update.
};

/// Noise-free run without landmarks: no vision update ever happens, so the filter's
/// navigation state is pure IMU propagation from the static initialisation.
DeadReckoning deadReckon(double imu_rate, double duration) {
  TrialConfig cfg = calib::scenario("convergence");
  calib::applyOverride(&cfg, "noise_free=1");
  calib::applyOverride(&cfg, "landmarks=0");
  cfg.sim.duration = duration;
  cfg.sim.imu.rate = imu_rate;
  cfg.calib.imu_rate = imu_rate;

  CalibSimulator sim(cfg.sim, 1);
  ExtrinsicCalibrator cal(cfg.calib, sim.truthExtrinsic(0.0));
  DeadReckoning out;
  cal.setFrameCallback([&](const FrameReport& rep) {
    ++out.frames;
    if (rep.updated) ++out.updates;
  });
  streamRun(sim, cal, [&](int) {
    if (!cal.initialised()) return;
    const KinematicState truth = sim.truthState(cal.time());
    out.max_attitude =
        std::max(out.max_attitude, math::boxMinus(truth.quaternion, cal.attitude()).norm());
  });
  const KinematicState truth = sim.truthState(cal.time());
  out.end_time = cal.time();
  out.final_attitude = math::boxMinus(truth.quaternion, cal.attitude()).norm();
  out.final_velocity = (truth.velocity - cal.velocity()).norm();
  return out;
}

/// Feed IMU samples until the static initialisation has run; returns the averaged window.
std::vector<sensors::ImuSample> runStaticInit(CalibSimulator& sim, ExtrinsicCalibrator& cal) {
  std::vector<sensors::ImuSample> window;
  for (int i = 0; i < sim.imuCount() && !cal.initialised(); ++i) {
    window.push_back(sim.imu(i));
    cal.addImu(window.back());
  }
  return window;
}

/// Largest per-axis sample standard deviation of the gyro (or accelerometer) over a window.
double maxAxisStd(const std::vector<sensors::ImuSample>& window, bool gyro) {
  Vec3 mean = Vec3::Zero();
  for (const auto& s : window) mean += gyro ? s.gyro : s.accel;
  mean /= static_cast<double>(window.size());
  Vec3 var = Vec3::Zero();
  for (const auto& s : window) var += ((gyro ? s.gyro : s.accel) - mean).cwiseAbs2();
  var /= static_cast<double>(window.size() - 1);
  return var.cwiseSqrt().maxCoeff();
}

/// World "down" axis seen in the IMU frame, \f$R_{WI}^\top e_z\f$.
Vec3 downInBody(const Vec4& q_wi) { return math::quatToRotation(q_wi).transpose() * Vec3::UnitZ(); }

/// Landmark ids of the wrong correspondences in a frame, found by comparing every observation
/// with the noise-free projection of its landmark through the hidden truth.
///
/// Threshold 5 px: an inlier's pixels carry 0.5 px noise per coordinate (5 px = 10 sigma;
/// the largest inlier deviation in the whole outlier run is 2.5 px), while an outlier is a
/// uniformly random pixel, within 5 px of the truth with probability
/// pi * 25 / (640 * 480) ~ 3e-4 (and harmless if it is).
std::set<int> corruptedIds(const CalibSimulator& sim, const calib::StereoFrame& frame,
                           double capture_time) {
  const KinematicState s = sim.truthState(capture_time);
  const Extrinsic e = sim.truthExtrinsic(capture_time);
  const Mat3 R_wi = math::quatToRotation(s.quaternion);
  const Mat3 R_wc = R_wi * e.rotation();
  const Vec3 p_wc = s.position + R_wi * e.p_ic;
  const calib::StereoIntrinsics& cam = sim.config().camera;
  std::set<int> bad;
  for (const auto& o : frame.observations) {
    const Vec3 P = R_wc.transpose() * (sim.landmarks()[static_cast<std::size_t>(o.id)] - p_wc);
    const double dev = std::max((o.left - cam.project(P)).norm(),
                                (o.right - cam.project(cam.leftToRight(P))).norm());
    if (dev > 5.0) bad.insert(o.id);
  }
  return bad;
}

}  // namespace

// -------------------------------------------------------------------------------------------
// IMU propagation
// -------------------------------------------------------------------------------------------

TEST_CASE("IMU-only propagation tracks the noise-free truth with a second-order error",
          "[calib][propagation]") {
  // Regression for a fixed bug: propagate() took its first sample by const reference to the
  // member segment_start_, which it overwrites before reading it, so every step integrated the
  // END sample over the whole interval. That is a rectangle rule, i.e. an effective dt/2 =
  // 2.5 ms time shift and |omega| dt/2 ~ 2 mrad of attitude error on this trajectory. (A
  // scratch mutant re-creating the aliasing fails every check below: 3.3 mrad worst attitude,
  // 1.75e-3 m/s, error ratios 2.0 and 2.2.)
  //
  // With noise-free data, zero biases and no landmarks, nothing but the integration scheme
  // separates the filter from the truth (static init is exact and yaw 0 is the simulator's
  // world frame), so its truncation error is all that remains: measured 1.9e-5 rad worst case
  // and 9.0e-4 m/s after 20 s at 200 Hz. The velocity error grows linearly because a mean
  // tilt error of ~5 urad (the integrator's coning error) leaks gravity: 9.81 * 5e-6 * 18.5 s
  // ~ 9e-4 m/s.
  const DeadReckoning r200 = deadReckon(200.0, 20.0);
  REQUIRE(r200.end_time == Approx(20.0));
  REQUIRE(r200.frames > 300);
  REQUIRE(r200.updates == 0);  // pure propagation, no vision update can mask an error

  INFO("200 Hz: final attitude " << r200.final_attitude << " rad, worst " << r200.max_attitude
                                 << " rad, velocity " << r200.final_velocity << " m/s");
  CHECK(r200.final_attitude < 1e-4);
  CHECK(r200.max_attitude < 1e-4);
  CHECK(r200.final_velocity < 1e-3);

  // Order of accuracy, independent of the absolute level: the trapezoidal scheme's global error
  // is O(dt^2), so halving dt divides it by 4 (the bug's rectangle rule is O(dt): ratio 2).
  // At 200 Hz |omega| dt ~ 4e-3, so higher-order terms move the ratio by well under 0.5.
  const DeadReckoning r400 = deadReckon(400.0, 20.0);
  REQUIRE(r400.updates == 0);
  const double attitude_ratio = r200.max_attitude / r400.max_attitude;
  const double velocity_ratio = r200.final_velocity / r400.final_velocity;
  INFO("error ratio 200 Hz / 400 Hz: attitude " << attitude_ratio << ", velocity "
                                                << velocity_ratio);
  CHECK(attitude_ratio == Approx(4.0).margin(0.5));
  CHECK(velocity_ratio == Approx(4.0).margin(0.5));
}

// -------------------------------------------------------------------------------------------
// Fixed point and convergence
// -------------------------------------------------------------------------------------------

TEST_CASE("with noise-free data and a perfect prior the estimate stays at the truth",
          "[calib]") {
  TrialConfig cfg = calib::scenario("convergence");
  calib::applyOverride(&cfg, "noise_free=1");
  calib::applyOverride(&cfg, "prior_rot_deg=0");
  calib::applyOverride(&cfg, "prior_trans_m=0");
  const TrialResult r = calib::runTrial(cfg, 3);

  REQUIRE(r.ok);
  REQUIRE(r.static_init_ok);
  REQUIRE(r.frames_fused > 1000);
  // The truth is a fixed point up to the integrator's O(dt^2) truncation error (~1e-5 rad of
  // pose error): measured 1.4e-4 deg and 0.09 mm at the end, two orders below the bounds.
  CHECK(r.final_rotation_error_deg < 0.01);
  CHECK(r.final_translation_error < 5e-4);
  // ... and it never wanders away on the way. The worst moment (measured 0.004 deg, 0.2 mm) is
  // just after the motion starts, when the barely excited extrinsic still has its 5 deg prior
  // sigma and absorbs a large share of the ~1e-6 rad propagation error; it then decays.
  double worst_rotation = 0.0, worst_translation = 0.0;
  for (const auto& s : r.series) {
    worst_rotation = std::max(worst_rotation, s.rotation_error_deg);
    worst_translation = std::max(worst_translation, s.translation_error);
  }
  REQUIRE(r.series.size() > 100);
  CHECK(worst_rotation < 0.01);
  CHECK(worst_translation < 5e-4);
  CHECK(r.converged);
  CHECK(r.convergence_time == 0.0);
  // At the truth with noise-free pixels the residuals are only the propagation error
  // (<= 1e-5 rad * 400 px = 4e-3 px) against an assumed sigma of 0.5 px, so NIS/dof is at most
  // (4e-3 / 0.5)^2 ~ 6e-5 (measured 1e-8). A model/simulator mismatch would show up here.
  CHECK(r.mean_nis_per_dof < 0.01);
}

TEST_CASE("the convergence scenario reaches 0.5 deg / 1 cm with a consistent filter",
          "[calib][convergence]") {
  const int seed = GENERATE(1, 2, 3);
  const TrialResult r =
      calib::runTrial(calib::scenario("convergence"), static_cast<std::uint64_t>(seed));
  INFO("seed " << seed << ": rotation " << r.final_rotation_error_deg << " deg, translation "
               << r.final_translation_error << " m, NIS/dof " << r.mean_nis_per_dof
               << ", converged at " << r.convergence_time << " s");
  REQUIRE(r.ok);
  CHECK(r.static_init_ok);
  CHECK(r.converged);
  CHECK(r.final_rotation_error_deg < 0.5);
  CHECK(r.final_translation_error < 0.01);
  CHECK(r.convergence_time >= 0.0);
  CHECK(r.convergence_time < 60.0);
  CHECK(r.frames_fused > 1000);  // ~1170 frames after the 1.5 s static window
  // E[NIS/dof] = 1 for a consistent filter. The mean is over ~1170 frames of >= 40 dof each,
  // so its sampling std is < 0.01: the [0.7, 1.3] band only admits model errors (a pixel sigma
  // mis-scaled by 20 % already moves it to 0.69 or 1.44).
  CHECK(r.mean_nis_per_dof >= 0.7);
  CHECK(r.mean_nis_per_dof <= 1.3);
}

TEST_CASE("a trial is reproducible bit for bit from its seed", "[calib][montecarlo]") {
  TrialConfig cfg = calib::scenario("convergence");
  calib::applyOverride(&cfg, "duration=10");
  const TrialResult a = calib::runTrial(cfg, 77);
  const TrialResult b = calib::runTrial(cfg, 77);
  const TrialResult c = calib::runTrial(cfg, 78);
  REQUIRE(a.ok);
  REQUIRE(a.series.size() == b.series.size());
  CHECK(a.final_rotation_error_deg == b.final_rotation_error_deg);
  CHECK(a.final_translation_error == b.final_translation_error);
  CHECK(a.mean_nis_per_dof == b.mean_nis_per_dof);
  for (std::size_t i = 0; i < a.series.size(); ++i) {
    INFO("sample " << i);
    CHECK(a.series[i].rotation_error_deg == b.series[i].rotation_error_deg);
    CHECK(a.series[i].translation_error == b.series[i].translation_error);
  }
  CHECK(c.final_rotation_error_deg != a.final_rotation_error_deg);
}

// -------------------------------------------------------------------------------------------
// Covariance health
// -------------------------------------------------------------------------------------------

TEST_CASE("the error covariance stays symmetric and positive semi-definite after every frame",
          "[calib][covariance]") {
  TrialConfig cfg = calib::scenario("convergence");
  cfg.sim.duration = 15.0;
  CalibSimulator sim(cfg.sim, 11);
  ExtrinsicCalibrator cal(cfg.calib, offsetPrior(sim.truthExtrinsic(0.0), 5.0 * kDeg, 0.05));

  int frames = 0, updates = 0, non_finite = 0;
  double worst_asymmetry = 0.0;  // relative to the largest entry
  double worst_eigenvalue = std::numeric_limits<double>::infinity();
  double worst_eigenvalue_time = -1.0;
  cal.setFrameCallback([&](const FrameReport& rep) {
    const MatX& P = cal.covariance();
    ++frames;
    if (rep.updated) ++updates;
    if (!P.allFinite()) {
      ++non_finite;
      return;
    }
    const double scale = P.cwiseAbs().maxCoeff();
    worst_asymmetry = std::max(worst_asymmetry, (P - P.transpose()).cwiseAbs().maxCoeff() / scale);
    const Eigen::SelfAdjointEigenSolver<MatX> es(P, Eigen::EigenvaluesOnly);
    if (es.eigenvalues().minCoeff() < worst_eigenvalue) {
      worst_eigenvalue = es.eigenvalues().minCoeff();
      worst_eigenvalue_time = rep.time;
    }
  });
  streamRun(sim, cal);

  REQUIRE(frames == 271);           // frames at 1.5, 1.55, ..., 15.0 s
  REQUIRE(updates > frames * 9 / 10);  // the update path was exercised on most frames
  CHECK(non_finite == 0);
  // symmetrize() after propagation and after cloning makes P exactly symmetric; allow
  // rounding only.
  CHECK(worst_asymmetry <= 1e-14);
  // P is singular by construction (the clone duplicates the current pose, the time offset is
  // not estimated), so its smallest eigenvalues are 0 up to the eigen-solver's rounding,
  // eps * ||P|| ~ 2e-16 * 0.02 ~ 5e-18 here. -1e-12 still catches any real loss of
  // definiteness, which would show up at the scale of the smallest variances (~1e-7).
  INFO("most negative eigenvalue " << worst_eigenvalue << " at t = " << worst_eigenvalue_time);
  CHECK(worst_eigenvalue >= -1e-12);

  // Not estimated: the time offset keeps zero variance and its prior value.
  const MatX& P = cal.covariance();
  CHECK(P(ExtrinsicCalibrator::kIdxTimeOffset, ExtrinsicCalibrator::kIdxTimeOffset) == 0.0);
  CHECK(cal.timeOffset() == 0.0);
  // Information was actually gained on the extrinsic (prior: 5 deg / 5 cm per axis).
  const Eigen::Matrix<double, 6, 1> sigma = cal.extrinsicCovariance().diagonal().cwiseSqrt();
  CHECK(sigma.head<3>().maxCoeff() < 0.2 * kDeg);
  CHECK(sigma.tail<3>().maxCoeff() < 0.01);
}

// -------------------------------------------------------------------------------------------
// Static initialisation
// -------------------------------------------------------------------------------------------

TEST_CASE("static initialisation levels the filter from gravity and fixes yaw to zero",
          "[calib][init]") {
  TrialConfig cfg = calib::scenario("convergence");
  // A tilted, yawed start so roll and pitch are actually exercised.
  cfg.sim.trajectory.base_euler = Vec3(0.12, -0.08, 0.6);
  REQUIRE(cfg.sim.trajectory.static_duration >= cfg.calib.static_init_duration);

  SECTION("noise-free data: roll, pitch and biases are exact") {
    calib::applyOverride(&cfg, "noise_free=1");
    CalibSimulator sim(cfg.sim, 5);
    ExtrinsicCalibrator cal(cfg.calib, sim.truthExtrinsic(0.0));
    const auto window = runStaticInit(sim, cal);

    REQUIRE(cal.initialised());
    CHECK(window.size() == 301);  // 1.5 s at 200 Hz, both ends included
    CHECK(cal.time() == Approx(1.5));
    CHECK(cal.staticInitPassed());
    const Vec3 est = math::quatToEuler(cal.attitude());
    const Vec3 tru = sim.truthState(cal.time()).euler;
    // Only rounding separates them (asin/atan2 of an exact gravity vector).
    CHECK(est(0) == Approx(tru(0)).margin(1e-9));
    CHECK(est(1) == Approx(tru(1)).margin(1e-9));
    CHECK(est(2) == Approx(0.0).margin(1e-12));  // yaw is the estimator's gauge, not 0.6
    CHECK(cal.gyroBias().norm() < 1e-12);
    CHECK(cal.accelBias().norm() == 0.0);
    CHECK(cal.velocity().norm() == 0.0);
  }

  SECTION("default noise: the tilt error is what the accelerometer bias explains") {
    CalibSimulator sim(cfg.sim, 5);
    ExtrinsicCalibrator cal(cfg.calib, sim.truthExtrinsic(0.0));
    const auto window = runStaticInit(sim, cal);
    REQUIRE(cal.initialised());
    REQUIRE(maxAxisStd(window, true) < cfg.calib.static_gyro_std_max);
    REQUIRE(maxAxisStd(window, false) < cfg.calib.static_accel_std_max);
    CHECK(cal.staticInitPassed());

    const Vec3 est = math::quatToEuler(cal.attitude());
    const Vec3 tru = sim.truthState(cal.time()).euler;
    CHECK(std::abs(est(0) - tru(0)) < 1.0 * kDeg);
    CHECK(std::abs(est(1) - tru(1)) < 1.0 * kDeg);
    CHECK(est(2) == Approx(0.0).margin(1e-12));

    // The levelling uses the mean specific force, so an accelerometer bias b tilts it by at most
    // atan(|b| / (g - |b|)). Margin 0.02 m/s^2 (~4 sigma) covers the window mean of the white
    // noise (0.05 / sqrt(301) = 2.9e-3 per axis) and the bias walk over 1.5 s (3.7e-3).
    const double ba = sim.truthAccelBias().norm();
    const double tilt_bound = std::atan((ba + 0.02) / (constants::kGravity - ba - 0.02));
    const Vec3 d_est = downInBody(cal.attitude());
    const Vec3 d_tru = downInBody(sim.truthState(cal.time()).quaternion);
    const double tilt = std::atan2(d_est.cross(d_tru).norm(), d_est.dot(d_tru));
    INFO("tilt error " << tilt << " rad, bound " << tilt_bound << " rad (|b_a| = " << ba << ")");
    CHECK(tilt <= tilt_bound);

    // Gyro bias = window mean: sigma ~ sqrt(2.0e-4^2 (white noise) + 2.1e-4^2 (walk)) = 3e-4
    // per axis; 1e-3 is > 3 sigma.
    CHECK((cal.gyroBias() - sim.truthGyroBias()).cwiseAbs().maxCoeff() < 1e-3);
  }
}

TEST_CASE("a start-up window that contains motion fails the stillness test",
          "[calib][init]") {
  TrialConfig cfg = calib::scenario("convergence");
  // The motion ramps in at 0.5 s, inside the 1.5 s averaging window.
  cfg.sim.trajectory.static_duration = 0.5;
  REQUIRE(cfg.sim.trajectory.static_duration < cfg.calib.static_init_duration);
  CalibSimulator sim(cfg.sim, 5);
  ExtrinsicCalibrator cal(cfg.calib, sim.truthExtrinsic(0.0));
  const auto window = runStaticInit(sim, cal);

  // Precondition: the window really violates the stillness threshold (measured 0.08 rad/s
  // against 0.02 rad/s), so the outcome below is the test, not an accident of the trajectory.
  const double gyro_std = maxAxisStd(window, true);
  INFO("gyro std over the window " << gyro_std << " rad/s");
  REQUIRE(gyro_std > 2.0 * cfg.calib.static_gyro_std_max);

  CHECK_FALSE(cal.staticInitPassed());
  CHECK(cal.initialised());  // the filter still starts (and reports the failed test)
  CHECK(cal.time() == Approx(cfg.calib.static_init_duration));
}

// -------------------------------------------------------------------------------------------
// Observability
// -------------------------------------------------------------------------------------------

TEST_CASE("yaw-only motion leaves the lever arm along the rotation axis unobservable",
          "[calib][observability]") {
  // With roll and pitch amplitudes at 0 the IMU only rotates about its own z axis (R_WI =
  // Rz(psi)). The camera centre is p_WI + R_WI p_IC, so the z component of the lever arm only
  // adds a constant world-z offset to every camera position, indistinguishable from the
  // (unobservable) absolute position: no measurement constrains it.
  //
  // Note on frames: p_IC and its error state are expressed in the IMU frame (P^I = R_IC P^C +
  // p_IC and the update adds dx straight onto p_IC), so the unobservable lever-arm direction is
  // e_z of the IMU itself. R_IC^T e_z would be the right mapping for the camera-frame rotation
  // error dphi_IC, not for dp_IC (projecting on it gives only ~2x between the runs).
  struct Outcome {
    double sigma_axis = 0.0;  ///< Lever-arm 1-sigma along IMU z [m]
    Vec3 major_axis = Vec3::Zero();  ///< Direction of the largest lever-arm uncertainty
    double rotation_error_deg = 0.0;
  };
  auto run = [](const char* name) {
    const TrialConfig cfg = calib::scenario(name);
    CalibSimulator sim(cfg.sim, 1);
    ExtrinsicCalibrator cal(cfg.calib, offsetPrior(sim.truthExtrinsic(0.0), 5.0 * kDeg, 0.05));
    streamRun(sim, cal);
    const Mat3 Pt = cal.extrinsicCovariance().bottomRightCorner<3, 3>();
    const Eigen::SelfAdjointEigenSolver<Mat3> es(Pt);
    Outcome o;
    o.sigma_axis = std::sqrt(Vec3::UnitZ().dot(Pt * Vec3::UnitZ()));
    o.major_axis = es.eigenvectors().col(2);
    o.rotation_error_deg = calib::rotationErrorDeg(cal.extrinsic(), sim.truthExtrinsic(cal.time()));
    return o;
  };

  const TrialConfig yaw_cfg = calib::scenario("yaw_only");
  REQUIRE(yaw_cfg.sim.trajectory.euler_amplitude.head<2>().norm() == 0.0);
  const Outcome full = run("convergence");
  const Outcome yaw = run("yaw_only");
  INFO("lever-arm sigma along IMU z: full motion " << full.sigma_axis << " m, yaw only "
                                                   << yaw.sigma_axis << " m");
  // Measured ~52x; >= 5x is the observability claim.
  CHECK(yaw.sigma_axis >= 5.0 * full.sigma_axis);
  // Essentially nothing was learned along the axis: the prior sigma is 5 cm (the random walk
  // adds only 1e-5 * sqrt(60) = 8e-5 m), linearisation leaks a little information.
  CHECK(yaw.sigma_axis > 0.8 * yaw_cfg.calib.init_extrinsic_translation_sigma);
  // And that axis is the filter's least certain lever-arm direction.
  CHECK(std::abs(yaw.major_axis.dot(Vec3::UnitZ())) > 0.99);
  // With full excitation the lever arm is pinned to the millimetre level along the same axis.
  CHECK(full.sigma_axis < 0.003);
}

// -------------------------------------------------------------------------------------------
// Time offset
// -------------------------------------------------------------------------------------------

TEST_CASE("an 8 ms camera-IMU time offset is estimated to better than 1 ms within 60 s",
          "[calib][time_offset]") {
  const TrialConfig cfg = calib::scenario("time_offset");
  REQUIRE(cfg.calib.estimate_time_offset);
  REQUIRE(cfg.sim.time_offset == Approx(0.008));
  REQUIRE(cfg.calib.init_time_offset == 0.0);  // starts 8 ms off
  const TrialResult r = calib::runTrial(cfg, 1);
  INFO("time-offset error " << r.final_time_offset_error * 1e3 << " ms");
  REQUIRE(r.ok);
  CHECK(r.final_time_offset_error < 1e-3);
  CHECK(r.converged);
  CHECK(r.final_rotation_error_deg < 0.5);
  CHECK(r.final_translation_error < 0.01);
}

// -------------------------------------------------------------------------------------------
// Change detection
// -------------------------------------------------------------------------------------------

TEST_CASE("the change detector catches a knock and the estimate recovers quickly",
          "[calib][change][slow]") {
  struct Outcome {
    std::vector<double> event_times;
    double error_10s_after_deg = -1.0;  ///< Rotation error 10 s after the knock [deg]
    double final_rotation_error_deg = 0.0;
    double final_translation_error = 0.0;
    int change_events = 0;
  };
  auto run = [](const char* name) {
    const TrialConfig cfg = calib::scenario(name);
    const double step_time = cfg.sim.drift.step_time;
    CalibSimulator sim(cfg.sim, 3);
    ExtrinsicCalibrator cal(cfg.calib, offsetPrior(sim.truthExtrinsic(0.0), 5.0 * kDeg, 0.05));
    Outcome o;
    cal.setFrameCallback([&](const FrameReport& rep) {
      if (rep.change_detected) o.event_times.push_back(rep.time);
      if (o.error_10s_after_deg < 0.0 && rep.time >= step_time + 10.0)
        o.error_10s_after_deg =
            calib::rotationErrorDeg(cal.extrinsic(), sim.truthExtrinsic(rep.time));
    });
    streamRun(sim, cal);
    const Extrinsic truth = sim.truthExtrinsic(cal.time());
    o.final_rotation_error_deg = calib::rotationErrorDeg(cal.extrinsic(), truth);
    o.final_translation_error = calib::translationError(cal.extrinsic(), truth);
    o.change_events = cal.changeEvents();
    return o;
  };

  const TrialConfig cfg = calib::scenario("knock");
  REQUIRE(cfg.calib.detect_change);
  const double step_time = cfg.sim.drift.step_time;
  REQUIRE(step_time == Approx(90.0));

  const Outcome with = run("knock");
  INFO("events: " << with.change_events << ", first at "
                  << (with.event_times.empty() ? -1.0 : with.event_times.front()) << " s");
  REQUIRE(with.change_events >= 1);
  REQUIRE(static_cast<int>(with.event_times.size()) == with.change_events);
  // No false alarm in the ~85 s of tests before the knock: under no change T ~ chi^2_6
  // (threshold 40, tail ~ 3e-7 per test; the largest T before the knock is ~7); the knock is
  // caught within one test window (40 updates = 2 s; measured 0.2 s).
  CHECK(with.event_times.front() >= step_time);
  CHECK(with.event_times.front() <= step_time + 2.0);
  CHECK(with.error_10s_after_deg < 0.1);
  CHECK(with.final_rotation_error_deg < 0.1);
  CHECK(with.final_translation_error < 0.01);

  // Ablation: without the detector the filter can only follow the 1.5 deg step through its
  // small extrinsic random walk, so 10 s later most of the step is still there.
  const Outcome without = run("knock_no_detector");
  CHECK(without.change_events == 0);
  CHECK(without.event_times.empty());
  INFO("10 s after the knock: " << with.error_10s_after_deg << " deg with the detector, "
                                << without.error_10s_after_deg << " deg without");
  CHECK(without.error_10s_after_deg > 0.3);
  CHECK(without.error_10s_after_deg > 5.0 * with.error_10s_after_deg);
}

// -------------------------------------------------------------------------------------------
// Outliers
// -------------------------------------------------------------------------------------------

TEST_CASE("5 % wrong correspondences are rejected and the calibration still converges",
          "[calib][outliers]") {
  const TrialConfig cfg = calib::scenario("outliers");
  REQUIRE(cfg.sim.outlier_fraction == Approx(0.05));
  REQUIRE(cfg.sim.time_offset == 0.0);  // camera stamps are exposure times below
  CalibSimulator sim(cfg.sim, 2);
  ExtrinsicCalibrator cal(cfg.calib, offsetPrior(sim.truthExtrinsic(0.0), 5.0 * kDeg, 0.05));

  long matched = 0, used = 0, gated = 0;
  double nis_sum = 0.0;
  int updates = 0;
  cal.setFrameCallback([&](const FrameReport& rep) {
    matched += rep.matched;
    used += rep.used;
    gated += rep.gated;
    if (rep.updated) {
      nis_sum += rep.nis_per_dof;
      ++updates;
    }
  });
  // Ground truth of the corruption, per frame: (exposure time, landmark ids, corrupted ids).
  struct FrameTruth {
    double time;
    std::vector<int> ids;
    std::set<int> corrupted;
  };
  std::vector<FrameTruth> frames;
  streamRun(sim, cal, NoOp{}, [&](int k, const calib::StereoFrame& frame) {
    FrameTruth f{frame.stamp, {}, corruptedIds(sim, frame, sim.captureTime(k))};
    for (const auto& o : frame.observations) f.ids.push_back(o.id);
    frames.push_back(std::move(f));
  });

  // Rebuild the filter's landmark pairs: consecutive frames from the first one fused (at the end
  // of the static window). A pair is corrupted when either of its two observations is.
  long corrupted_obs = 0, observations = 0;
  for (const auto& f : frames) {
    observations += static_cast<long>(f.ids.size());
    corrupted_obs += static_cast<long>(f.corrupted.size());
  }
  long pairs = 0, corrupted_pairs = 0;
  for (std::size_t k = 1; k < frames.size(); ++k) {
    if (frames[k - 1].time < cfg.calib.static_init_duration - 1e-9) continue;
    const std::set<int> previous(frames[k - 1].ids.begin(), frames[k - 1].ids.end());
    for (int id : frames[k].ids) {
      if (previous.count(id) == 0) continue;
      ++pairs;
      if (frames[k].corrupted.count(id) != 0 || frames[k - 1].corrupted.count(id) != 0)
        ++corrupted_pairs;
    }
  }
  const long clean_pairs = pairs - corrupted_pairs;
  INFO("pairs " << pairs << " (filter matched " << matched << "), corrupted " << corrupted_pairs
                << ", used " << used << ", gated " << gated << ", dropped before the gate "
                << matched - used - gated);
  REQUIRE(pairs == matched);  // the reconstruction pairs exactly what the filter paired
  // Precondition on the data: 5 % of observations are corrupted, so 1 - 0.95^2 = 9.75 % of the
  // pairs (binomial std over 1.2e5 pairs ~ 1e-3; +-0.005 is ~4 sigma).
  REQUIRE(static_cast<double>(corrupted_obs) / static_cast<double>(observations) ==
          Approx(0.05).margin(0.005));
  REQUIRE(static_cast<double>(corrupted_pairs) / static_cast<double>(pairs) ==
          Approx(0.0975).margin(0.005));

  // Every corrupted pair is rejected (by the chi-squared gate, or earlier when the landmark
  // cannot be triangulated in front of both cameras), so no more landmarks are fused than there
  // are clean pairs: a leaked outlier would have to be offset by a clean pair rejected by chance.
  CHECK(used <= clean_pairs);
  // ... and the clean pairs are kept: the 99.9 % gate rejects ~0.1 % of them (measured 0.09 %).
  CHECK(static_cast<double>(used) >= 0.995 * static_cast<double>(clean_pairs));
  CHECK(gated > 0);

  const Extrinsic truth = sim.truthExtrinsic(cal.time());
  const double rotation_deg = calib::rotationErrorDeg(cal.extrinsic(), truth);
  const double translation = calib::translationError(cal.extrinsic(), truth);
  const double mean_nis = nis_sum / updates;
  INFO("rotation " << rotation_deg << " deg, translation " << translation << " m, NIS/dof "
                   << mean_nis);
  REQUIRE(updates > 1000);
  CHECK(rotation_deg < 0.5);
  CHECK(translation < 0.01);
  // With the outliers rejected, the inliers keep the filter consistent (see the convergence
  // case for the band).
  CHECK(mean_nis >= 0.7);
  CHECK(mean_nis <= 1.3);
}

TEST_CASE("without the landmark gate the same outliers wreck the calibration",
          "[calib][outliers]") {
  // Ablation of the case above: same scenario and seed, gate disabled. A single wrong
  // correspondence is a residual of hundreds of pixels against a 0.5 px sigma, so one per frame
  // is enough to drag the filter away (measured 10-49 deg, NIS/dof ~ 2000 on seeds 1-3).
  TrialConfig cfg = calib::scenario("outliers");
  calib::applyOverride(&cfg, "landmark_gate=0");
  const TrialResult r = calib::runTrial(cfg, 2);
  INFO("rotation " << r.final_rotation_error_deg << " deg, translation "
                   << r.final_translation_error << " m, NIS/dof " << r.mean_nis_per_dof);
  CHECK_FALSE(r.converged);
  CHECK(!(r.final_rotation_error_deg < 5.0));  // also true for a NaN estimate
  CHECK(!(r.mean_nis_per_dof < 10.0));
}
