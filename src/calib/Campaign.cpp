#include "aether/calib/Campaign.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "aether/math/Rotation.hpp"
#include "aether/util/Random.hpp"

namespace aether::calib {

namespace {

constexpr std::uint64_t kStreamPrior = 31;

Vec3 randomUnitVector(util::Rng& rng) {
  for (;;) {
    const Vec3 v = rng.gaussian3();
    const double n = v.norm();
    if (n > 1e-6) return v / n;
  }
}

/// Per-frame scoring record (kept for every fused frame).
struct FrameScore {
  double time;
  double rot_deg;
  double trans;
};

}  // namespace

TrialResult runTrial(const TrialConfig& config, std::uint64_t seed, int index) {
  const auto wall_start = std::chrono::steady_clock::now();
  TrialResult result;
  result.index = index;
  result.seed = seed;

  CalibSimulator sim(config.sim, seed);

  // --- Prior: truth perturbed by exactly the configured magnitudes ---------------------
  util::Rng rng_prior(util::deriveSeed(seed, kStreamPrior, 0));
  const Extrinsic truth0 = sim.truthExtrinsic(0.0);
  Extrinsic prior;
  prior.q_ic = math::boxPlus(truth0.q_ic, config.prior_rotation_error * randomUnitVector(rng_prior));
  prior.p_ic = truth0.p_ic + config.prior_translation_error * randomUnitVector(rng_prior);

  CalibratorConfig calib_config = config.calib;
  calib_config.max_frame_latency =
      std::max(calib_config.max_frame_latency, config.frame_delivery_latency + 0.05);
  ExtrinsicCalibrator calibrator(calib_config, prior);

  std::vector<FrameScore> scores;
  scores.reserve(static_cast<std::size_t>(sim.frameCount()));
  double next_series = 0.0;
  double nees_sum = 0.0, nis_sum = 0.0, used_sum = 0.0;
  int nees_count = 0, nis_count = 0;
  double track_sq = 0.0, track_max = 0.0, track_max_t = 0.0;
  int track_count = 0;
  bool horizon_recorded = false;

  calibrator.setFrameCallback([&](const FrameReport& rep) {
    const Extrinsic est = calibrator.extrinsic();
    const Extrinsic tru = sim.truthExtrinsic(rep.time);
    const Vec3 drot = rotationErrorVector(est, tru);
    const Vec3 dpos = tru.p_ic - est.p_ic;
    const double rot_deg = drot.norm() * constants::kRadToDeg;
    const double trans = dpos.norm();
    scores.push_back({rep.time, rot_deg, trans});
    if (rep.change_detected) result.change_times.push_back(rep.time);
    if (rep.updated) {
      ++result.frames_fused;
      nis_sum += rep.nis_per_dof;
      used_sum += rep.used;
      ++nis_count;
    }

    Eigen::Matrix<double, 6, 1> e;
    e << drot, dpos;
    const Eigen::Matrix<double, 6, 6> Pe = calibrator.extrinsicCovariance();
    const double nees = e.dot(Pe.ldlt().solve(e));
    if (rep.time >= 10.0 && std::isfinite(nees)) {
      nees_sum += nees;
      ++nees_count;
    }

    // Errors "at the horizon" are those of the last frame fused at or before it. (With a
    // camera-IMU time offset the last exposure can fall a few ms short of the horizon.)
    if (rep.time <= config.convergence_horizon + 1e-9) {
      result.horizon_rotation_error_deg = rot_deg;
      result.horizon_translation_error = trans;
      horizon_recorded = true;
    }
    if (config.tracking_start >= 0.0 && rep.time >= config.tracking_start) {
      track_sq += rot_deg * rot_deg;
      track_max = std::max(track_max, rot_deg);
      track_max_t = std::max(track_max_t, trans);
      ++track_count;
    }

    if (config.series_interval > 0.0 && rep.time + 1e-9 >= next_series) {
      next_series += config.series_interval;
      SeriesSample s;
      s.time = rep.time;
      s.rotation_error_deg = rot_deg;
      s.translation_error = trans;
      s.rotation_error = drot;
      s.translation_error_vec = dpos;
      s.rotation_sigma = Pe.diagonal().head<3>().cwiseMax(0.0).cwiseSqrt();
      s.translation_sigma = Pe.diagonal().tail<3>().cwiseMax(0.0).cwiseSqrt();
      s.nees = nees;
      s.truth_drift_deg = rotationErrorDeg(tru, truth0);
      s.estimate_drift_deg = rotationErrorDeg(est, truth0);
      s.time_offset_truth = config.sim.time_offset;
      s.time_offset_estimate = calibrator.timeOffset();
      s.gyro_bias_error = (sim.truthGyroBias() - calibrator.gyroBias()).norm();
      s.accel_bias_error = (sim.truthAccelBias() - calibrator.accelBias()).norm();
      s.nis_per_dof = rep.nis_per_dof;
      s.landmarks_used = rep.used;
      const KinematicState truth_state = sim.truthState(rep.time);
      const MatX& P = calibrator.covariance();
      s.velocity_error = truth_state.velocity - calibrator.velocity();
      s.attitude_error = math::boxMinus(truth_state.quaternion, calibrator.attitude());
      const Mat3 Pv = P.block<3, 3>(ExtrinsicCalibrator::kIdxVelocity, ExtrinsicCalibrator::kIdxVelocity);
      const Mat3 Pa = P.block<3, 3>(ExtrinsicCalibrator::kIdxAttitude, ExtrinsicCalibrator::kIdxAttitude);
      s.velocity_nees = s.velocity_error.dot(Pv.ldlt().solve(s.velocity_error));
      s.attitude_nees = s.attitude_error.dot(Pa.ldlt().solve(s.attitude_error));
      result.series.push_back(s);
    }
  });

  // --- Stream the data in arrival order: IMU samples as they are measured, each stereo
  // frame `frame_delivery_latency` after its exposure ------------------------------------
  int k = 0;
  for (int i = 0; i < sim.imuCount(); ++i) {
    const double t = sim.imuTime(i);
    while (k < sim.frameCount() && sim.captureTime(k) + config.frame_delivery_latency <= t) {
      calibrator.addFrame(sim.frame(k));
      ++k;
    }
    calibrator.addImu(sim.imu(i));
  }
  while (k < sim.frameCount()) calibrator.addFrame(sim.frame(k++));  // still in flight at the end
  calibrator.flush();
  result.frames_dropped = calibrator.droppedFrames();

  // --- Scores --------------------------------------------------------------------------
  const Extrinsic est = calibrator.extrinsic();
  const Extrinsic tru = sim.truthExtrinsic(calibrator.time());
  result.final_rotation_error_deg = rotationErrorDeg(est, tru);
  result.final_translation_error = translationError(est, tru);
  result.truth_drift_total_deg = rotationErrorDeg(tru, truth0);
  result.final_time_offset_error = std::abs(config.sim.time_offset - calibrator.timeOffset());
  result.final_gyro_bias_error = (sim.truthGyroBias() - calibrator.gyroBias()).norm();
  result.final_accel_bias_error = (sim.truthAccelBias() - calibrator.accelBias()).norm();
  result.mean_nees = nees_count > 0 ? nees_sum / nees_count : 0.0;
  result.mean_nis_per_dof = nis_count > 0 ? nis_sum / nis_count : 0.0;
  result.mean_landmarks_used = nis_count > 0 ? used_sum / nis_count : 0.0;
  result.change_events = calibrator.changeEvents();
  result.static_init_ok = calibrator.staticInitPassed();
  // A run shorter than the horizon cannot certify convergence "within the horizon".
  const bool horizon_covered =
      horizon_recorded && config.sim.duration >= config.convergence_horizon - 1e-9;
  result.tracking_samples = track_count;
  if (track_count > 0) {
    result.tracking_rms_error_deg = std::sqrt(track_sq / track_count);
    result.tracking_max_error_deg = track_max;
    result.tracking_max_translation = track_max_t;
  }

  const double rth = config.rotation_threshold_deg;
  const double tth = config.translation_threshold;
  const auto good = [&](const FrameScore& s) { return s.rot_deg < rth && s.trans < tth; };
  // Start of the last unbroken run of below-threshold frames in [from, until]: the time of
  // the first frame after the last failing one (or `from` if none failed).
  const auto settledSince = [&](double from, double until) {
    double t = from;
    for (std::size_t i = 0; i < scores.size(); ++i) {
      if (scores[i].time < from - 1e-9) continue;
      if (scores[i].time > until + 1e-9) break;
      if (!good(scores[i])) t = i + 1 < scores.size() ? scores[i + 1].time : scores[i].time;
    }
    return t;
  };

  result.converged = horizon_covered && result.horizon_rotation_error_deg < rth &&
                     result.horizon_translation_error < tth;
  if (result.converged)
    result.convergence_time = settledSince(0.0, config.convergence_horizon);

  if (config.tracking_start >= 0.0 && track_count > 0 && result.final_rotation_error_deg < rth &&
      result.final_translation_error < tth) {
    result.recovery_time =
        settledSince(config.tracking_start, scores.back().time) - config.tracking_start;
  }

  result.ok = std::isfinite(result.final_rotation_error_deg) &&
              std::isfinite(result.final_translation_error) && !scores.empty();
  result.wall_clock_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return result;
}

std::vector<std::string> scenarioNames() {
  return {"convergence", "drift",    "drift_frozen", "knock",
          "knock_no_detector", "time_offset", "time_offset_ignored", "yaw_only", "outliers"};
}

TrialConfig scenario(const std::string& name) {
  TrialConfig c;
  c.name = name;
  c.calib.camera = c.sim.camera;
  c.calib.pixel_sigma = c.sim.pixel_noise;

  if (name == "convergence") {
    c.sim.duration = 60.0;
  } else if (name == "drift" || name == "drift_frozen") {
    c.sim.duration = 660.0;
    c.sim.drift.start_time = 60.0;
    c.tracking_start = 60.0;
    c.series_interval = 2.0;
    if (name == "drift_frozen") {
      // Calibrate once and trust it: no random walk and no change detector.
      c.calib.extrinsic_rotation_walk = 0.0;
      c.calib.extrinsic_translation_walk = 0.0;
      c.calib.detect_change = false;
    }
  } else if (name == "knock" || name == "knock_no_detector") {
    c.sim.duration = 180.0;
    c.sim.drift.step_time = 90.0;
    c.sim.drift.step_rotation = Vec3(1.0, -0.6, 0.5).normalized() * 1.5 * constants::kDegToRad;
    c.sim.drift.step_translation = Vec3(0.012, -0.010, 0.012);
    c.calib.detect_change = (name == "knock");  // on by default; the ablation turns it off
    c.tracking_start = 90.0;
  } else if (name == "time_offset") {
    c.sim.duration = 60.0;
    c.sim.time_offset = 0.008;
  } else if (name == "time_offset_ignored") {
    // Assume the clocks are synchronised when the camera actually lags by 2.5 ms.
    c.sim.duration = 60.0;
    c.sim.time_offset = 0.0025;
    c.calib.estimate_time_offset = false;
  } else if (name == "yaw_only") {
    c.sim.duration = 60.0;
    c.sim.trajectory.euler_amplitude = Vec3(0.0, 0.0, 0.45);
  } else if (name == "outliers") {
    c.sim.duration = 60.0;
    c.sim.outlier_fraction = 0.05;
  } else {
    throw std::invalid_argument("unknown scenario '" + name + "'");
  }
  return c;
}

void applyOverride(TrialConfig* c, const std::string& assignment) {
  const auto eq = assignment.find('=');
  if (eq == std::string::npos)
    throw std::invalid_argument("override '" + assignment + "' is not key=value");
  const std::string key = assignment.substr(0, eq);
  double v = 0.0;
  try {
    v = std::stod(assignment.substr(eq + 1));
  } catch (const std::exception&) {
    throw std::invalid_argument("override '" + assignment + "' has a non-numeric value");
  }
  if (!std::isfinite(v)) throw std::invalid_argument("override '" + assignment + "' is not finite");
  if (key == "duration") {
    c->sim.duration = v;
  } else if (key == "prior_rot_deg") {
    c->prior_rotation_error = v * constants::kDegToRad;
  } else if (key == "prior_trans_m") {
    c->prior_translation_error = v;
  } else if (key == "pixel_noise") {
    c->sim.pixel_noise = v;
  } else if (key == "pixel_sigma") {
    c->calib.pixel_sigma = v;
  } else if (key == "rot_walk") {
    c->calib.extrinsic_rotation_walk = v;
  } else if (key == "trans_walk") {
    c->calib.extrinsic_translation_walk = v;
  } else if (key == "landmarks") {
    if (!(v >= 0.0 && v <= 1e6)) throw std::invalid_argument("landmarks must be in [0, 1e6]");
    c->sim.scene.landmarks = static_cast<int>(v);
  } else if (key == "camera_rate") {
    c->sim.camera_rate = v;
  } else if (key == "time_offset") {
    c->sim.time_offset = v;
  } else if (key == "estimate_time_offset") {
    c->calib.estimate_time_offset = v != 0.0;
  } else if (key == "outlier_fraction") {
    c->sim.outlier_fraction = v;
  } else if (key == "landmark_gate") {
    c->calib.landmark_gate = v;
  } else if (key == "detect_change") {
    c->calib.detect_change = v != 0.0;
  } else if (key == "change_threshold") {
    c->calib.change_threshold = v;
  } else if (key == "knock_rot_deg") {
    c->sim.drift.step_rotation = c->sim.drift.step_rotation.normalized() * v * constants::kDegToRad;
  } else if (key == "knock_trans_m") {
    c->sim.drift.step_translation = c->sim.drift.step_translation.normalized() * v;
  } else if (key == "reopen_rot_deg") {
    c->calib.reopen_rotation_sigma = v * constants::kDegToRad;
  } else if (key == "reopen_trans_m") {
    c->calib.reopen_translation_sigma = v;
  } else if (key == "change_window") {
    if (!(v >= 1.0 && v <= 1e6)) throw std::invalid_argument("change_window must be in [1, 1e6]");
    c->calib.change_window = static_cast<int>(v);
  } else if (key == "drift_rate_deg_per_min") {
    c->sim.drift.rotation_rate = v * constants::kDegToRad / 60.0;
  } else if (key == "drift_start") {
    c->sim.drift.start_time = v;
  } else if (key == "tracking_start") {
    c->tracking_start = v;
  } else if (key == "roll_pitch_scale") {
    c->sim.trajectory.euler_amplitude.head<2>() *= v;
  } else if (key == "yaw_scale") {
    c->sim.trajectory.euler_amplitude(2) *= v;
  } else if (key == "gyro_noise_scale") {
    c->sim.imu.gyro_noise_density *= v;
    c->calib.gyro_noise *= v;
  } else if (key == "accel_noise_scale") {
    c->sim.imu.accel_noise_density *= v;
    c->calib.accel_noise *= v;
  } else if (key == "yaw_gauge_noise") {
    c->calib.yaw_gauge_noise = v;
  } else if (key == "disjoint_pairs") {
    c->calib.disjoint_pairs = v != 0.0;
  } else if (key == "init_attitude_deg") {
    c->calib.init_attitude_sigma = v * constants::kDegToRad;
  } else if (key == "init_velocity_sigma") {
    c->calib.init_velocity_sigma = v;
  } else if (key == "init_accel_bias_sigma") {
    c->calib.init_accel_bias_sigma = v;
  } else if (key == "accel_bias_initial") {
    c->sim.imu.accel_bias_initial = v;
  } else if (key == "gyro_bias_initial") {
    c->sim.imu.gyro_bias_initial = v;
  } else if (key == "noise_free") {
    if (v != 0.0) {
      c->sim.imu.gyro_noise_density = 0.0;
      c->sim.imu.accel_noise_density = 0.0;
      c->sim.imu.gyro_bias_initial = 0.0;
      c->sim.imu.accel_bias_initial = 0.0;
      c->sim.imu.gyro_bias_walk = 0.0;
      c->sim.imu.accel_bias_walk = 0.0;
      c->sim.pixel_noise = 0.0;
    }
  } else if (key == "frame_latency") {
    c->frame_delivery_latency = v;
  } else if (key == "series_interval") {
    c->series_interval = v;
  } else {
    throw std::invalid_argument("unknown override key '" + key + "'");
  }
}

}  // namespace aether::calib
