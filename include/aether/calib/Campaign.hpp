/// \file Campaign.hpp
/// \brief One calibration trial (simulate + estimate + score) and the named scenarios.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "aether/calib/CalibSimulator.hpp"
#include "aether/calib/ExtrinsicCalibrator.hpp"

namespace aether::calib {

/// \brief Everything that defines one trial apart from its seed.
struct TrialConfig {
  std::string name = "convergence";  ///< Scenario name (for reports).
  SimConfig sim;                     ///< Truth generator.
  CalibratorConfig calib;            ///< Estimator tuning.

  /// The prior handed to the estimator is the per-trial truth perturbed by exactly these
  /// magnitudes, in a random direction drawn per trial.
  double prior_rotation_error = 5.0 * constants::kDegToRad;  ///< [rad]
  double prior_translation_error = 0.05;                     ///< [m]

  double rotation_threshold_deg = 0.5;   ///< Convergence threshold on rotation [deg]
  double translation_threshold = 0.01;   ///< Convergence threshold on the lever arm [m]
  /// Convergence is judged on [0, convergence_horizon] (the "within 60 s" requirement).
  double convergence_horizon = 60.0;
  /// Drift-tracking error is scored on [tracking_start, end] (< 0 disables).
  double tracking_start = -1.0;

  double series_interval = 0.5;  ///< Sampling of the recorded time series [s] (<= 0: none)
  /// Each stereo frame is handed to the calibrator this long after its exposure, as a camera
  /// driver would deliver it [s]. The calibrator is run with `max_frame_latency` of at least
  /// this plus 50 ms, so it lags the newest IMU sample and still fuses every frame at its
  /// exposure time.
  double frame_delivery_latency = 0.03;
};

/// \brief One row of a recorded time series (estimate vs. hidden truth).
struct SeriesSample {
  double time = 0.0;              ///< IMU-clock time [s]
  double rotation_error_deg = 0.0;///< Total extrinsic rotation error [deg]
  double translation_error = 0.0; ///< Lever-arm error [m]
  Vec3 rotation_error = Vec3::Zero();    ///< Per-axis rotation error, camera frame [rad]
  Vec3 translation_error_vec = Vec3::Zero();  ///< Per-axis lever-arm error [m]
  Vec3 rotation_sigma = Vec3::Zero();    ///< Filter 1-sigma, rotation [rad]
  Vec3 translation_sigma = Vec3::Zero(); ///< Filter 1-sigma, lever arm [m]
  double nees = 0.0;              ///< Extrinsic NEES (6 dof)
  double truth_drift_deg = 0.0;   ///< Truth rotation relative to the initial truth mount [deg]
  double estimate_drift_deg = 0.0;///< Estimate rotation relative to the initial truth mount [deg]
  double time_offset_truth = 0.0; ///< [s]
  double time_offset_estimate = 0.0;  ///< [s]
  double gyro_bias_error = 0.0;   ///< \f$\|b_g - \hat b_g\|\f$ [rad/s]
  double accel_bias_error = 0.0;  ///< \f$\|b_a - \hat b_a\|\f$ [m/s^2]
  double nis_per_dof = 0.0;       ///< Vision update NIS per dof (last frame)
  int landmarks_used = 0;         ///< Landmarks fused at the last frame
  /// Navigation-state diagnostics. The estimator's world frame is fixed by the static
  /// initialisation (yaw 0, origin at the start point), which coincides with the simulator's
  /// world frame, so velocity and attitude errors are meaningful (up to slow yaw drift).
  Vec3 velocity_error = Vec3::Zero();   ///< Truth minus estimate, world [m/s]
  Vec3 attitude_error = Vec3::Zero();   ///< Log(R_est^T R_true), body frame [rad]
  double velocity_nees = 0.0;           ///< 3-dof NEES of the velocity
  double attitude_nees = 0.0;           ///< 3-dof NEES of the attitude
  double pixel_sigma = 0.0;             ///< Pixel noise the filter assumes (adapted) [px]
  CalibrationState state = CalibrationState::kInitialising;  ///< status().state
  double rotation_bound_deg = 0.0;      ///< status().rotation_bound [deg]
  double translation_bound = 0.0;       ///< status().translation_bound [m]
};

/// \brief Outcome and scores of one trial.
struct TrialResult {
  int index = 0;
  std::uint64_t seed = 0;
  bool ok = false;                      ///< Ran to completion with a finite estimate.
  std::string error;

  double final_rotation_error_deg = 0.0;   ///< At the end of the run [deg]
  double final_translation_error = 0.0;    ///< At the end of the run [m]
  double horizon_rotation_error_deg = 0.0; ///< At the convergence horizon [deg]
  double horizon_translation_error = 0.0;  ///< At the convergence horizon [m]
  /// Both errors below threshold at the horizon (false if the run ended before it).
  bool converged = false;
  /// Earliest time after which both errors stay below their thresholds up to the horizon
  /// (-1 when not converged).
  double convergence_time = -1.0;
  int tracking_samples = 0;              ///< Frames scored in the tracking window.
  double tracking_max_error_deg = 0.0;   ///< Max rotation error over the tracking window [deg]
  double tracking_rms_error_deg = 0.0;   ///< RMS rotation error over the tracking window [deg]
  double tracking_max_translation = 0.0; ///< Max lever-arm error over the window [m]
  double truth_drift_total_deg = 0.0;    ///< How far the truth moved in the run [deg]
  /// Time from tracking_start until both errors are below the convergence thresholds for
  /// good (-1 when they are not below them at the end) [s].
  double recovery_time = -1.0;
  double final_time_offset_error = 0.0;  ///< \f$|t_d - \hat t_d|\f$ at the end [s]
  double final_gyro_bias_error = 0.0;    ///< [rad/s]
  double final_accel_bias_error = 0.0;   ///< [m/s^2]
  double mean_nees = 0.0;          ///< Mean extrinsic NEES after the first 10 s (6 expected)
  double mean_nis_per_dof = 0.0;   ///< Mean vision NIS per dof (1 expected)
  double mean_landmarks_used = 0.0;
  int frames_fused = 0;
  int frames_dropped = 0;          ///< Frames that arrived too late to be fused (0 expected).
  int change_events = 0;
  std::vector<double> change_times;  ///< When the change detector fired [s]

  // Self-assessment (ExtrinsicCalibrator::status()) scored against the hidden truth.
  CalibrationStatus final_status;    ///< Status at the end of the run.
  bool status_converged_at_horizon = false;  ///< Status said "converged" at the horizon.
  /// Earliest time after which the status says "converged" up to the horizon (-1 if never).
  double status_convergence_time = -1.0;
  int status_converged_frames = 0;   ///< Frames on which the status said "converged".
  /// Frames on which the status said "converged" while the true error was outside the
  /// converged bounds (rotation or lever arm). This is the number that must stay at 0.
  int false_confidence_frames = 0;
  double final_pixel_sigma = 0.0;    ///< Pixel noise the filter ended up assuming [px]
  bool static_init_ok = false;
  double wall_clock_seconds = 0.0;
  std::vector<SeriesSample> series;
};

/// \brief Simulate, estimate and score one trial.
TrialResult runTrial(const TrialConfig& config, std::uint64_t seed, int index = 0);

/// \brief Named scenarios used by the apps and the README.
///
/// - `convergence`: 60 s, 5 deg / 5 cm prior error (success criterion 1).
/// - `drift`: converge for 60 s, then the truth rotates at 0.1 deg/min for 10 min
///   (success criterion 2).
/// - `drift_frozen`: same, with the extrinsic random walk switched off (calibrate-once baseline).
/// - `knock`: a 1.5 deg / 2 cm step at t = 90 s, with the change detector.
/// - `knock_no_detector`: same without the detector.
/// - `time_offset`: 8 ms camera-IMU offset (estimated online, as in every scenario).
/// - `time_offset_ignored`: 2.5 ms offset with time-offset estimation switched off.
/// - `noise_misspecified`: 1.0 px pixel noise while the filter is configured for 0.5 px
///   (the online noise estimate corrects it); `noise_misspecified_fixed`: same, adaptation off.
/// - `yaw_only`: roll and pitch excitation removed (observability ablation).
/// - `outliers`: 5 % wrong correspondences.
/// \throws std::invalid_argument on an unknown name.
TrialConfig scenario(const std::string& name);

/// Names accepted by scenario().
std::vector<std::string> scenarioNames();

/// \brief Override one parameter of a trial configuration from a `key=value` string.
///
/// Used by the apps' `--set` option for sweeps and ablations. Keys (units as in the structs):
/// `duration`, `prior_rot_deg`, `prior_trans_m`, `pixel_noise` (truth), `pixel_sigma`
/// (assumed), `rot_walk`, `trans_walk`, `landmarks`, `camera_rate`, `time_offset` (truth),
/// `estimate_time_offset` (0/1), `outlier_fraction`, `landmark_gate`, `detect_change` (0/1),
/// `adapt_pixel_noise` (0/1), `drift_rate_deg_per_min`, `drift_start`, `tracking_start`, `roll_pitch_scale`,
/// `yaw_scale`, `gyro_noise_scale`, `accel_noise_scale`, `series_interval`.
/// \throws std::invalid_argument on an unknown key or malformed value.
void applyOverride(TrialConfig* config, const std::string& assignment);

}  // namespace aether::calib
