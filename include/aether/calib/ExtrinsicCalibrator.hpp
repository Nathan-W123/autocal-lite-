/// \file ExtrinsicCalibrator.hpp
/// \brief Online, targetless stereo/IMU extrinsic calibration with an error-state EKF.
#pragma once

#include <cmath>
#include <deque>
#include <functional>
#include <string>
#include <vector>

#include "aether/calib/Extrinsic.hpp"
#include "aether/calib/StereoCamera.hpp"
#include "aether/core/Constants.hpp"
#include "aether/core/Types.hpp"
#include "aether/sensors/Imu.hpp"

namespace aether::calib {

/// \brief Tuning of the calibrator.
struct CalibratorConfig {
  StereoIntrinsics camera;                ///< Known stereo intrinsics and baseline.
  double gravity = constants::kGravity;   ///< [m/s^2]

  // --- IMU noise, in Aether6's ImuConfig parameterisation ---------------------------
  double imu_rate = 200.0;          ///< Nominal IMU rate, converts per-sample sigmas [Hz]
  double gyro_noise = 0.0035;       ///< Gyro white noise, per-sample sigma [rad/s]
  double accel_noise = 0.05;        ///< Accelerometer white noise, per-sample sigma [m/s^2]
  double gyro_bias_walk = 3.0e-4;   ///< Gyro bias random walk [rad/s/sqrt(s)]
  double accel_bias_walk = 3.0e-3;  ///< Accel bias random walk [m/s^2/sqrt(s)]

  // --- Extrinsic process model ----------------------------------------------------
  /// Random-walk intensity of the extrinsic rotation [rad/sqrt(s)]. Non-zero is what lets the
  /// filter keep tracking a mount that slowly moves; zero freezes the estimate once the
  /// covariance has collapsed (the "calibrate once" baseline).
  double extrinsic_rotation_walk = 1.0e-4;
  double extrinsic_translation_walk = 1.0e-5;  ///< Lever-arm random walk [m/sqrt(s)]
  double time_offset_walk = 0.0;               ///< Time-offset random walk [s/sqrt(s)]

  // --- Vision measurement ---------------------------------------------------------
  /// Pixel noise, 1-sigma per coordinate [px]. With `adapt_pixel_noise` this is only the
  /// starting value: the filter rescales it to the noise it actually measures.
  double pixel_sigma = 0.5;
  /// Estimate the pixel noise online. After the QR compression, the part of the stacked
  /// residual that no state can explain is pure measurement noise with a known number of
  /// degrees of freedom, so it measures the true noise level independently of the state
  /// covariance. Its running average rescales `pixel_sigma`.
  bool adapt_pixel_noise = true;
  int noise_adaptation_frames = 20;   ///< Averaging window of the noise estimate [frames]
  double min_noise_scale = 0.1;       ///< Bounds on (adapted / configured variance) [-]
  double max_noise_scale = 100.0;
  /// Per-landmark chi-squared gate on the 5-dof null-space residual (99.9 % = 20.5).
  double landmark_gate = 20.5;
  int min_landmarks = 8;         ///< Skip the update when fewer landmarks survive.
  double min_disparity = 1.0;    ///< Landmarks with a smaller anchor disparity are skipped [px]
  int triangulation_iterations = 3;  ///< Gauss-Newton refinements of each landmark.

  // --- Static initialisation ------------------------------------------------------
  double static_init_duration = 1.5;  ///< IMU averaging window at start-up [s]
  double static_gyro_std_max = 0.02;  ///< Stillness test on the gyro [rad/s]
  double static_accel_std_max = 0.2;  ///< Stillness test on the accelerometer [m/s^2]

  // --- Initial uncertainty (1-sigma) ----------------------------------------------
  double init_attitude_sigma = 2.0 * constants::kDegToRad;  ///< [rad]
  double init_velocity_sigma = 0.05;                         ///< [m/s]
  double init_position_sigma = 0.01;                         ///< [m]
  double init_gyro_bias_sigma = 0.002;                       ///< [rad/s]
  double init_accel_bias_sigma = 0.15;                       ///< [m/s^2]
  double init_extrinsic_rotation_sigma = 5.0 * constants::kDegToRad;  ///< [rad]
  double init_extrinsic_translation_sigma = 0.05;                     ///< [m]

  // --- Camera-IMU time offset -----------------------------------------------------
  /// Estimate \f$t_d\f$ online. On by default: the lever arm is strongly coupled to timing, and
  /// even 2.5 ms of unmodelled camera latency biases it by more than a centimetre.
  bool estimate_time_offset = true;
  double init_time_offset = 0.0;       ///< Prior \f$t_d\f$ [s]
  double init_time_offset_sigma = 0.01;///< [s]
  /// The estimate is clamped to +-this [s]. It is a configuration bound, not a fit: it must
  /// exceed the largest offset expected, because a clamped estimate leaves the rest of the
  /// state inconsistent with it.
  double max_time_offset = 0.05;

  // --- Change detection ("knock" recovery) -----------------------------------------
  /// Re-open the extrinsic covariance when the filter keeps being pulled in one direction
  /// by more than its own covariance allows. Over a window of updates, the summed extrinsic
  /// corrections \f$\Delta = \sum K_i\nu_i\f$ of a consistent filter have covariance
  /// \f$C = \sum (P^-_i - P^+_i)\f$ (innovations are white), so
  /// \f$T = \Delta^\top C^{-1}\Delta \sim \chi^2_6\f$ while the mount behaves as modelled
  /// and grows quadratically after a step change.
  bool detect_change = true;
  int change_window = 40;               ///< Updates in the test window (2 s at 20 Hz).
  double change_threshold = 40.0;       ///< Threshold on T (chi^2_6 tail 4.6e-7).
  /// 1-sigma the extrinsic covariance is re-opened to when a change is detected. Sized for a
  /// knock, not for the initial CAD-level uncertainty: a much larger value makes the first
  /// updates after the reset overshoot. A bigger change simply trips the detector again.
  double reopen_rotation_sigma = 1.0 * constants::kDegToRad;  ///< [rad]
  double reopen_translation_sigma = 0.01;                      ///< [m]
  /// The detector is armed only once the calibration has converged (every rotation and
  /// lever-arm sigma below these), so neither the initial convergence nor the re-convergence
  /// after a reset is mistaken for a knock.
  double change_arm_rotation_sigma = 0.1 * constants::kDegToRad;  ///< [rad]
  double change_arm_translation_sigma = 0.005;                    ///< [m]

  /// How late a stereo frame may be delivered after its exposure [s]. The filter keeps the
  /// newest IMU samples in a buffer and lags the newest one by this much, so a frame that
  /// arrives late is still fused at its exact exposure time. 0 means frames must be queued
  /// before the IMU passes their exposure (no lag). Frames later than this are dropped.
  double max_frame_latency = 0.0;

  bool exact_discretisation = false;   ///< Van Loan instead of the series (Aether6 option).

  /// Random-walk intensity injected along the unobservable global-yaw direction of the
  /// error state [rad/sqrt(s)] (0 disables). See ExtrinsicCalibrator::yawNullDirection().
  double yaw_gauge_noise = 0.0;

  /// Experimental: fuse disjoint frame pairs (0-1, 2-3, ...) so every observation is used
  /// exactly once, instead of every consecutive pair.
  bool disjoint_pairs = false;

  // --- Calibration status (see ExtrinsicCalibrator::status()) ---------------------
  /// The status bounds are this many filter sigmas along the least-certain direction [-].
  /// 2 is enough because the filter covariance is conservative (NEES ~2 against 6, so a
  /// filter sigma is ~1.7 true sigmas): in the evaluation the status never claimed
  /// convergence with the true error outside the bounds, except between a knock and its
  /// detection.
  double status_sigma_multiplier = 2.0;
  /// The calibration counts as converged once the bound along the least-certain direction
  /// is below these. The defaults are the acceptance criteria of the evaluation.
  double converged_rotation_bound = 0.5 * constants::kDegToRad;  ///< [rad]
  double converged_translation_bound = 0.01;                      ///< [m]
  double converged_time_offset_bound = 0.001;                     ///< [s]
  /// Time constant of the running mean of the body rate used to judge excitation [s].
  double excitation_window = 5.0;
  /// RMS body rate about the axes that would observe the weakest direction below which the
  /// status asks for more motion rather than more time [rad/s].
  double min_excitation_rate = 0.1;
  /// Below this RMS rate (but above the minimum) the status says convergence is slow and
  /// names the rotations that would speed it up [rad/s].
  double good_excitation_rate = 0.3;
};

/// \brief What the calibration is doing, for an operator or a supervising process.
enum class CalibrationState {
  kInitialising,   ///< Collecting the static start-up window.
  kCalibrating,    ///< Running, not yet within the converged bounds.
  kConverged,      ///< Every extrinsic (and time-offset) bound is within tolerance.
  kRecalibrating,  ///< A mount change was detected; re-converging to the new mount.
};

/// Lower-case name of a state ("initialising", "calibrating", ...).
const char* toString(CalibrationState state);

/// \brief Self-assessment of the calibration, derived from the filter covariance.
///
/// The bounds are `status_sigma_multiplier` filter sigmas (default 2) along the least-certain
/// direction, so a direction the motion has not excited stays visible even when the other
/// two are well determined. The weakest directions are unit vectors in the IMU frame.
struct CalibrationStatus {
  CalibrationState state = CalibrationState::kInitialising;
  double time = 0.0;                    ///< Filter time [s]
  double rotation_bound = 0.0;          ///< k-sigma, least-certain rotation direction [rad]
  double translation_bound = 0.0;       ///< k-sigma, least-certain lever-arm direction [m]
  double time_offset_bound = 0.0;       ///< k-sigma of t_d (0 when not estimated) [s]
  Vec3 weakest_rotation_axis = Vec3::UnitX();     ///< IMU frame, unit
  Vec3 weakest_translation_axis = Vec3::UnitX();  ///< IMU frame, unit
  bool rotation_converged = false;
  bool translation_converged = false;
  bool time_offset_converged = false;
  /// RMS body rate about IMU x, y and z over the last `excitation_window` [rad/s].
  Vec3 excitation = Vec3::Zero();
  double pixel_sigma = 0.0;             ///< Pixel noise in use (adapted) [px]
  std::string guidance;                 ///< One-line advice for the operator.
};

/// \brief Per-frame filter diagnostics, reported after every processed stereo frame.
struct FrameReport {
  double time = 0.0;        ///< IMU-clock time at which the frame was fused [s]
  int observations = 0;     ///< Observations in the frame.
  int matched = 0;          ///< Landmarks also seen in the previous frame.
  int used = 0;             ///< Landmarks that passed the gate and were fused.
  int gated = 0;            ///< Landmarks rejected by the gate.
  bool updated = false;     ///< True when an EKF update was applied.
  double nis_per_dof = 0.0; ///< Normalised innovation squared per degree of freedom.
  double noise_scale = 1.0; ///< Adapted / configured pixel-noise variance after this frame.
  double change_statistic = 0.0; ///< Change-detector statistic T (chi^2_6 under no change).
  bool change_detected = false;  ///< Change detector fired on this frame.
};

/// \brief Targetless stereo/IMU extrinsic calibrator (error-state EKF, single stochastic clone).
///
/// **State.** The nominal state is the IMU pose, velocity and biases needed to propagate
/// with the IMU, the extrinsic, the time offset, and one stochastic clone of the IMU pose
/// at the previous stereo frame. There are no landmarks and no sliding window in the state.
/// Error-state layout (28 scalars, Aether6 ordering for the first five blocks):
///
/// | Index  | Block | Meaning |
/// |--------|-------|---------|
/// | 0..2   | \f$\delta p\f$       | IMU position, world [m] |
/// | 3..5   | \f$\delta v\f$       | IMU velocity, world [m/s] |
/// | 6..8   | \f$\delta\theta\f$   | IMU attitude, body-frame rotation vector [rad] |
/// | 9..11  | \f$\delta b_g\f$     | gyro bias [rad/s] |
/// | 12..14 | \f$\delta b_a\f$     | accelerometer bias [m/s^2] |
/// | 15..17 | \f$\delta\phi_{IC}\f$| extrinsic rotation, camera-frame rotation vector [rad] |
/// | 18..20 | \f$\delta p_{IC}\f$  | extrinsic lever arm [m] |
/// | 21     | \f$\delta t_d\f$     | camera-IMU time offset [s] |
/// | 22..24 | \f$\delta\theta_c\f$ | cloned attitude at the previous frame [rad] |
/// | 25..27 | \f$\delta p_c\f$     | cloned position at the previous frame [m] |
///
/// **Propagation.** IMU samples drive the nominal state with a trapezoidal (midpoint) rule
/// and the covariance with Aether6's error dynamics. Only the 15-state navigation block has
/// dynamics; the extrinsic, time offset and clone are random walks or constants, so the
/// propagation is done block-wise.
///
/// **Update.** At every stereo frame, each landmark seen in both the previous and the
/// current frame contributes 8 reprojection residuals (left and right pixels at both
/// frames), predicted with the *current extrinsic estimate* and a landmark position
/// triangulated from the data. The landmark is then marginalised exactly by projecting the
/// residual onto the left null space of its Jacobian (5 residuals per landmark remain), a
/// per-landmark chi-squared gate removes outliers, the stack is compressed by QR to at most
/// 28 rows, and Aether6's Joseph-form update is applied. The stochastic clone is then moved
/// to the current pose.
///
/// No ground truth enters the filter: it starts from an IMU-only static initialisation and
/// a nominal (CAD) extrinsic prior.
class ExtrinsicCalibrator {
 public:
  static constexpr int kStateDim = 28;   ///< Error-state dimension.
  static constexpr int kIdxPosition = 0;
  static constexpr int kIdxVelocity = 3;
  static constexpr int kIdxAttitude = 6;
  static constexpr int kIdxGyroBias = 9;
  static constexpr int kIdxAccelBias = 12;
  static constexpr int kIdxExtRotation = 15;
  static constexpr int kIdxExtTranslation = 18;
  static constexpr int kIdxTimeOffset = 21;
  static constexpr int kIdxCloneAttitude = 22;
  static constexpr int kIdxClonePosition = 25;
  static constexpr int kNavDim = 15;     ///< Size of the block with IMU dynamics.

  /// \param config Tuning.
  /// \param prior  Nominal (e.g. CAD) extrinsic the filter starts from.
  ExtrinsicCalibrator(const CalibratorConfig& config, const Extrinsic& prior);

  /// Feed one IMU sample (strictly increasing time). The filter then advances to
  /// `sample.time - max_frame_latency`, fusing every queued stereo frame whose exposure time,
  /// `stamp + t_d_hat`, it passes at that exact time.
  void addImu(const sensors::ImuSample& sample);
  /// Queue a stereo frame (non-decreasing stamps). It may arrive up to `max_frame_latency`
  /// after its exposure. A frame whose exposure lies less than half an IMU period before the
  /// filter time is fused at the filter time; an older one is dropped (see droppedFrames()).
  void addFrame(StereoFrame frame);
  /// Process every buffered IMU sample and the frames they pass (end of a data set).
  void flush();
  /// Frames exposed after the filter started that arrived too late to be fused.
  int droppedFrames() const { return dropped_frames_; }

  /// Called after every fused frame.
  void setFrameCallback(std::function<void(const FrameReport&)> cb) { callback_ = std::move(cb); }

  /// True once the static initialisation has completed.
  bool initialised() const { return initialised_; }
  /// False if the start-up window failed the stillness test (the filter still starts).
  bool staticInitPassed() const { return static_ok_; }
  /// Current filter time (IMU clock) [s].
  double time() const { return time_; }

  /// Current extrinsic estimate.
  Extrinsic extrinsic() const { return {q_ic_, p_ic_}; }
  /// Current time-offset estimate [s].
  double timeOffset() const { return td_; }
  Vec3 gyroBias() const { return bg_; }   ///< Current gyro-bias estimate [rad/s]
  Vec3 accelBias() const { return ba_; }  ///< Current accel-bias estimate [m/s^2]
  Vec3 velocity() const { return v_; }    ///< Current velocity estimate (world) [m/s]
  Vec4 attitude() const { return q_wi_; } ///< Current attitude estimate \f$q_{WI}\f$
  /// Full error covariance.
  const MatX& covariance() const { return P_; }
  /// 6x6 covariance of \f$[\delta\phi_{IC}, \delta p_{IC}]\f$.
  Eigen::Matrix<double, 6, 6> extrinsicCovariance() const {
    return P_.block<6, 6>(kIdxExtRotation, kIdxExtRotation);
  }
  /// Number of times the change detector re-opened the extrinsic covariance.
  int changeEvents() const { return change_events_; }
  /// Pixel noise currently assumed by the filter (adapted when `adapt_pixel_noise`) [px].
  double pixelSigma() const { return cfg_.pixel_sigma * std::sqrt(noise_scale_); }
  /// Self-assessment: state, k-sigma bounds, weakest directions and operator guidance.
  CalibrationStatus status() const;
  /// Configuration in use.
  const CalibratorConfig& config() const { return cfg_; }

 private:
  void initialiseFromStatic();
  void processImuUntil(double t);
  void stepImu(const sensors::ImuSample& sample);
  void propagate(sensors::ImuSample a, const sensors::ImuSample& b);
  void fuseFrame(const StereoFrame& frame, const Vec3& gyro_at_frame);
  void injectAndReset(const VecX& dx);
  void cloneCurrentPose(const Vec3& omega, double shift);
  void reopenExtrinsic();
  VecX yawNullDirection() const;

  CalibratorConfig cfg_;

  // Nominal state.
  Vec4 q_wi_ = Vec4(1, 0, 0, 0);
  Vec3 v_ = Vec3::Zero();
  Vec3 p_ = Vec3::Zero();
  Vec3 bg_ = Vec3::Zero();
  Vec3 ba_ = Vec3::Zero();
  Vec4 q_ic_ = Vec4(1, 0, 0, 0);
  Vec3 p_ic_ = Vec3::Zero();
  double td_ = 0.0;
  // Stochastic clone of the IMU pose at the previous frame, with the rates needed for the
  // time-offset Jacobian.
  Vec4 clone_q_ = Vec4(1, 0, 0, 0);
  Vec3 clone_p_ = Vec3::Zero();
  Vec3 clone_omega_ = Vec3::Zero();
  Vec3 clone_v_ = Vec3::Zero();
  bool has_clone_ = false;
  std::vector<StereoObservation> clone_observations_;

  MatX P_;
  double time_ = 0.0;

  // IMU/frame bookkeeping.
  bool initialised_ = false;
  bool static_ok_ = false;
  std::vector<sensors::ImuSample> init_samples_;
  sensors::ImuSample segment_start_;  ///< IMU sample (possibly interpolated) at time_.
  std::deque<StereoFrame> frames_;
  std::deque<sensors::ImuSample> imu_buffer_;  ///< Samples not yet processed (fixed lag).
  double last_imu_time_ = -1e300;
  double init_time_ = 0.0;  ///< Filter start (end of the static window) [s]
  int dropped_frames_ = 0;

  // Change detection.
  /// Per-update extrinsic correction and covariance reduction, for the change detector.
  std::deque<std::pair<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 6, 6>>> change_window_;
  int change_events_ = 0;
  bool recalibrating_ = false;  ///< Set by a detected change, cleared on re-convergence.

  // Pixel-noise adaptation: adapted / configured variance.
  double noise_scale_ = 1.0;

  // Running second moment of the bias-corrected body rate (excitation).
  Mat3 rate_moment_ = Mat3::Zero();

  std::function<void(const FrameReport&)> callback_;
};

}  // namespace aether::calib
