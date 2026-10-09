/// \file CalibSimulator.hpp
/// \brief Truth generator: scripted trajectory, landmark scene, IMU, stereo camera and a
///        (hidden) IMU-camera extrinsic that can drift or be knocked.
#pragma once

#include <cstdint>
#include <vector>

#include "aether/calib/Extrinsic.hpp"
#include "aether/calib/StereoCamera.hpp"
#include "aether/calib/Trajectory.hpp"
#include "aether/core/Constants.hpp"
#include "aether/sensors/Imu.hpp"
#include "aether/util/Random.hpp"

namespace aether::calib {

/// \brief Landmark field. Landmarks are drawn uniformly in an axis-aligned box of the world
/// frame, placed in front of the rig (the rig looks north, towards +x).
struct SceneConfig {
  int landmarks = 200;                    ///< Number of 3-D landmarks.
  Vec3 box_min = Vec3(2.5, -4.5, -3.0);   ///< Box lower corner, NED [m]
  Vec3 box_max = Vec3(8.0, 4.5, 3.0);     ///< Box upper corner, NED [m]
};

/// \brief Time variation of the true extrinsic after the rig has been calibrated.
///
/// Rotation drift is a constant-rate rotation about a fixed camera-frame axis,
/// \f$R_{IC}(t) = R_{IC,0}\,\mathrm{Exp}(\hat a\,\dot\phi\,(t - t_0))\f$ for \f$t > t_0\f$; the
/// lever arm drifts linearly. A "knock" is an instantaneous step applied at `step_time`.
struct DriftConfig {
  double start_time = -1.0;   ///< Drift onset \f$t_0\f$ [s] (< 0 disables drift)
  double rotation_rate = 0.1 * constants::kDegToRad / 60.0;  ///< \f$\dot\phi\f$ [rad/s] (0.1 deg/min)
  Vec3 rotation_axis = Vec3::Zero();  ///< Unit axis in the camera frame; zero = random per trial
  Vec3 translation_rate = Vec3::Zero();  ///< Lever-arm drift [m/s]

  double step_time = -1.0;             ///< Knock time [s] (< 0 disables)
  Vec3 step_rotation = Vec3::Zero();   ///< Knock rotation vector, camera frame [rad]
  Vec3 step_translation = Vec3::Zero();///< Knock lever-arm change [m]
};

/// \brief Full simulation configuration.
struct SimConfig {
  double duration = 60.0;        ///< Simulated time [s]
  double gravity = constants::kGravity;  ///< [m/s^2]

  TrajectoryConfig trajectory;   ///< Scripted motion.
  sensors::ImuConfig imu;        ///< Aether6 IMU model (200 Hz, white noise + bias walk).
  StereoIntrinsics camera;       ///< Known stereo intrinsics.
  double camera_rate = 20.0;     ///< Stereo frame rate [Hz]
  double pixel_noise = 0.5;      ///< Gaussian pixel noise, 1-sigma per coordinate [px]
  double image_margin = 2.0;     ///< Observations closer than this to the border are dropped [px]
  double min_depth = 0.5;        ///< Minimum landmark depth for an observation [m]
  double time_offset = 0.0;      ///< True camera-IMU offset \f$t_d\f$: exposure at IMU time stamp + t_d [s]
  double outlier_fraction = 0.0; ///< Fraction of correspondences replaced by random pixels [-]
  SceneConfig scene;             ///< Landmark field.

  /// Nominal mount; the per-trial truth is this perturbed by the dispersions below.
  Extrinsic mount{forwardLookingMount(), Vec3(0.06, -0.04, 0.02)};
  double mount_rotation_dispersion = 2.0 * constants::kDegToRad;  ///< 1-sigma per axis [rad]
  double mount_translation_dispersion = 0.02;                       ///< 1-sigma per axis [m]

  DriftConfig drift;             ///< Post-calibration drift / knock.
};

/// \brief Deterministic stereo/IMU data generator with a hidden truth extrinsic.
///
/// Everything random (scene, truth mount, drift axis, IMU errors, pixel noise) is drawn from
/// streams derived from one seed, so a trial is reproducible bit for bit. The estimator only
/// ever sees `imu(i)` and `frame(k)`; the truth accessors are for scoring.
class CalibSimulator {
 public:
  CalibSimulator(const SimConfig& config, std::uint64_t seed);

  /// Number of IMU samples in the run.
  int imuCount() const { return imu_count_; }
  /// Number of stereo frames in the run.
  int frameCount() const { return frame_count_; }
  /// Time of IMU sample \a i [s].
  double imuTime(int i) const { return static_cast<double>(i) / cfg_.imu.rate; }
  /// Camera-clock stamp of frame \a k [s].
  double frameStamp(int k) const { return captureTime(k) - cfg_.time_offset; }
  /// IMU-clock exposure time of frame \a k [s].
  double captureTime(int k) const { return static_cast<double>(k) / cfg_.camera_rate; }

  /// IMU sample \a i. Must be requested with consecutive indices (the bias walk is stateful).
  sensors::ImuSample imu(int i);
  /// Stereo frame \a k. Must be requested with consecutive indices (the noise stream is
  /// sequential).
  StereoFrame frame(int k);

  /// True extrinsic at IMU-clock time \a t.
  Extrinsic truthExtrinsic(double t) const;
  /// Truth kinematics at time \a t.
  KinematicState truthState(double t) const { return trajectory_.evaluate(t); }
  /// Truth IMU biases after the most recent `imu()` call.
  const Vec3& truthGyroBias() const { return imu_model_.gyroBias(); }
  const Vec3& truthAccelBias() const { return imu_model_.accelBias(); }  ///< \copydoc truthGyroBias
  /// Landmark positions (world frame).
  const std::vector<Vec3>& landmarks() const { return landmarks_; }
  /// Drift axis actually used (camera frame, unit).
  const Vec3& driftAxis() const { return drift_axis_; }
  /// Configuration in use.
  const SimConfig& config() const { return cfg_; }

 private:
  SimConfig cfg_;
  FigureEightTrajectory trajectory_;
  sensors::ImuModel imu_model_;
  util::Rng rng_pixels_;
  std::vector<Vec3> landmarks_;
  Extrinsic mount_truth_;
  Vec3 drift_axis_ = Vec3::UnitX();
  int imu_count_ = 0;
  int frame_count_ = 0;
};

}  // namespace aether::calib
