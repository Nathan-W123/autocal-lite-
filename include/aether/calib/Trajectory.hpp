/// \file Trajectory.hpp
/// \brief Scripted kinematic trajectory: a figure-eight with roll, pitch and yaw excitation.
///
/// Replaces Aether6's flight dynamics. Calibration needs a *known, smooth, persistently
/// exciting* motion, not physics, so the IMU pose is a closed-form function of time with
/// analytic first and second derivatives:
///
/// - **Translation**: a figure-eight (Lissajous 1:2) in the plane facing the scene,
///   \f$p(t) = c + s(t)\,[A_n\sin(2\omega t'+\varphi_n),\ A_e\sin(\omega t'),\ A_d\sin(2\omega t')]\f$.
/// - **Rotation**: 3-2-1 Euler angles oscillating at three incommensurate frequencies,
///   \f$\eta_i(t) = \eta_{0,i} + s(t)\,a_i\sin(\omega_i t' + \phi_i)\f$. Rotation about at
///   least two non-parallel axes is what makes all six extrinsic parameters observable: with
///   rotation about a single axis the lever-arm component along that axis cancels out of
///   every measurement.
/// - **Start**: the rig is stationary for `static_duration` seconds (used by the estimator's
///   static initialisation, exactly as an operator would hold a rig still), then every
///   excursion is scaled by a C2-continuous quintic ramp \f$s(t)\f$ over `ramp_duration`, so
///   acceleration and angular rate start from zero without a jump.
///
/// \f$t' = t - t_{\text{static}}\f$. All quantities are in Aether6's NED world frame with the
/// IMU frame forward-right-down.
#pragma once

#include "aether/core/Types.hpp"

namespace aether::calib {

/// \brief Trajectory shape parameters.
struct TrajectoryConfig {
  double static_duration = 2.0;  ///< Initial stationary period [s]
  double ramp_duration = 3.0;    ///< Quintic ramp-in of every excursion [s]

  Vec3 center = Vec3::Zero();    ///< Figure-eight centre, NED [m]
  double period = 8.0;           ///< Figure-eight period [s]
  double amp_north = 0.15;       ///< Fore-aft excursion (towards/away from the scene) [m]
  double amp_east = 0.9;         ///< Lateral half-width of the figure-eight [m]
  double amp_down = 0.45;        ///< Vertical half-height of the figure-eight [m]
  double phase_north = 0.7;      ///< Fore-aft phase [rad]

  Vec3 base_euler = Vec3::Zero();                       ///< Mean roll, pitch, yaw [rad]
  Vec3 euler_amplitude = Vec3(0.40, 0.35, 0.45);        ///< Roll, pitch, yaw amplitude [rad]
  Vec3 euler_frequency = Vec3(0.29, 0.41, 0.23);        ///< Roll, pitch, yaw frequency [Hz]
  Vec3 euler_phase = Vec3(0.0, 1.1, 2.3);               ///< Roll, pitch, yaw phase [rad]
};

/// \brief Kinematic truth at one instant (IMU frame I relative to world W).
struct KinematicState {
  double time = 0.0;               ///< [s]
  Vec3 position = Vec3::Zero();    ///< \f$p_{WI}\f$, NED [m]
  Vec3 velocity = Vec3::Zero();    ///< \f$v_{WI}\f$, NED [m/s]
  Vec3 acceleration = Vec3::Zero();///< \f$a_{WI}\f$, NED [m/s^2]
  Vec4 quaternion = Vec4(1, 0, 0, 0); ///< \f$q_{WI}\f$ (body to world) [-]
  Vec3 angular_rate = Vec3::Zero();///< \f$\omega_{WI}^I\f$, body frame [rad/s]
  Vec3 euler = Vec3::Zero();       ///< Roll, pitch, yaw [rad]

  /// Specific force sensed by an ideal accelerometer at the IMU origin,
  /// \f$f^I = R_{WI}^\top (a^W - g^W)\f$ with \f$g^W = [0,0,g]\f$ [m/s^2].
  Vec3 specificForce(double gravity) const;
};

/// \brief Closed-form figure-eight trajectory with analytic derivatives.
class FigureEightTrajectory {
 public:
  explicit FigureEightTrajectory(const TrajectoryConfig& config = {});

  /// Evaluate the kinematic state at time \a t [s].
  KinematicState evaluate(double t) const;

  /// Configuration in use.
  const TrajectoryConfig& config() const { return cfg_; }

  /// Quintic ramp \f$s(t)\f$ and its first two time derivatives (exposed for testing).
  Vec3 envelope(double t) const;

 private:
  TrajectoryConfig cfg_;
};

}  // namespace aether::calib
