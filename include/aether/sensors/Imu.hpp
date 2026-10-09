/// \file Imu.hpp
/// \brief Simulated IMU with white noise, initial bias and bias random walk.
///
/// Reused from Aether6 (`include/aether/sensors/Sensors.hpp`). `ImuConfig` and `ImuSample`
/// are Aether6's structures unchanged, and `ImuModel` is the IMU branch of Aether6's
/// `SensorSuite` lifted out of the multi-sensor suite: same noise model, same SplitMix64
/// stream for the IMU, same sampling scheduler. The only change is the input: the aircraft
/// truth state is replaced by a body rate and specific force supplied by the caller, so the
/// model can sit behind any kinematic truth generator.
#pragma once

#include <cstdint>

#include "aether/core/Types.hpp"
#include "aether/util/Random.hpp"

namespace aether::sensors {

/// \brief Inertial measurement unit configuration.
struct ImuConfig {
  double rate = 200.0;               ///< Sample rate [Hz]
  double gyro_noise_density = 0.0035;///< Gyro white noise \f$\sigma\f$ at the sample rate [rad/s]
  double accel_noise_density = 0.05; ///< Accelerometer white noise \f$\sigma\f$ [m/s^2]
  double gyro_bias_initial = 0.010;  ///< 1-sigma of the initial gyro bias [rad/s]
  double accel_bias_initial = 0.08;  ///< 1-sigma of the initial accelerometer bias [m/s^2]
  double gyro_bias_walk = 3.0e-4;    ///< Gyro bias random-walk intensity [rad/s/sqrt(s)]
  double accel_bias_walk = 3.0e-3;   ///< Accel bias random-walk intensity [m/s^2/sqrt(s)]
};

/// \brief IMU sample: body angular rate and specific force.
struct ImuSample {
  double time = 0.0;          ///< Sample time [s]
  Vec3 gyro = Vec3::Zero();   ///< Measured body rate [rad/s]
  Vec3 accel = Vec3::Zero();  ///< Measured specific force in body axes [m/s^2]
};

/// \brief Deterministic IMU error model (Aether6 `SensorSuite`, IMU branch).
///
/// The biases are drawn once at construction from the configured 1-sigma values and then
/// evolve as random walks advanced by one sampling period per sample. The white noise is
/// specified as the per-sample standard deviation at the configured rate, exactly as in
/// Aether6.
class ImuModel {
 public:
  /// \param config IMU configuration.
  /// \param seed   Master seed; the IMU stream is derived from it as in Aether6.
  ImuModel(const ImuConfig& config, std::uint64_t seed);

  /// Corrupt one truth sample.
  /// \param time            Sample time [s].
  /// \param true_rate       True body angular rate [rad/s].
  /// \param true_specific_force True specific force in body axes [m/s^2].
  ImuSample measure(double time, const Vec3& true_rate, const Vec3& true_specific_force);

  /// Truth value of the current gyro bias [rad/s] (for estimator error metrics).
  const Vec3& gyroBias() const { return gyro_bias_; }
  /// Truth value of the current accelerometer bias [m/s^2].
  const Vec3& accelBias() const { return accel_bias_; }
  /// Configuration in use.
  const ImuConfig& config() const { return cfg_; }

 private:
  ImuConfig cfg_;
  util::Rng rng_;
  Vec3 gyro_bias_ = Vec3::Zero();
  Vec3 accel_bias_ = Vec3::Zero();
};

}  // namespace aether::sensors
