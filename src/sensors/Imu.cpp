#include "aether/sensors/Imu.hpp"

#include <cmath>

namespace aether::sensors {

namespace {
constexpr std::uint64_t kStreamImu = 1;  // Aether6's IMU stream id
}  // namespace

ImuModel::ImuModel(const ImuConfig& config, std::uint64_t seed)
    : cfg_(config), rng_(util::deriveSeed(seed, kStreamImu, 0)) {
  // Same draw order as Aether6's SensorSuite constructor (gyro, accel per axis).
  for (int i = 0; i < 3; ++i) {
    gyro_bias_(i) = rng_.gaussian(0.0, cfg_.gyro_bias_initial);
    accel_bias_(i) = rng_.gaussian(0.0, cfg_.accel_bias_initial);
  }
}

ImuSample ImuModel::measure(double time, const Vec3& true_rate, const Vec3& true_specific_force) {
  // Bias random walk advanced over the elapsed sampling interval.
  const double walk_dt = 1.0 / cfg_.rate;
  for (int i = 0; i < 3; ++i) {
    gyro_bias_(i) += cfg_.gyro_bias_walk * std::sqrt(walk_dt) * rng_.gaussian();
    accel_bias_(i) += cfg_.accel_bias_walk * std::sqrt(walk_dt) * rng_.gaussian();
  }
  ImuSample out;
  out.time = time;
  for (int i = 0; i < 3; ++i) {
    out.gyro(i) = true_rate(i) + gyro_bias_(i) + cfg_.gyro_noise_density * rng_.gaussian();
    out.accel(i) =
        true_specific_force(i) + accel_bias_(i) + cfg_.accel_noise_density * rng_.gaussian();
  }
  return out;
}

}  // namespace aether::sensors
