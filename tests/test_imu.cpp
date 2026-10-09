/// \file test_imu.cpp
/// \brief Simulated IMU error model (`sensors/Imu.hpp`): reproducibility, white-noise level,
///        initial bias and bias random-walk statistics, and the noise-free limit.
///
/// The model is Aether6's `SensorSuite` IMU branch: per-sample white-noise sigma, an initial
/// bias drawn once at construction, and a bias random walk advanced by 1/rate per sample.
/// The statistical tolerances are ~5 sigma of the estimator used, so the fixed-seed results
/// are far from the thresholds while a wrong scaling (e.g. a missing sqrt or 1/rate) is not.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

#include "aether/sensors/Imu.hpp"
#include "aether/util/Random.hpp"

using namespace aether;
using Catch::Approx;

namespace {

/// Configuration with every error source switched off.
sensors::ImuConfig quietConfig(double rate = 200.0) {
  sensors::ImuConfig c;
  c.rate = rate;
  c.gyro_noise_density = 0.0;
  c.accel_noise_density = 0.0;
  c.gyro_bias_initial = 0.0;
  c.accel_bias_initial = 0.0;
  c.gyro_bias_walk = 0.0;
  c.accel_bias_walk = 0.0;
  return c;
}

/// A smoothly varying, non-trivial truth signal (rates and specific force of a gentle
/// manoeuvre) so equality checks are not satisfied by accident with zeros.
Vec3 truthRate(double t) { return Vec3(0.4 * std::sin(1.3 * t), -0.25 * std::cos(0.7 * t), 0.1); }
Vec3 truthForce(double t) {
  return Vec3(0.8 * std::cos(0.9 * t), 0.3 * std::sin(2.1 * t), -9.80665 + 0.2 * std::sin(t));
}

struct Moments {
  double n = 0.0, sum = 0.0, sumsq = 0.0;
  void add(double v) {
    n += 1.0;
    sum += v;
    sumsq += v * v;
  }
  double mean() const { return sum / n; }
  double stddev() const { return std::sqrt(sumsq / n - mean() * mean()); }
  double meanSquare() const { return sumsq / n; }
};

}  // namespace

TEST_CASE("the IMU model is bitwise reproducible for a given seed", "[imu][determinism]") {
  const sensors::ImuConfig cfg;  // defaults: every error source active
  sensors::ImuModel a(cfg, 424242), b(cfg, 424242), c(cfg, 424243);
  REQUIRE(a.gyroBias() == b.gyroBias());
  REQUIRE(a.accelBias() == b.accelBias());
  REQUIRE(a.gyroBias() != c.gyroBias());

  bool differs_from_other_seed = false;
  for (int k = 0; k < 2000; ++k) {
    const double t = k / cfg.rate;
    const auto sa = a.measure(t, truthRate(t), truthForce(t));
    const auto sb = b.measure(t, truthRate(t), truthForce(t));
    const auto sc = c.measure(t, truthRate(t), truthForce(t));
    REQUIRE(sa.time == t);
    REQUIRE(sa.gyro == sb.gyro);
    REQUIRE(sa.accel == sb.accel);
    differs_from_other_seed = differs_from_other_seed || sa.gyro != sc.gyro;
  }
  REQUIRE(a.gyroBias() == b.gyroBias());
  REQUIRE(a.accelBias() == b.accelBias());
  REQUIRE(differs_from_other_seed);
}

TEST_CASE("the initial biases come from Aether6's IMU stream in Aether6's draw order",
          "[imu][determinism]") {
  // ImuModel seeds its generator with deriveSeed(seed, stream 1, 0) and draws the initial
  // biases interleaved per axis (gyro x, accel x, gyro y, ...), as Aether6's SensorSuite.
  sensors::ImuConfig cfg;
  const std::uint64_t seed = 987654321;
  const sensors::ImuModel imu(cfg, seed);
  util::Rng ref(util::deriveSeed(seed, 1, 0));
  for (int i = 0; i < 3; ++i) {
    REQUIRE(imu.gyroBias()(i) == ref.gaussian(0.0, cfg.gyro_bias_initial));
    REQUIRE(imu.accelBias()(i) == ref.gaussian(0.0, cfg.accel_bias_initial));
  }
  REQUIRE(imu.config().rate == cfg.rate);
  REQUIRE(imu.config().gyro_noise_density == cfg.gyro_noise_density);
}

TEST_CASE("with every error source off the IMU returns the truth exactly", "[imu]") {
  sensors::ImuModel imu(quietConfig(), 5);
  REQUIRE(imu.gyroBias() == Vec3::Zero());
  REQUIRE(imu.accelBias() == Vec3::Zero());
  for (int k = 0; k < 500; ++k) {
    const double t = 0.005 * k;
    const auto s = imu.measure(t, truthRate(t), truthForce(t));
    REQUIRE(s.time == t);
    REQUIRE(s.gyro == truthRate(t));
    REQUIRE(s.accel == truthForce(t));
  }
  REQUIRE(imu.gyroBias() == Vec3::Zero());
  REQUIRE(imu.accelBias() == Vec3::Zero());
}

TEST_CASE("without white noise the measurement is truth plus the current advanced bias",
          "[imu][bias]") {
  // measure() first advances the random walk by one period and then applies the bias, so
  // gyroBias()/accelBias() read after the call are exactly the biases in that sample.
  sensors::ImuConfig cfg;
  cfg.gyro_noise_density = 0.0;
  cfg.accel_noise_density = 0.0;
  sensors::ImuModel imu(cfg, 99);
  Vec3 prev_bg = imu.gyroBias();
  int bias_moved = 0;
  for (int k = 0; k < 200; ++k) {
    const double t = k / cfg.rate;
    const auto s = imu.measure(t, truthRate(t), truthForce(t));
    REQUIRE(s.gyro == truthRate(t) + imu.gyroBias());
    REQUIRE(s.accel == truthForce(t) + imu.accelBias());
    bias_moved += (imu.gyroBias() != prev_bg) ? 1 : 0;
    prev_bg = imu.gyroBias();
  }
  REQUIRE(bias_moved == 200);  // the walk advances on every sample

  // With the walk switched off the initial bias is held constant (bit for bit).
  cfg.gyro_bias_walk = 0.0;
  cfg.accel_bias_walk = 0.0;
  sensors::ImuModel frozen(cfg, 99);
  const Vec3 bg0 = frozen.gyroBias(), ba0 = frozen.accelBias();
  REQUIRE(bg0.norm() > 0.0);
  for (int k = 0; k < 200; ++k) {
    const auto s = frozen.measure(k / cfg.rate, Vec3::Zero(), Vec3::Zero());
    REQUIRE(s.gyro == bg0);
    REQUIRE(s.accel == ba0);
  }
  REQUIRE(frozen.gyroBias() == bg0);
  REQUIRE(frozen.accelBias() == ba0);
}

TEST_CASE("white noise has the configured per-sample standard deviation and is white",
          "[imu][noise][statistics]") {
  // N = 100000 samples per axis. Sample-std relative sigma is 1/sqrt(2N) = 0.22 %, so 1.2 %
  // is ~5 sigma; the sample mean has sigma s/sqrt(N); correlations have sigma 1/sqrt(N) =
  // 0.0032, bounded at 0.016 (~5 sigma).
  auto cfg = quietConfig();
  cfg.gyro_noise_density = 0.0035;
  cfg.accel_noise_density = 0.05;
  sensors::ImuModel imu(cfg, 2026);
  const int n = 100000;
  Moments g[3], a[3];
  double cross_ga = 0.0, lag1_g = 0.0, lag1_a = 0.0;
  Vec3 prev_g = Vec3::Zero(), prev_a = Vec3::Zero();
  for (int k = 0; k < n; ++k) {
    const double t = k / cfg.rate;
    const auto s = imu.measure(t, truthRate(t), truthForce(t));
    const Vec3 eg = s.gyro - truthRate(t);
    const Vec3 ea = s.accel - truthForce(t);
    for (int i = 0; i < 3; ++i) {
      g[i].add(eg(i));
      a[i].add(ea(i));
    }
    cross_ga += (eg(0) / cfg.gyro_noise_density) * (ea(0) / cfg.accel_noise_density);
    if (k > 0) {
      lag1_g += eg(1) * prev_g(1) / (cfg.gyro_noise_density * cfg.gyro_noise_density);
      lag1_a += ea(2) * prev_a(2) / (cfg.accel_noise_density * cfg.accel_noise_density);
    }
    prev_g = eg;
    prev_a = ea;
  }
  for (int i = 0; i < 3; ++i) {
    INFO("axis " << i << ": gyro std " << g[i].stddev() << ", accel std " << a[i].stddev());
    REQUIRE(g[i].stddev() == Approx(cfg.gyro_noise_density).epsilon(0.012));
    REQUIRE(a[i].stddev() == Approx(cfg.accel_noise_density).epsilon(0.012));
    REQUIRE(std::abs(g[i].mean()) < 5.0 * cfg.gyro_noise_density / std::sqrt(n));
    REQUIRE(std::abs(a[i].mean()) < 5.0 * cfg.accel_noise_density / std::sqrt(n));
  }
  REQUIRE(std::abs(cross_ga / n) < 0.016);       // gyro and accel noise independent
  REQUIRE(std::abs(lag1_g / (n - 1)) < 0.016);   // no sample-to-sample correlation
  REQUIRE(std::abs(lag1_a / (n - 1)) < 0.016);
}

TEST_CASE("the white-noise sigma is per sample and does not scale with the IMU rate",
          "[imu][noise]") {
  // Aether6's parameterisation: gyro/accel_noise_density is the per-sample sigma at the
  // configured rate (the calibrator converts it with sigma^2/rate). The draw sequence does
  // not depend on the configured magnitudes either, so with the same seed the noise at
  // sigma = 0.02 is exactly twice the noise at sigma = 0.01 (multiplying by 2 is exact).
  auto c1 = quietConfig(200.0);
  c1.gyro_noise_density = 0.01;
  c1.accel_noise_density = 0.01;
  auto c2 = quietConfig(1000.0);  // different rate: must not matter for white noise
  c2.gyro_noise_density = 0.02;
  c2.accel_noise_density = 0.02;
  sensors::ImuModel m1(c1, 314), m2(c2, 314);
  for (int k = 0; k < 1000; ++k) {
    const auto s1 = m1.measure(0.0, Vec3::Zero(), Vec3::Zero());
    const auto s2 = m2.measure(0.0, Vec3::Zero(), Vec3::Zero());
    REQUIRE(s2.gyro == 2.0 * s1.gyro);
    REQUIRE(s2.accel == 2.0 * s1.accel);
  }
}

TEST_CASE("the initial bias spread matches the configured 1-sigma values",
          "[imu][bias][statistics]") {
  // 3000 seeds x 3 axes = 9000 independent draws: relative sigma of the sample std is
  // 1/sqrt(2*9000) = 0.75 %, bounded at 4 % (~5 sigma).
  auto cfg = quietConfig();
  cfg.gyro_bias_initial = 0.010;
  cfg.accel_bias_initial = 0.08;
  Moments g, a;
  for (std::uint64_t seed = 1; seed <= 3000; ++seed) {
    const sensors::ImuModel imu(cfg, seed);
    for (int i = 0; i < 3; ++i) {
      g.add(imu.gyroBias()(i));
      a.add(imu.accelBias()(i));
    }
  }
  REQUIRE(std::sqrt(g.meanSquare()) == Approx(cfg.gyro_bias_initial).epsilon(0.04));
  REQUIRE(std::sqrt(a.meanSquare()) == Approx(cfg.accel_bias_initial).epsilon(0.04));
  REQUIRE(std::abs(g.mean()) < 5.0 * cfg.gyro_bias_initial / std::sqrt(9000.0));
  REQUIRE(std::abs(a.mean()) < 5.0 * cfg.accel_bias_initial / std::sqrt(9000.0));
}

TEST_CASE("bias random-walk variance grows as walk^2 * N / rate", "[imu][bias][statistics]") {
  // Starting from zero bias, after N samples each axis is walk*sqrt(1/rate) times a sum of N
  // unit gaussians, so E[b^2] = walk^2 * N / rate. M = 3000 seeds x 3 axes = 9000 samples:
  // the relative sigma of the mean-square estimate is sqrt(2/9000) = 1.5 %, bounded at
  // 7.5 % (~5 sigma). Two rates with the same elapsed time (2 s) must give the same
  // variance, which pins the 1/rate step; the half-time checkpoint pins linear growth.
  for (double rate : {200.0, 100.0}) {
    auto cfg = quietConfig(rate);
    cfg.gyro_bias_walk = 3.0e-4;
    cfg.accel_bias_walk = 3.0e-3;
    const int steps = static_cast<int>(2.0 * rate);
    const int half = steps / 2;
    Moments g_half, a_half, g_end, a_end;
    for (std::uint64_t seed = 1; seed <= 3000; ++seed) {
      sensors::ImuModel imu(cfg, seed);
      for (int k = 1; k <= steps; ++k) {
        imu.measure(k / rate, Vec3::Zero(), Vec3::Zero());
        if (k == half)
          for (int i = 0; i < 3; ++i) {
            g_half.add(imu.gyroBias()(i));
            a_half.add(imu.accelBias()(i));
          }
      }
      for (int i = 0; i < 3; ++i) {
        g_end.add(imu.gyroBias()(i));
        a_end.add(imu.accelBias()(i));
      }
    }
    const double var_g_end = cfg.gyro_bias_walk * cfg.gyro_bias_walk * steps / rate;
    const double var_a_end = cfg.accel_bias_walk * cfg.accel_bias_walk * steps / rate;
    const double var_g_half = cfg.gyro_bias_walk * cfg.gyro_bias_walk * half / rate;
    const double var_a_half = cfg.accel_bias_walk * cfg.accel_bias_walk * half / rate;
    INFO("rate " << rate << " Hz: gyro E[b^2] " << g_end.meanSquare() << " vs " << var_g_end
                 << ", accel E[b^2] " << a_end.meanSquare() << " vs " << var_a_end);
    REQUIRE(g_end.meanSquare() == Approx(var_g_end).epsilon(0.075));
    REQUIRE(a_end.meanSquare() == Approx(var_a_end).epsilon(0.075));
    REQUIRE(g_half.meanSquare() == Approx(var_g_half).epsilon(0.075));
    REQUIRE(a_half.meanSquare() == Approx(var_a_half).epsilon(0.075));
    // Zero-mean walk: |mean| < 5 sigma / sqrt(9000).
    REQUIRE(std::abs(g_end.mean()) < 5.0 * std::sqrt(var_g_end / 9000.0));
    REQUIRE(std::abs(a_end.mean()) < 5.0 * std::sqrt(var_a_end / 9000.0));
  }
}
