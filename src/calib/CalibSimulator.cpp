#include "aether/calib/CalibSimulator.hpp"

#include <cmath>

#include "aether/math/Rotation.hpp"

namespace aether::calib {

namespace {
// Stream ids for util::deriveSeed. The IMU uses Aether6's own derivation inside ImuModel.
constexpr std::uint64_t kStreamScene = 21;
constexpr std::uint64_t kStreamMount = 22;
constexpr std::uint64_t kStreamPixels = 23;
constexpr std::uint64_t kStreamDrift = 24;

Vec3 randomUnitVector(util::Rng& rng) {
  for (;;) {
    const Vec3 v = rng.gaussian3();
    const double n = v.norm();
    if (n > 1e-6) return v / n;
  }
}
}  // namespace

CalibSimulator::CalibSimulator(const SimConfig& config, std::uint64_t seed)
    : cfg_(config),
      trajectory_(config.trajectory),
      imu_model_(config.imu, seed),
      rng_pixels_(util::deriveSeed(seed, kStreamPixels, 0)) {
  imu_count_ = static_cast<int>(std::floor(cfg_.duration * cfg_.imu.rate + 1e-9)) + 1;
  frame_count_ = static_cast<int>(std::floor(cfg_.duration * cfg_.camera_rate + 1e-9)) + 1;

  // --- Scene ------------------------------------------------------------------------
  util::Rng rng_scene(util::deriveSeed(seed, kStreamScene, 0));
  landmarks_.reserve(static_cast<std::size_t>(std::max(cfg_.scene.landmarks, 0)));
  for (int i = 0; i < cfg_.scene.landmarks; ++i) {
    Vec3 p;
    for (int a = 0; a < 3; ++a) p(a) = rng_scene.uniform(cfg_.scene.box_min(a), cfg_.scene.box_max(a));
    landmarks_.push_back(p);
  }

  // --- Truth mount: the nominal mount plus a per-trial manufacturing dispersion -----------
  util::Rng rng_mount(util::deriveSeed(seed, kStreamMount, 0));
  const Vec3 dtheta = cfg_.mount_rotation_dispersion * rng_mount.gaussian3();
  const Vec3 dp = cfg_.mount_translation_dispersion * rng_mount.gaussian3();
  mount_truth_.q_ic = math::boxPlus(cfg_.mount.q_ic, dtheta);
  mount_truth_.p_ic = cfg_.mount.p_ic + dp;

  // --- Drift axis --------------------------------------------------------------------
  util::Rng rng_drift(util::deriveSeed(seed, kStreamDrift, 0));
  const Vec3 random_axis = randomUnitVector(rng_drift);
  drift_axis_ = cfg_.drift.rotation_axis.norm() > 1e-9 ? cfg_.drift.rotation_axis.normalized()
                                                        : random_axis;
}

Extrinsic CalibSimulator::truthExtrinsic(double t) const {
  Extrinsic e = mount_truth_;
  const DriftConfig& d = cfg_.drift;
  if (d.start_time >= 0.0 && t > d.start_time) {
    const double dt = t - d.start_time;
    e.q_ic = math::boxPlus(e.q_ic, drift_axis_ * (d.rotation_rate * dt));
    e.p_ic += d.translation_rate * dt;
  }
  if (d.step_time >= 0.0 && t >= d.step_time) {
    e.q_ic = math::boxPlus(e.q_ic, d.step_rotation);
    e.p_ic += d.step_translation;
  }
  return e;
}

sensors::ImuSample CalibSimulator::imu(int i) {
  const double t = imuTime(i);
  const KinematicState k = trajectory_.evaluate(t);
  return imu_model_.measure(t, k.angular_rate, k.specificForce(cfg_.gravity));
}

StereoFrame CalibSimulator::frame(int k) {
  StereoFrame out;
  const double t_capture = captureTime(k);
  out.stamp = frameStamp(k);

  const KinematicState s = trajectory_.evaluate(t_capture);
  const Extrinsic ext = truthExtrinsic(t_capture);
  const Mat3 R_wi = math::quatToRotation(s.quaternion);
  const Mat3 R_wc = R_wi * ext.rotation();
  const Vec3 p_wc = s.position + R_wi * ext.p_ic;
  const StereoIntrinsics& cam = cfg_.camera;

  out.observations.reserve(landmarks_.size());
  for (std::size_t j = 0; j < landmarks_.size(); ++j) {
    const Vec3 P = R_wc.transpose() * (landmarks_[j] - p_wc);
    if (P.z() < cfg_.min_depth) continue;
    const Vec2 uv_l = cam.project(P);
    const Vec2 uv_r = cam.project(cam.leftToRight(P));
    if (!cam.inImage(uv_l, cfg_.image_margin) || !cam.inImage(uv_r, cfg_.image_margin)) continue;

    StereoObservation o;
    o.id = static_cast<int>(j);
    // One draw per statement: argument evaluation order differs between compilers.
    double noise[4];
    for (double& n : noise) n = rng_pixels_.gaussian();
    o.left = uv_l + cfg_.pixel_noise * Vec2(noise[0], noise[1]);
    o.right = uv_r + cfg_.pixel_noise * Vec2(noise[2], noise[3]);
    if (cfg_.outlier_fraction > 0.0 && rng_pixels_.uniform() < cfg_.outlier_fraction) {
      // A wrong correspondence: a random point with a plausible positive disparity.
      // The left pixel starts 40 px in so the right one stays on the sensor.
      const double u = rng_pixels_.uniform(40.0, cam.width - 1.0);
      const double v = rng_pixels_.uniform(0.0, cam.height - 1.0);
      o.left = Vec2(u, v);
      o.right = Vec2(o.left.x() - rng_pixels_.uniform(2.0, 40.0), o.left.y());
    }
    out.observations.push_back(o);
  }
  return out;
}

}  // namespace aether::calib
