#include "aether/calib/ExtrinsicCalibrator.hpp"

#include <algorithm>
#include <cmath>

#include "aether/calib/MeasurementModel.hpp"
#include "aether/estimation/EskfCore.hpp"
#include "aether/math/LinearAlgebra.hpp"
#include "aether/math/Rotation.hpp"

namespace aether::calib {

namespace {

constexpr int kNx = ExtrinsicCalibrator::kStateDim;
constexpr int kNn = ExtrinsicCalibrator::kNavDim;
constexpr int kNr = kNx - kNn;  // extrinsic, time offset and clone: no dynamics

/// Linear interpolation of an IMU sample at time \a t inside [a.time, b.time].
sensors::ImuSample interpolate(const sensors::ImuSample& a, const sensors::ImuSample& b,
                               double t) {
  const double span = b.time - a.time;
  const double alpha = span > 0.0 ? std::clamp((t - a.time) / span, 0.0, 1.0) : 1.0;
  sensors::ImuSample m;
  m.time = t;
  m.gyro = a.gyro + alpha * (b.gyro - a.gyro);
  m.accel = a.accel + alpha * (b.accel - a.accel);
  return m;
}

}  // namespace

ExtrinsicCalibrator::ExtrinsicCalibrator(const CalibratorConfig& config, const Extrinsic& prior)
    : cfg_(config), q_ic_(math::quatNormalize(prior.q_ic)), p_ic_(prior.p_ic),
      td_(config.init_time_offset) {
  P_ = MatX::Zero(kNx, kNx);
}

void ExtrinsicCalibrator::addFrame(StereoFrame frame) {
  std::sort(frame.observations.begin(), frame.observations.end(),
            [](const StereoObservation& a, const StereoObservation& b) { return a.id < b.id; });
  frames_.push_back(std::move(frame));
}

void ExtrinsicCalibrator::addImu(const sensors::ImuSample& sample) {
  if (!initialised_) {
    init_samples_.push_back(sample);
    if (sample.time - init_samples_.front().time >= cfg_.static_init_duration - 1e-9)
      initialiseFromStatic();
    return;
  }
  if (!(sample.time > time_)) return;  // IMU must move forward in time

  const double late_tolerance = 0.5 / cfg_.imu_rate;
  while (!frames_.empty()) {
    const double tau = frames_.front().stamp + td_;
    if (tau > sample.time) break;
    if (tau < time_ - late_tolerance) {  // exposure before the filter's present: unusable
      frames_.pop_front();
      continue;
    }
    const sensors::ImuSample at_frame = interpolate(segment_start_, sample, std::max(tau, time_));
    if (at_frame.time > time_) propagate(segment_start_, at_frame);
    fuseFrame(frames_.front(), at_frame.gyro);
    frames_.pop_front();
  }
  propagate(segment_start_, sample);
}

void ExtrinsicCalibrator::initialiseFromStatic() {
  const double n = static_cast<double>(init_samples_.size());
  Vec3 gyro_mean = Vec3::Zero(), accel_mean = Vec3::Zero();
  for (const auto& s : init_samples_) {
    gyro_mean += s.gyro;
    accel_mean += s.accel;
  }
  gyro_mean /= n;
  accel_mean /= n;
  Vec3 gyro_var = Vec3::Zero(), accel_var = Vec3::Zero();
  for (const auto& s : init_samples_) {
    gyro_var += (s.gyro - gyro_mean).cwiseAbs2();
    accel_var += (s.accel - accel_mean).cwiseAbs2();
  }
  const double denom = std::max(n - 1.0, 1.0);
  static_ok_ = (gyro_var / denom).cwiseSqrt().maxCoeff() < cfg_.static_gyro_std_max &&
               (accel_var / denom).cwiseSqrt().maxCoeff() < cfg_.static_accel_std_max;

  // At rest the accelerometer senses f = -R^T g^W, so the world "down" axis seen in the
  // body frame is -f/|f| = [-sin(theta), sin(phi)cos(theta), cos(phi)cos(theta)]. Yaw and
  // position are unobservable and set to zero (they define the estimator's world frame).
  const Vec3 down = -accel_mean.normalized();
  const double pitch = std::asin(std::clamp(-down.x(), -1.0, 1.0));
  const double roll = std::atan2(down.y(), down.z());
  q_wi_ = math::eulerToQuat(roll, pitch, 0.0);
  v_.setZero();
  p_.setZero();
  bg_ = gyro_mean;
  ba_.setZero();

  P_.setZero();
  auto setBlock = [&](int idx, int size, double sigma) {
    P_.block(idx, idx, size, size) = MatX::Identity(size, size) * sigma * sigma;
  };
  setBlock(kIdxPosition, 3, cfg_.init_position_sigma);
  setBlock(kIdxVelocity, 3, cfg_.init_velocity_sigma);
  setBlock(kIdxAttitude, 3, cfg_.init_attitude_sigma);
  setBlock(kIdxGyroBias, 3, cfg_.init_gyro_bias_sigma);
  setBlock(kIdxAccelBias, 3, cfg_.init_accel_bias_sigma);
  setBlock(kIdxExtRotation, 3, cfg_.init_extrinsic_rotation_sigma);
  setBlock(kIdxExtTranslation, 3, cfg_.init_extrinsic_translation_sigma);
  P_(kIdxTimeOffset, kIdxTimeOffset) =
      cfg_.estimate_time_offset ? cfg_.init_time_offset_sigma * cfg_.init_time_offset_sigma : 0.0;

  segment_start_ = init_samples_.back();
  time_ = segment_start_.time;
  init_samples_.clear();
  init_samples_.shrink_to_fit();
  initialised_ = true;
}

void ExtrinsicCalibrator::propagate(sensors::ImuSample a, const sensors::ImuSample& b) {
  // `a` is taken by value: callers pass segment_start_, which is overwritten right here.
  const double dt = b.time - a.time;
  segment_start_ = b;
  if (!(dt > 0.0)) return;
  time_ = b.time;

  // --- Nominal state: trapezoidal (second-order) integration of the two samples ---------
  const Vec3 w0 = a.gyro - bg_, w1 = b.gyro - bg_;
  const Vec3 f0 = a.accel - ba_, f1 = b.accel - ba_;
  const Vec3 w_mean = 0.5 * (w0 + w1);
  const Vec3 f_mean = 0.5 * (f0 + f1);
  const Mat3 R0 = math::quatToRotation(q_wi_);
  const Vec4 q1 = math::boxPlus(q_wi_, w_mean * dt);
  const Mat3 R1 = math::quatToRotation(q1);
  const Vec3 accel_w = 0.5 * (R0 * f0 + R1 * f1) + Vec3(0.0, 0.0, cfg_.gravity);
  p_ += v_ * dt + 0.5 * accel_w * dt * dt;
  v_ += accel_w * dt;
  q_wi_ = q1;

  // --- Error dynamics of the navigation block (Aether6 ErrorStateEkf::predict) ----------
  const Mat3 R_mid = math::quatToRotation(math::boxPlus(q1, -0.5 * w_mean * dt));
  MatX F = MatX::Zero(kNn, kNn);
  F.block<3, 3>(kIdxPosition, kIdxVelocity) = Mat3::Identity();
  F.block<3, 3>(kIdxVelocity, kIdxAttitude) = -R_mid * math::skew(f_mean);
  F.block<3, 3>(kIdxVelocity, kIdxAccelBias) = -R_mid;
  F.block<3, 3>(kIdxAttitude, kIdxAttitude) = -math::skew(w_mean);
  F.block<3, 3>(kIdxAttitude, kIdxGyroBias) = -Mat3::Identity();

  // The configured white noise is a per-sample sigma (Aether6's sensor parameterisation);
  // the equivalent continuous spectral density is sigma^2 / rate.
  MatX Qc = MatX::Zero(kNn, kNn);
  Qc.block<3, 3>(kIdxVelocity, kIdxVelocity) =
      Mat3::Identity() * cfg_.accel_noise * cfg_.accel_noise / cfg_.imu_rate;
  Qc.block<3, 3>(kIdxAttitude, kIdxAttitude) =
      Mat3::Identity() * cfg_.gyro_noise * cfg_.gyro_noise / cfg_.imu_rate;
  Qc.block<3, 3>(kIdxGyroBias, kIdxGyroBias) =
      Mat3::Identity() * cfg_.gyro_bias_walk * cfg_.gyro_bias_walk;
  Qc.block<3, 3>(kIdxAccelBias, kIdxAccelBias) =
      Mat3::Identity() * cfg_.accel_bias_walk * cfg_.accel_bias_walk;

  MatX Phi, Qd;
  estimation::discretize(F, Qc, dt, cfg_.exact_discretisation, &Phi, &Qd);

  // Block-wise P <- blkdiag(Phi, I) P blkdiag(Phi, I)^T + Q: only the navigation block has
  // dynamics, so the extrinsic/time-offset/clone block is touched only by its process noise.
  P_.topLeftCorner(kNn, kNn) = Phi * P_.topLeftCorner(kNn, kNn) * Phi.transpose() + Qd;
  P_.topRightCorner(kNn, kNr) = Phi * P_.topRightCorner(kNn, kNr);
  P_.bottomLeftCorner(kNr, kNn) = P_.topRightCorner(kNn, kNr).transpose();
  const double q_rot = cfg_.extrinsic_rotation_walk * cfg_.extrinsic_rotation_walk * dt;
  const double q_pos = cfg_.extrinsic_translation_walk * cfg_.extrinsic_translation_walk * dt;
  for (int i = 0; i < 3; ++i) {
    P_(kIdxExtRotation + i, kIdxExtRotation + i) += q_rot;
    P_(kIdxExtTranslation + i, kIdxExtTranslation + i) += q_pos;
  }
  if (cfg_.estimate_time_offset)
    P_(kIdxTimeOffset, kIdxTimeOffset) += cfg_.time_offset_walk * cfg_.time_offset_walk * dt;
  if (cfg_.yaw_gauge_noise > 0.0) {
    const VecX n = yawNullDirection();
    P_ += (cfg_.yaw_gauge_noise * cfg_.yaw_gauge_noise * dt) * (n * n.transpose());
  }
  P_ = math::symmetrize(P_);
}

VecX ExtrinsicCalibrator::yawNullDirection() const {
  // Error-state image of a rotation of the whole world about the gravity axis: every
  // relative measurement is invariant to it.
  const Vec3 ez(0.0, 0.0, 1.0);
  VecX n = VecX::Zero(kNx);
  n.segment<3>(kIdxPosition) = ez.cross(p_);
  n.segment<3>(kIdxVelocity) = ez.cross(v_);
  n.segment<3>(kIdxAttitude) = math::quatToRotation(q_wi_).transpose() * ez;
  if (has_clone_) {
    n.segment<3>(kIdxCloneAttitude) = math::quatToRotation(clone_q_).transpose() * ez;
    n.segment<3>(kIdxClonePosition) = ez.cross(clone_p_);
  }
  return n;
}

void ExtrinsicCalibrator::fuseFrame(const StereoFrame& frame, const Vec3& gyro_at_frame) {
  FrameReport report;
  report.time = time_;
  report.observations = static_cast<int>(frame.observations.size());

  if (!has_clone_) {
    cloneCurrentPose(gyro_at_frame - bg_);
    clone_observations_ = frame.observations;
    if (callback_) callback_(report);
    return;
  }

  const StereoIntrinsics& cam = cfg_.camera;
  const Mat3 R_ic = math::quatToRotation(q_ic_);
  ViewPose anchor{math::quatToRotation(clone_q_), clone_p_, clone_omega_, clone_v_};
  ViewPose current{math::quatToRotation(q_wi_), p_, gyro_at_frame - bg_, v_};
  const double var_px = cfg_.pixel_sigma * cfg_.pixel_sigma;

  // Landmarks in both frames (both observation lists are sorted by id).
  std::vector<std::pair<std::size_t, std::size_t>> matches;
  {
    std::size_t i = 0, j = 0;
    const auto& A = clone_observations_;
    const auto& B = frame.observations;
    while (i < A.size() && j < B.size()) {
      if (A[i].id < B[j].id) {
        ++i;
      } else if (B[j].id < A[i].id) {
        ++j;
      } else {
        matches.emplace_back(i++, j++);
      }
    }
  }
  report.matched = static_cast<int>(matches.size());

  MatX H_stack(5 * static_cast<Eigen::Index>(matches.size()), kNx);
  VecX r_stack(5 * static_cast<Eigen::Index>(matches.size()));
  int used = 0;

  using Mat83 = Eigen::Matrix<double, 8, 3>;
  using Vec8 = Eigen::Matrix<double, 8, 1>;
  for (const auto& [ia, ic] : matches) {
    const StereoObservation& oa = clone_observations_[ia];
    const StereoObservation& oc = frame.observations[ic];
    Vec8 z;
    z << oa.left, oa.right, oc.left, oc.right;

    // Landmark linearisation point: stereo triangulation at the anchor, refined by
    // Gauss-Newton over all four views with the current state estimate.
    Vec3 P_anchor;
    if (!cam.triangulate(oa.left, oa.right, &P_anchor, cfg_.min_disparity)) continue;
    Vec3 f = anchor.p_wi + anchor.R_wi * (R_ic * P_anchor + p_ic_);
    StereoViewJacobians va, vc;
    bool ok = true;
    for (int it = 0; it <= cfg_.triangulation_iterations; ++it) {
      va = stereoView(cam, anchor, R_ic, p_ic_, f);
      vc = stereoView(cam, current, R_ic, p_ic_, f);
      if (!va.valid || !vc.valid) {
        ok = false;
        break;
      }
      if (it == cfg_.triangulation_iterations) break;
      Vec8 r;
      r << z.head<4>() - va.h, z.tail<4>() - vc.h;
      Mat83 Jf;
      Jf << va.d_feature, vc.d_feature;
      f += (Jf.transpose() * Jf).ldlt().solve(Jf.transpose() * r);
    }
    if (!ok) continue;

    Vec8 r;
    r << z.head<4>() - va.h, z.tail<4>() - vc.h;
    Mat83 Hf;
    Hf << va.d_feature, vc.d_feature;
    Eigen::Matrix<double, 8, kNx> Hx = Eigen::Matrix<double, 8, kNx>::Zero();
    Hx.block<4, 3>(0, kIdxCloneAttitude) = va.d_att;
    Hx.block<4, 3>(0, kIdxClonePosition) = va.d_pos;
    Hx.block<4, 3>(0, kIdxExtRotation) = va.d_ext_rot;
    Hx.block<4, 3>(0, kIdxExtTranslation) = va.d_ext_pos;
    Hx.block<4, 3>(4, kIdxAttitude) = vc.d_att;
    Hx.block<4, 3>(4, kIdxPosition) = vc.d_pos;
    Hx.block<4, 3>(4, kIdxExtRotation) = vc.d_ext_rot;
    Hx.block<4, 3>(4, kIdxExtTranslation) = vc.d_ext_pos;
    if (cfg_.estimate_time_offset) {
      Hx.block<4, 1>(0, kIdxTimeOffset) = va.d_time;
      Hx.block<4, 1>(4, kIdxTimeOffset) = vc.d_time;
    }

    // Marginalise the landmark: project onto the left null space of Hf. The projector is
    // orthonormal, so isotropic pixel noise stays isotropic with the same variance.
    const Eigen::HouseholderQR<Mat83> qr(Hf);
    const Eigen::Matrix<double, 8, 8> Q = qr.householderQ();
    const Eigen::Matrix<double, 8, 5> N = Q.rightCols<5>();
    const Eigen::Matrix<double, 5, 1> r_o = N.transpose() * r;
    const Eigen::Matrix<double, 5, kNx> H_o = N.transpose() * Hx;

    // Per-landmark Mahalanobis gate (rejects bad correspondences).
    if (cfg_.landmark_gate > 0.0) {
      Eigen::Matrix<double, 5, 5> S = H_o * P_ * H_o.transpose();
      S.diagonal().array() += var_px;
      const double d2 = r_o.dot(S.ldlt().solve(r_o));
      if (!(d2 <= cfg_.landmark_gate)) {
        ++report.gated;
        continue;
      }
    }
    H_stack.middleRows<5>(5 * used) = H_o;
    r_stack.segment<5>(5 * used) = r_o;
    ++used;
  }
  report.used = used;

  if (used >= cfg_.min_landmarks) {
    const Eigen::Index rows = 5 * used;
    MatX H = H_stack.topRows(rows);
    VecX r = r_stack.head(rows);
    double residual_outside = 0.0;  // part of the residual no state can explain
    if (rows > kNx) {
      // Compress the stack: H = Q [T; 0], so the update only needs T and Q^T r.
      const Eigen::HouseholderQR<MatX> qr(H);
      const VecX qtr = qr.householderQ().adjoint() * r;
      H = qr.matrixQR().topRows(kNx).triangularView<Eigen::Upper>();
      residual_outside = qtr.tail(rows - kNx).squaredNorm();
      r = qtr.head(kNx);
    }
    const MatX R = MatX::Identity(H.rows(), H.rows()) * var_px;
    const Eigen::Matrix<double, 6, 6> P_ext_prior = P_.block<6, 6>(kIdxExtRotation, kIdxExtRotation);
    const estimation::UpdateResult upd = estimation::josephUpdate(P_, r, H, R, 0.0);
    if (upd.accepted) {
      const Eigen::Matrix<double, 6, 6> reduction =
          P_ext_prior - P_.block<6, 6>(kIdxExtRotation, kIdxExtRotation);
      injectAndReset(upd.dx);
      report.updated = true;
      const double dof = static_cast<double>(rows);
      const double nis = upd.nis + residual_outside / var_px;
      report.nis_per_dof = nis / dof;

      const VecX ext_var = P_.diagonal().segment<6>(kIdxExtRotation);
      const bool armed =
          ext_var.head<3>().maxCoeff() <
              cfg_.change_arm_rotation_sigma * cfg_.change_arm_rotation_sigma &&
          ext_var.tail<3>().maxCoeff() <
              cfg_.change_arm_translation_sigma * cfg_.change_arm_translation_sigma;
      if (cfg_.detect_change && !armed) change_window_.clear();
      if (cfg_.detect_change && armed) {
        change_window_.emplace_back(upd.dx.segment<6>(kIdxExtRotation), reduction);
        if (static_cast<int>(change_window_.size()) > cfg_.change_window)
          change_window_.pop_front();
        if (static_cast<int>(change_window_.size()) == cfg_.change_window) {
          Eigen::Matrix<double, 6, 1> sum_dx = Eigen::Matrix<double, 6, 1>::Zero();
          Eigen::Matrix<double, 6, 6> sum_cov = Eigen::Matrix<double, 6, 6>::Zero();
          for (const auto& [d, c] : change_window_) {
            sum_dx += d;
            sum_cov += c;
          }
          // A floor keeps the test defined when the extrinsic has no process noise and the
          // window carries almost no information.
          sum_cov.diagonal().array() += 1e-14;
          report.change_statistic = sum_dx.dot(sum_cov.ldlt().solve(sum_dx));
          if (report.change_statistic > cfg_.change_threshold) {
            reopenExtrinsic();
            change_window_.clear();
            report.change_detected = true;
          }
        }
      }
    }
  }

  // The current pose becomes the anchor of the next frame.
  if (cfg_.disjoint_pairs) {
    has_clone_ = false;
  } else {
    cloneCurrentPose(gyro_at_frame - bg_);
    clone_observations_ = frame.observations;
  }
  if (callback_) callback_(report);
}

void ExtrinsicCalibrator::injectAndReset(const VecX& dx) {
  p_ += dx.segment<3>(kIdxPosition);
  v_ += dx.segment<3>(kIdxVelocity);
  q_wi_ = math::boxPlus(q_wi_, dx.segment<3>(kIdxAttitude));
  bg_ += dx.segment<3>(kIdxGyroBias);
  ba_ += dx.segment<3>(kIdxAccelBias);
  q_ic_ = math::boxPlus(q_ic_, dx.segment<3>(kIdxExtRotation));
  p_ic_ += dx.segment<3>(kIdxExtTranslation);
  if (cfg_.estimate_time_offset)
    td_ = std::clamp(td_ + dx(kIdxTimeOffset), -cfg_.max_time_offset, cfg_.max_time_offset);
  clone_q_ = math::boxPlus(clone_q_, dx.segment<3>(kIdxCloneAttitude));
  clone_p_ += dx.segment<3>(kIdxClonePosition);

  estimation::applyAttitudeReset(P_, dx, {kIdxAttitude, kIdxExtRotation, kIdxCloneAttitude});
}

void ExtrinsicCalibrator::cloneCurrentPose(const Vec3& omega) {
  clone_q_ = q_wi_;
  clone_p_ = p_;
  clone_omega_ = omega;
  clone_v_ = v_;
  has_clone_ = true;

  // P <- J P J^T where J copies the current-pose error into the clone slots (the old clone
  // is discarded).
  MatX J = MatX::Identity(kNx, kNx);
  J.block<6, kNx>(kIdxCloneAttitude, 0).setZero();
  J.block<3, 3>(kIdxCloneAttitude, kIdxAttitude) = Mat3::Identity();
  J.block<3, 3>(kIdxClonePosition, kIdxPosition) = Mat3::Identity();
  P_ = math::symmetrize(J * P_ * J.transpose());
}

void ExtrinsicCalibrator::reopenExtrinsic() {
  // Forget what the filter knew about the extrinsic: drop its correlations and re-open its
  // uncertainty, so the next updates re-estimate it around the current value.
  for (int b : {kIdxExtRotation, kIdxExtTranslation}) {
    P_.middleRows<3>(b).setZero();
    P_.middleCols<3>(b).setZero();
  }
  const double sr = cfg_.reopen_rotation_sigma;
  const double st = cfg_.reopen_translation_sigma;
  P_.block<3, 3>(kIdxExtRotation, kIdxExtRotation) = Mat3::Identity() * sr * sr;
  P_.block<3, 3>(kIdxExtTranslation, kIdxExtTranslation) = Mat3::Identity() * st * st;
  ++change_events_;
}

}  // namespace aether::calib
