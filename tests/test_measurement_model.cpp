/// \file test_measurement_model.cpp
/// \brief Stereo reprojection model: predicted pixels, validity, every analytic Jacobian of
///        calib::stereoView against central differences, and the landmark marginalisation
///        (left null-space projection) used by the extrinsic calibrator.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "aether/calib/Extrinsic.hpp"
#include "aether/calib/MeasurementModel.hpp"
#include "aether/calib/StereoCamera.hpp"
#include "aether/core/Constants.hpp"
#include "aether/math/NumericalDiff.hpp"
#include "aether/math/Rotation.hpp"
#include "aether/util/Random.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::StereoIntrinsics;
using calib::StereoViewJacobians;
using calib::ViewPose;

/// Intrinsics with fx != fy and an off-centre principal point, so a swapped focal length or
/// principal-point coordinate cannot cancel out.
StereoIntrinsics testCamera() {
  StereoIntrinsics c;
  c.fx = 410.0;
  c.fy = 395.0;
  c.cx = 315.5;
  c.cy = 242.25;
  c.width = 640;
  c.height = 480;
  c.baseline = 0.12;
  return c;
}

/// \f$\mathrm{Exp}(w)\f$ as a rotation matrix, with the library's exponential map.
Mat3 expR(const Vec3& w) { return math::quatToRotation(math::expMapQuat(w)); }

/// One linearisation point of the stereo model, plus the left-camera point it was built from.
struct Config {
  ViewPose pose;
  Mat3 R_ic = Mat3::Identity();
  Vec3 p_ic = Vec3::Zero();
  Vec3 f_w = Vec3::Zero();
  Vec3 P_left = Vec3::Zero();  ///< The landmark in the left camera frame (ground truth).
};

/// World point whose left-camera coordinates are \a P_left: the inverse of the model,
/// \f$f^W = p_{WI} + R_{WI}(R_{IC}P + p_{IC})\f$.
Vec3 landmarkFromCamera(const ViewPose& pose, const Mat3& R_ic, const Vec3& p_ic,
                        const Vec3& P_left) {
  return pose.p_wi + pose.R_wi * (R_ic * P_left + p_ic);
}

/// Uniform draw with magnitude in [lo, hi] and a random sign.
double signedUniform(util::Rng& rng, double lo, double hi) {
  const double m = rng.uniform(lo, hi);
  return rng.uniform() < 0.5 ? -m : m;
}

/// A generic, valid configuration: arbitrary IMU attitude, the CAD forward-looking mount
/// composed with a few degrees of extra rotation, a lever arm of up to 10 cm, nonzero body rate
/// and velocity on every axis, and a landmark in front of both cameras at 0.5-20 m depth.
Config randomConfig(util::Rng& rng) {
  Config c;
  c.pose.R_wi = math::quatToRotation(
      math::quatNormalize(Vec4(rng.gaussian(), rng.gaussian(), rng.gaussian(), rng.gaussian())));
  c.pose.p_wi = Vec3(rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(-2.0, 0.0));
  c.pose.omega = Vec3(signedUniform(rng, 0.2, 1.0), signedUniform(rng, 0.2, 1.0),
                      signedUniform(rng, 0.2, 1.0));
  c.pose.v_wi = Vec3(signedUniform(rng, 0.3, 2.0), signedUniform(rng, 0.3, 2.0),
                     signedUniform(rng, 0.3, 2.0));
  const double d = 4.0 * constants::kDegToRad;
  const Vec3 extra(signedUniform(rng, 0.5 * d, d), signedUniform(rng, 0.5 * d, d),
                   signedUniform(rng, 0.5 * d, d));
  c.R_ic = math::quatToRotation(math::boxPlus(calib::forwardLookingMount(), extra));
  c.p_ic = Vec3(rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1));
  // Depth log-uniform in [0.5, 20] m; bearing within (roughly) the 640x480 field of view.
  const double Z = 0.5 * std::exp(rng.uniform(0.0, std::log(40.0)));
  c.P_left = Vec3(rng.uniform(-0.7, 0.7) * Z, rng.uniform(-0.55, 0.55) * Z, Z);
  c.f_w = landmarkFromCamera(c.pose, c.R_ic, c.p_ic, c.P_left);
  return c;
}

std::vector<Config> randomConfigs(std::uint64_t seed, int n) {
  util::Rng rng(seed);
  std::vector<Config> out;
  for (int i = 0; i < n; ++i) out.push_back(randomConfig(rng));
  return out;
}

/// Central-difference step [rad, m or s]. With x_scale = 1 and x = 0, numericalJacobian uses
/// exactly h = 1e-6 for every component.
constexpr double kStep = 1e-6;

/// Relative tolerance on each Jacobian column, \f$\|a_j - n_j\| \le 10^{-5}\max(\|n_j\|, 1)\f$.
///
/// Error budget of the central difference with h = 1e-6:
/// - round-off: pixels are <= ~1500 px and accurate to a few ulps (~1e-12 px including the
///   rotation-matrix products), so (fp - fm) / 2h carries <= ~1e-6 px/unit;
/// - truncation: h^2/6 |h'''|, with |h'''| <= ~1e5 px/unit^3 at the nearest (0.5 m) points,
///   i.e. <= ~1e-8 px/unit.
/// Jacobian columns are >= ~10 px/unit (fx / 20 m for translations, more for rotations), so
/// the numerical error is <= ~1e-7 relative: 1e-5 leaves a 100x margin, while any modelling
/// mistake (sign, missing term, left vs right perturbation) is an O(1) relative error. The
/// 1 px/unit floor only matters for a near-zero column, where it allows 1e-5 px/unit absolute.
constexpr double kRelTol = 1e-5;
constexpr double kColumnFloor = 1.0;

/// Compare an analytic Jacobian with a numerical one column by column.
void checkJacobian(const char* name, const MatX& analytic, const MatX& numeric) {
  INFO("Jacobian " << name);
  REQUIRE(analytic.rows() == numeric.rows());
  REQUIRE(analytic.cols() == numeric.cols());
  for (Eigen::Index j = 0; j < analytic.cols(); ++j) {
    const double err = (analytic.col(j) - numeric.col(j)).norm();
    const double scale = std::max(numeric.col(j).norm(), kColumnFloor);
    INFO("column " << j << "\n  analytic " << analytic.col(j).transpose() << "\n  numeric  "
                   << numeric.col(j).transpose() << "\n  relative error " << err / scale);
    CHECK(err <= kRelTol * scale);
  }
}

/// Pixels of a perturbed view; clears *all_valid if the perturbed view is invalid.
VecX pixels(const StereoIntrinsics& cam, const ViewPose& pose, const Mat3& R_ic,
            const Vec3& p_ic, const Vec3& f_w, bool* all_valid) {
  const StereoViewJacobians v = calib::stereoView(cam, pose, R_ic, p_ic, f_w);
  if (!v.valid) *all_valid = false;
  return v.h;
}

/// Independent reference for the predicted pixels: transform the landmark into the left
/// camera by hand and apply the pinhole formula to it and to its right-camera image.
Eigen::Matrix<double, 4, 1> referencePixels(const StereoIntrinsics& cam, const ViewPose& pose,
                                            const Mat3& R_ic, const Vec3& p_ic,
                                            const Vec3& f_w) {
  const Vec3 q = pose.R_wi.transpose() * (f_w - pose.p_wi);
  const Vec3 P = R_ic.transpose() * (q - p_ic);
  Eigen::Matrix<double, 4, 1> h;
  h << cam.fx * P.x() / P.z() + cam.cx, cam.fy * P.y() / P.z() + cam.cy,
      cam.fx * (P.x() - cam.baseline) / P.z() + cam.cx, cam.fy * P.y() / P.z() + cam.cy;
  return h;
}

}  // namespace

TEST_CASE("stereoView predicts the left and right pinhole projections of the landmark",
          "[calib][measurement]") {
  const StereoIntrinsics cam = testCamera();
  for (const Config& c : randomConfigs(20240611, 40)) {
    const StereoViewJacobians v = calib::stereoView(cam, c.pose, c.R_ic, c.p_ic, c.f_w);
    INFO("P_left = " << c.P_left.transpose());
    REQUIRE(v.valid);

    // Built from the inverse transform, so the world->camera chain must give back P_left and
    // its pixels: [pi(P); pi(P - [b,0,0])].
    const Eigen::Matrix<double, 4, 1> ref = referencePixels(cam, c.pose, c.R_ic, c.p_ic, c.f_w);
    Eigen::Matrix<double, 4, 1> from_truth;
    from_truth << cam.project(c.P_left), cam.project(c.P_left - Vec3(cam.baseline, 0.0, 0.0));
    // Round-off only: the world round trip loses ~eps * |f_w| ~ 1e-15 m, i.e. ~1e-12 px.
    CHECK((v.h - ref).norm() <= 1e-9);
    CHECK((v.h - from_truth).norm() <= 1e-9);

    // Rectified pair: same image row, disparity fx * b / Z.
    CHECK(v.h(3) == Approx(v.h(1)).margin(1e-9));
    CHECK(v.h(0) - v.h(2) == Approx(cam.fx * cam.baseline / c.P_left.z()).epsilon(1e-9));
  }
}

TEST_CASE("stereoView Jacobians match central differences under the right-perturbation "
          "convention",
          "[calib][measurement][jacobian]") {
  const StereoIntrinsics cam = testCamera();
  int n_configs = 0;
  for (const Config& c : randomConfigs(7, 25)) {
    ++n_configs;
    const StereoViewJacobians v = calib::stereoView(cam, c.pose, c.R_ic, c.p_ic, c.f_w);
    INFO("configuration " << n_configs << ", P_left = " << c.P_left.transpose()
                          << ", omega = " << c.pose.omega.transpose()
                          << ", v = " << c.pose.v_wi.transpose());
    REQUIRE(v.valid);
    bool all_valid = true;
    const VecX zero3 = VecX::Zero(3);

    // d_att: R_wi -> R_wi Exp(dtheta).
    const math::VectorFunction f_att = [&](const VecX& x) -> VecX {
      ViewPose p = c.pose;
      p.R_wi = c.pose.R_wi * expR(x);
      return pixels(cam, p, c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    checkJacobian("d_att", v.d_att, math::numericalJacobian(f_att, zero3, kStep, 1.0));

    // The same perturbation applied through the quaternion boxplus of the filter's
    // injection step must agree with the matrix form.
    const Vec4 q_wi = [&] {
      const Eigen::Quaterniond qe(c.pose.R_wi);
      return math::quatNormalize(Vec4(qe.w(), qe.x(), qe.y(), qe.z()));
    }();
    const math::VectorFunction f_att_q = [&](const VecX& x) -> VecX {
      ViewPose p = c.pose;
      p.R_wi = math::quatToRotation(math::boxPlus(q_wi, Vec3(x)));
      return pixels(cam, p, c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    checkJacobian("d_att (boxPlus)", v.d_att, math::numericalJacobian(f_att_q, zero3, kStep, 1.0));

    // d_pos: p_wi -> p_wi + dp.
    const math::VectorFunction f_pos = [&](const VecX& x) -> VecX {
      ViewPose p = c.pose;
      p.p_wi = c.pose.p_wi + Vec3(x);
      return pixels(cam, p, c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    checkJacobian("d_pos", v.d_pos, math::numericalJacobian(f_pos, zero3, kStep, 1.0));

    // d_ext_rot: R_ic -> R_ic Exp(dphi).
    const math::VectorFunction f_ext_rot = [&](const VecX& x) -> VecX {
      return pixels(cam, c.pose, c.R_ic * expR(x), c.p_ic, c.f_w, &all_valid);
    };
    checkJacobian("d_ext_rot", v.d_ext_rot,
                  math::numericalJacobian(f_ext_rot, zero3, kStep, 1.0));

    // d_ext_pos: p_ic -> p_ic + dp.
    const math::VectorFunction f_ext_pos = [&](const VecX& x) -> VecX {
      return pixels(cam, c.pose, c.R_ic, c.p_ic + Vec3(x), c.f_w, &all_valid);
    };
    checkJacobian("d_ext_pos", v.d_ext_pos,
                  math::numericalJacobian(f_ext_pos, zero3, kStep, 1.0));

    // d_feature: f_w -> f_w + df.
    const math::VectorFunction f_feat = [&](const VecX& x) -> VecX {
      return pixels(cam, c.pose, c.R_ic, c.p_ic, c.f_w + Vec3(x), &all_valid);
    };
    checkJacobian("d_feature", v.d_feature, math::numericalJacobian(f_feat, zero3, kStep, 1.0));

    // d_time: the exposure happened at tau + dt, when the pose was
    // (R_wi Exp(omega dt), p_wi + v dt) with omega and v held fixed.
    const math::VectorFunction f_time = [&](const VecX& x) -> VecX {
      const double dt = x(0);
      ViewPose p = c.pose;
      p.R_wi = c.pose.R_wi * expR(c.pose.omega * dt);
      p.p_wi = c.pose.p_wi + c.pose.v_wi * dt;
      return pixels(cam, p, c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    checkJacobian("d_time", v.d_time,
                  math::numericalJacobian(f_time, VecX::Zero(1), kStep, 1.0));

    CHECK(all_valid);  // every perturbed view stayed in front of the cameras

    // Sensitivity of the check itself: a LEFT perturbation, Exp(dtheta) R, has the Jacobian
    // d_att R^T (resp. d_ext_rot R_ic^T), which differs from the right-perturbation one by
    // O(1) for these non-trivial rotations, so it must fail the tolerance by a wide margin.
    const math::VectorFunction f_att_left = [&](const VecX& x) -> VecX {
      ViewPose p = c.pose;
      p.R_wi = expR(x) * c.pose.R_wi;
      return pixels(cam, p, c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    const MatX left_att = math::numericalJacobian(f_att_left, zero3, kStep, 1.0);
    CHECK((left_att - v.d_att).norm() > 100.0 * kRelTol * v.d_att.norm());
    const math::VectorFunction f_ext_left = [&](const VecX& x) -> VecX {
      return pixels(cam, c.pose, expR(x) * c.R_ic, c.p_ic, c.f_w, &all_valid);
    };
    const MatX left_ext = math::numericalJacobian(f_ext_left, zero3, kStep, 1.0);
    CHECK((left_ext - v.d_ext_rot).norm() > 100.0 * kRelTol * v.d_ext_rot.norm());
  }
  CHECK(n_configs == 25);
}

TEST_CASE("stereoView Jacobians obey the model's structural identities",
          "[calib][measurement][jacobian]") {
  const StereoIntrinsics cam = testCamera();
  for (const Config& c : randomConfigs(99, 20)) {
    const StereoViewJacobians v = calib::stereoView(cam, c.pose, c.R_ic, c.p_ic, c.f_w);
    REQUIRE(v.valid);
    const double scale = v.d_feature.norm();
    // The landmark and the IMU position only enter through f_w - p_wi.
    CHECK((v.d_feature + v.d_pos).norm() <= 1e-12 * scale);
    // Moving the camera by dp in the IMU frame is moving the landmark by -R_wi dp in the
    // world: d_ext_pos = -d_feature R_wi, i.e. d_ext_pos R_wi^T = -d_feature.
    CHECK((v.d_ext_pos * c.pose.R_wi.transpose() + v.d_feature).norm() <= 1e-12 * scale);
    // Chain rule of the exposure-time model: dh/dt = d_att omega + d_pos v.
    const Eigen::Matrix<double, 4, 1> chain = v.d_att * c.pose.omega + v.d_pos * c.pose.v_wi;
    CHECK((v.d_time - chain).norm() <= 1e-12 * std::max(chain.norm(), 1.0));
    // The rows of the right camera share the y-row with the left camera (rectified pair).
    CHECK((v.d_feature.row(3) - v.d_feature.row(1)).norm() <= 1e-12 * scale);
  }
}

TEST_CASE("stereoView is invalid behind the camera or closer than min_depth",
          "[calib][measurement]") {
  const StereoIntrinsics cam = testCamera();
  util::Rng rng(31337);
  Config base = randomConfig(rng);
  // Camera 30 cm ahead of the IMU along its forward axis, so "in front of the IMU" and "in
  // front of the camera" differ: the depth must be measured in the camera frame.
  base.p_ic = Vec3(0.30, 0.02, -0.01);
  auto view = [&](const Vec3& P_left, double min_depth) {
    const Vec3 f = landmarkFromCamera(base.pose, base.R_ic, base.p_ic, P_left);
    return calib::stereoView(cam, base.pose, base.R_ic, base.p_ic, f, min_depth);
  };

  SECTION("behind the camera") {
    CHECK_FALSE(view(Vec3(0.1, -0.2, -2.0), 0.05).valid);
    CHECK_FALSE(view(Vec3(0.0, 0.0, -1e-3), 0.05).valid);
    CHECK_FALSE(view(Vec3(0.3, 0.1, 0.0), 0.05).valid);  // in the image plane
    // Between the IMU and the camera: in front of the IMU (positive forward coordinate in the
    // IMU frame) but 10 cm behind the camera.
    const Vec3 P_between(0.0, 0.0, -0.10);
    const Vec3 q_between = base.R_ic * P_between + base.p_ic;
    REQUIRE(q_between.x() > 0.15);
    CHECK_FALSE(view(P_between, 0.05).valid);
  }

  SECTION("closer than the default min_depth (0.05 m)") {
    CHECK_FALSE(view(Vec3(0.0, 0.0, 0.04), 0.05).valid);
    CHECK_FALSE(view(Vec3(0.01, -0.01, 0.049), 0.05).valid);
    CHECK(view(Vec3(0.0, 0.0, 0.051), 0.05).valid);
    // The default argument is 0.05 m.
    const Vec3 f_close = landmarkFromCamera(base.pose, base.R_ic, base.p_ic, Vec3(0, 0, 0.04));
    const Vec3 f_ok = landmarkFromCamera(base.pose, base.R_ic, base.p_ic, Vec3(0, 0, 0.06));
    CHECK_FALSE(calib::stereoView(cam, base.pose, base.R_ic, base.p_ic, f_close).valid);
    CHECK(calib::stereoView(cam, base.pose, base.R_ic, base.p_ic, f_ok).valid);
  }

  SECTION("custom min_depth") {
    CHECK_FALSE(view(Vec3(0.2, 0.1, 1.9), 2.0).valid);
    CHECK(view(Vec3(0.2, 0.1, 2.1), 2.0).valid);
    CHECK_FALSE(view(Vec3(0.2, 0.1, 9.9), 10.0).valid);
  }

  SECTION("validity does not depend on the image bounds") {
    // In front of both cameras but far outside the 640x480 image: the projection is still
    // defined, and image-bounds culling is the simulator's job, not the model's.
    const StereoViewJacobians v = view(Vec3(5.0, -4.0, 1.0), 0.05);
    CHECK(v.valid);
    CHECK_FALSE(cam.inImage(v.h.head<2>()));
    CHECK(v.h.allFinite());
  }

  // The right camera is a pure +x translation of the left one (no relative rotation), so its
  // depth equals the left depth and "closer than min_depth in either camera" reduces to the
  // left depth; the boundary cases above therefore cover both cameras.
  const Vec3 P(0.4, -0.2, 3.0);
  CHECK(cam.leftToRight(P).z() == P.z());
}

TEST_CASE("the landmark null-space projection of two stereo views removes the landmark error",
          "[calib][measurement][marginalisation]") {
  const StereoIntrinsics cam = testCamera();
  util::Rng rng(4242);
  using Mat83 = Eigen::Matrix<double, 8, 3>;
  using Vec8 = Eigen::Matrix<double, 8, 1>;
  for (int trial = 0; trial < 10; ++trial) {
    // Anchor view from a generic configuration; the current view is the anchor moved by up to
    // 30 cm and rotated by up to ~6 deg, with the landmark >= 2 m away so both views see it.
    Config c = randomConfig(rng);
    c.P_left.z() = std::max(c.P_left.z(), 2.0);
    c.f_w = landmarkFromCamera(c.pose, c.R_ic, c.p_ic, c.P_left);
    const ViewPose anchor = c.pose;
    ViewPose current = c.pose;
    current.R_wi = c.pose.R_wi * expR(Vec3(rng.uniform(-0.06, 0.06), rng.uniform(-0.06, 0.06),
                                            rng.uniform(-0.06, 0.06)));
    current.p_wi = c.pose.p_wi + Vec3(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
                                      rng.uniform(-0.3, 0.3));
    const StereoViewJacobians va = calib::stereoView(cam, anchor, c.R_ic, c.p_ic, c.f_w);
    const StereoViewJacobians vc = calib::stereoView(cam, current, c.R_ic, c.p_ic, c.f_w);
    INFO("trial " << trial << ", P_left = " << c.P_left.transpose());
    REQUIRE(va.valid);
    REQUIRE(vc.valid);

    // Exactly as in ExtrinsicCalibrator::fuseFrame.
    Mat83 Hf;
    Hf << va.d_feature, vc.d_feature;
    const Eigen::HouseholderQR<Mat83> qr(Hf);
    const Eigen::Matrix<double, 8, 8> Q = qr.householderQ();
    const Eigen::Matrix<double, 8, 5> N = Q.rightCols<5>();

    // Hf has full column rank (a single stereo view already fixes the depth), so its left
    // null space is exactly 5-dimensional and spanned by N.
    const Eigen::JacobiSVD<Mat83> svd(Hf);
    REQUIRE(svd.singularValues()(2) > 1e-3 * svd.singularValues()(0));

    // Householder QR is backward stable: ||N^T Hf|| ~ c * eps * ||Hf|| with c = O(10) for an
    // 8x3 matrix, and Q is orthogonal to a few ulps. 1e-12 is a >100x margin on both.
    CHECK((N.transpose() * Hf).norm() <= 1e-12 * Hf.norm());
    CHECK((N.transpose() * N - Eigen::Matrix<double, 5, 5>::Identity()).norm() <= 1e-12);

    // Functional consequence: the projected residual is blind to the landmark position to
    // first order. Moving the landmark by s along a direction changes the stacked pixels by
    // ~|Hf| s (~1 px at s = 1 cm), but the projected change is the second-order remainder,
    // so it must shrink 4x when s is halved (up to an O(s / Z) <= 1% third-order correction).
    auto stacked = [&](const Vec3& f) {
      Vec8 h;
      h << calib::stereoView(cam, anchor, c.R_ic, c.p_ic, f).h,
          calib::stereoView(cam, current, c.R_ic, c.p_ic, f).h;
      return h;
    };
    const Vec8 h0 = stacked(c.f_w);
    for (const Vec3& dir : {Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1),
                            Vec3(0.6, -0.48, 0.64)}) {
      const double s = 0.01;
      const Vec8 dh = stacked(c.f_w + s * dir) - h0;
      const Vec8 dh_half = stacked(c.f_w + 0.5 * s * dir) - h0;
      const double proj = (N.transpose() * dh).norm();
      const double proj_half = (N.transpose() * dh_half).norm();
      INFO("direction " << dir.transpose() << ": |dh| = " << dh.norm() << ", |N^T dh| = "
                        << proj << ", |N^T dh(s/2)| = " << proj_half);
      // Second-order remainder is <= ~(s / Z) |dh| with s / Z <= 0.005.
      CHECK(proj <= 1e-2 * dh.norm());
      REQUIRE(proj_half > 0.0);
      CHECK(proj / proj_half == Approx(4.0).epsilon(0.05));
    }
  }
}
