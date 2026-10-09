/// \file test_stereo.cpp
/// \brief Rectified pinhole stereo pair: projection, projection Jacobian, triangulation and
///        image-bounds checks.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <vector>

#include "aether/calib/StereoCamera.hpp"
#include "aether/math/NumericalDiff.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::StereoIntrinsics;

/// Intrinsics with fx != fy and a non-centred principal point so that a swapped focal length
/// or principal-point coordinate cannot go unnoticed.
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

/// Points in the left-camera frame at depths from 0.5 m to 25 m, on and off the optical axis.
std::vector<Vec3> testPoints() {
  std::vector<Vec3> pts;
  for (double z : {0.5, 1.0, 2.5, 5.0, 10.0, 25.0}) {
    for (double xn : {-0.6, 0.0, 0.35}) {
      for (double yn : {-0.4, 0.0, 0.2}) pts.emplace_back(xn * z, yn * z, z);
    }
  }
  return pts;
}

}  // namespace

TEST_CASE("pinhole projection follows the documented formula", "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();
  // Optical axis lands on the principal point at any depth.
  for (double z : {0.3, 4.0, 50.0}) {
    const Vec2 uv = cam.project(Vec3(0.0, 0.0, z));
    CHECK(uv.x() == Approx(cam.cx).margin(1e-12));
    CHECK(uv.y() == Approx(cam.cy).margin(1e-12));
  }
  // A ray at 45 deg in x (y) is displaced by exactly fx (fy).
  const Vec2 ux = cam.project(Vec3(3.0, 0.0, 3.0));
  CHECK(ux.x() == Approx(cam.cx + cam.fx));
  CHECK(ux.y() == Approx(cam.cy));
  const Vec2 uy = cam.project(Vec3(0.0, -2.0, 2.0));
  CHECK(uy.x() == Approx(cam.cx));
  CHECK(uy.y() == Approx(cam.cy - cam.fy));
  // Scale invariance along the ray.
  const Vec3 P(0.7, -0.3, 2.2);
  const Vec2 a = cam.project(P);
  const Vec2 b = cam.project(3.7 * P);
  CHECK((a - b).norm() == Approx(0.0).margin(1e-12));
}

TEST_CASE("the right camera is the left camera shifted by the baseline along +x",
          "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();
  for (const Vec3& P : testPoints()) {
    const Vec3 Pr = cam.leftToRight(P);
    CHECK(Pr.x() == P.x() - cam.baseline);
    CHECK(Pr.y() == P.y());
    CHECK(Pr.z() == P.z());
    const Vec2 uvl = cam.project(P);
    const Vec2 uvr = cam.project(Pr);
    INFO("P = " << P.transpose());
    // Rectified pair: same row, disparity fx * b / Z (> 0 for a point in front).
    CHECK(uvr.y() == uvl.y());
    CHECK(uvl.x() - uvr.x() == Approx(cam.fx * cam.baseline / P.z()).epsilon(1e-12));
  }
}

TEST_CASE("project then triangulate recovers the point at every depth", "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();
  for (const Vec3& P : testPoints()) {
    const Vec2 uvl = cam.project(P);
    const Vec2 uvr = cam.project(cam.leftToRight(P));
    Vec3 Pt = Vec3::Constant(std::numeric_limits<double>::quiet_NaN());
    INFO("P = " << P.transpose());
    REQUIRE(cam.triangulate(uvl, uvr, &Pt));
    // Round-off only: the disparity (>= 1.97 px at 25 m) is a difference of pixel values of
    // magnitude <= ~1000 px, so its relative error is <= 1e-13 and so is that of Z = fx b / d.
    CHECK((Pt - P).norm() <= 1e-10 * P.norm());
  }
}

TEST_CASE("triangulation averages the two image rows", "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();
  const Vec3 P(0.4, -0.25, 3.0);
  const Vec2 uvl = cam.project(P);
  const Vec2 uvr = cam.project(cam.leftToRight(P));
  Vec3 clean, split;
  REQUIRE(cam.triangulate(uvl, uvr, &clean));
  // Opposite vertical errors on the two images cancel in the averaged row.
  REQUIRE(cam.triangulate(uvl + Vec2(0.0, 0.7), uvr - Vec2(0.0, 0.7), &split));
  CHECK((split - clean).norm() == Approx(0.0).margin(1e-12));
  // A common vertical offset moves Y by offset * Z / fy and leaves X and Z alone.
  Vec3 shifted;
  REQUIRE(cam.triangulate(uvl + Vec2(0.0, 2.0), uvr + Vec2(0.0, 2.0), &shifted));
  CHECK(shifted.x() == Approx(clean.x()).margin(1e-12));
  CHECK(shifted.z() == Approx(clean.z()).margin(1e-12));
  CHECK(shifted.y() - clean.y() == Approx(2.0 * clean.z() / cam.fy).margin(1e-12));
}

TEST_CASE("triangulation rejects disparities below the minimum", "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();
  const Vec2 left(300.0, 200.0);
  Vec3 P = Vec3::Zero();

  // Default minimum 0.5 px.
  CHECK_FALSE(cam.triangulate(left, left - Vec2(0.3, 0.0), &P));   // below
  CHECK_FALSE(cam.triangulate(left, left, &P));                     // zero
  CHECK_FALSE(cam.triangulate(left, left + Vec2(5.0, 0.0), &P));   // negative (behind)
  REQUIRE(cam.triangulate(left, left - Vec2(0.6, 0.0), &P));       // just above
  CHECK(P.z() == Approx(cam.fx * cam.baseline / 0.6));

  // Custom minimum.
  CHECK_FALSE(cam.triangulate(left, left - Vec2(1.5, 0.0), &P, 2.0));
  REQUIRE(cam.triangulate(left, left - Vec2(2.5, 0.0), &P, 2.0));
  CHECK(P.z() == Approx(cam.fx * cam.baseline / 2.5));

  // Non-finite pixels must be rejected rather than produce a NaN point.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(cam.triangulate(Vec2(nan, 200.0), left, &P));
  CHECK_FALSE(cam.triangulate(left, Vec2(nan, 200.0), &P));
}

TEST_CASE("projection Jacobian matches central differences", "[calib][stereo][jacobian]") {
  const StereoIntrinsics cam = testCamera();
  for (const Vec3& P : testPoints()) {
    const Eigen::Matrix<double, 2, 3> J = cam.projectionJacobian(P);
    const math::VectorFunction f = [&cam](const VecX& x) -> VecX {
      return cam.project(Vec3(x(0), x(1), x(2)));
    };
    // x_scale = 1 makes every step h = 1e-6 * max(|x_j|, 1) >= 1e-6 m. Round-off is then
    // eps * |u| / h <= 2.2e-16 * 1000 / 1e-6 ~ 2e-7 px/m and truncation h^2/6 * |pi'''| is
    // < 1e-8 even at 0.5 m depth, against Jacobian norms >= fx / 25 ~ 16 px/m: a relative
    // tolerance of 1e-6 is >= 5x the numerical error.
    const MatX Jn = math::numericalJacobian(f, VecX(P), 1e-6, 1.0);
    INFO("P = " << P.transpose());
    CHECK((J - Jn).norm() <= 1e-6 * J.norm());
    // Structural zeros and the ray-invariance identity J * P = 0.
    CHECK(J(0, 1) == 0.0);
    CHECK(J(1, 0) == 0.0);
    CHECK((J * P).norm() == Approx(0.0).margin(1e-12 * J.norm() * P.norm()));
  }
}

TEST_CASE("inImage honours the image bounds and the margin", "[calib][stereo]") {
  const StereoIntrinsics cam = testCamera();  // 640 x 480: valid pixels [0, 639] x [0, 479]
  // No margin: the corner pixel centres are inside, anything beyond them is outside.
  CHECK(cam.inImage(Vec2(0.0, 0.0)));
  CHECK(cam.inImage(Vec2(639.0, 479.0)));
  CHECK(cam.inImage(Vec2(320.0, 240.0)));
  CHECK_FALSE(cam.inImage(Vec2(-1e-6, 100.0)));
  CHECK_FALSE(cam.inImage(Vec2(100.0, -1e-6)));
  CHECK_FALSE(cam.inImage(Vec2(639.0 + 1e-6, 100.0)));
  CHECK_FALSE(cam.inImage(Vec2(100.0, 479.0 + 1e-6)));

  // A 2 px margin shrinks the valid box to [2, 637] x [2, 477] on every side.
  const double m = 2.0;
  CHECK(cam.inImage(Vec2(2.0, 2.0), m));
  CHECK(cam.inImage(Vec2(637.0, 477.0), m));
  CHECK_FALSE(cam.inImage(Vec2(1.999, 240.0), m));
  CHECK_FALSE(cam.inImage(Vec2(320.0, 1.999), m));
  CHECK_FALSE(cam.inImage(Vec2(637.001, 240.0), m));
  CHECK_FALSE(cam.inImage(Vec2(320.0, 477.001), m));
  CHECK_FALSE(cam.inImage(Vec2(0.0, 0.0), m));  // inside the image, but within the margin

  // Non-finite pixels are never inside.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_FALSE(cam.inImage(Vec2(nan, 240.0)));
  CHECK_FALSE(cam.inImage(Vec2(320.0, nan)));
}
