#include "aether/calib/StereoCamera.hpp"

namespace aether::calib {

Vec2 StereoIntrinsics::project(const Vec3& P) const {
  const double iz = 1.0 / P.z();
  return Vec2(fx * P.x() * iz + cx, fy * P.y() * iz + cy);
}

Eigen::Matrix<double, 2, 3> StereoIntrinsics::projectionJacobian(const Vec3& P) const {
  const double iz = 1.0 / P.z();
  const double iz2 = iz * iz;
  Eigen::Matrix<double, 2, 3> J;
  J << fx * iz, 0.0, -fx * P.x() * iz2,
       0.0, fy * iz, -fy * P.y() * iz2;
  return J;
}

bool StereoIntrinsics::inImage(const Vec2& uv, double margin) const {
  return uv.x() >= margin && uv.x() <= width - 1.0 - margin && uv.y() >= margin &&
         uv.y() <= height - 1.0 - margin;
}

bool StereoIntrinsics::triangulate(const Vec2& left, const Vec2& right, Vec3* P_left,
                                   double min_disparity) const {
  const double disparity = left.x() - right.x();
  if (!(disparity > min_disparity)) return false;
  const double Z = fx * baseline / disparity;
  // Average the two rows: in a rectified pair both carry the same y information.
  const double v = 0.5 * (left.y() + right.y());
  *P_left = Vec3((left.x() - cx) * Z / fx, (v - cy) * Z / fy, Z);
  return true;
}

}  // namespace aether::calib
