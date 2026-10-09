#include "aether/calib/MeasurementModel.hpp"

#include "aether/math/Rotation.hpp"

namespace aether::calib {

StereoViewJacobians stereoView(const StereoIntrinsics& cam, const ViewPose& pose,
                               const Mat3& R_ic, const Vec3& p_ic, const Vec3& f_w,
                               double min_depth) {
  StereoViewJacobians out;
  const Mat3 A = R_ic.transpose();
  const Vec3 q = pose.R_wi.transpose() * (f_w - pose.p_wi);
  const Vec3 P = A * (q - p_ic);
  const Vec3 P_r = cam.leftToRight(P);
  // Written as !(z >= min) so a NaN depth is rejected too.
  if (!(P.z() >= min_depth) || !(P_r.z() >= min_depth)) return out;

  out.h.head<2>() = cam.project(P);
  out.h.tail<2>() = cam.project(P_r);

  // The right camera differs from the left by a constant offset in the camera frame, so
  // both share every chain-rule factor after dh/dP.
  Eigen::Matrix<double, 4, 3> J;
  J.topRows<2>() = cam.projectionJacobian(P);
  J.bottomRows<2>() = cam.projectionJacobian(P_r);

  const Eigen::Matrix<double, 4, 3> JA = J * A;
  const Mat3 Rt = pose.R_wi.transpose();
  out.d_att = JA * math::skew(q);
  out.d_pos = -JA * Rt;
  out.d_ext_rot = J * math::skew(P);
  out.d_ext_pos = -JA;
  out.d_feature = JA * Rt;
  out.d_time = JA * (math::skew(q) * pose.omega - Rt * pose.v_wi);
  out.valid = true;
  return out;
}

}  // namespace aether::calib
