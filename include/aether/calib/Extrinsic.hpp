/// \file Extrinsic.hpp
/// \brief IMU-camera extrinsic transform and the error metrics used to score it.
#pragma once

#include <cmath>

#include "aether/core/Constants.hpp"
#include "aether/core/Types.hpp"
#include "aether/math/Rotation.hpp"

namespace aether::calib {

/// \brief Rigid transform of the left camera relative to the IMU, \f$T_{IC}\f$.
///
/// `q_ic` rotates camera-frame vectors into the IMU frame and `p_ic` is the camera optical
/// centre expressed in the IMU frame (the lever arm). A point maps as
/// \f$P^I = R_{IC}P^C + p_{IC}\f$. (Kalibr's `T_cam_imu` is the inverse of this transform.)
struct Extrinsic {
  Vec4 q_ic = Vec4(1, 0, 0, 0);  ///< \f$q_{IC}\f$ [-]
  Vec3 p_ic = Vec3::Zero();      ///< \f$p_{IC}\f$ [m]

  /// Rotation matrix \f$R_{IC}\f$.
  Mat3 rotation() const { return math::quatToRotation(q_ic); }
};

/// \brief Rotation error \f$\delta\phi = \mathrm{Log}(R_{est}^\top R_{true})\f$ [rad], i.e. the
/// right (camera-frame) perturbation with \f$R_{true} = R_{est}\,\mathrm{Exp}(\delta\phi)\f$ —
/// the same convention as the filter's error state, so it can be scored against \f$P\f$.
inline Vec3 rotationErrorVector(const Extrinsic& estimate, const Extrinsic& truth) {
  return math::boxMinus(truth.q_ic, estimate.q_ic);
}

/// \brief Total rotation error angle [deg].
inline double rotationErrorDeg(const Extrinsic& estimate, const Extrinsic& truth) {
  return rotationErrorVector(estimate, truth).norm() * constants::kRadToDeg;
}

/// \brief Lever-arm error \f$\|p_{est} - p_{true}\|\f$ [m].
inline double translationError(const Extrinsic& estimate, const Extrinsic& truth) {
  return (estimate.p_ic - truth.p_ic).norm();
}

/// \brief The nominal "CAD" mount: camera looking along the IMU's forward axis.
///
/// Camera axes in IMU (forward-right-down) coordinates: \f$x_C\f$ = right = \f$y_I\f$,
/// \f$y_C\f$ = down = \f$z_I\f$, \f$z_C\f$ = forward = \f$x_I\f$.
inline Vec4 forwardLookingMount() {
  Mat3 R;
  R << 0.0, 0.0, 1.0,
       1.0, 0.0, 0.0,
       0.0, 1.0, 0.0;
  const Eigen::Quaterniond q(R);
  return math::quatNormalize(Vec4(q.w(), q.x(), q.y(), q.z()));
}

}  // namespace aether::calib
