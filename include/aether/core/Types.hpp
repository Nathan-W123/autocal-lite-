/// \file Types.hpp
/// \brief Fundamental vector/matrix aliases.
///
/// Reused from Aether6 (`include/aether/core/Types.hpp`). The aircraft-specific types
/// (13-state rigid-body vector, control inputs, wrench) are not needed by the calibrator and
/// were dropped; the aliases and conventions below are unchanged.
///
/// \par Coordinate frames used by AutoCal-lite
/// - **World (W)**: local-level North-East-Down frame, exactly Aether6's NED frame. Gravity is
///   \f$g^W = [0, 0, +g]^\top\f$. Its origin and heading are arbitrary: the calibrator never
///   needs (and cannot observe) absolute position or yaw.
/// - **IMU (I)**: forward-right-down body frame of the inertial sensor (Aether6's body frame).
/// - **Camera (C)**: left camera of the stereo pair, OpenCV convention: \f$x\f$ right,
///   \f$y\f$ down, \f$z\f$ along the optical axis. The right camera (C1) is the left camera
///   translated by the baseline along \f$+x_C\f$ (rectified pair).
///
/// \par Attitude convention
/// Hamilton quaternions stored \f$[w,x,y,z]\f$; \f$q_{AB}\f$ rotates a B-frame vector into
/// frame A, \f$v^A = R(q_{AB})\,v^B\f$ (Aether6's \f$q_{nb}\f$ is \f$q_{WI}\f$ here).
#pragma once

#include <Eigen/Dense>

namespace aether {

using Vec2 = Eigen::Vector2d;   ///< 2-vector, double precision.
using Vec3 = Eigen::Vector3d;   ///< 3-vector, double precision.
using Vec4 = Eigen::Vector4d;   ///< 4-vector, double precision.
using Mat3 = Eigen::Matrix3d;   ///< 3x3 matrix, double precision.
using VecX = Eigen::VectorXd;   ///< Dynamically sized column vector.
using MatX = Eigen::MatrixXd;   ///< Dynamically sized matrix.

}  // namespace aether
