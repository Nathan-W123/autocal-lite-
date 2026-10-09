// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
/// \file Rotation.hpp
/// \brief Quaternion / rotation-matrix / Euler-angle utilities (Hamilton convention).
///
/// All quaternions in Aether6 are Hamilton quaternions stored as \f$[w,x,y,z]\f$ and
/// interpreted as \f$q_{nb}\f$: they rotate a **body** vector into the **NED** frame,
/// \f$v^n = R(q)\,v^b\f$. Euler angles are the aerospace 3-2-1 (yaw-pitch-roll) sequence
/// \f$(\psi,\theta,\phi)\f$ applied as \f$R = R_z(\psi)R_y(\theta)R_x(\phi)\f$.
#pragma once

#include <cmath>

#include "aether/core/Constants.hpp"
#include "aether/core/Types.hpp"

namespace aether::math {

/// \brief Skew-symmetric cross-product matrix, \f$[a]_\times b = a \times b\f$.
inline Mat3 skew(const Vec3& a) {
  Mat3 s;
  s << 0.0, -a.z(), a.y(),
       a.z(), 0.0, -a.x(),
      -a.y(), a.x(), 0.0;
  return s;
}

/// \brief Hamilton quaternion product \f$a \otimes b\f$ for \f$[w,x,y,z]\f$ storage.
inline Vec4 quatMultiply(const Vec4& a, const Vec4& b) {
  const double aw = a(0), ax = a(1), ay = a(2), az = a(3);
  const double bw = b(0), bx = b(1), by = b(2), bz = b(3);
  Vec4 r;
  r(0) = aw * bw - ax * bx - ay * by - az * bz;
  r(1) = aw * bx + ax * bw + ay * bz - az * by;
  r(2) = aw * by - ax * bz + ay * bw + az * bx;
  r(3) = aw * bz + ax * by - ay * bx + az * bw;
  return r;
}

/// \brief Quaternion conjugate (inverse for unit quaternions).
inline Vec4 quatConjugate(const Vec4& q) { return Vec4(q(0), -q(1), -q(2), -q(3)); }

/// \brief Return \a q rescaled to unit norm, with a sign convention of \f$q_w \ge 0\f$.
/// Falls back to the identity quaternion if the input is numerically degenerate.
inline Vec4 quatNormalize(const Vec4& q) {
  const double n = q.norm();
  if (!(n > 1e-12)) return Vec4(1.0, 0.0, 0.0, 0.0);
  Vec4 out = q / n;
  if (out(0) < 0.0) out = -out;
  return out;
}

/// \brief Rotation matrix \f$R_{nb}\f$ from the quaternion: \f$v^n = R_{nb} v^b\f$.
inline Mat3 quatToRotation(const Vec4& q) {
  const Vec4 u = quatNormalize(q);
  const double w = u(0), x = u(1), y = u(2), z = u(3);
  Mat3 R;
  R << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
       2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
       2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y);
  return R;
}

/// \brief Rotate a body-frame vector into NED: \f$v^n = R_{nb} v^b\f$ (quaternion sandwich).
inline Vec3 rotateBodyToNed(const Vec4& q, const Vec3& vb) {
  const Vec4 qv(0.0, vb.x(), vb.y(), vb.z());
  const Vec4 r = quatMultiply(quatMultiply(q, qv), quatConjugate(q));
  return Vec3(r(1), r(2), r(3));
}

/// \brief Rotate a NED vector into body axes: \f$v^b = R_{nb}^\top v^n\f$.
inline Vec3 rotateNedToBody(const Vec4& q, const Vec3& vn) {
  return rotateBodyToNed(quatConjugate(q), vn);
}

/// \brief Convert 3-2-1 Euler angles (roll \f$\phi\f$, pitch \f$\theta\f$, yaw \f$\psi\f$, all rad)
/// to the quaternion \f$q_{nb}\f$.
inline Vec4 eulerToQuat(double roll, double pitch, double yaw) {
  const double cr = std::cos(0.5 * roll), sr = std::sin(0.5 * roll);
  const double cp = std::cos(0.5 * pitch), sp = std::sin(0.5 * pitch);
  const double cy = std::cos(0.5 * yaw), sy = std::sin(0.5 * yaw);
  Vec4 q;
  q(0) = cr * cp * cy + sr * sp * sy;
  q(1) = sr * cp * cy - cr * sp * sy;
  q(2) = cr * sp * cy + sr * cp * sy;
  q(3) = cr * cp * sy - sr * sp * cy;
  return quatNormalize(q);
}

/// \brief Convert \f$q_{nb}\f$ to 3-2-1 Euler angles \f$(\phi,\theta,\psi)\f$ in rad.
/// Pitch is clamped at \f$\pm\pi/2\f$ to keep the conversion well defined at the singularity.
inline Vec3 quatToEuler(const Vec4& q) {
  const Vec4 u = quatNormalize(q);
  const double w = u(0), x = u(1), y = u(2), z = u(3);
  const double sinp = 2.0 * (w * y - z * x);
  const double clamped = sinp > 1.0 ? 1.0 : (sinp < -1.0 ? -1.0 : sinp);
  Vec3 e;
  e(0) = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));  // roll
  e(1) = std::asin(clamped);                                              // pitch
  e(2) = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));  // yaw
  return e;
}

/// \brief Exponential map from a body rotation vector \f$\delta\theta\f$ [rad] to a quaternion.
/// Uses a Taylor expansion near zero to stay accurate and branch-safe.
inline Vec4 expMapQuat(const Vec3& dtheta) {
  const double angle = dtheta.norm();
  Vec4 q;
  if (angle < 1e-8) {
    q << 1.0, 0.5 * dtheta.x(), 0.5 * dtheta.y(), 0.5 * dtheta.z();
  } else {
    const double s = std::sin(0.5 * angle) / angle;
    q << std::cos(0.5 * angle), s * dtheta.x(), s * dtheta.y(), s * dtheta.z();
  }
  return quatNormalize(q);
}

/// \brief Logarithmic map: smallest body rotation vector \f$\delta\theta\f$ [rad] with
/// \f$q = \exp(\delta\theta/2)\f$.
inline Vec3 logMapQuat(const Vec4& q) {
  const Vec4 u = quatNormalize(q);  // enforces w >= 0, i.e. the short way round
  const Vec3 v(u(1), u(2), u(3));
  const double nv = v.norm();
  if (nv < 1e-12) return Vec3::Zero();
  const double angle = 2.0 * std::atan2(nv, u(0));
  return v * (angle / nv);
}

/// \brief Right-multiplicative "boxplus": \f$q \boxplus \delta\theta = q \otimes \exp(\delta\theta/2)\f$.
inline Vec4 boxPlus(const Vec4& q, const Vec3& dtheta) {
  return quatNormalize(quatMultiply(q, expMapQuat(dtheta)));
}

/// \brief Right-multiplicative "boxminus": the \f$\delta\theta\f$ with \f$q_a = q_b \boxplus \delta\theta\f$.
inline Vec3 boxMinus(const Vec4& qa, const Vec4& qb) {
  return logMapQuat(quatMultiply(quatConjugate(qb), qa));
}

/// \brief Wrap an angle to \f$(-\pi, \pi]\f$ [rad].
inline double wrapPi(double a) {
  constexpr double kTwoPi = 2.0 * constants::kPi;
  a = std::fmod(a + constants::kPi, kTwoPi);
  if (a <= 0.0) a += kTwoPi;
  return a - constants::kPi;
}

}  // namespace aether::math
