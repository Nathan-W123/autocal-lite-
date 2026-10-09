/// \file MeasurementModel.hpp
/// \brief Stereo reprojection model of a world point and its analytic Jacobians.
///
/// A landmark \f$f^W\f$ seen from an IMU pose \f$(R_{WI}, p_{WI})\f$ through the extrinsic
/// \f$(R_{IC}, p_{IC})\f$ lands at
/// \f{align*}{
///   q &= R_{WI}^\top (f^W - p_{WI})            && \text{point in the IMU frame}
///   \\ P &= R_{IC}^\top (q - p_{IC})           && \text{point in the left camera}
///   \\ h &= [\pi(P);\ \pi(P - b\,e_x)]         && \text{left and right pixels (4 rows)}
/// \f}
/// The error state perturbs every rotation on the right, \f$R = \hat R\,\mathrm{Exp}(\delta\theta)\f$,
/// so with \f$J = \partial h/\partial P\f$ (4x3) and \f$A = \hat R_{IC}^\top\f$:
/// \f{align*}{
///   \partial h/\partial\delta\theta_{WI} &= J A [\hat q]_\times &
///   \partial h/\partial\delta p_{WI}     &= -J A \hat R_{WI}^\top
///   \\ \partial h/\partial\delta\phi_{IC} &= J [\hat P]_\times &
///   \partial h/\partial\delta p_{IC}     &= -J A
///   \\ \partial h/\partial\delta f^W      &= J A \hat R_{WI}^\top &
///   \partial h/\partial\delta t_d        &= J A([\hat q]_\times\hat\omega - \hat R_{WI}^\top \hat v)
/// \f}
/// The last row follows from the exposure-time model: the image was taken at
/// \f$\hat\tau + \delta t_d\f$, when the pose was
/// \f$(\hat R\,\mathrm{Exp}(\hat\omega\,\delta t_d),\ \hat p + \hat v\,\delta t_d)\f$.
#pragma once

#include "aether/calib/StereoCamera.hpp"
#include "aether/core/Types.hpp"

namespace aether::calib {

/// \brief Linearisation point of one stereo view.
struct ViewPose {
  Mat3 R_wi = Mat3::Identity();  ///< IMU attitude \f$R_{WI}\f$.
  Vec3 p_wi = Vec3::Zero();      ///< IMU position \f$p_{WI}\f$ [m]
  Vec3 omega = Vec3::Zero();     ///< Bias-corrected body rate at the view time [rad/s]
  Vec3 v_wi = Vec3::Zero();      ///< World velocity at the view time [m/s]
};

/// \brief Predicted pixels of one stereo view and every Jacobian of them.
struct StereoViewJacobians {
  bool valid = false;                    ///< False if the point is behind either camera.
  Eigen::Matrix<double, 4, 1> h;         ///< [uL, vL, uR, vR] [px]
  Eigen::Matrix<double, 4, 3> d_att;     ///< w.r.t. \f$\delta\theta_{WI}\f$
  Eigen::Matrix<double, 4, 3> d_pos;     ///< w.r.t. \f$\delta p_{WI}\f$
  Eigen::Matrix<double, 4, 3> d_ext_rot; ///< w.r.t. \f$\delta\phi_{IC}\f$
  Eigen::Matrix<double, 4, 3> d_ext_pos; ///< w.r.t. \f$\delta p_{IC}\f$
  Eigen::Matrix<double, 4, 3> d_feature; ///< w.r.t. \f$\delta f^W\f$
  Eigen::Matrix<double, 4, 1> d_time;    ///< w.r.t. \f$\delta t_d\f$
};

/// \brief Evaluate the stereo projection of \a f_w and its Jacobians.
/// \param min_depth Points closer than this in either camera make the view invalid [m].
StereoViewJacobians stereoView(const StereoIntrinsics& cam, const ViewPose& pose,
                               const Mat3& R_ic, const Vec3& p_ic, const Vec3& f_w,
                               double min_depth = 0.05);

}  // namespace aether::calib
