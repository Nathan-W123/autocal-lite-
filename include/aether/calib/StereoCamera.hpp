/// \file StereoCamera.hpp
/// \brief Rectified pinhole stereo pair: projection, Jacobians, triangulation, observations.
#pragma once

#include <vector>

#include "aether/core/Types.hpp"

namespace aether::calib {

/// \brief Known intrinsics of a rectified stereo pair (both cameras share them).
///
/// The right camera C1 is the left camera C translated by `baseline` along \f$+x_C\f$, with
/// no relative rotation, so a point \f$P\f$ in the left frame is \f$P - [b,0,0]^\top\f$ in the
/// right frame.
struct StereoIntrinsics {
  double fx = 400.0;       ///< Focal length, x [px]
  double fy = 400.0;       ///< Focal length, y [px]
  double cx = 320.0;       ///< Principal point, x [px]
  double cy = 240.0;       ///< Principal point, y [px]
  int width = 640;         ///< Image width [px]
  int height = 480;        ///< Image height [px]
  double baseline = 0.12;  ///< Stereo baseline [m]

  /// Pinhole projection \f$\pi(P) = [f_x X/Z + c_x,\ f_y Y/Z + c_y]\f$ of a point in a camera frame.
  Vec2 project(const Vec3& P) const;
  /// Jacobian \f$\partial\pi/\partial P\f$ (2x3).
  Eigen::Matrix<double, 2, 3> projectionJacobian(const Vec3& P) const;
  /// True when the pixel lies inside the image with a `margin` [px] border.
  bool inImage(const Vec2& uv, double margin = 0.0) const;
  /// Point in the right camera frame given the point in the left camera frame.
  Vec3 leftToRight(const Vec3& P_left) const { return P_left - Vec3(baseline, 0.0, 0.0); }
  /// Rectified-stereo triangulation of a left/right pixel pair into the left camera frame.
  /// Returns false when the disparity is below `min_disparity` [px].
  bool triangulate(const Vec2& left, const Vec2& right, Vec3* P_left,
                   double min_disparity = 0.5) const;
};

/// \brief One landmark seen in both images of a stereo frame (the provided correspondence).
struct StereoObservation {
  int id = -1;              ///< Landmark identifier (the "correspondence").
  Vec2 left = Vec2::Zero(); ///< Pixel in the left image [px]
  Vec2 right = Vec2::Zero();///< Pixel in the right image [px]
};

/// \brief All stereo observations of one exposure.
struct StereoFrame {
  /// Timestamp in the camera clock [s]. The exposure happened at IMU-clock time
  /// `stamp + t_d`, where \f$t_d\f$ is the (possibly unknown) camera-IMU time offset.
  double stamp = 0.0;
  std::vector<StereoObservation> observations;  ///< Sorted by landmark id.
};

}  // namespace aether::calib
