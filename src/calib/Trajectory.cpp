#include "aether/calib/Trajectory.hpp"

#include <cmath>

#include "aether/core/Constants.hpp"
#include "aether/math/Rotation.hpp"

namespace aether::calib {

namespace {
constexpr double kTwoPi = 2.0 * constants::kPi;

/// Value and first two derivatives of a*sin(w t + phi).
struct Sinusoid {
  double f, df, ddf;
};
Sinusoid sinusoid(double amplitude, double omega, double phase, double t) {
  const double s = std::sin(omega * t + phase);
  const double c = std::cos(omega * t + phase);
  return {amplitude * s, amplitude * omega * c, -amplitude * omega * omega * s};
}
}  // namespace

Vec3 KinematicState::specificForce(double gravity) const {
  return math::rotateNedToBody(quaternion, acceleration - Vec3(0.0, 0.0, gravity));
}

FigureEightTrajectory::FigureEightTrajectory(const TrajectoryConfig& config) : cfg_(config) {}

Vec3 FigureEightTrajectory::envelope(double t) const {
  const double t0 = cfg_.static_duration;
  const double T = cfg_.ramp_duration;
  if (t <= t0) return Vec3::Zero();
  if (!(T > 0.0) || t >= t0 + T) return Vec3(1.0, 0.0, 0.0);
  const double u = (t - t0) / T;
  const double u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u;
  // s = 10u^3 - 15u^4 + 6u^5: s, s', s'' all vanish at u = 0 and s', s'' vanish at u = 1.
  return Vec3(10.0 * u3 - 15.0 * u4 + 6.0 * u5,
              (30.0 * u2 - 60.0 * u3 + 30.0 * u4) / T,
              (60.0 * u - 180.0 * u2 + 120.0 * u3) / (T * T));
}

KinematicState FigureEightTrajectory::evaluate(double t) const {
  KinematicState k;
  k.time = t;
  const Vec3 env = envelope(t);
  const double s = env(0), ds = env(1), dds = env(2);
  const double tp = t - cfg_.static_duration;

  // --- Translation: c + s(t) f(t') ------------------------------------------------
  const double w = kTwoPi / cfg_.period;
  const Sinusoid fn = sinusoid(cfg_.amp_north, 2.0 * w, cfg_.phase_north, tp);
  const Sinusoid fe = sinusoid(cfg_.amp_east, w, 0.0, tp);
  const Sinusoid fd = sinusoid(cfg_.amp_down, 2.0 * w, 0.0, tp);
  const Vec3 f(fn.f, fe.f, fd.f);
  const Vec3 df(fn.df, fe.df, fd.df);
  const Vec3 ddf(fn.ddf, fe.ddf, fd.ddf);
  k.position = cfg_.center + s * f;
  k.velocity = ds * f + s * df;
  k.acceleration = dds * f + 2.0 * ds * df + s * ddf;

  // --- Rotation: eta0 + s(t) g(t') -------------------------------------------------
  Vec3 g, dg;
  for (int i = 0; i < 3; ++i) {
    const Sinusoid si = sinusoid(cfg_.euler_amplitude(i), kTwoPi * cfg_.euler_frequency(i),
                                 cfg_.euler_phase(i), tp);
    g(i) = si.f;
    dg(i) = si.df;
  }
  const Vec3 eta = cfg_.base_euler + s * g;
  const Vec3 deta = ds * g + s * dg;
  k.euler = eta;
  k.quaternion = math::eulerToQuat(eta(0), eta(1), eta(2));

  // Body rate from 3-2-1 Euler rates (R = Rz(psi) Ry(theta) Rx(phi)).
  const double sphi = std::sin(eta(0)), cphi = std::cos(eta(0));
  const double sth = std::sin(eta(1)), cth = std::cos(eta(1));
  k.angular_rate = Vec3(deta(0) - deta(2) * sth,
                        deta(1) * cphi + deta(2) * sphi * cth,
                        -deta(1) * sphi + deta(2) * cphi * cth);
  return k;
}

}  // namespace aether::calib
