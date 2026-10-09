// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
/// \file Constants.hpp
/// \brief Physical and numerical constants used throughout Aether6 (all SI units).
#pragma once

namespace aether::constants {

/// Standard gravitational acceleration at mean sea level [m/s^2] (WGS-84 / ISO 80000-3).
inline constexpr double kGravity = 9.80665;

/// Sea-level standard atmospheric density [kg/m^3] (ISA).
inline constexpr double kRhoSeaLevel = 1.225;

/// Sea-level standard atmospheric pressure [Pa] (ISA).
inline constexpr double kPressureSeaLevel = 101325.0;

/// Sea-level standard temperature [K] (ISA).
inline constexpr double kTemperatureSeaLevel = 288.15;

/// Troposphere temperature lapse rate [K/m] (ISA, magnitude; temperature decreases with altitude).
inline constexpr double kLapseRate = 0.0065;

/// Specific gas constant for dry air [J/(kg*K)].
inline constexpr double kGasConstantAir = 287.0528;

/// Ratio of specific heats for dry air [-].
inline constexpr double kGamma = 1.4;

/// Geopotential altitude of the tropopause [m].
inline constexpr double kTropopauseAltitude = 11000.0;

/// Minimum airspeed used to guard aerodynamic normalisation terms [m/s].
/// Below this the dimensionless rate terms (p*b/(2*Va) etc.) and the
/// alpha/beta computation would blow up, so the model clamps to this value.
inline constexpr double kMinAirspeed = 1.0;

/// Pi (kept local so the code does not depend on non-standard M_PI).
inline constexpr double kPi = 3.14159265358979323846;

/// Degrees per radian [deg/rad].
inline constexpr double kRadToDeg = 180.0 / kPi;

/// Radians per degree [rad/deg].
inline constexpr double kDegToRad = kPi / 180.0;

}  // namespace aether::constants
