/// \file test_status.cpp
/// \brief Online pixel-noise estimation and the calibrator's self-assessment (status()).
///
/// The end-to-end cases drive whole 60 s trials through `calib::runTrial` (about 1 s each in
/// Release) and are tagged [slow] so the unoptimised Debug CI job can skip them.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "aether/calib/CalibSimulator.hpp"
#include "aether/calib/Campaign.hpp"
#include "aether/calib/ExtrinsicCalibrator.hpp"
#include "aether/core/Constants.hpp"

using namespace aether;
using Catch::Approx;

namespace {

using calib::CalibrationState;
using calib::ExtrinsicCalibrator;
using calib::TrialConfig;
using calib::TrialResult;

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

}  // namespace

// -------------------------------------------------------------------------------------------
// Pixel-noise estimation
// -------------------------------------------------------------------------------------------

TEST_CASE("the pixel-noise estimate recovers the true noise whatever it was configured to",
          "[calib][noise][slow]") {
  // The estimate averages ~20 frames of a ~500-dof chi-square, so its relative spread is a
  // few per cent; 5 % leaves room without hiding a bias.
  struct Case {
    double truth, configured;
  };
  for (const Case c : {Case{0.5, 0.5}, Case{1.0, 0.5}, Case{0.5, 1.0}}) {
    INFO("true " << c.truth << " px, configured " << c.configured << " px");
    TrialConfig cfg = calib::scenario("convergence");
    cfg.sim.duration = 30.0;
    cfg.sim.pixel_noise = c.truth;
    cfg.calib.pixel_sigma = c.configured;
    const TrialResult r = calib::runTrial(cfg, 3);
    REQUIRE(r.ok);
    CHECK(r.final_pixel_sigma == Approx(c.truth).epsilon(0.05));
    // With the noise right, the vision innovations are consistent again.
    CHECK(r.mean_nis_per_dof == Approx(1.0).margin(0.1));
    CHECK(r.final_rotation_error_deg < 0.5);
    CHECK(r.final_translation_error < 0.01);
  }
}

TEST_CASE("with adaptation off the configured pixel noise is used unchanged",
          "[calib][noise]") {
  TrialConfig cfg = calib::scenario("noise_misspecified_fixed");
  REQUIRE_FALSE(cfg.calib.adapt_pixel_noise);
  cfg.sim.duration = 8.0;
  const TrialResult r = calib::runTrial(cfg, 3);
  REQUIRE(r.ok);
  CHECK(r.final_pixel_sigma == cfg.calib.pixel_sigma);
  // The 2x noise shows up in the innovations instead (variance ratio 4, partly gated).
  CHECK(r.mean_nis_per_dof > 1.5);
}

TEST_CASE("estimating the noise turns a misconfigured filter from confidently wrong to right",
          "[calib][noise][slow]") {
  // 1.0 px of real noise, 0.5 px configured. Without adaptation the filter claims a
  // precision it does not have, and its status vouches for an estimate outside the bounds.
  const TrialResult fixed = calib::runTrial(calib::scenario("noise_misspecified_fixed"), 5);
  const TrialResult adapted = calib::runTrial(calib::scenario("noise_misspecified"), 5);
  REQUIRE(fixed.ok);
  REQUIRE(adapted.ok);
  INFO("fixed: NEES " << fixed.mean_nees << ", false-confidence frames "
                      << fixed.false_confidence_frames << "; adapted: NEES " << adapted.mean_nees);
  CHECK(fixed.mean_nees > 12.0);
  CHECK(adapted.mean_nees < 6.0);
  CHECK(adapted.false_confidence_frames == 0);
  CHECK(adapted.converged);
}

// -------------------------------------------------------------------------------------------
// Status
// -------------------------------------------------------------------------------------------

TEST_CASE("every calibration state has a name", "[calib][status]") {
  CHECK(std::string(calib::toString(CalibrationState::kInitialising)) == "initialising");
  CHECK(std::string(calib::toString(CalibrationState::kCalibrating)) == "calibrating");
  CHECK(std::string(calib::toString(CalibrationState::kConverged)) == "converged");
  CHECK(std::string(calib::toString(CalibrationState::kRecalibrating)) == "recalibrating");
}

TEST_CASE("before the static window has been collected the status asks to hold still",
          "[calib][status]") {
  const TrialConfig cfg = calib::scenario("convergence");
  calib::CalibSimulator sim(cfg.sim, 1);
  ExtrinsicCalibrator cal(cfg.calib, sim.truthExtrinsic(0.0));
  for (int i = 0; i < 100; ++i) cal.addImu(sim.imu(i));  // 0.5 s, the window is 1.5 s
  const calib::CalibrationStatus s = cal.status();
  CHECK(s.state == CalibrationState::kInitialising);
  CHECK(contains(s.guidance, "hold the rig still"));
}

TEST_CASE("the status certifies convergence only when the truth is inside its bounds",
          "[calib][status][slow]") {
  const TrialConfig cfg = calib::scenario("convergence");
  for (const std::uint64_t seed : {1u, 2u, 3u}) {
    INFO("seed " << seed);
    const TrialResult r = calib::runTrial(cfg, seed);
    REQUIRE(r.ok);
    const calib::CalibrationStatus& s = r.final_status;
    CHECK(s.state == CalibrationState::kConverged);
    CHECK(contains(s.guidance, "calibrated"));
    CHECK(s.rotation_bound <= cfg.calib.converged_rotation_bound);
    CHECK(s.translation_bound <= cfg.calib.converged_translation_bound);
    // The bound is on the least-certain direction, so it covers the actual error.
    CHECK(r.final_rotation_error_deg * constants::kDegToRad <= s.rotation_bound);
    CHECK(r.final_translation_error <= s.translation_bound);
    CHECK(r.status_converged_at_horizon);
    CHECK(r.false_confidence_frames == 0);
    // It certifies after the truth got there, never before.
    CHECK(r.status_convergence_time >= r.convergence_time);
    // The default figure-eight excites all three axes well.
    CHECK(s.excitation.minCoeff() > cfg.calib.good_excitation_rate);
  }
}

TEST_CASE("with yaw-only motion the status names the unobservable lever-arm axis",
          "[calib][status][slow]") {
  const TrialResult r = calib::runTrial(calib::scenario("yaw_only"), 1);
  REQUIRE(r.ok);
  const calib::CalibrationStatus& s = r.final_status;
  INFO("guidance: " << s.guidance);
  CHECK(s.state == CalibrationState::kCalibrating);
  CHECK_FALSE(s.translation_converged);
  // The lever arm along the rotation axis (IMU z) is the one the motion cannot observe.
  CHECK(std::abs(s.weakest_translation_axis.z()) > 0.95);
  // No roll or pitch rate, plenty of yaw rate.
  CHECK(s.excitation.x() < 0.02);
  CHECK(s.excitation.y() < 0.02);
  CHECK(s.excitation.z() > 0.2);
  CHECK(contains(s.guidance, "needs motion"));
  CHECK(contains(s.guidance, "lever arm along IMU z"));
  CHECK(contains(s.guidance, "roll"));
  CHECK(contains(s.guidance, "pitch"));
  CHECK(r.false_confidence_frames == 0);
}

TEST_CASE("after a knock the status says recalibrating and then converges again",
          "[calib][status][slow]") {
  const TrialConfig cfg = calib::scenario("knock");
  calib::CalibSimulator sim(cfg.sim, 2);
  calib::CalibratorConfig cc = cfg.calib;
  ExtrinsicCalibrator cal(cc, sim.truthExtrinsic(0.0));
  bool converged_before = false, recalibrating_after = false;
  double detected_at = -1.0;
  cal.setFrameCallback([&](const calib::FrameReport& rep) {
    const CalibrationState state = cal.status().state;
    if (rep.time < cfg.sim.drift.step_time && state == CalibrationState::kConverged)
      converged_before = true;
    if (rep.change_detected && detected_at < 0.0) detected_at = rep.time;
    if (detected_at >= 0.0 && state == CalibrationState::kRecalibrating) recalibrating_after = true;
  });
  int k = 0;
  for (int i = 0; i < sim.imuCount(); ++i) {
    const double t = sim.imuTime(i);
    while (k < sim.frameCount() && sim.frameStamp(k) <= t + 0.1) cal.addFrame(sim.frame(k++));
    cal.addImu(sim.imu(i));
  }
  CHECK(converged_before);
  CHECK(detected_at > cfg.sim.drift.step_time);
  CHECK(recalibrating_after);
  const calib::CalibrationStatus end = cal.status();
  CHECK(end.state == CalibrationState::kConverged);
}

TEST_CASE("after a knock the status is wrong only until the knock is detected",
          "[calib][status][slow]") {
  const TrialConfig cfg = calib::scenario("knock");
  const TrialResult r = calib::runTrial(cfg, 4);
  REQUIRE(r.ok);
  REQUIRE(r.change_events == 1);
  // Frames between the knock and its detection (20 Hz) are the only possible false claims.
  const double latency = r.change_times.front() - cfg.sim.drift.step_time;
  const int latency_frames = static_cast<int>(std::lround(latency * cfg.sim.camera_rate)) + 1;
  INFO("detection latency " << latency << " s, false-confidence frames "
                            << r.false_confidence_frames);
  CHECK(r.false_confidence_frames <= latency_frames);
}
