/// \file test_status.cpp
/// \brief Online pixel-noise estimation and the calibrator's self-assessment (status()).
///
/// The end-to-end cases drive whole 60 s trials through `calib::runTrial` (about 1 s each in
/// Release) and are tagged [slow] so the unoptimised Debug CI job can skip them.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <string>

#include "aether/calib/CalibSimulator.hpp"
#include "aether/calib/Campaign.hpp"
#include "aether/calib/ExtrinsicCalibrator.hpp"
#include "aether/core/Constants.hpp"
#include "aether/mc/MonteCarlo.hpp"

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
  // The estimate averages ~20 frames of a median over ~100 landmarks, so its relative spread
  // is a few per cent; 5 % leaves room without hiding a bias. The 7x cases are the kind a
  // post-gate estimate cannot recover from: the gate (sized by the wrong noise) rejects the
  // data, or the covariance collapses before the estimate has caught up. (They stay inside the
  // 10x clamp of min/max_noise_scale.)
  struct Case {
    double truth, configured;
  };
  for (const Case c : {Case{0.5, 0.5}, Case{1.0, 0.5}, Case{0.5, 1.0}, Case{0.5, 0.07},
                       Case{0.5, 3.5}}) {
    INFO("true " << c.truth << " px, configured " << c.configured << " px");
    TrialConfig cfg = calib::scenario("convergence");
    cfg.sim.duration = 30.0;
    cfg.sim.pixel_noise = c.truth;
    cfg.calib.pixel_sigma = c.configured;
    const TrialResult r = calib::runTrial(cfg, 3);
    REQUIRE(r.ok);
    CHECK(r.final_pixel_sigma == Approx(c.truth).epsilon(0.05));
    // The noise level now matches the residuals the state cannot explain...
    CHECK(r.mean_noise_check == Approx(1.0).margin(0.1));
    // ...and, separately, the state-dependent innovations match H P H^T + R.
    CHECK(r.mean_nis_per_dof == Approx(1.0).margin(0.15));
    CHECK(r.final_rotation_error_deg < 0.5);
    CHECK(r.final_translation_error < 0.01);
    CHECK(r.false_confidence_frames == 0);
  }
}

TEST_CASE("wrong correspondences do not inflate the pixel-noise estimate",
          "[calib][noise][outliers][slow]") {
  // A wrong match is hundreds of pixels; through the least-squares projection it would leak
  // into every landmark's residual and inflate the median. The coarse gate and the trimmed
  // refit keep it out, also when the configured noise is far off.
  for (const double configured : {0.5, 0.07, 3.5}) {
    INFO("configured " << configured << " px");
    TrialConfig cfg = calib::scenario("outliers");
    cfg.sim.duration = 30.0;
    cfg.calib.pixel_sigma = configured;
    const TrialResult r = calib::runTrial(cfg, 2);
    REQUIRE(r.ok);
    CHECK(r.final_pixel_sigma == Approx(cfg.sim.pixel_noise).epsilon(0.05));
    CHECK(r.mean_noise_check == Approx(1.0).margin(0.1));
    CHECK(r.mean_nis_per_dof == Approx(1.0).margin(0.15));
    CHECK(r.false_confidence_frames == 0);
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
  // The 2x noise shows up in the residuals instead (variance ratio 4, partly gated).
  CHECK(r.mean_noise_check > 2.0);
}

TEST_CASE("estimating the noise turns a misconfigured filter from confidently wrong to right",
          "[calib][noise][slow]") {
  // 1.0 px of real noise, 0.5 px configured. Without adaptation the filter claims a
  // precision it does not have (NEES far above 6); with it, the covariance is honest again.
  const TrialResult fixed = calib::runTrial(calib::scenario("noise_misspecified_fixed"), 5);
  const TrialResult adapted = calib::runTrial(calib::scenario("noise_misspecified"), 5);
  REQUIRE(fixed.ok);
  REQUIRE(adapted.ok);
  INFO("fixed: NEES " << fixed.mean_nees << ", false-confidence frames "
                      << fixed.false_confidence_frames << "; adapted: NEES " << adapted.mean_nees);
  // Margins from the 50-run campaigns: fixed NEES 6.5-90, adapted at most ~3.7.
  CHECK(fixed.mean_nees > 6.0);
  CHECK(fixed.mean_nees > 3.0 * adapted.mean_nees);
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

TEST_CASE("the status certifies convergence only when the truth is inside tolerance",
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
    CHECK(r.status_converged_at_horizon);
    // Never "converged" with the truth outside the acceptance tolerance.
    CHECK(r.false_confidence_frames == 0);
    // In this nominal scenario it certifies after the truth got there, not before.
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
  // Between the knock and its detection the status cannot know; after the detection it must
  // not certify the new mount until the truth is inside tolerance. Without loosening the
  // navigation states on a re-open, the filter re-converged over-confident and re-certified
  // a lever arm still >1 cm off in about 1 run in 6 (campaign trials 6, 9, 18 were among
  // them), so this runs those trials and a few more.
  TrialConfig cfg = calib::scenario("knock");
  cfg.series_interval = 1.0 / cfg.sim.camera_rate;  // every frame
  const double tol_deg = cfg.calib.converged_rotation_bound * constants::kRadToDeg;
  const double tol_m = cfg.calib.converged_translation_bound;
  for (const int trial : {0, 1, 2, 3, 6, 9, 18, 19}) {
    INFO("campaign trial " << trial);
    const std::uint64_t seed = mc::trialSeed(mc::RunnerConfig{}.master_seed, trial);
    const TrialResult r = calib::runTrial(cfg, seed, trial);
    REQUIRE(r.ok);
    REQUIRE(r.change_events == 1);
    const double detected = r.change_times.front();
    int false_before = 0, false_after = 0;
    for (const calib::SeriesSample& s : r.series) {
      const bool wrong = s.rotation_error_deg > tol_deg || s.translation_error > tol_m;
      if (s.state != CalibrationState::kConverged || !wrong) continue;
      CHECK(s.time >= cfg.sim.drift.step_time);  // never before the knock
      ++(s.time < detected ? false_before : false_after);
    }
    INFO("detected " << detected - cfg.sim.drift.step_time << " s after the knock");
    CHECK(false_after == 0);
    // Only the frames of the detection latency (20 Hz) can be wrong.
    CHECK(false_before <= static_cast<int>(std::lround((detected - cfg.sim.drift.step_time) *
                                                       cfg.sim.camera_rate)) + 1);
  }
}
