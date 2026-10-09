/// \file autocal_run.cpp
/// \brief Run one calibration trial and write its time series.
///
///   autocal_run [--scenario NAME] [--seed N] [--output DIR]
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "aether/calib/Campaign.hpp"
#include "aether/mc/MonteCarlo.hpp"
#include "aether/util/Csv.hpp"

using namespace aether;

namespace {

void usage() {
  std::cout << "usage: autocal_run [--scenario NAME] [--seed N] [--output DIR] [--set key=value]...\n"
               "  scenarios:";
  for (const auto& n : calib::scenarioNames()) std::cout << " " << n;
  std::cout << "\n  --seed N     master seed (default 987654321)\n"
               "  --trial I    run trial I of that seed, i.e. the same run as autocal_mc's trial I"
               " (default 0)\n"
               "  --output DIR write DIR/<scenario>_series.csv (default results/runs)\n";
}

void writeSeries(const std::string& path, const calib::TrialResult& r) {
  util::CsvWriter w(path, {"time_s", "rot_err_deg", "trans_err_m", "rot_err_x_rad",
                           "rot_err_y_rad", "rot_err_z_rad", "trans_err_x_m", "trans_err_y_m",
                           "trans_err_z_m", "rot_sigma_x_rad", "rot_sigma_y_rad",
                           "rot_sigma_z_rad", "trans_sigma_x_m", "trans_sigma_y_m",
                           "trans_sigma_z_m", "nees", "truth_drift_deg", "estimate_drift_deg",
                           "td_truth_s", "td_estimate_s", "gyro_bias_err", "accel_bias_err",
                           "nis_per_dof", "landmarks_used", "vel_err_x", "vel_err_y", "vel_err_z",
                           "att_err_x", "att_err_y", "att_err_z", "vel_nees", "att_nees"});
  for (const auto& s : r.series) {
    w.writeRow({s.time, s.rotation_error_deg, s.translation_error, s.rotation_error.x(),
                s.rotation_error.y(), s.rotation_error.z(), s.translation_error_vec.x(),
                s.translation_error_vec.y(), s.translation_error_vec.z(), s.rotation_sigma.x(),
                s.rotation_sigma.y(), s.rotation_sigma.z(), s.translation_sigma.x(),
                s.translation_sigma.y(), s.translation_sigma.z(), s.nees, s.truth_drift_deg,
                s.estimate_drift_deg, s.time_offset_truth, s.time_offset_estimate,
                s.gyro_bias_error, s.accel_bias_error, s.nis_per_dof,
                static_cast<double>(s.landmarks_used), s.velocity_error.x(),
                s.velocity_error.y(), s.velocity_error.z(), s.attitude_error.x(),
                s.attitude_error.y(), s.attitude_error.z(), s.velocity_nees,
                s.attitude_nees});
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string name = "convergence";
  std::uint64_t master_seed = 987654321ULL;
  int trial = 0;
  std::string output = "results/runs";
  std::vector<std::string> overrides;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--scenario") {
      name = next();
    } else if (a == "--seed") {
      master_seed = std::stoull(next());
    } else if (a == "--trial") {
      trial = std::stoi(next());
    } else if (a == "--output") {
      output = next();
    } else if (a == "--set") {
      overrides.push_back(next());
    } else if (a == "--help" || a == "-h") {
      usage();
      return 0;
    } else {
      usage();
      return 2;
    }
  }

  calib::TrialConfig cfg;
  try {
    cfg = calib::scenario(name);
    for (const auto& o : overrides) calib::applyOverride(&cfg, o);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    usage();
    return 2;
  }
  const calib::TrialResult r = calib::runTrial(cfg, mc::trialSeed(master_seed, trial), trial);

  std::filesystem::create_directories(output);
  const std::string path = output + "/" + name + "_series.csv";
  writeSeries(path, r);

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "scenario            " << name << "  (" << cfg.sim.duration << " s simulated, "
            << r.wall_clock_seconds << " s wall)\n"
            << "static init         " << (r.static_init_ok ? "passed" : "FAILED") << "\n"
            << "frames fused        " << r.frames_fused << "  (mean " << r.mean_landmarks_used
            << " landmarks)\n"
            << "error @ horizon     " << r.horizon_rotation_error_deg << " deg, "
            << 100.0 * r.horizon_translation_error << " cm  -> "
            << (r.converged ? "CONVERGED" : "not converged") << "\n"
            << "convergence time    " << r.convergence_time << " s\n"
            << "final error         " << r.final_rotation_error_deg << " deg, "
            << 100.0 * r.final_translation_error << " cm\n";
  if (cfg.tracking_start >= 0.0)
    std::cout << "tracking error      max " << r.tracking_max_error_deg << " deg, rms "
              << r.tracking_rms_error_deg << " deg (truth moved " << r.truth_drift_total_deg
              << " deg); max lever-arm " << 100.0 * r.tracking_max_translation << " cm\n";
  if (cfg.calib.estimate_time_offset)
    std::cout << "time offset error   " << 1000.0 * r.final_time_offset_error << " ms\n";
  std::cout << "bias error          gyro " << r.final_gyro_bias_error << " rad/s, accel "
            << r.final_accel_bias_error << " m/s^2\n"
            << "consistency         mean NEES " << r.mean_nees << " (6 expected), mean NIS/dof "
            << r.mean_nis_per_dof << " (1 expected)\n"
            << "change events       " << r.change_events;
  for (double t : r.change_times) std::cout << (t == r.change_times.front() ? "  at " : ", ") << t << " s";
  std::cout << "\n"
            << "series              " << path << "\n";
  return r.ok ? 0 : 1;
}
