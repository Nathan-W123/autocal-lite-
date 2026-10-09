/// \file autocal_mc.cpp
/// \brief Deterministic Monte-Carlo campaigns over the calibration scenarios.
///
///   autocal_mc [--scenario NAME[,NAME...]|all] [--trials N] [--seed N] [--threads N]
///              [--output DIR]
///
/// For every scenario it writes DIR/<scenario>/{trials.csv, envelope.csv, example_series.csv,
/// summary.json} and prints one row of the results table.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "aether/calib/Campaign.hpp"
#include "aether/mc/MonteCarlo.hpp"
#include "aether/util/Csv.hpp"

using namespace aether;

namespace {

void usage() {
  std::cout << "usage: autocal_mc [--scenario NAME[,NAME...]|all] [--trials N] [--seed N]\n"
               "                  [--threads N] [--output DIR] [--set key=value]...\n  scenarios:";
  for (const auto& n : calib::scenarioNames()) std::cout << " " << n;
  std::cout << "\n  defaults: --scenario convergence,drift --trials 50 --seed 987654321"
               " --output results/mc\n";
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty()) out.push_back(item);
  return out;
}

void writeSeries(const std::string& path, const std::vector<calib::SeriesSample>& series) {
  util::CsvWriter w(path, {"time_s", "rot_err_deg", "trans_err_m", "rot_sigma_max_deg",
                           "trans_sigma_max_m", "nees", "truth_drift_deg", "estimate_drift_deg",
                           "td_truth_s", "td_estimate_s", "nis_per_dof", "landmarks_used"});
  for (const auto& s : series)
    w.writeRow({s.time, s.rotation_error_deg, s.translation_error,
                s.rotation_sigma.maxCoeff() * constants::kRadToDeg,
                s.translation_sigma.maxCoeff(), s.nees, s.truth_drift_deg,
                s.estimate_drift_deg, s.time_offset_truth, s.time_offset_estimate,
                s.nis_per_dof, static_cast<double>(s.landmarks_used)});
}

/// Time-binned percentiles of the error histories across all trials.
void writeEnvelope(const std::string& path, const std::vector<calib::TrialResult>& trials,
                   double bin) {
  std::map<long, std::vector<double>> rot, trans, nees, drift_truth, drift_est;
  for (const auto& t : trials) {
    for (const auto& s : t.series) {
      const long k = std::lround(s.time / bin);
      rot[k].push_back(s.rotation_error_deg);
      trans[k].push_back(s.translation_error);
      nees[k].push_back(s.nees);
      drift_truth[k].push_back(s.truth_drift_deg);
      drift_est[k].push_back(s.estimate_drift_deg);
    }
  }
  util::CsvWriter w(path, {"time_s", "n", "rot_err_p50_deg", "rot_err_p95_deg",
                           "rot_err_max_deg", "trans_err_p50_m", "trans_err_p95_m",
                           "trans_err_max_m", "nees_mean", "truth_drift_p50_deg",
                           "estimate_drift_p50_deg"});
  for (const auto& [k, values] : rot) {
    const auto r = mc::computeStatistics("", "", values);
    const auto t = mc::computeStatistics("", "", trans[k]);
    const auto n = mc::computeStatistics("", "", nees[k]);
    const auto dt = mc::computeStatistics("", "", drift_truth[k]);
    const auto de = mc::computeStatistics("", "", drift_est[k]);
    w.writeRow({static_cast<double>(k) * bin, static_cast<double>(r.count), r.median, r.p95,
                r.max, t.median, t.p95, t.max, n.mean, dt.median, de.median});
  }
}

struct ScenarioSummary {
  std::string name;
  int trials = 0;
  int ok = 0;
  int converged = 0;
  int tracked = 0;           ///< Trials whose max tracking error met the threshold.
  bool has_tracking = false;
  double tracking_threshold_deg = 0.3;
  std::vector<mc::MetricStatistics> stats;
  double wall = 0.0;
  int threads = 1;
};

const mc::MetricStatistics& stat(const ScenarioSummary& s, const std::string& name) {
  for (const auto& m : s.stats)
    if (m.name == name) return m;
  static const mc::MetricStatistics empty;
  return empty;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> names = {"convergence", "drift"};
  mc::RunnerConfig runner;
  runner.trials = 50;
  std::string output = "results/mc";
  std::vector<std::string> overrides;
  constexpr double kTrackingThresholdDeg = 0.3;

  try {
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
        const std::string v = next();
        names = v == "all" ? calib::scenarioNames() : split(v);
      } else if (a == "--trials") {
        runner.trials = std::stoi(next());
      } else if (a == "--seed") {
        runner.master_seed = std::stoull(next());
      } else if (a == "--threads") {
        runner.threads = std::stoi(next());
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
    if (runner.trials < 0 || runner.trials > 1000000) throw std::invalid_argument("--trials out of range");
  } catch (const std::exception& e) {
    std::cerr << "autocal_mc: bad argument (" << e.what() << ")\n";
    usage();
    return 2;
  }

  std::vector<ScenarioSummary> summaries;
  for (const auto& name : names) {
    calib::TrialConfig cfg;
    try {
      cfg = calib::scenario(name);
      for (const auto& o : overrides) calib::applyOverride(&cfg, o);
    } catch (const std::exception& e) {
      std::cerr << e.what() << "\n";
      usage();
      return 2;
    }
    std::cout << "[" << name << "] " << runner.trials << " trials of " << cfg.sim.duration
              << " s\n";
    const auto run = mc::runCampaign<calib::TrialResult>(
        runner, [&cfg](int index, std::uint64_t seed) { return calib::runTrial(cfg, seed, index); });

    ScenarioSummary sum;
    sum.name = name;
    sum.trials = runner.trials;
    sum.wall = run.wall_clock_seconds;
    sum.threads = run.threads_used;
    sum.has_tracking = cfg.tracking_start >= 0.0;
    sum.tracking_threshold_deg = kTrackingThresholdDeg;
    std::vector<double> hr, ht, fr, ft, ct, tm, tr, tt, nees, nis, td, bg, ba, wall, lm, rec,
        events, latency;
    std::vector<calib::TrialResult> valid;
    for (std::size_t i = 0; i < run.results.size(); ++i) {
      const auto& t = run.results[i];
      if (!run.errors[i].empty() || !t.ok) continue;
      ++sum.ok;
      if (t.converged) ++sum.converged;
      if (sum.has_tracking && t.tracking_samples > 0 &&
          t.tracking_max_error_deg < kTrackingThresholdDeg)
        ++sum.tracked;
      hr.push_back(t.horizon_rotation_error_deg);
      ht.push_back(100.0 * t.horizon_translation_error);
      fr.push_back(t.final_rotation_error_deg);
      ft.push_back(100.0 * t.final_translation_error);
      if (t.converged) ct.push_back(t.convergence_time);
      tm.push_back(t.tracking_max_error_deg);
      tr.push_back(t.tracking_rms_error_deg);
      tt.push_back(100.0 * t.tracking_max_translation);
      if (t.recovery_time >= 0.0) rec.push_back(t.recovery_time);
      // Detection latency: first change event at or after a scripted knock.
      if (cfg.sim.drift.step_time >= 0.0) {
        for (double event_time : t.change_times) {
          if (event_time >= cfg.sim.drift.step_time - 1e-9) {
            latency.push_back(event_time - cfg.sim.drift.step_time);
            break;
          }
        }
      }
      events.push_back(static_cast<double>(t.change_events));
      nees.push_back(t.mean_nees);
      nis.push_back(t.mean_nis_per_dof);
      td.push_back(1000.0 * t.final_time_offset_error);
      bg.push_back(t.final_gyro_bias_error);
      ba.push_back(t.final_accel_bias_error);
      wall.push_back(t.wall_clock_seconds);
      lm.push_back(t.mean_landmarks_used);
      valid.push_back(t);
    }
    sum.stats = {mc::computeStatistics("horizon_rotation_error", "deg", hr),
                 mc::computeStatistics("horizon_translation_error", "cm", ht),
                 mc::computeStatistics("final_rotation_error", "deg", fr),
                 mc::computeStatistics("final_translation_error", "cm", ft),
                 mc::computeStatistics("convergence_time", "s", ct),
                 mc::computeStatistics("tracking_max_error", "deg", tm),
                 mc::computeStatistics("tracking_rms_error", "deg", tr),
                 mc::computeStatistics("tracking_max_translation", "cm", tt),
                 mc::computeStatistics("recovery_time", "s", rec),
                 mc::computeStatistics("change_events", "-", events),
                 mc::computeStatistics("detection_latency", "s", latency),
                 mc::computeStatistics("mean_nees", "-", nees),
                 mc::computeStatistics("mean_nis_per_dof", "-", nis),
                 mc::computeStatistics("final_time_offset_error", "ms", td),
                 mc::computeStatistics("final_gyro_bias_error", "rad/s", bg),
                 mc::computeStatistics("final_accel_bias_error", "m/s^2", ba),
                 mc::computeStatistics("mean_landmarks_used", "-", lm),
                 mc::computeStatistics("trial_wall_clock", "s", wall)};

    // --- Files ---------------------------------------------------------------------
    namespace fs = std::filesystem;
    const std::string dir = output + "/" + name;
    fs::create_directories(dir);
    {
      util::CsvWriter w(dir + "/trials.csv",
                        {"index", "ok", "converged", "convergence_time_s",
                         "horizon_rot_err_deg", "horizon_trans_err_m", "final_rot_err_deg",
                         "final_trans_err_m", "tracking_max_err_deg", "tracking_rms_err_deg",
                         "tracking_max_trans_err_m", "truth_drift_total_deg", "recovery_time_s",
                         "final_td_err_s", "final_gyro_bias_err", "final_accel_bias_err",
                         "mean_nees", "mean_nis_per_dof", "mean_landmarks_used",
                         "frames_fused", "frames_dropped", "change_events", "static_init_ok",
                         "wall_clock_s"});
      for (std::size_t i = 0; i < run.results.size(); ++i) {
        const auto& t = run.results[i];
        const bool ok = run.errors[i].empty() && t.ok;
        // Trial i is reproduced exactly by `autocal_run --seed <master_seed> --trial i`.
        w.writeRow({static_cast<double>(i), ok ? 1.0 : 0.0, t.converged ? 1.0 : 0.0, t.convergence_time,
                    t.horizon_rotation_error_deg, t.horizon_translation_error,
                    t.final_rotation_error_deg, t.final_translation_error,
                    t.tracking_max_error_deg, t.tracking_rms_error_deg,
                    t.tracking_max_translation, t.truth_drift_total_deg, t.recovery_time,
                    t.final_time_offset_error, t.final_gyro_bias_error,
                    t.final_accel_bias_error, t.mean_nees, t.mean_nis_per_dof,
                    t.mean_landmarks_used, static_cast<double>(t.frames_fused),
                    static_cast<double>(t.frames_dropped),
                    static_cast<double>(t.change_events), t.static_init_ok ? 1.0 : 0.0,
                    t.wall_clock_seconds});
      }
    }
    writeEnvelope(dir + "/envelope.csv", valid, cfg.series_interval > 0 ? cfg.series_interval : 1.0);
    if (!valid.empty()) writeSeries(dir + "/example_series.csv", valid.front().series);
    {
      std::ofstream f(dir + "/summary.json");
      f << std::setprecision(10);
      f << "{\n  \"scenario\": \"" << name << "\",\n"
        << "  \"trials\": " << sum.trials << ",\n"
        << "  \"master_seed\": " << runner.master_seed << ",\n"
        << "  \"duration_s\": " << cfg.sim.duration << ",\n"
        << "  \"prior_rotation_error_deg\": " << cfg.prior_rotation_error * constants::kRadToDeg
        << ",\n"
        << "  \"prior_translation_error_m\": " << cfg.prior_translation_error << ",\n"
        << "  \"rotation_threshold_deg\": " << cfg.rotation_threshold_deg << ",\n"
        << "  \"translation_threshold_m\": " << cfg.translation_threshold << ",\n"
        << "  \"convergence_horizon_s\": " << cfg.convergence_horizon << ",\n"
        << "  \"completed\": " << sum.ok << ",\n"
        << "  \"converged\": " << sum.converged << ",\n"
        << "  \"convergence_rate\": "
        << (sum.trials > 0 ? static_cast<double>(sum.converged) / sum.trials : 0.0) << ",\n";
      if (sum.has_tracking) {
        f << "  \"tracking_start_s\": " << cfg.tracking_start << ",\n"
          << "  \"tracking_threshold_deg\": " << kTrackingThresholdDeg << ",\n"
          << "  \"tracked\": " << sum.tracked << ",\n"
          << "  \"tracking_rate\": "
          << (sum.trials > 0 ? static_cast<double>(sum.tracked) / sum.trials : 0.0) << ",\n";
      }
      f << "  \"threads\": " << run.threads_used << ",\n"
        << "  \"wall_clock_s\": " << run.wall_clock_seconds << ",\n"
        << "  \"statistics\": [\n";
      for (std::size_t i = 0; i < sum.stats.size(); ++i) {
        const auto& s = sum.stats[i];
        f << "    {\"name\": \"" << s.name << "\", \"unit\": \"" << s.unit
          << "\", \"count\": " << s.count << ", \"mean\": " << s.mean
          << ", \"stddev\": " << s.stddev << ", \"min\": " << s.min << ", \"p05\": " << s.p05
          << ", \"median\": " << s.median << ", \"p95\": " << s.p95 << ", \"p99\": " << s.p99
          << ", \"max\": " << s.max << "}" << (i + 1 < sum.stats.size() ? "," : "") << "\n";
      }
      f << "  ]\n}\n";
    }
    summaries.push_back(sum);
  }

  // --- Table ---------------------------------------------------------------------------
  std::cout << "\n" << std::fixed;
  std::cout << "| scenario | converged (<0.5 deg & <1 cm @ 60 s) | rot err @60s p50 / p95 / max "
               "[deg] | trans err @60s p50 / p95 / max [cm] | conv. time p50 / p95 [s] | "
               "tracking max err p50 / p95 / max [deg] | mean NEES (6) |\n";
  std::cout << "|---|---|---|---|---|---|---|\n";
  for (const auto& s : summaries) {
    const auto& r = stat(s, "horizon_rotation_error");
    const auto& t = stat(s, "horizon_translation_error");
    const auto& c = stat(s, "convergence_time");
    const auto& m = stat(s, "tracking_max_error");
    const auto& n = stat(s, "mean_nees");
    std::cout << std::setprecision(3) << "| " << s.name << " | " << s.converged << "/" << s.trials
              << " (" << std::setprecision(1) << 100.0 * s.converged / std::max(1, s.trials)
              << " %) | " << std::setprecision(3) << r.median << " / " << r.p95 << " / " << r.max
              << " | " << t.median << " / " << t.p95 << " / " << t.max << " | "
              << std::setprecision(1) << c.median << " / " << c.p95 << " | ";
    if (s.has_tracking)
      std::cout << std::setprecision(3) << m.median << " / " << m.p95 << " / " << m.max << " ("
                << s.tracked << "/" << s.trials << " < " << s.tracking_threshold_deg << ")";
    else
      std::cout << "-";
    std::cout << " | " << std::setprecision(2) << n.mean << " |\n";
  }
  for (const auto& s : summaries)
    std::cout << "  " << s.name << ": " << std::setprecision(1) << s.wall << " s wall on "
              << s.threads << " threads\n";
  return 0;
}
