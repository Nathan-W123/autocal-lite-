/// \file MonteCarlo.hpp
/// \brief Deterministic, thread-count-independent Monte-Carlo harness and statistics.
///
/// Reused from Aether6 (`include/aether/mc/MonteCarlo.hpp`, `src/mc/MonteCarlo.cpp`).
/// `MetricStatistics` and `computeStatistics` are unchanged. The work-stealing thread pool and
/// the per-trial seed derivation of Aether6's `MonteCarloRunner::run` are kept exactly, but
/// lifted out of the aircraft-specific runner into a template over the trial function, so any
/// experiment gets the same guarantee: trial \a i always sees the same seed, whatever the
/// number of worker threads or the completion order.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "aether/util/Random.hpp"

namespace aether::mc {

/// \brief Aggregate campaign statistics for one scalar metric.
struct MetricStatistics {
  std::string name;   ///< Metric name.
  std::string unit;   ///< Physical unit.
  double mean = 0.0;  ///< Sample mean.
  double stddev = 0.0;///< Sample standard deviation (n-1).
  double min = 0.0;   ///< Minimum.
  double p05 = 0.0;   ///< 5th percentile.
  double median = 0.0;///< Median.
  double p95 = 0.0;   ///< 95th percentile.
  double p99 = 0.0;   ///< 99th percentile.
  double max = 0.0;   ///< Maximum.
  int count = 0;      ///< Number of samples contributing.
};

/// \brief Compute mean/stddev/percentiles of a sample set.
MetricStatistics computeStatistics(std::string name, std::string unit, std::vector<double> data);

/// \brief Campaign settings shared by every experiment.
struct RunnerConfig {
  int trials = 50;                         ///< Number of trials.
  std::uint64_t master_seed = 987654321;   ///< Master seed; trial seeds derive from it.
  int threads = 0;                         ///< Worker threads (0 = hardware concurrency).
};

/// \brief Seed of trial \a index (Aether6's `kStreamRun` derivation).
inline std::uint64_t trialSeed(std::uint64_t master_seed, int index) {
  constexpr std::uint64_t kStreamRun = 15;
  return util::deriveSeed(master_seed, kStreamRun, static_cast<std::uint64_t>(index));
}

/// \brief Results of a campaign, ordered by trial index.
template <class Result>
struct CampaignRun {
  std::vector<Result> results;        ///< Per-trial results (default-constructed on exception).
  std::vector<std::string> errors;    ///< Per-trial exception text, empty when the trial ran.
  int threads_used = 1;               ///< Worker threads actually used.
  double wall_clock_seconds = 0.0;    ///< Campaign wall-clock time [s].
};

/// \brief Run `fn(index, seed)` for every trial on a pool of worker threads.
///
/// Each worker claims the next unprocessed index from an atomic counter (Aether6's scheme), so
/// the load balances itself, and writes its result into the slot of that index, so the output
/// order never depends on scheduling.
template <class Result, class TrialFn>
CampaignRun<Result> runCampaign(const RunnerConfig& cfg, TrialFn fn, bool verbose = true) {
  const auto wall_start = std::chrono::steady_clock::now();
  CampaignRun<Result> out;
  const int n = std::max(cfg.trials, 0);
  // Workers write into one-object-per-trial slots, never straight into a std::vector<Result>:
  // std::vector<bool> packs neighbouring trials into one word, so concurrent writes to
  // adjacent indices would race.
  struct Slot {
    Result value{};
  };
  std::vector<Slot> slots(static_cast<std::size_t>(n));
  out.errors.resize(static_cast<std::size_t>(n));

  int threads = cfg.threads > 0 ? cfg.threads
                                : static_cast<int>(std::thread::hardware_concurrency());
  threads = std::clamp(threads, 1, std::max(1, n));
  out.threads_used = threads;

  std::atomic<int> next_index{0};
  std::atomic<int> completed{0};
  std::mutex print_mutex;

  auto worker = [&]() {
    for (;;) {
      const int i = next_index.fetch_add(1);
      if (i >= n) return;
      try {
        slots[static_cast<std::size_t>(i)].value = fn(i, trialSeed(cfg.master_seed, i));
      } catch (const std::exception& e) {
        out.errors[static_cast<std::size_t>(i)] = std::string("exception: ") + e.what();
      }
      const int done = completed.fetch_add(1) + 1;
      if (verbose && (done % std::max(1, n / 20) == 0 || done == n)) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "  progress: " << done << "/" << n << " trials\r" << std::flush;
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  for (auto& t : pool) t.join();
  if (verbose) std::cout << "\n";

  out.results.reserve(slots.size());
  for (auto& slot : slots) out.results.push_back(std::move(slot.value));

  out.wall_clock_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
  return out;
}

}  // namespace aether::mc
