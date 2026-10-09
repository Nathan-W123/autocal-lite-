/// \file test_montecarlo.cpp
/// \brief Campaign statistics, per-trial seed derivation and the thread-count-independent
///        Monte-Carlo runner (`mc::runCampaign`).
///
/// The statistics case is ported unchanged from Aether6 (`tests/test_montecarlo.cpp`). The
/// aircraft-specific runner cases there do not apply; the guarantees they checked (ordered,
/// reproducible, thread-count-independent results) are re-tested here on the generic
/// `runCampaign` template with synthetic trial functions.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "aether/mc/MonteCarlo.hpp"
#include "aether/util/Random.hpp"

using namespace aether;
using Catch::Approx;

namespace {

/// What a synthetic trial reports: who it was, which seed it got, and a few draws.
struct ProbeResult {
  int index = -1;
  std::uint64_t seed = 0;
  std::vector<double> draws;
};

/// Bit pattern of a double, so "bitwise identical" is literally checked (== would also accept
/// +0.0 vs -0.0).
std::uint64_t bits(double v) {
  std::uint64_t b = 0;
  std::memcpy(&b, &v, sizeof b);
  return b;
}

/// Trial that consumes util::Rng(seed): a burst of gaussians, then a random-length tail so
/// trials have unequal cost and complete out of order on a multi-threaded pool.
ProbeResult rngTrial(int index, std::uint64_t seed) {
  util::Rng rng(seed);
  ProbeResult r;
  r.index = index;
  r.seed = seed;
  double acc = 0.0;
  for (int k = 0; k < 8; ++k) r.draws.push_back(rng.gaussian());
  const int tail = 1000 + static_cast<int>(rng.uniform(0.0, 20000.0));
  for (int k = 0; k < tail; ++k) acc += rng.gaussian(0.0, 0.1) * rng.uniform();
  r.draws.push_back(acc);
  return r;
}

mc::RunnerConfig campaign(int trials, int threads, std::uint64_t seed) {
  mc::RunnerConfig c;
  c.trials = trials;
  c.threads = threads;
  c.master_seed = seed;
  return c;
}

}  // namespace

TEST_CASE("statistics helper computes correct moments and percentiles", "[mc][statistics]") {
  std::vector<double> data;
  for (int i = 1; i <= 100; ++i) data.push_back(static_cast<double>(i));
  const auto s = mc::computeStatistics("test", "-", data);
  REQUIRE(s.count == 100);
  REQUIRE(s.mean == Approx(50.5));
  REQUIRE(s.min == Approx(1.0));
  REQUIRE(s.max == Approx(100.0));
  REQUIRE(s.median == Approx(50.5));
  REQUIRE(s.p05 == Approx(5.95));
  REQUIRE(s.p95 == Approx(95.05));
  // Sample standard deviation of 1..100 is sqrt(100*101/12) = 29.011...
  REQUIRE(s.stddev == Approx(29.0115).epsilon(1e-4));

  const auto empty = mc::computeStatistics("e", "-", {});
  REQUIRE(empty.count == 0);
  REQUIRE(empty.mean == Approx(0.0));

  const auto single = mc::computeStatistics("s", "-", {7.0});
  REQUIRE(single.mean == Approx(7.0));
  REQUIRE(single.stddev == Approx(0.0));
  REQUIRE(single.median == Approx(7.0));
}

TEST_CASE("statistics helper is order independent and keeps name and unit and p99",
          "[mc][statistics]") {
  // AutoCal-lite: the campaign table feeds unsorted per-trial errors straight in.
  std::vector<double> data;
  for (int i = 100; i >= 1; --i) data.push_back(static_cast<double>((i * 37) % 101));
  const auto s = mc::computeStatistics("rot_err", "deg", data);
  REQUIRE(s.name == "rot_err");
  REQUIRE(s.unit == "deg");
  // (i*37) mod 101 for i = 1..100 is a permutation of 1..100, so the statistics must equal
  // those of the sorted sequence: p99 sits at position 0.99*99 = 98.01 -> 99 + 0.01.
  REQUIRE(s.min == 1.0);
  REQUIRE(s.max == 100.0);
  REQUIRE(s.median == Approx(50.5));
  REQUIRE(s.p99 == Approx(99.01));
  REQUIRE(s.min <= s.p05);
  REQUIRE(s.p05 <= s.median);
  REQUIRE(s.median <= s.p95);
  REQUIRE(s.p95 <= s.p99);
  REQUIRE(s.p99 <= s.max);

  // Two samples: linear interpolation between them.
  const auto two = mc::computeStatistics("t", "-", {4.0, 2.0});
  REQUIRE(two.median == Approx(3.0));
  REQUIRE(two.p95 == Approx(2.0 + 0.95 * 2.0));
  REQUIRE(two.stddev == Approx(std::sqrt(2.0)));
}

TEST_CASE("trial seeds use Aether6's run-stream derivation and are stable",
          "[mc][determinism]") {
  // The derivation is pure 64-bit integer arithmetic, so the values are platform
  // independent. The golden values were computed independently (Python, from the SplitMix64
  // definition) for the default master seed: deriveSeed(987654321, stream 15, i).
  const std::uint64_t master = mc::RunnerConfig{}.master_seed;
  REQUIRE(master == 987654321ULL);
  REQUIRE(mc::trialSeed(master, 0) == 0xCEB379E7560A17DEULL);
  REQUIRE(mc::trialSeed(master, 1) == 0xCA4EA9D8235FD6A8ULL);
  REQUIRE(mc::trialSeed(master, 2) == 0x6C90B52B9AA39C41ULL);

  std::set<std::uint64_t> seen;
  for (int i = 0; i < 1000; ++i) {
    const std::uint64_t s = mc::trialSeed(master, i);
    REQUIRE(s == util::deriveSeed(master, 15, static_cast<std::uint64_t>(i)));
    REQUIRE(s == mc::trialSeed(master, i));  // a pure function of (master, index)
    seen.insert(s);
  }
  REQUIRE(seen.size() == 1000);  // no collisions between trial streams
  REQUIRE(mc::trialSeed(master, 0) != mc::trialSeed(master + 1, 0));
}

TEST_CASE("campaign results are stored by trial index with that trial's seed",
          "[mc][campaign]") {
  const auto cfg = campaign(40, 4, 20240917);
  // Make early indices the slowest so completion order is the reverse of claim order.
  const auto run = mc::runCampaign<ProbeResult>(
      cfg,
      [](int i, std::uint64_t seed) {
        std::this_thread::sleep_for(std::chrono::microseconds(50 * (40 - i)));
        ProbeResult r;
        r.index = i;
        r.seed = seed;
        return r;
      },
      false);

  REQUIRE(run.threads_used == 4);
  REQUIRE(run.results.size() == 40);
  REQUIRE(run.errors.size() == 40);
  REQUIRE(run.wall_clock_seconds >= 0.0);
  for (int i = 0; i < 40; ++i) {
    INFO("trial " << i);
    REQUIRE(run.results[static_cast<std::size_t>(i)].index == i);
    REQUIRE(run.results[static_cast<std::size_t>(i)].seed == mc::trialSeed(cfg.master_seed, i));
    REQUIRE(run.errors[static_cast<std::size_t>(i)].empty());
  }
}

TEST_CASE("one worker and four workers give bitwise identical campaigns",
          "[mc][determinism]") {
  const auto single = mc::runCampaign<ProbeResult>(campaign(24, 1, 5150), rngTrial, false);
  const auto multi = mc::runCampaign<ProbeResult>(campaign(24, 4, 5150), rngTrial, false);
  REQUIRE(single.threads_used == 1);
  REQUIRE(multi.threads_used == 4);
  REQUIRE(single.results.size() == multi.results.size());
  for (std::size_t i = 0; i < single.results.size(); ++i) {
    INFO("trial " << i);
    const auto& a = single.results[i];
    const auto& b = multi.results[i];
    REQUIRE(a.index == static_cast<int>(i));
    REQUIRE(a.index == b.index);
    REQUIRE(a.seed == b.seed);
    REQUIRE(a.draws.size() == b.draws.size());
    for (std::size_t k = 0; k < a.draws.size(); ++k) REQUIRE(bits(a.draws[k]) == bits(b.draws[k]));
    // And both equal running the trial directly, outside any pool.
    const ProbeResult direct = rngTrial(static_cast<int>(i), mc::trialSeed(5150, static_cast<int>(i)));
    for (std::size_t k = 0; k < a.draws.size(); ++k)
      REQUIRE(bits(a.draws[k]) == bits(direct.draws[k]));
  }

  // A different master seed changes every trial.
  const auto other = mc::runCampaign<ProbeResult>(campaign(24, 4, 5151), rngTrial, false);
  for (std::size_t i = 0; i < other.results.size(); ++i)
    REQUIRE(other.results[i].draws.front() != single.results[i].draws.front());
}

TEST_CASE("a throwing trial is recorded in errors without stopping the others",
          "[mc][campaign]") {
  for (int threads : {1, 4}) {
    INFO("threads = " << threads);
    const auto run = mc::runCampaign<ProbeResult>(
        campaign(30, threads, 77),
        [](int i, std::uint64_t seed) {
          if (i % 7 == 3) throw std::runtime_error("trial " + std::to_string(i) + " diverged");
          return rngTrial(i, seed);
        },
        false);
    REQUIRE(run.results.size() == 30);
    REQUIRE(run.errors.size() == 30);
    int failures = 0;
    for (int i = 0; i < 30; ++i) {
      const auto k = static_cast<std::size_t>(i);
      INFO("trial " << i);
      if (i % 7 == 3) {
        ++failures;
        REQUIRE(run.errors[k] == "exception: trial " + std::to_string(i) + " diverged");
        // The slot keeps its default-constructed value.
        REQUIRE(run.results[k].index == -1);
        REQUIRE(run.results[k].draws.empty());
      } else {
        REQUIRE(run.errors[k].empty());
        REQUIRE(run.results[k].index == i);
        REQUIRE(run.results[k].seed == mc::trialSeed(77, i));
        REQUIRE(run.results[k].draws.size() == 9);
      }
    }
    REQUIRE(failures == 4);  // i = 3, 10, 17, 24
  }
}

TEST_CASE("worker count is clamped to the number of trials", "[mc][campaign]") {
  auto trivial = [](int i, std::uint64_t) { return i; };
  // More threads than trials: never spawn idle workers.
  const auto few = mc::runCampaign<int>(campaign(3, 8, 1), trivial, false);
  REQUIRE(few.threads_used == 3);
  REQUIRE(few.results == std::vector<int>{0, 1, 2});
  // threads = 0 means hardware concurrency, still at least 1 and at most the trial count.
  const auto hw = mc::runCampaign<int>(campaign(5, 0, 1), trivial, false);
  REQUIRE(hw.threads_used >= 1);
  REQUIRE(hw.threads_used <= 5);
  REQUIRE(hw.results == std::vector<int>{0, 1, 2, 3, 4});
  // Zero or negative trial counts produce an empty, well-formed campaign.
  for (int n : {0, -3}) {
    const auto none = mc::runCampaign<int>(campaign(n, 4, 1), trivial, false);
    REQUIRE(none.results.empty());
    REQUIRE(none.errors.empty());
    REQUIRE(none.threads_used == 1);
  }
}

TEST_CASE("a bool-valued campaign keeps every per-trial result", "[mc][campaign]") {
  // Regression for a data race found by this test: runCampaign used to write each result
  // straight into a std::vector<Result>. For Result = bool that is the bit-packed
  // std::vector<bool>, so two workers writing neighbouring indices did an unsynchronised
  // read-modify-write of the same machine word and one write could be lost. Results now go
  // into one-object-per-trial slots. The trial pattern alternates true/false so a lost
  // update is visible, and a start barrier (each worker's first trial waits until all four
  // workers are inside a trial) guarantees the workers really overlap; before the fix the
  // race showed up in essentially every campaign on a multi-core machine.
  constexpr int kWorkers = 4;
  const int trials = 65536;
  int corrupted_campaigns = 0;
  for (int rep = 0; rep < 20 && corrupted_campaigns == 0; ++rep) {
    std::atomic<int> arrived{0};
    const auto run = mc::runCampaign<bool>(
        campaign(trials, kWorkers, static_cast<std::uint64_t>(rep)),
        [&arrived](int i, std::uint64_t) {
          thread_local bool registered = false;  // fresh for every new worker thread
          if (!registered) {
            registered = true;
            arrived.fetch_add(1);
            while (arrived.load() < kWorkers) std::this_thread::yield();
          }
          return (i & 1) == 0;
        },
        false);
    int wrong = 0;
    for (int i = 0; i < trials; ++i)
      wrong += (run.results[static_cast<std::size_t>(i)] != ((i & 1) == 0)) ? 1 : 0;
    if (wrong > 0) ++corrupted_campaigns;
  }
  REQUIRE(corrupted_campaigns == 0);
}
