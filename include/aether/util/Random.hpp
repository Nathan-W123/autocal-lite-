// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
/// \file Random.hpp
/// \brief Deterministic, reproducible random-number generation.
///
/// Every stochastic component in Aether6 (sensor noise, gusts, Monte-Carlo dispersions)
/// draws from an explicitly seeded `Rng`. Seeds are derived from a single master seed via
/// SplitMix64 so that a trial index maps to a fixed, platform-independent stream.
#pragma once

#include <cstdint>
#include <random>

#include "aether/core/Types.hpp"

namespace aether::util {

/// \brief SplitMix64 seed mixer (Steele et al., 2014). Used to derive independent,
/// well-separated seeds from a master seed and a stream index.
inline std::uint64_t splitMix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/// \brief Derive a reproducible sub-seed from a master seed, a stream id and a trial index.
inline std::uint64_t deriveSeed(std::uint64_t master, std::uint64_t stream, std::uint64_t index) {
  return splitMix64(splitMix64(master ^ (stream * 0xD1B54A32D192ED03ULL)) + index);
}

/// \brief Thin deterministic wrapper around std::mt19937_64 with the draws Aether6 needs.
class Rng {
 public:
  explicit Rng(std::uint64_t seed = 0) : engine_(seed), seed_(seed) {}

  /// Standard normal sample, \f$\mathcal{N}(0,1)\f$.
  double gaussian() { return normal_(engine_); }

  /// Normal sample with the given mean and standard deviation.
  double gaussian(double mean, double stddev) { return mean + stddev * normal_(engine_); }

  /// Uniform sample on [lo, hi).
  double uniform(double lo = 0.0, double hi = 1.0) {
    return lo + (hi - lo) * uniform_(engine_);
  }

  /// Vector of independent standard normals.
  Vec3 gaussian3() { return Vec3(gaussian(), gaussian(), gaussian()); }

  /// Seed this generator was constructed with.
  std::uint64_t seed() const { return seed_; }

  /// Underlying engine (exposed for std distributions that need it).
  std::mt19937_64& engine() { return engine_; }

 private:
  std::mt19937_64 engine_;
  std::normal_distribution<double> normal_{0.0, 1.0};
  std::uniform_real_distribution<double> uniform_{0.0, 1.0};
  std::uint64_t seed_;
};

}  // namespace aether::util
