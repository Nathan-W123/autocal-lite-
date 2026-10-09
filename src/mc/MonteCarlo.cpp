// Reused from Aether6 (src/mc/MonteCarlo.cpp): computeStatistics is unchanged. The thread pool
// now lives in the runCampaign template in the header.
#include "aether/mc/MonteCarlo.hpp"

#include <cmath>

namespace aether::mc {

MetricStatistics computeStatistics(std::string name, std::string unit,
                                   std::vector<double> data) {
  MetricStatistics s;
  s.name = std::move(name);
  s.unit = std::move(unit);
  s.count = static_cast<int>(data.size());
  if (data.empty()) return s;
  std::sort(data.begin(), data.end());
  const double n = static_cast<double>(data.size());
  double sum = 0.0;
  for (double v : data) sum += v;
  s.mean = sum / n;
  double sq = 0.0;
  for (double v : data) sq += (v - s.mean) * (v - s.mean);
  s.stddev = data.size() > 1 ? std::sqrt(sq / (n - 1.0)) : 0.0;
  auto percentile = [&](double p) {
    const double pos = p * (n - 1.0);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = std::min(lo + 1, data.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return data[lo] * (1.0 - frac) + data[hi] * frac;
  };
  s.min = data.front();
  s.max = data.back();
  s.p05 = percentile(0.05);
  s.median = percentile(0.50);
  s.p95 = percentile(0.95);
  s.p99 = percentile(0.99);
  return s;
}

}  // namespace aether::mc
