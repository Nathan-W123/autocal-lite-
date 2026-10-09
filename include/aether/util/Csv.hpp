// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
/// \file Csv.hpp
/// \brief Minimal buffered CSV writer with a fixed header.
#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace aether::util {

/// \brief Writes rows of doubles to a CSV file with a fixed, named header.
///
/// The number of values in every row must match the header; a mismatch throws. Values are
/// written with `precision` significant digits (default 9, which round-trips single-precision
/// exactly and keeps file sizes reasonable for long runs).
class CsvWriter {
 public:
  /// \param path Output file path (parent directories must already exist).
  /// \param header Column names.
  /// \param precision Significant digits written per value.
  CsvWriter(const std::string& path, std::vector<std::string> header, int precision = 9);

  /// Append a row. \throws std::invalid_argument if the size does not match the header.
  void writeRow(const std::vector<double>& values);

  /// Flush and close the file.
  void close();

  /// Number of rows written so far.
  std::size_t rows() const { return rows_; }

  /// Column names.
  const std::vector<std::string>& header() const { return header_; }

  ~CsvWriter();

 private:
  std::ofstream file_;
  std::vector<std::string> header_;
  std::size_t rows_ = 0;
};

}  // namespace aether::util
