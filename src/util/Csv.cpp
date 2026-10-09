// Reused verbatim from Aether6 (github.com/Nathan-W123/Aether6, MIT, commit 6b8dc56).
#include "aether/util/Csv.hpp"

#include <iomanip>
#include <stdexcept>

namespace aether::util {

CsvWriter::CsvWriter(const std::string& path, std::vector<std::string> header, int precision)
    : file_(path), header_(std::move(header)) {
  if (!file_) throw std::runtime_error("CsvWriter: cannot open '" + path + "' for writing");
  file_ << std::setprecision(precision);
  for (std::size_t i = 0; i < header_.size(); ++i) {
    if (i) file_ << ',';
    file_ << header_[i];
  }
  file_ << '\n';
}

void CsvWriter::writeRow(const std::vector<double>& values) {
  if (values.size() != header_.size())
    throw std::invalid_argument("CsvWriter: row has " + std::to_string(values.size()) +
                                " values but the header has " + std::to_string(header_.size()));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) file_ << ',';
    file_ << values[i];
  }
  file_ << '\n';
  ++rows_;
}

void CsvWriter::close() {
  if (file_.is_open()) file_.close();
}

CsvWriter::~CsvWriter() { close(); }

}  // namespace aether::util
