#pragma once

#include "NvInfer.h"
#include "tensorrt_flow/tensorrt/trt_logger.hpp"
#include "tensorrt_flow/tensorrt/trt_utils.hpp"

#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <vector>


namespace tensorrt_flow {
namespace calibrator {

struct TrtCalibratorParamter {
  size_t      batch_size{1};
  std::string calibrator_dir_path;
  std::string regex_pattern = ".*";
  // if calibration_table_path = "", don't save
  std::string calibration_table_path{""};
  bool        use_calibration_cache{true};
};

template<typename CalibratorInterface> class TrtCalibrator : public CalibratorInterface {
public:
  TrtCalibrator(const TrtCalibratorParamter& parameter)
    : CalibratorInterface()
    , parameter_(parameter) {
    std::regex pattern(parameter.regex_pattern);
    for (const auto& entry : std::filesystem::directory_iterator(parameter.calibrator_dir_path)) {
      if (std::filesystem::is_regular_file(entry.path())) {
        if (std::regex_match(entry.path().string(), pattern)) {
          file_path_list_.push_back(entry);
          LOG_INFO("find calibration file %s", entry.path().filename().string().c_str());
        } else {
          LOG_WARN("jump %s in calibration path", entry.path().filename().string().c_str());
        }
      }
    }
  }
  virtual ~TrtCalibrator() noexcept = default;

  // override
  virtual int32_t getBatchSize() const noexcept override { return parameter_.batch_size; }
  // override
  virtual void const* readCalibrationCache(std::size_t& length) noexcept override {
    calibration_cache_.clear();

    std::filesystem::path file_path(parameter_.calibration_table_path);

    if (!std::filesystem::exists(file_path)) { return nullptr; }

    std::ifstream input(parameter_.calibration_table_path, std::ios::binary);
    input >> std::noskipws;
    if (parameter_.use_calibration_cache && input.good()) { std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calibration_cache_)); }

    length = calibration_cache_.size();

    void* output = nullptr;
    if (length) {
      LOG_INFO("Using cached calibration table to build INT8 trt engine...");
      output = calibration_cache_.data();
    } else {
      LOG_INFO("Creating new calibration table to build INT8 trt engine...");
    }
    return output;
  }

  // override
  void writeCalibrationCache(const void* ptr, std::size_t length) noexcept override {
    std::filesystem::path file_path(parameter_.calibration_table_path);
    if (!std::filesystem::exists(file_path)) {
      if (!std::filesystem::exists(file_path.parent_path())) { std::filesystem::create_directories(file_path.parent_path()); }
    }
    std::ofstream output(parameter_.calibration_table_path);
    output.write(reinterpret_cast<const char*>(ptr), length);
    output.close();
  }

  // override
  virtual bool getBatch(void* bindings[], char const* names[], int32_t nbBindings) noexcept override {

    // no sufficient file
    if (file_index_ + parameter_.batch_size >= file_path_list_.size()) { return false; }

    std::vector<std::string> file_path_names;
    for (size_t index = 0; index < parameter_.batch_size; ++index) { file_path_names.emplace_back(file_path_list_[file_index_++].path().string()); }
    auto device_memories = process(file_path_names);

    if (device_memories.size() != nbBindings) {
      LOG_ERROR("load calibration data file in TrtCalibrator, device_memories size : %d, nbBindings : %d", device_memories.size(), nbBindings);
      return false;
    }


    for (size_t index = 0; index < device_memories.size(); ++index) { bindings[index] = device_memories[index]; }

    return true;
  }

  virtual std::vector<void*> process(const std::vector<std::string>& file_path_name_list) = 0;

protected:
  TrtCalibratorParamter                         parameter_{};
  std::vector<std::filesystem::directory_entry> file_path_list_{};
  size_t                                        file_index_{0};
  std::vector<char>                             calibration_cache_{};
};


}   // namespace calibrator


}   // namespace tensorrt_flow