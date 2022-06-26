/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINNIE_BENCHMARK_BENCHMARK_BASE_H_
#define MINNIE_BENCHMARK_BENCHMARK_BASE_H_

#include <signal.h>
#include <random>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <map>
#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <cfloat>
#include <utility>
#ifndef BENCHMARK_CLIP_JSON
#include <nlohmann/json.hpp>
#endif
#include "include/model.h"
#include "include/api/types.h"
#include "include/api/format.h"
#include "tools/common/flag_parser.h"
#include "src/common/file_utils.h"
#include "src/common/utils.h"
#include "ir/dtype/type_id.h"
#include "schema/model_generated.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
#define BENCHMARK_LOG_ERROR(str)   \
  do {                             \
    MS_LOG(ERROR) << str;          \
    std::cerr << str << std::endl; \
  } while (0);

enum MS_API InDataType { kImage = 0, kBinary = 1 };

enum MS_API AiModelDescription_Frequency {
  AiModelDescription_Frequency_LOW = 1,
  AiModelDescription_Frequency_MEDIUM = 2,
  AiModelDescription_Frequency_HIGH = 3,
  AiModelDescription_Frequency_EXTREME = 4
};

enum MS_API DumpMode { DUMP_MODE_ALL = 0, DUMP_MODE_INPUT = 1, DUMP_MODE_OUTPUT = 2 };

constexpr float relativeTolerance = 1e-5;
constexpr float absoluteTolerance = 1e-8;
constexpr int CosineErrMaxVal = 2;
constexpr float kFloatMSEC = 1000.0f;

constexpr int kNumPrintMin = 5;
constexpr const char *DELIM_COLON = ":";
constexpr const char *DELIM_COMMA = ",";
constexpr const char *DELIM_SLASH = "/";
constexpr size_t kEncMaxLen = 16;

extern const std::unordered_map<int, std::string> kTypeIdMap;
extern const std::unordered_map<mindspore::Format, std::string> kTensorFormatMap;

const std::unordered_map<std::string, mindspore::ModelType> ModelTypeMap{
  {"MindIR_Lite", mindspore::ModelType::kMindIR_Lite}, {"MindIR", mindspore::ModelType::kMindIR}};

namespace dump {
constexpr auto kConfigPath = "MINDSPORE_DUMP_CONFIG";
constexpr auto kSettings = "common_dump_settings";
constexpr auto kMode = "dump_mode";
constexpr auto kPath = "path";
constexpr auto kNetName = "net_name";
constexpr auto kInputOutput = "input_output";
constexpr auto kKernels = "kernels";
}  // namespace dump

#ifdef ENABLE_ARM64
struct PerfResult {
  int64_t nr;
  struct {
    int64_t value;
    int64_t id;
  } values[2];
};
struct PerfCount {
  int64_t value[2];
};
#endif

struct MS_API CheckTensor {
  CheckTensor(const std::vector<size_t> &shape, const std::vector<float> &data,
              const std::vector<std::string> &strings_data = {""}) {
    this->shape = shape;
    this->data = data;
    this->strings_data = strings_data;
  }
  std::vector<size_t> shape;
  std::vector<float> data;
  std::vector<std::string> strings_data;
};

class MS_API BenchmarkFlags : public virtual FlagParser {
 public:
  BenchmarkFlags() {
    // common
    AddFlag(&BenchmarkFlags::model_file_, "modelFile", "Input model file", "");
    AddFlag(&BenchmarkFlags::model_type_, "modelType", "Input model type. MindIR | MindIR_Lite", "MindIR");
    AddFlag(&BenchmarkFlags::in_data_file_, "inDataFile", "Input data file, if not set, use random input", "");
    AddFlag(&BenchmarkFlags::config_file_, "configFile", "Config file", "");
    AddFlag(&BenchmarkFlags::device_, "device", "CPU | GPU | NPU | Ascend310 | Ascend710", "CPU");
    AddFlag(&BenchmarkFlags::cpu_bind_mode_, "cpuBindMode", "Input 0 for NO_BIND, 1 for HIGHER_CPU, 2 for MID_CPU.", 1);
    // MarkPerformance
    AddFlag(&BenchmarkFlags::loop_count_, "loopCount", "Run loop count", 10);
    AddFlag(&BenchmarkFlags::num_threads_, "numThreads", "Run threads number", 2);
    AddFlag(&BenchmarkFlags::enable_fp16_, "enableFp16", "Enable float16", false);
    AddFlag(&BenchmarkFlags::enable_parallel_, "enableParallel", "Enable subgraph parallel : true | false", false);
    AddFlag(&BenchmarkFlags::warm_up_loop_count_, "warmUpLoopCount", "Run warm up loop", 3);
    AddFlag(&BenchmarkFlags::time_profiling_, "timeProfiling", "Run time profiling", false);
    AddFlag(&BenchmarkFlags::perf_profiling_, "perfProfiling",
            "Perf event profiling(only instructions statics enabled currently)", false);
    AddFlag(&BenchmarkFlags::perf_event_, "perfEvent", "CYCLE|CACHE|STALL", "CYCLE");
    // MarkAccuracy
    AddFlag(&BenchmarkFlags::benchmark_data_file_, "benchmarkDataFile", "Benchmark data file path", "");
    AddFlag(&BenchmarkFlags::benchmark_data_type_, "benchmarkDataType",
            "Benchmark data type. FLOAT | INT32 | INT8 | UINT8", "FLOAT");
    AddFlag(&BenchmarkFlags::accuracy_threshold_, "accuracyThreshold", "Threshold of accuracy", 0.5);
    AddFlag(&BenchmarkFlags::cosine_distance_threshold_, "cosineDistanceThreshold", "cosine distance threshold", -1.1);
    AddFlag(&BenchmarkFlags::resize_dims_in_, "inputShapes",
            "Shape of input data, the format should be NHWC. e.g. 1,32,32,32:1,1,32,32,1", "");
    AddFlag(&BenchmarkFlags::decrypt_key_str_, "decryptKey",
            "The key used to decrypt the file, expressed in hexadecimal characters. Only support AES-GCM and the key "
            "length is 16.",
            "");
    AddFlag(&BenchmarkFlags::crypto_lib_path_, "cryptoLibPath", "Pass the crypto library path.", "");
    AddFlag(&BenchmarkFlags::enable_parallel_predict_, "enableParallelPredict", "Enable model parallel : true | false",
            false);
    AddFlag(&BenchmarkFlags::parallel_num_, "parallelNum", "parallel num of parallel predict", 2);
    AddFlag(&BenchmarkFlags::parallel_task_num_, "parallelTaskNum", "parallel task num of parallel predict", 2);
    AddFlag(&BenchmarkFlags::workers_num_, "workersNum", "works num of parallel predict", 2);
#ifdef ENABLE_OPENGL_TEXTURE
    AddFlag(&BenchmarkFlags::enable_gl_texture_, "enableGLTexture", "Enable GlTexture2D", false);
#endif
  }

  ~BenchmarkFlags() override = default;

  void InitInputDataList();

  void InitResizeDimsList();

 public:
  // common
  bool enable_parallel_predict_ = false;
  int parallel_num_ = 2;
  int parallel_task_num_ = 2;
  int workers_num_ = 2;
  std::string model_file_;
  std::string in_data_file_;
  std::string config_file_;
  std::string model_type_;
  std::vector<std::string> input_data_list_;
  InDataType in_data_type_ = kBinary;
  std::string in_data_type_in_ = "bin";
  int cpu_bind_mode_ = 1;
  // MarkPerformance
  int loop_count_ = 10;
  int num_threads_ = 2;
  bool enable_fp16_ = false;
#ifdef ENABLE_OPENGL_TEXTURE
  bool enable_gl_texture_ = false;
#endif
  bool enable_parallel_ = false;
  int warm_up_loop_count_ = 3;
  // MarkAccuracy
  std::string benchmark_data_file_;
  std::string benchmark_data_type_ = "FLOAT";
  float accuracy_threshold_ = 0.5;
  float cosine_distance_threshold_ = -1.1;
  // Resize
  std::string resize_dims_in_;
  std::vector<std::vector<int>> resize_dims_;

  std::string device_ = "CPU";
  bool time_profiling_ = false;
  bool perf_profiling_ = false;
  std::string perf_event_ = "CYCLE";
  bool dump_tensor_data_ = false;
  bool print_tensor_data_ = false;
  std::string decrypt_key_str_;
  std::string dec_mode_ = "AES-GCM";
  std::string crypto_lib_path_;
};

class MS_API BenchmarkBase {
 public:
  explicit BenchmarkBase(BenchmarkFlags *flags) : flags_(flags) {}

  virtual ~BenchmarkBase();

  int Init();
  virtual int RunBenchmark() = 0;

 protected:
  virtual int LoadInput() = 0;

  virtual int GenerateInputData() = 0;

  int GenerateRandomData(size_t size, void *data, int data_type);

  virtual int ReadInputFile() = 0;

  int ReadCalibData();

  int ReadTensorData(std::ifstream &in_file_stream, const std::string &tensor_name, const std::vector<size_t> &dims);

  virtual int GetDataTypeByTensorName(const std::string &tensor_name) = 0;

  virtual int CompareOutput() = 0;

  int CompareStringData(const std::string &name, const std::vector<std::string> &calib_strings,
                        const std::vector<std::string> &output_strings);

  int InitDumpConfigFromJson(char *path);

  int InitCallbackParameter();

  virtual int InitTimeProfilingCallbackParameter() = 0;

  virtual int InitPerfProfilingCallbackParameter() = 0;

  virtual int InitDumpTensorDataCallbackParameter() = 0;

  virtual int InitPrintTensorDataCallbackParameter() = 0;

  int PrintResult(const std::vector<std::string> &title, const std::map<std::string, std::pair<int, float>> &result);

#ifdef ENABLE_ARM64
  int PrintPerfResult(const std::vector<std::string> &title,
                      const std::map<std::string, std::pair<int, struct PerfCount>> &result);
#endif

  // tensorData need to be converter first
  template <typename T, typename ST>
  float CompareData(const std::string &nodeName, const std::vector<ST> &msShape, const void *tensor_data) {
    const T *msTensorData = static_cast<const T *>(tensor_data);
    auto iter = this->benchmark_data_.find(nodeName);
    if (iter != this->benchmark_data_.end()) {
      std::vector<size_t> castedMSShape;
      size_t shapeSize = 1;
      for (ST dim : msShape) {
        castedMSShape.push_back(dim);
        shapeSize *= dim;
      }

      CheckTensor *calibTensor = iter->second;
      if (calibTensor->shape != castedMSShape) {
        std::ostringstream oss;
        oss << "Shape of mslite output(";
        for (auto dim : castedMSShape) {
          oss << dim << ",";
        }
        oss << ") and shape source model output(";
        for (auto dim : calibTensor->shape) {
          oss << dim << ",";
        }
        oss << ") are different";
        std::cerr << oss.str() << std::endl;
        MS_LOG(ERROR) << oss.str().c_str();
        return RET_ERROR;
      }
      size_t errorCount = 0;
      float meanError = 0;
      std::cout << "Data of node " << nodeName << " : ";
      for (size_t j = 0; j < shapeSize; j++) {
        if (j < 50) {
          std::cout << static_cast<float>(msTensorData[j]) << " ";
        }

        if (std::is_same<T, float>::value && (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j]))) {
          std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
          MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
          return RET_ERROR;
        }

        auto tolerance = absoluteTolerance + relativeTolerance * fabs(calibTensor->data.at(j));
        auto absoluteError = std::fabs(msTensorData[j] - calibTensor->data.at(j));
        if (absoluteError > tolerance) {
          if (fabs(calibTensor->data.at(j) - 0.0f) < FLT_EPSILON) {
            if (absoluteError > 1e-5) {
              meanError += absoluteError;
              errorCount++;
            } else {
              continue;
            }
          } else {
            // just assume that atol = rtol
            meanError += absoluteError / (fabs(calibTensor->data.at(j)) + FLT_MIN);
            errorCount++;
          }
        }
      }
      std::cout << std::endl;
      if (meanError > 0.0f) {
        meanError /= errorCount;
      }

      if (meanError <= 0.0000001) {
        std::cout << "Mean bias of node/tensor " << nodeName << " : 0%" << std::endl;
      } else {
        std::cout << "Mean bias of node/tensor " << nodeName << " : " << meanError * 100 << "%" << std::endl;
      }
      return meanError;
    } else {
      MS_LOG(INFO) << "%s is not in Source Model output", nodeName.c_str();
      return RET_ERROR;
    }
  }

  void GetMeanError(double sum_a, double sum_b, double dot_sum, float *mean_error) {
    if (fabs(sum_a) < DBL_EPSILON && fabs(sum_b) < FLT_EPSILON) {
      *mean_error = 1;
    } else if (fabs(sum_a * sum_b) < DBL_EPSILON) {
      if (fabs(sum_a) < FLT_EPSILON || fabs(sum_b) < FLT_EPSILON) {
        *mean_error = 1;
      } else {
        *mean_error = 0;
      }
    } else {
      *mean_error = dot_sum / (sqrt(sum_a) * sqrt(sum_b));
    }
  }

  // tensorData need to be converter first
  template <typename T, typename ST>
  int CompareDatabyCosineDistance(const std::string &nodeName, const std::vector<ST> &msShape, const void *tensor_data,
                                  float *mean_error) {
    if (mean_error == nullptr) {
      MS_LOG(ERROR) << "mean_error is nullptr";
      return RET_ERROR;
    }
    if (tensor_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      return RET_ERROR;
    }
    const T *msTensorData = static_cast<const T *>(tensor_data);
    auto iter = this->benchmark_data_.find(nodeName);
    if (iter != this->benchmark_data_.end()) {
      std::vector<size_t> castedMSShape;
      size_t shapeSize = 1;
      for (int64_t dim : msShape) {
        castedMSShape.push_back(size_t(dim));
        shapeSize *= dim;
      }

      CheckTensor *calibTensor = iter->second;
      if (calibTensor->shape != castedMSShape) {
        std::ostringstream oss;
        oss << "Shape of mslite output(";
        for (auto dim : castedMSShape) {
          oss << dim << ",";
        }
        oss << ") and shape source model output(";
        for (auto dim : calibTensor->shape) {
          oss << dim << ",";
        }
        oss << ") are different";
        std::cerr << oss.str() << std::endl;
        MS_LOG(ERROR) << oss.str().c_str();
        return RET_ERROR;
      }

      double dot_sum = 0;
      double sum_a = 0;
      double sum_b = 0;
      std::cout << "Data of node " << nodeName << " : ";
      for (size_t j = 0; j < shapeSize; j++) {
        if (j < 50) {
          std::cout << static_cast<float>(msTensorData[j]) << " ";
        }

        if (std::is_same<T, float>::value && (std::isnan(msTensorData[j]) || std::isinf(msTensorData[j]))) {
          std::cerr << "Output tensor has nan or inf data, compare fail" << std::endl;
          MS_LOG(ERROR) << "Output tensor has nan or inf data, compare fail";
          return RET_ERROR;
        }
        dot_sum += static_cast<double>(msTensorData[j]) * calibTensor->data.at(j);
        sum_a += static_cast<double>(msTensorData[j]) * msTensorData[j];
        sum_b += static_cast<double>(calibTensor->data.at(j)) * calibTensor->data.at(j);
      }
      GetMeanError(sum_a, sum_b, dot_sum, mean_error);
      std::cout << std::endl;
      std::cout << "Mean cosine distance of node/tensor " << nodeName << " : " << (*mean_error) * 100 << "%"
                << std::endl;
      return RET_OK;
    } else {
      MS_LOG(ERROR) << "%s is not in Source Model output", nodeName.c_str();
      return RET_ERROR;
    }
  }

  template <typename T, typename Distribution>
  void FillInputData(size_t size, void *data, Distribution distribution) {
    MS_ASSERT(data != nullptr);
    size_t elements_num = size / sizeof(T);
    (void)std::generate_n(static_cast<T *>(data), elements_num,
                          [&]() { return static_cast<T>(distribution(random_engine_)); });
  }

  int CheckThreadNumValid();

  int CheckModelValid();

  int CheckDeviceTypeValid();

 protected:
  BenchmarkFlags *flags_;
  std::vector<std::string> benchmark_tensor_names_;
  std::unordered_map<std::string, CheckTensor *> benchmark_data_;
  std::unordered_map<std::string, int> data_type_map_{
    {"FLOAT", kNumberTypeFloat}, {"INT8", kNumberTypeInt8}, {"INT32", kNumberTypeInt32}, {"UINT8", kNumberTypeUInt8}};
  int msCalibDataType = kNumberTypeFloat;

  // callback parameters
  uint64_t op_begin_ = 0;
  int op_call_times_total_ = 0;
  float op_cost_total_ = 0.0f;
  std::map<std::string, std::pair<int, float>> op_times_by_type_;
  std::map<std::string, std::pair<int, float>> op_times_by_name_;
#ifndef BENCHMARK_CLIP_JSON
  // dump data
  nlohmann::json dump_cfg_json_;
#endif
  std::string dump_file_output_dir_;
#ifdef ENABLE_ARM64
  int perf_fd = 0;
  int perf_fd2 = 0;
  float op_cost2_total_ = 0.0f;
  std::map<std::string, std::pair<int, struct PerfCount>> op_perf_by_type_;
  std::map<std::string, std::pair<int, struct PerfCount>> op_perf_by_name_;
#endif
  std::mt19937 random_engine_;
};
#ifdef SUPPORT_NNIE
int SvpSysInit();
int SvpSysExit();
#endif

}  // namespace mindspore::lite
#endif  // MINNIE_BENCHMARK_BENCHMARK_BASE_H_
