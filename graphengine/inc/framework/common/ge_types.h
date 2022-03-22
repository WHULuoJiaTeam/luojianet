/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef INC_FRAMEWORK_COMMON_GE_TYPES_H_
#define INC_FRAMEWORK_COMMON_GE_TYPES_H_

#include <cstdint>

#include <string>
#include <vector>

#include "framework/common/fmk_error_codes.h"
#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "external/ge/ge_api_types.h"

namespace ge {
enum RuntimeType { HOST = 0, DEVICE = 1 };

enum class PerfLevel : int32_t {
  GEN_TASK_WITH_FUSION = -1,
  GEN_TASK_WITHOUT_L2FUSION = 3,
  GEN_TASK_WITHOUT_FUSION = 4
};

enum FrameworkType {
  CAFFE = 0,
  LUOJIANET = 1,
  TENSORFLOW = 3,
  ANDROID_NN = 4,
  ONNX = 5,
};

enum class GraphStage : int64_t { GRAPH_STAGE_FUZZ = 0, GRAPH_STAGE_RESERVED };

const char *const kGraphDumpStage = "DumpStage";

const std::map<std::string, std::string> kFwkTypeToStr = {
    {"0", "Caffe"}, {"1", "LuoJiaNet"}, {"3", "TensorFlow"}, {"4", "Android_NN"}, {"5", "Onnx"}};

enum OpEngineType {
  ENGINE_SYS = 0,  // default engine
  ENGINE_AICORE = 1,
  ENGINE_VECTOR = 2,
  ENGINE_AICUBE = 3,   // not support
  ENGINE_AIVECTOR = 4  // not support
};

enum InputAippType { DATA_WITHOUT_AIPP = 0, DATA_WITH_STATIC_AIPP, DATA_WITH_DYNAMIC_AIPP, DYNAMIC_AIPP_NODE };

const char_t *const GE_ENGINE_ATTR_MEM_TYPE_HBM = "HBM";
const char_t *const GE_OPTION_EXEC_PLACEMENT = "ge.exec.placement";

// profiling data

const std::string kTaskTypeAicore = "AI_CORE";
const std::string kTaskTypeAicpu = "AI_CPU";
const std::string kTaskTypeInvalid = "TASK_TYPE_INVALID";
const std::string kTaskTypeFftsPlus = "FFTS_PLUS";

// dynamic execute mode
const char_t *const kLazyRecompile = "lazy_recompile";

// Data cache, including data address and length
struct DataBuffer {
 public:
  void *data;       // Data address
  uint64_t length;  // Data length
  bool isDataSupportMemShare = false;
  uint32_t placement = 0U;
  DataBuffer(void *data_in, const uint64_t data_len, const bool is_support_mem_share, const uint32_t placement = 0U)
      : data(data_in), length(data_len), isDataSupportMemShare(is_support_mem_share), placement(placement) {}

  DataBuffer() : data(nullptr), length(0UL), isDataSupportMemShare(false) {}
};

///
/// @ingroup domi_ome
/// @brief External input data
///
struct InputData {
  uint32_t index;                            // Index of input data
  uint32_t timestamp;                        // Data creation time
  uint32_t timeout;                          // Processing timeout
  uint32_t model_id;                         // Model ID required for data processing
  uint64_t request_id = 0UL;                 // Request ID
  std::vector<DataBuffer> blobs;             // Actual input data, currently only supports one input
  bool is_dynamic_batch = false;             // Whether is dynamic batch size scene, default:false
  std::string batch_label;                   // Gear used for current inference in dynamic batch scene
  std::vector<std::vector<int64_t>> shapes;  // Input shapes
};

/// Output result structure definition
struct OutputData {
  uint32_t index;     // Index of input data
  uint32_t model_id;  // The model ID corresponding to the processing result
  /// Output data cache, arranged in sequence of output operators.
  /// If the operator has multiple outputs,
  /// the data buffer order of the operator is the same as that defined in the
  /// offline model
  std::vector<DataBuffer> blobs;
};

// The definition of command data structure
struct Command {
  std::string cmd_type;                 // Command type
  std::vector<std::string> cmd_params;  // Command params
  uint64_t module_index;                // prof module
};

// The definition of I/O shape description
struct ShapeDescription {
  int64_t num = 0L;
  int64_t channel = 0L;
  int64_t height = 0L;
  int64_t width = 0L;
  std::vector<int64_t> dims;
  std::vector<std::pair<int64_t, int64_t>> shape_ranges;
};

// Definition of input and output description information
struct InputOutputDescInfo {
  std::string name;
  uint64_t size;
  uint32_t data_type;
  ShapeDescription shape_info;
};

// Definition of model io dims
struct InputOutputDims {
  std::string name;
  size_t dim_num;
  uint32_t size;
  std::vector<int64_t> dims;
};

// Definition of model io dims
struct OriginInputInfo {
  Format format;
  DataType data_type;
  uint32_t dim_num;
};

// The structure of AIPP info
struct AippConfigInfo {
  int8_t aipp_mode;
  int8_t input_format;
  int32_t src_image_size_w;
  int32_t src_image_size_h;
  int8_t crop;
  int32_t load_start_pos_w;
  int32_t load_start_pos_h;
  int32_t crop_size_w;
  int32_t crop_size_h;
  int8_t resize;
  int32_t resize_output_w;
  int32_t resize_output_h;
  int8_t padding;
  int32_t left_padding_size;
  int32_t right_padding_size;
  int32_t top_padding_size;
  int32_t bottom_padding_size;
  int8_t csc_switch;
  int8_t rbuv_swap_switch;
  int8_t ax_swap_switch;
  int8_t single_line_mode;
  int32_t matrix_r0c0;
  int32_t matrix_r0c1;
  int32_t matrix_r0c2;
  int32_t matrix_r1c0;
  int32_t matrix_r1c1;
  int32_t matrix_r1c2;
  int32_t matrix_r2c0;
  int32_t matrix_r2c1;
  int32_t matrix_r2c2;
  int32_t output_bias_0;
  int32_t output_bias_1;
  int32_t output_bias_2;
  int32_t input_bias_0;
  int32_t input_bias_1;
  int32_t input_bias_2;
  int32_t mean_chn_0;
  int32_t mean_chn_1;
  int32_t mean_chn_2;
  int32_t mean_chn_3;
  float32_t min_chn_0;
  float32_t min_chn_1;
  float32_t min_chn_2;
  float32_t min_chn_3;
  float32_t var_reci_chn_0;
  float32_t var_reci_chn_1;
  float32_t var_reci_chn_2;
  float32_t var_reci_chn_3;
  int8_t support_rotation;
  uint32_t related_input_rank;
  uint32_t max_src_image_size;
};

// The structure of offline Modeldata
struct ModelData {
  void *model_data = nullptr;  // Model binary data start addr
  uint32_t model_len = 0U;     // Model binary data length
  int32_t priority = 0;        // Model priority
  std::string key;             // Key path for encrypt model, Empty for unencrypt
  std::string om_name;         // om file name, used for data dump
};

// The definition of Model information
struct ModelInfo {
  uint32_t version = 0U;
  std::string name;
  bool is_encrypt = false;  //  0:unencrypt, 1:encrypt
  std::vector<ShapeDescription> input_desc;
  std::vector<ShapeDescription> output_desc;
  uint8_t reserved[3] = {0U};  // 3-byte reserved field
};

// Asynchronous callback interface, implemented by the caller
class GE_FUNC_VISIBILITY ModelListener {
 public:
  virtual ~ModelListener() {}
  ///
  /// @brief Asynchronous callback interface
  /// @param [in] model_id   Model ID of the callback
  /// @param [in] data_index Index of the input_data
  /// @param [in] resultCode Execution results
  ///
  virtual Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t result_code,
                               std::vector<ge::Tensor> &outputs) = 0;

  virtual void SetCallback(const RunAsyncCallback &callback){};

  virtual uint32_t GetResultCode() {
    return 0U;
  };

  virtual Status ResetResult() {
    return SUCCESS;
  };
};

// OMM configuration item
struct Options {
  int64_t session_id;
  int32_t device_id;
  std::string job_id;
  bool isUseHcom;
  bool isUseHvd;
  bool deployMode;
  bool isAICPUMode;
  bool enable_atomic;
  std::string podName;
  int64_t rankId;
  std::string rankTableFile;
  int32_t ge_hccl_flag = 0;
  int32_t physical_device_id;
  std::string profiling_mode;
  std::string profiling_options;
};

// Profiling info of task
struct TaskDescInfo {
  std::string model_name;
  std::string op_name;
  std::string op_type;
  uint32_t block_dim;
  uint32_t task_id;
  uint32_t stream_id;
  std::string shape_type;
  int64_t cur_iter_num;
  std::string task_type;
  std::vector<Format> input_format;
  std::vector<std::vector<int64_t>> input_shape;
  std::vector<DataType> input_data_type;
  std::vector<Format> output_format;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<DataType> output_data_type;
  uint32_t context_id;
};

struct OpDescInfo {
  std::string op_name;
  std::string op_type;
  uint32_t task_id = 0U;
  uint32_t stream_id = 0U;
  uint32_t imply_type = 0U;
  uint32_t block_dim = 0U;
  std::string op_file_path;
  std::string dev_func;
  std::string tvm_magic;
  uint32_t tiling_key = 0U;
  uintptr_t args = 0U;
  std::string tiling_data;
  std::string node_info;
  std::vector<int64_t> workspace_bytes;
  std::vector<Format> input_format;
  std::vector<std::vector<int64_t>> input_shape;
  std::vector<DataType> input_data_type;
  std::vector<void *> input_addrs;
  std::vector<int64_t> input_size;
  std::vector<Format> output_format;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<DataType> output_data_type;
  std::vector<void *> output_addrs;
  std::vector<int64_t> output_size;
};
struct ModelDumpConfig {
  std::string model_name;
  std::vector<std::string> layers;
};

struct DumpConfig {
  std::string dump_path;
  std::string dump_mode;
  std::string dump_status;
  std::string dump_op_switch;
  std::string dump_debug;
  std::vector<ModelDumpConfig> dump_list;
};
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_GE_TYPES_H_
