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

#ifndef FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_
#define FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_

#include <map>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/types.h"
#include "runtime/kernel.h"

namespace fe {
const uint32_t L2_MAXDATANUM = 8;
struct FusionOpSrc {
  uint32_t src_op_id;
  ge::AnchorPtr src_anchor;
  int32_t fusion_src_index;
  int32_t fusion_dst_index;
};

struct FusionOpDst {
  uint32_t dst_op_id;
  ge::AnchorPtr dst_anchor;
};

struct FusionDataFlow {
  std::pair<ge::AnchorPtr, ge::AnchorPtr> edge;
  std::pair<std::string, ge::AnchorPtr> node_dataindex_pair;
};

typedef struct tag_l2_fusion_data {
  uint32_t l2Index;
  uint64_t l2Addr;
  uint64_t l2PageNum;
} L2FusionData_t;
typedef std::map<uint64_t, L2FusionData_t> L2FusionDataMap_t;

typedef struct tag_fe_sm_desc {
  rtL2Ctrl_t l2ctrl;
  std::string node_name[L2_MAXDATANUM];
  uint8_t output_index[L2_MAXDATANUM];
} fe_sm_desc_t;

typedef struct TagTaskL2FusionInfo {
  std::string node_name;
  fe_sm_desc_t l2_info;
  L2FusionDataMap_t input;
  L2FusionDataMap_t output;
  uint32_t is_used;
} TaskL2FusionInfo_t;

using L2FusionInfoPtr = std::shared_ptr<TaskL2FusionInfo_t>;

typedef struct ToOpStruct {
  int64_t op_l1_space = 0;
  std::vector<int64_t> op_l1_fusion_type;
  int64_t op_l1_workspace_flag = 0; // for workspace flag
  int64_t op_l1_workspace_size = 0;
  std::vector<std::vector<int64_t>> slice_input_shape;
  std::vector<std::vector<int64_t>> slice_output_shape;
  std::vector<std::vector<int64_t>>
      slice_input_offset; // conv & pooling & ReadSelect
  std::vector<std::vector<int64_t>> slice_output_offset; // WriteSelect
  std::vector<uint32_t> total_shape;
  uint32_t split_index = 0;
  ToOpStruct() {
    // set invalid value for essential variable
    op_l1_space = -1;
    op_l1_workspace_size = -1;
  }
} ToOpStruct_t;

enum SlicePattern {
  ELEMENT_WISE = 0,
  ELEMENT_WISE_BROADCAST,
  BROADCAST,
  SLIDING_WINDOW,
  SLIDING_WINDOW_DECONV,
  CUBE_MATMUL,
  SLICE_PATTERN_REDUCE,
  SLICE_PATTERN_RESIZE,
  SLICE_PATTERN_SCATTER,
  SLICE_PATTERN_SEGMENT,
  PATTERN_RESERVED
};

enum OpImplType {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,   // custom constant op
  EN_IMPL_CUSTOM_TIK,                // custom tik op
  EN_IMPL_CUSTOM_TBE,                // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,           // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,            // Huawei built-in cce op
  EN_IMPL_HW_TIK,                    // Huawei built-in tik op
  EN_IMPL_HW_TBE,                    // Huawei built-in tbe op
  EN_IMPL_RL,                        // RL op
  EN_IMPL_PLUGIN_TBE,                // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,        // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,    // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE, // custom tbe op
  EN_RESERVED                        // reserved value
};

enum AOEOption {
  AOE_OPT_USE_KB = 0,
  AOE_OPT_NOT_USE_KB,
  AOE_OPT_RESERVED
};

struct FEOpsStoreInfo {
  int32_t priority;
  std::string fe_ops_store_name;
  OpImplType op_impl_type;
  std::string cfg_file_path;
  std::string op_impl_file_path;
  bool need_pre_compile;
  bool need_compile;
  FEOpsStoreInfo() : priority(0), fe_ops_store_name(), op_impl_type(EN_RESERVED), cfg_file_path(), op_impl_file_path(),
                     need_pre_compile(false), need_compile(false) {}
  FEOpsStoreInfo(int32_t priority_value, std::string ops_store_name_value,  OpImplType op_impl_type_value,
                 std::string cfg_file_path_value, std::string op_impl_file_path_value,
                 bool need_pre_compile_value, bool need_compile_value)
                 : priority(priority_value), fe_ops_store_name(ops_store_name_value), op_impl_type(op_impl_type_value),
                   cfg_file_path(cfg_file_path_value), op_impl_file_path(op_impl_file_path_value),
                   need_pre_compile(need_pre_compile_value), need_compile(need_compile_value) {}
  FEOpsStoreInfo(int32_t priority_value, std::string ops_store_name_value,  OpImplType op_impl_type_value,
                 std::string cfg_file_path_value, std::string op_impl_file_path_value)
                 : priority(priority_value), fe_ops_store_name(ops_store_name_value), op_impl_type(op_impl_type_value),
                   cfg_file_path(cfg_file_path_value), op_impl_file_path(op_impl_file_path_value),
                   need_pre_compile(false), need_compile(false) {}
};

enum ISAArchVersion { EN_ISA_ARCH_V100 = 0, EN_ISA_ARCH_V200, EN_ISA_ARCH_V210 };

// Don't change the order, only add new mode in the end.
enum AppendArgsMode { NO_ARGS = 0, L2_BUFFER_ARGS = 1, L2_CACHE_ARGS = 999};

enum BufferFusionMode { EN_OPTIMIZE_DISABLE = 0, EN_L2_BUFFER, EN_L2_FUSION };

enum BufferOptimize { EN_UNKNOWN_OPTIMIZE = 0, EN_OFF_OPTIMIZE, EN_L1_OPTIMIZE, EN_L2_OPTIMIZE };

enum AutoTuneMode { TUNE_MODE_NO_TUNE = 0, TUNE_MODE_AUTO_TUNE, TUNE_MODE_RL_TUNE, TUNE_MODE_AUTO_AND_RL_TUNE };

enum PrecisionPolicy { WHITE = 0, BLACK = 1, GRAY = 2 };

enum OpPattern {
  OP_PATTERN_OP_KERNEL = 0,
  OP_PATTERN_OP_CUSTOMIZE,
  OP_PATTERN_FORMAT_AGNOSTIC,
  OP_PATTERN_BROADCAST,
  OP_PATTERN_REDUCE
};

enum OpParamType { REQUIRED = 0, OPTIONAL, DYNAMIC, RESERVED };

enum OpConstValueDepend { CONST_IGNORE = 0, CONST_REQUIRED, CONST_OPTIONAL };

enum OpReduceType { REDUCE_MEAN = 0, REDUCE_ADD, REDUCE_MAX, REDUCE_MIN };

enum OpL1FusionType { L1FUSION_DISABLE = 0, L1FUSION_BASIC, L1FUSION_INPUT_CTR };
}
#endif  // FUSION_ENGINE_INC_COMMON_AICORE_UTIL_TYPES_H_
