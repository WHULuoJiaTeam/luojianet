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

#ifndef INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
#define INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_

#include <memory>
#include <vector>

#include "framework/common/op/attr_value_util.h"
#include "register/register_types.h"
#include "register/register_error_codes.h"
#include "framework/common/util.h"
#include "graph/attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "proto/insert_op.pb.h"

namespace ge {

// Add Sub Mul
GE_FUNC_VISIBILITY extern const uint32_t ADD_INPUT_NUM;
GE_FUNC_VISIBILITY extern const uint32_t SUB_INPUT_NUM;
GE_FUNC_VISIBILITY extern const uint32_t MUL_INPUT_NUM;

// Permute
GE_FUNC_VISIBILITY extern const int32_t PERMUTE_ORDER_NUM;

// Ssd PriroBox
GE_FUNC_VISIBILITY extern const double SSD_PRIORBOX_ASPECT_RATIO_VALUE;

GE_FUNC_VISIBILITY extern const uint32_t STRIDEDSLICE_INPUT_NUM;

// Switch
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_INPUT_NUM;
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_OUTPUT_NUM;
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_FALSE_OUTPUT;
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_TRUE_OUTPUT;
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_DATA_INPUT;
GE_FUNC_VISIBILITY extern const uint32_t SWITCH_PRED_INPUT;

// Merge
GE_FUNC_VISIBILITY extern const int32_t MERGE_DATA_OUTPUT;
GE_FUNC_VISIBILITY extern const int32_t MERGE_INDEX_OUTPUT;

// FunctionOp
GE_FUNC_VISIBILITY extern const uint32_t IF_COND_INPUT;
GE_FUNC_VISIBILITY extern const uint32_t FOR_START_INPUT;
GE_FUNC_VISIBILITY extern const uint32_t FOR_LIMIT_INPUT;
GE_FUNC_VISIBILITY extern const uint32_t FOR_DELTA_INPUT;
GE_FUNC_VISIBILITY extern const uint32_t FOR_DATA_INPUT;

GE_FUNC_VISIBILITY extern const int32_t NORMAL_TENSOR_SIZE;
/*lint -e148*/
class GE_FUNC_VISIBILITY OpUtils {
 public:
  ///
  /// @ingroup domi_ome
  /// @brief Check whether check_value is in [min_enum_value, max_enum_value]
  /// @return true Within
  /// @return false out of range
  //
  static inline bool CheckEnumValid(int32_t check_value, int32_t min_enum_value, int32_t max_enum_value) {
    return check_value < min_enum_value ? false : (check_value >= max_enum_value ? false : true);
  }

  ///
  /// @ingroup domi_omg
  /// @brief Determine whether to manually calculate the tensor size based on the values of format and dim
  /// @param [in] format, Format information of the tensor
  /// @param [in] real_dim_cnt, Tensor dim
  /// @return true Manually calculate the size based on dim and datatype
  /// @return false skip
  ///
  static bool IsComputDimsSize(const int32_t format, const uint32_t real_dim_cnt);

  ///
  /// @brief Extract AIPP parameters from AttrDefMap and splice them
  /// @param [in] aipp_attr attr of operator
  /// @param [out] aipp_params aipp parameters
  /// @return enum of tagCCAippInputFormat
  ///

  static Status ConvertAippParams(const NamedAttrs &aipp_attr, domi::AippOpParams &aipp_params);
  template <typename T>
  static void SliceData(const std::vector<char *> &input, int64_t chunk_size, std::vector<char *> &output,
                        int64_t begin, int64_t out_dim, int64_t stride);
  template <typename T>
  static Status SetDataByDataType(size_t out_size, const std::vector<char *> &chunk_input,
                                  const std::vector<char *> &chunk_output, GeTensor *output);
  template <typename T>
  static Status SetOutputSliceDataByDataType(void *data, int64_t data_size, const std::vector<int64_t> &input_dims,
                                             const std::vector<int64_t> &begin, const std::vector<int64_t> &output_dims,
                                             ge::GeTensor *output, const std::vector<int64_t> &stride);
  static Status SetOutputSliceData(void *const data, const int64_t data_size, const int32_t data_type,
                                   const std::vector<int64_t> &input_dims, const std::vector<int64_t> &begin,
                                   const std::vector<int64_t> &output_dims, GeTensor *const output,
                                   const std::vector<int64_t> &stride);
  static Status GetShapeDataFromConstTensor(const ConstGeTensorPtr &tensor, const DataType type,
                                            std::vector<int64_t> &dims);
};
/*lint +e148*/
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_OP_GE_OP_UTILS_H_
