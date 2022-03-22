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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_adapter.h"

using std::vector;

namespace ge {
class ModelUtils {
 public:
  ModelUtils() = default;
  ~ModelUtils() = default;

  ///
  /// @ingroup ge
  /// @brief Get input size.
  /// @return vector<uint32_t>
  ///
  static vector<int64_t> GetInputSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get output size.
  /// @return vector<uint32_t>
  ///
  static vector<int64_t> GetOutputSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get workspace size.
  /// @return vector<uint32_t>
  ///
  static vector<int64_t> GetWorkspaceSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get weight size.
  /// @return vector<uint32_t>
  ///
  static vector<int64_t> GetWeightSize(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get weights.
  /// @return vector<ConstGeTensorPtr>
  ///
  static vector<ConstGeTensorPtr> GetWeights(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get AiCpuOp Input descriptor.
  /// @return vector<::tagCcAICPUTensor>
  ///
  static vector<::tagCcAICPUTensor> GetInputDescs(ConstOpDescPtr op_desc);
  ///
  /// @ingroup ge
  /// @brief Get AiCpuOp Output descriptor.
  /// @return vector<::tagCcAICPUTensor>
  ///
  static vector<::tagCcAICPUTensor> GetOutputDescs(ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get input data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetInputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc);
  ///
  /// @ingroup ge
  /// @brief Get output data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetOutputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get workspace data address.
  /// @return vector<void*>
  ///
  static vector<void *> GetWorkspaceDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc);

  ///
  /// @ingroup ge
  /// @brief Get memory runtime base.
  /// @return Status
  ///
  static Status GetRtAddress(const RuntimeParam &model_param, uintptr_t logic_addr, uint8_t *&mem_addr);

 private:
  ///
  /// @ingroup ge
  /// @brief Get variable address.
  /// @return Status
  ///
  static Status GetVarAddr(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc, int64_t offset,
                           int64_t tensor_size, uint8_t *&var_addr);
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_UTILS_H_
