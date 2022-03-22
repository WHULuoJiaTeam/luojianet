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

#include "graph/load/model_manager/model_utils.h"
#include <string>
#include "framework/common/debug/log.h"
#include "framework/common/op/ge_op_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "external/graph/types.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "common/math/math_util.h"

#define VALIDATE_MEM_RANGE(OP, TOTAL_SIZE, OFFSET, SIZE)                                                    \
  do {                                                                                                      \
    if (ge::CheckInt64AddOverflow((OFFSET), (SIZE)) != SUCCESS) {                                           \
      GELOGE(PARAM_INVALID, "Int64 %ld and %ld addition can result in overflow!",                           \
             static_cast<int64_t>(OFFSET), static_cast<int64_t>(SIZE));                                     \
      return {};                                                                                            \
    }                                                                                                       \
    int64_t range = (OFFSET) + (SIZE);                                                                      \
    if ((TOTAL_SIZE) < static_cast<uint64_t>(range)) {                                                      \
      REPORT_INNER_ERROR("E19999",                                                                          \
                         "Node:%s(%s) memory out of range, offset:%ld, size:%ld, exceed total size:%lu.",   \
                         OP->GetName().c_str(), OP->GetType().c_str(), (OFFSET), (SIZE), (TOTAL_SIZE));     \
      GELOGE(OUT_OF_MEMORY,                                                                                 \
             "[Check][Param]Node:%s(%s) memory out of range, offset:%ld, size:%ld, exceed total size:%lu.", \
             OP->GetName().c_str(), OP->GetType().c_str(), (OFFSET), (SIZE), (TOTAL_SIZE));                 \
      return {};                                                                                            \
    }                                                                                                       \
  } while (0)

namespace ge {
///
/// @ingroup ge
/// @brief Get input size.
/// @return vector<int64_t>
///
vector<int64_t> ModelUtils::GetInputSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_input_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_size);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < inputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Get size from TensorDesc failed, op : %s, input index : %zu", op_desc->GetName().c_str(), i);
      continue);

    GELOGI("GetInputSize op: %s, index: %zu, size:%ld", op_desc->GetName().c_str(), i, tensor_size);
    v_input_size.push_back(tensor_size);
  }

  return v_input_size;
}

///
/// @ingroup ge
/// @brief Get output size.
/// @return vector<int64_t>
///
vector<int64_t> ModelUtils::GetOutputSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_output_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_size);

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_size;);

  for (size_t i = 0; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int64_t tensor_size = 0;
    GE_IF_BOOL_EXEC(
      TensorUtils::GetSize(*tensor_desc, tensor_size) != GRAPH_SUCCESS,
      GELOGI("Get size from TensorDesc failed, op : %s, output index : %zu", op_desc->GetName().c_str(), i);
      continue);

    GELOGI("GetOutputSize op: %s, index: %zu, size:%ld", op_desc->GetName().c_str(), i, tensor_size);
    v_output_size.push_back(tensor_size);
  }

  return v_output_size;
}

///
/// @ingroup ge
/// @brief Get workspace size.
/// @return vector<int64_t>
///
vector<int64_t> ModelUtils::GetWorkspaceSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_workspace_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_size);

  const vector<int64_t> v_workspace_num = op_desc->GetWorkspace();
  const vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_num.size() != v_workspace_bytes.size()) {
    GELOGW("workspace_num[%zu]!= workspace_bytes[%zu]", v_workspace_num.size(), v_workspace_bytes.size());
    return v_workspace_size;
  }

  for (auto workspace_bytes : v_workspace_bytes) {
    v_workspace_size.push_back(workspace_bytes);
  }

  return v_workspace_size;
}

///
/// @ingroup ge
/// @brief Get weight size.
/// @return vector<int64_t>
///
vector<int64_t> ModelUtils::GetWeightSize(ConstOpDescPtr op_desc) {
  vector<int64_t> v_weight_size;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weight_size);

  // const op, get weight directly
  const string type_name = op_desc->GetType();
  if ((type_name == "Const") || (type_name == "Constant")) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weight_size.push_back(TensorUtils::GetWeightSize(weight));
    }

    return v_weight_size;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      int64_t tensor_size = 0;
      (void)TensorUtils::GetSize(*tensor_desc, tensor_size);
      v_weight_size.push_back(tensor_size);
    }
  }

  return v_weight_size;
}

///
/// @ingroup ge
/// @brief Get weights.
/// @return vector<ConstGeTensorPtr>
///
vector<ConstGeTensorPtr> ModelUtils::GetWeights(ConstOpDescPtr op_desc) {
  vector<ConstGeTensorPtr> v_weights;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_weights);

  // const op, get weight directly
  const string op_type = op_desc->GetType();
  if ((op_type == "Const") || (op_type == "Constant")) {
    ConstGeTensorPtr weight = nullptr;
    if (AttrUtils::GetTensor(*op_desc, ATTR_NAME_WEIGHTS, weight)) {
      v_weights.push_back(weight);
    }

    return v_weights;
  }

  // other ops get weight from connected constop
  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
      if (tensor_desc == nullptr) {
        GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
        continue;
      }

      ConstGeTensorPtr weight = nullptr;
      if (AttrUtils::GetTensor(*tensor_desc, ATTR_NAME_WEIGHTS, weight)) {
        v_weights.push_back(weight);
      }
    }
  }

  return v_weights;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Input descriptor.
/// @return vector<::tagCcAICPUTensor>
///
vector<::tagCcAICPUTensor> ModelUtils::GetInputDescs(ConstOpDescPtr op_desc) {
  // AiCpuOp::GetInputDescs
  vector<::opTensor_t> v_input_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_descs);

  const size_t inputs_size = op_desc->GetAllInputsSize();
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();

  for (size_t i = 0; i < inputs_size; ++i) {
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {  // skip Const input node
      continue;
    }

    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0;
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = tensor_desc->GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = tensor_desc->GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = (j < tmp.dim_cnt ? tensor_desc->GetShape().GetDim(j) : 1);
    }

    v_input_descs.push_back(tmp);
  }

  return v_input_descs;
}

///
/// @ingroup ge
/// @brief Get AiCpuOp Output descriptor.
/// @return vector<::tagCcAICPUTensor>
///
vector<::tagCcAICPUTensor> ModelUtils::GetOutputDescs(ConstOpDescPtr op_desc) {
  // AiCpuOp::GetOutputDescs
  vector<::opTensor_t> v_output_descs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_descs);

  // init op output opTensor_t struct
  const size_t output_num = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_num; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    uint32_t dim_cnt = 0;
    GE_CHK_BOOL_EXEC_WARN(TensorUtils::GetRealDimCnt(*tensor_desc, dim_cnt) == GRAPH_SUCCESS, continue,
                          "Get dim_cnt failed");

    opTensor_t tmp;
    uint32_t tmp_fmt = tensor_desc->GetFormat();
    tmp.format = tagOpTensorFormat(tmp_fmt);
    tmp.dim_cnt = static_cast<int32_t>(dim_cnt);
    uint32_t tmp_type = tensor_desc->GetDataType();
    tmp.data_type = tagOpDataType(tmp_type);

    for (int32_t j = 0; j < 4; j++) {  // 4 dims
      tmp.dim[j] = (j < tmp.dim_cnt ? tensor_desc->GetShape().GetDim(j) : 1);
    }

    v_output_descs.push_back(tmp);
  }

  return v_output_descs;
}

///
/// @ingroup ge
/// @brief Get input data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetInputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_input_data_addr;  // init as:buf_base + op_def_->input(i));
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_input_data_addr);
  uint64_t session_id = model_param.session_id;

  const size_t inputs_size = op_desc->GetInputsSize();
  const vector<int64_t> v_input_offset = op_desc->GetInputOffset();

  const string op_type = op_desc->GetType();

  size_t non_const_index = 0;
  const vector<bool> v_is_input_const = op_desc->GetIsInputConst();
  vector<int64_t> v_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != inputs_size)) {
    REPORT_INNER_ERROR("E19999", "Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != input_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_INPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), inputs_size,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return v_input_data_addr;
  }
  for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
    GE_IF_BOOL_EXEC(tensor_desc == nullptr, GELOGD("Op: %s, Index: %zu, has no input", op_desc->GetName().c_str(), i);
                    continue;)
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    if ((i < v_is_input_const.size()) && v_is_input_const[i]) {
      // Add weights address to input
      int64_t data_offset = 0;
      GE_CHK_STATUS(TensorUtils::GetDataOffset(*tensor_desc, data_offset));
      int64_t weight_size = 0;
      // The reason why GetTensorSizeInBytes is used here is that the weight is allocated based on the size of
      // TensorData in function AdjustConstWeightSize. and the size is zero when the tensor is empty.
      GE_CHK_STATUS(TensorUtils::GetTensorSizeInBytes(*tensor_desc, weight_size));
      VALIDATE_MEM_RANGE(op_desc, model_param.weight_size, data_offset, weight_size);
      uint8_t *weight_addr = model_param.weight_base + data_offset;
      v_input_data_addr.push_back(weight_addr);
      GELOGI("[IMAS]GetInputDataAddrs graph_%u type[C] name[%s] input[%zu] memaddr[%p]", model_param.graph_id,
             op_desc->GetName().c_str(), i, weight_addr);
      non_const_index++;
      continue;
    }

    GE_IF_BOOL_EXEC(non_const_index >= v_input_offset.size(), break);

    int64_t input_offset = v_input_offset[non_const_index];
    non_const_index++;
    int64_t inner_offset = 0;
    (void)ge::AttrUtils::GetInt(op_desc->MutableInputDesc(i), ATTR_NAME_INNER_OFFSET, inner_offset);
    GE_IF_BOOL_EXEC(model_param.var_size != 0
                    && ge::VarManager::Instance(session_id)->IsVarAddr(input_offset - inner_offset),
                    uint8_t *variable_addr = nullptr;
                    GE_CHK_STATUS_EXEC(GetVarAddr(model_param, op_desc, input_offset - inner_offset,
                                                  tensor_size + inner_offset, variable_addr), return {});
                    variable_addr += inner_offset;
                    v_input_data_addr.push_back(variable_addr);
                    GELOGI("[IMAS]GetInputDataAddrs graph_%u type[V] name[%s] input[%lu] memaddr[%p]",
                           model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
                    continue);

    int64_t mem_type;
    bool tensor_has_mem_type = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, mem_type);
    // feature maps
    void *mem_addr = nullptr;
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {  // fusion
      mem_addr = reinterpret_cast<uint8_t *>(static_cast<intptr_t>(input_offset));
      v_input_data_addr.push_back(mem_addr);
    } else if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_TS_4G) {
      // The input size and peer output size may be not consecutive, therefore, the tensor_size is not been checked.
      VALIDATE_MEM_RANGE(op_desc, model_param.mem_size, input_offset, static_cast<int64_t>(0));
      mem_addr = model_param.ts_mem_mall->Acquire(input_offset, static_cast<uint64_t>(tensor_size));
      v_input_data_addr.push_back(mem_addr);
    } else if (tensor_has_mem_type && mem_type == RT_MEMORY_P2P_DDR) {
      uint8_t *p2p_mem_addr = model_param.memory_infos.at(RT_MEMORY_P2P_DDR).memory_base + v_input_offset[i];
      v_input_data_addr.push_back(p2p_mem_addr);
      GELOGI("[IMAS]GetInputDataAddrs graph_%u type[P] name[%s] input[%zu] memaddr[%p]", model_param.graph_id,
             op_desc->GetName().c_str(), i, p2p_mem_addr);
      continue;
    } else {
      // The input size and peer output size may be not consecutive, therefore, the tensor_size is not been checked.
      VALIDATE_MEM_RANGE(op_desc, model_param.mem_size, input_offset, static_cast<int64_t>(0));
      mem_addr = model_param.mem_base + input_offset;
      v_input_data_addr.push_back(mem_addr);
    }
    GELOGI("[IMAS]GetInputDataAddrs graph_%u type[F] name[%s] input[%zu] memaddr[%p]", model_param.graph_id,
           op_desc->GetName().c_str(), i, mem_addr);
  }

  return v_input_data_addr;
}

///
/// @ingroup ge
/// @brief Get variable address.
/// @return Status
///
Status ModelUtils::GetVarAddr(const RuntimeParam &model_param, const ConstOpDescPtr &op_desc, int64_t offset,
                              int64_t tensor_size, uint8_t *&var_addr) {
  rtMemType_t mem_type = ge::VarManager::Instance(model_param.session_id)->GetVarMemType(offset);
  switch (mem_type) {
    case RT_MEMORY_RDMA_HBM:
      if (offset < 0) {
        REPORT_INNER_ERROR("E19999", "Param offset:%ld < 0, check invalid", offset);
        GELOGE(PARAM_INVALID, "[Check][Param] Param offset:%ld cannot be negative", offset);
        return PARAM_INVALID;
      }
      var_addr = reinterpret_cast<uint8_t *>(static_cast<uintptr_t>(offset));
      break;
    case RT_MEMORY_HBM:
      VALIDATE_MEM_RANGE(op_desc, model_param.var_size, offset - model_param.logic_var_base, tensor_size);
      var_addr = model_param.var_base + offset - model_param.logic_var_base;
      break;
    default:
      REPORT_INNER_ERROR("E19999", "Get mem_type:%d for offset:%ld is unsupported, check invalid", mem_type, offset);
      GELOGE(PARAM_INVALID, "[Check][Param] Get mem_type:%d for offset:%ld is unsupported, check invalid",
             mem_type, offset);
      return PARAM_INVALID;
  }
  GE_CHECK_NOTNULL(var_addr);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get output data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetOutputDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_output_data_addr;  // init as:buf_base + op_def_->output(i)
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_output_data_addr);
  uint64_t session_id = model_param.session_id;

  const size_t outputs_size = op_desc->GetOutputsSize();
  const vector<int64_t> v_output_offset = op_desc->GetOutputOffset();
  GE_IF_BOOL_EXEC(v_output_offset.size() != outputs_size,
                  GELOGW("Output param invalid: output_offset=%zu, outputs=%zu.", v_output_offset.size(), outputs_size);
                  return v_output_data_addr);
  vector<int64_t> v_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  if (has_mem_type_attr && (v_memory_type.size() != outputs_size)) {
    REPORT_INNER_ERROR("E19999", "Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s), check invalid",
                       ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s, memory_type.size:%zu != output_desc.size:%zu, op:%s(%s)",
           ATTR_NAME_OUTPUT_MEM_TYPE_LIST.c_str(), v_memory_type.size(), outputs_size,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return v_output_data_addr;
  }
  for (size_t i = 0; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc->MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %zu, Tensor Desc is null", op_desc->GetName().c_str(), i);
      continue;
    }

    int32_t calc_type = 0;
    bool ret = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (ret && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))) {
      GELOGD("%s is an optional output, the address don't need to be saved.", tensor_desc->GetName().c_str());
      continue;
    }
    int64_t inner_offset = 0;
    (void)ge::AttrUtils::GetInt(op_desc->MutableOutputDesc(i), ATTR_NAME_INNER_OFFSET, inner_offset);
    int64_t tensor_size = 0;
    GE_CHK_STATUS_EXEC(TensorUtils::GetSize(*tensor_desc, tensor_size), return {});
    GE_IF_BOOL_EXEC(model_param.var_size != 0
                    && ge::VarManager::Instance(session_id)->IsVarAddr(v_output_offset[i] - inner_offset),
                    uint8_t *variable_addr = nullptr;
                    GE_CHK_STATUS_EXEC(GetVarAddr(model_param, op_desc, v_output_offset[i] - inner_offset,
                                                  tensor_size + inner_offset, variable_addr), return {});
                    variable_addr += inner_offset;
                    v_output_data_addr.push_back(variable_addr);
                    GELOGI("[IMAS]GetOutputDataAddrs graph_%u type[V] name[%s] output[%zu] memaddr[%p]",
                           model_param.graph_id, op_desc->GetName().c_str(), i, variable_addr);
                    continue);

    int64_t mem_type;
    bool tensor_has_mem_type = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_TENSOR_MEM_TYPE, mem_type);
    // feature maps
    void *mem_addr = nullptr;
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {  // fusion
      mem_addr = reinterpret_cast<uint8_t *>(static_cast<intptr_t>(v_output_offset[i]));
      v_output_data_addr.push_back(mem_addr);
    } else if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_TS_4G) {
      VALIDATE_MEM_RANGE(op_desc, model_param.mem_size, v_output_offset[i], tensor_size);
      mem_addr = model_param.ts_mem_mall->Acquire(v_output_offset[i], static_cast<uint64_t>(tensor_size));
      v_output_data_addr.push_back(mem_addr);
    } else if (tensor_has_mem_type && mem_type == RT_MEMORY_P2P_DDR) {
      uint8_t *p2p_mem_addr = model_param.memory_infos.at(RT_MEMORY_P2P_DDR).memory_base + v_output_offset[i];
      v_output_data_addr.push_back(p2p_mem_addr);
      GELOGI("[IMAS]GetOutputDataAddrs graph_%u type[P] name[%s] output[%zu] memaddr[%p]", model_param.graph_id,
             op_desc->GetName().c_str(), i, p2p_mem_addr);
      continue;
    } else {
      VALIDATE_MEM_RANGE(op_desc, model_param.mem_size, v_output_offset[i], tensor_size);
      mem_addr = static_cast<uint8_t *>(model_param.mem_base + v_output_offset[i]);
      v_output_data_addr.push_back(mem_addr);
    }
    GELOGI("[IMAS]GetOutputDataAddrs graph_%u type[F] name[%s] output[%zu] memaddr[%p]", model_param.graph_id,
           op_desc->GetName().c_str(), i, mem_addr);
  }
  return v_output_data_addr;
}

///
/// @ingroup ge
/// @brief Get workspace data address.
/// @return vector<void*>
///
vector<void *> ModelUtils::GetWorkspaceDataAddrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  vector<void *> v_workspace_data_addr;
  GE_CHECK_NOTNULL_EXEC(op_desc, return v_workspace_data_addr);

  const vector<int64_t> v_workspace_offset = op_desc->GetWorkspace();
  const vector<int64_t> v_workspace_bytes = op_desc->GetWorkspaceBytes();
  if (v_workspace_offset.size() != v_workspace_bytes.size()) {
    GELOGW("v_workspace_offset.size()[%zu] != v_workspace_bytes.size()[%zu]", v_workspace_offset.size(),
           v_workspace_bytes.size());
    return v_workspace_data_addr;
  }

  vector<bool> workspace_reuse_flag;
  bool has_workspace_reuse = ge::AttrUtils::GetListBool(op_desc, "workspace_reuse_flag", workspace_reuse_flag);
  vector<int64_t> v_memory_type;
  vector<int64_t> workspace_memory_type;
  bool has_mem_type_attr = ge::AttrUtils::GetListInt(op_desc, TVM_ATTR_NAME_WORKSPACE_TYPE, v_memory_type);
  bool has_mem_type_workspace =
    ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_TYPE_LIST, workspace_memory_type);

  vector<int32_t> workspace_no_reuse_scope;
  bool has_workspace_no_reuse_scope =
    ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE, workspace_no_reuse_scope);

  for (size_t i = 0; i < v_workspace_bytes.size(); ++i) {
    // Temporary solution, the aicpu workspace of multiple images cannot be shared.
    bool aicpu_work_space = (has_workspace_reuse && i < workspace_reuse_flag.size() && !workspace_reuse_flag[i] &&
                             !model_param.is_single_op);
    if (aicpu_work_space) {
      void *mem_addr = model_param.aicpu_mem_mall->Acquire(v_workspace_offset[i], v_workspace_bytes[i]);
      v_workspace_data_addr.push_back(mem_addr);
      GELOGI(
        "[IMAS]GetWorkspaceDataAddrs graph_%u type[F] name[%s] aicpu workspace[%zu]  offset[%ld] bytes[%ld] "
        "memaddr[%p]",
        model_param.graph_id, op_desc->GetName().c_str(), i, v_workspace_offset[i], v_workspace_bytes[i], mem_addr);
      continue;
    } else if (has_mem_type_workspace && workspace_memory_type[i] == RT_MEMORY_P2P_DDR) {
      int64_t p2p_workspace_offset = v_workspace_offset[i];
      int64_t p2p_workspace_bytes = v_workspace_bytes[i];
      uint8_t *p2p_mem_addr = p2p_workspace_bytes == 0
                                ? nullptr
                                : model_param.memory_infos.at(RT_MEMORY_P2P_DDR).memory_base + p2p_workspace_offset;
      v_workspace_data_addr.push_back(p2p_mem_addr);
      GELOGI(
        "[IMAS]GetWorkspaceDataAddrs graph_%u type[P] name[%s] p2p workspace[%zu]  offset[%ld] bytes[%ld] "
        "memaddr[%p]",
        model_param.graph_id, op_desc->GetName().c_str(), i, p2p_workspace_offset, p2p_workspace_bytes, p2p_mem_addr);
      continue;
    }
    if (has_mem_type_attr && v_memory_type[i] == RT_MEMORY_L1) {
      v_workspace_data_addr.push_back(reinterpret_cast<uint8_t *>(static_cast<intptr_t>(v_workspace_offset[i])));
      GELOGI("[IMAS]GetWorkspaceDataAddrs graph_%u type[L1] name[%s], mem_addr[workspace index %zu]:0x%lx",
             model_param.graph_id, op_desc->GetName().c_str(), i, v_workspace_offset[i]);
    } else if (v_workspace_bytes[i] == 0) {
      v_workspace_data_addr.push_back(nullptr);
      GELOGI("[IMAS]GetWorkspaceDataAddrs graph_%u type[F] name[%s] workspace[%zu] offset[%ld] bytes[%ld] Null addr",
             model_param.graph_id, op_desc->GetName().c_str(), i, v_workspace_offset[i], v_workspace_bytes[i]);
    } else {
      VALIDATE_MEM_RANGE(op_desc, model_param.mem_size, v_workspace_offset[i], v_workspace_bytes[i]);
      uint8_t *mem_addr = nullptr;
      bool session_scope_memory = (has_workspace_no_reuse_scope) && (i < workspace_no_reuse_scope.size());
      if (session_scope_memory) {
        mem_addr = model_param.memory_infos.at(kSessionScopeMemory | RT_MEMORY_HBM).memory_base + v_workspace_offset[i];
      } else {
        mem_addr = model_param.mem_base + v_workspace_offset[i];
      }
      v_workspace_data_addr.push_back(mem_addr);
      GELOGI("[IMAS]GetWorkspaceDataAddrs graph_%u type[F] name[%s] workspace[%zu] offset[%ld] bytes[%ld] memaddr[%p]",
             model_param.graph_id, op_desc->GetName().c_str(), i, v_workspace_offset[i], v_workspace_bytes[i],
             mem_addr);
    }
  }

  return v_workspace_data_addr;
}

///
/// @ingroup ge
/// @brief Get runtime memory address.
/// @return Status
///
Status ModelUtils::GetRtAddress(const RuntimeParam &param, uintptr_t logic_addr, uint8_t *&mem_addr) {
  uint8_t *runtime_base_addr = nullptr;
  if ((param.logic_mem_base <= logic_addr) && (logic_addr < param.logic_mem_base + param.mem_size)) {
    runtime_base_addr = param.mem_base - param.logic_mem_base;
    GELOGI("The logic addr:0x%lx is data address, base:0x%lx, size:%lu", logic_addr, param.logic_mem_base,
           param.mem_size);
  } else if ((param.logic_weight_base <= logic_addr) && (logic_addr < param.logic_weight_base + param.weight_size)) {
    runtime_base_addr = param.weight_base - param.logic_weight_base;
    GELOGI("The logic addr:0x%lx is weight address, base:0x%lx, size:%lu", logic_addr, param.logic_weight_base,
           param.weight_size);
  } else if ((param.logic_var_base <= logic_addr) && (logic_addr < param.logic_var_base + param.var_size)) {
    runtime_base_addr = param.var_base - param.logic_var_base;
    GELOGI("The logic addr:0x%lx is variable address, base:0x%lx, size:%lu", logic_addr, param.logic_var_base,
           param.var_size);
  } else if (logic_addr != 0) {
    mem_addr = nullptr;
    REPORT_INNER_ERROR("E19999", "Check param logic addr:0x%lx abnormal", logic_addr);
    GELOGE(PARAM_INVALID, "[Check][Param] The logic addr:0x%lx is abnormal", logic_addr);
    return PARAM_INVALID;
  }

  mem_addr = runtime_base_addr + logic_addr;
  return SUCCESS;
}
}  // namespace ge
