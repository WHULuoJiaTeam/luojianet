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

#include "graph/manager/trans_var_data_utils.h"

#include "framework/common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/manager/graph_var_manager.h"
#include "external/graph/types.h"
#include "graph/utils/type_utils.h"
#include "common/thread_pool.h"
#include <algorithm>

namespace ge {
namespace {
class RtContextSwitchGuard {
 public:
  RtContextSwitchGuard(rtCtxMode_t mode, uint32_t device_id) : last_(nullptr), current_(nullptr) {
    auto ret = rtCtxGetCurrent(&last_);
    if (ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtCtxGetCurrent failed, device_id:%u, ret:0x%X,",
                        device_id, ret);
      GELOGE(RT_FAILED, "[Call][RtCtxGetCurrent] Failed to get current context, device_id:%u, ret:0x%X",
             device_id, ret);
      return;
    }

    ret = rtCtxCreate(&current_, mode, static_cast<int32_t>(device_id));
    if (ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtCtxCreate failed, device_id:%u, ret:0x%X,",
                        device_id, ret);
      GELOGE(RT_FAILED, "[Call][RtCtxCreate] Failed to create new context for device:%u, ret:%d", device_id, ret);
      return;
    }

    ret = rtCtxSetCurrent(current_);
    if (ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtCtxSetCurrent failed, device_id:%u, ret:0x%X", device_id, ret);
      GELOGE(RT_FAILED, "[Call][RtCtxSetCurrent] failed, device_id:%u, ret:0x%X", device_id, ret);
      return;
    }
    GELOGD("Create and switch rt context %p type %d for device %u, backup last %p.", current_, mode, device_id, last_);
  }

  ~RtContextSwitchGuard() {
    if (current_ != nullptr) {
      auto ret = rtCtxDestroy(current_);
      GELOGD("Destory current context %p result %d", current_, ret);
    }
    if (last_ != nullptr) {
      auto ret = rtCtxSetCurrent(last_);
      GELOGD("Recovery last context %p result %d.", last_, ret);
    }
  }

 private:
  rtContext_t last_;
  rtContext_t current_;
};

int64_t CalcVarSizeInBytes(const GeTensorDesc &desc) {
  int64_t var_size = GetSizeByDataType(desc.GetDataType());
  if (var_size <= 0) {
    REPORT_INNER_ERROR("E19999", "Data type:%s in desc, it's size:%ld < 0, check invalid",
                       TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str(), var_size);
    GELOGE(PARAM_INVALID, "[Calc][VarDataSize] by data type %s failed.",
           TypeUtils::DataTypeToSerialString(desc.GetDataType()).c_str());
    return -1;
  }
  auto shape = desc.GetShape();
  auto dim_num = shape.GetDimNum();
  for (size_t dim_index = 0; dim_index < dim_num; ++dim_index) {
    var_size *= shape.GetDim(dim_index);
  }
  return var_size;
}

Status CopyVarToDevice(const NodePtr &var, const formats::TransResult &trans_result, void *var_addr) {
  GELOGD("Copy var %s from host to device, size %zu", var->GetName().c_str(), trans_result.length);
  auto ret = rtMemcpy(var_addr, trans_result.length, reinterpret_cast<void *>(trans_result.data.get()),
                      trans_result.length, RT_MEMCPY_HOST_TO_DEVICE);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, op:%s(%s), size:%lu, ret:0x%X,", var->GetName().c_str(),
                      var->GetType().c_str(), trans_result.length, ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, op:%s(%s), size:%lu, ret:0x%X,", var->GetName().c_str(),
           var->GetType().c_str(), trans_result.length, ret);
    return RT_FAILED;
  }
  return SUCCESS;
}

Status CopyVarFromDevice(uint64_t session_id, const NodePtr &var, std::unique_ptr<uint8_t[]> &var_data,
                         const GeTensorDesc &input_desc) {
  uint8_t *var_logic = nullptr;
  GE_CHECK_NOTNULL(var);
  auto ret = VarManager::Instance(session_id)->GetVarAddr(var->GetName(), input_desc, &var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][VarAddr] failed, node:%s, session_id:%lu, ret:%d",
           var->GetName().c_str(), session_id, ret);
    return INTERNAL_ERROR;
  }

  uint8_t *var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM);
  if (var_addr == nullptr) {
    REPORT_CALL_ERROR("E19999", "Get variable memory addr failed, mem_type:%d, op:%s(%s), session_id:%lu",
                      RT_MEMORY_HBM, var->GetName().c_str(), var->GetType().c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][VarMemoryAddr] failed, mem_type:%d, op:%s(%s), session_id:%lu",
           RT_MEMORY_HBM, var->GetName().c_str(), var->GetType().c_str(), session_id);
    return INTERNAL_ERROR;
  }

  int64_t var_size_bytes = CalcVarSizeInBytes(input_desc);
  if (var_size_bytes <= 0) {
    return INTERNAL_ERROR;
  }

  std::unique_ptr<uint8_t[]> var_host(new(std::nothrow) uint8_t[var_size_bytes]);
  if (var_host == nullptr) {
    REPORT_CALL_ERROR("E19999", "New host memory failed, size:%ld, op:%s(%s), session_id:%lu",
                      var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id);
    GELOGE(OUT_OF_MEMORY, "[New][Memory] for rt-host failed, size:%ld, op:%s(%s), session_id:%lu",
           var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id);
    return OUT_OF_MEMORY;
  }

  ret = rtMemcpy(reinterpret_cast<void *>(var_host.get()), var_size_bytes, reinterpret_cast<void *>(var_addr),
                 var_size_bytes, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%ld, op:%s(%s), session_id:%lu, ret:0x%X",
                      var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id, ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%ld, op:%s(%s), session_id:%lu, ret:0x%X",
           var_size_bytes, var->GetName().c_str(), var->GetType().c_str(), session_id, ret);
    return RT_FAILED;
  }

  GELOGD("Copy var %s from device to host, size %ld", var->GetName().c_str(), var_size_bytes);
  var_data.swap(var_host);

  GELOGI("var_logic:%p, var_addr:%p", var_logic, var_addr);

  return SUCCESS;
}

Status TransVarOnHost(uint8_t *var_data, const VarTransRoad &trans_road, formats::TransResult &result) {
  formats::TransResult result_last_time{};
  bool use_init_data = true;
  for (const auto &trans_info : trans_road) {
    if (trans_info.node_type == RESHAPE || trans_info.node_type == REFORMAT) {
      GELOGD("Skip to trans variable data on the reshape/reformat node");
      continue;
    }
    uint8_t *src_data = nullptr;
    if (use_init_data) {
      src_data = var_data;
      use_init_data = false;
    } else {
      src_data = result_last_time.data.get();
    }

    formats::TransResult tmp_result{};
    if (trans_info.node_type == TRANSDATA || trans_info.node_type == TRANSPOSED) {
      auto src_format = trans_info.input.GetFormat();
      auto src_shape = trans_info.input.GetShape().GetDims();
      auto dst_format = trans_info.output.GetFormat();
      auto dst_shape = trans_info.output.GetShape().GetDims();
      auto data_type = trans_info.input.GetDataType();
      GELOGD("Trans format from %s to %s, shape %s to %s, data-type %s",
             TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
             formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(dst_shape).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str());
      auto ret = formats::TransFormat({src_data, src_format, dst_format, src_shape, dst_shape, data_type}, tmp_result);
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Trans format from %s to %s, shape %s to %s failed, data type:%s, ret:%u,",
                          TypeUtils::FormatToSerialString(src_format).c_str(),
                          TypeUtils::FormatToSerialString(dst_format).c_str(),
                          formats::ShapeToString(src_shape).c_str(),
                          formats::ShapeToString(dst_shape).c_str(),
                          TypeUtils::DataTypeToSerialString(data_type).c_str(), ret);
        GELOGE(INTERNAL_ERROR, "[Trans][Format] from %s to %s, shape %s to %s failed, data type %s error code %u",
               TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
               formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(dst_shape).c_str(),
               TypeUtils::DataTypeToSerialString(data_type).c_str(), ret);
        return ret;
      }
    } else if (trans_info.node_type == CAST) {
      auto input_shape = trans_info.input.GetShape();
      auto src_data_size = input_shape.GetShapeSize() == 0 ? 1 : input_shape.GetShapeSize();
      auto src_data_type = trans_info.input.GetDataType();
      auto dst_data_type = trans_info.output.GetDataType();
      GELOGD("Trans data type from %s to %s, input shape %s, data size %ld",
             TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
             TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
             src_data_size);
      auto ret = formats::TransDataType({src_data, static_cast<size_t>(src_data_size), src_data_type, dst_data_type},
                                        tmp_result);
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Trans data type from %s to %s failed, input shape %s, data size %ld, ret:%u",
                          TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
                          TypeUtils::DataTypeToSerialString(dst_data_type).c_str(),
                          formats::ShapeToString(input_shape).c_str(), src_data_size, ret);
        GELOGE(INTERNAL_ERROR, "[Trans][DataType] from %s to %s failed, input shape %s, data size %ld, error code %u",
               TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
               TypeUtils::DataTypeToSerialString(dst_data_type).c_str(), formats::ShapeToString(input_shape).c_str(),
               src_data_size, ret);
        return ret;
      }
    } else {
      REPORT_INNER_ERROR("E19999", "Trans var data failed, the trans type %s does not supported, check invalid",
                         trans_info.node_type.c_str());
      GELOGE(UNSUPPORTED, "[Trans][VarData] failed, the trans type %s does not supported",
             trans_info.node_type.c_str());
      return UNSUPPORTED;
    }
    result_last_time = tmp_result;
  }

  result = result_last_time;
  return SUCCESS;
}

/// re-alloc var memory on device using var-manager
/// free origin var memory(var manager does not support now)
/// @param session_id
/// @param var
/// @param var_size_bytes
/// @param var_device
/// @return
Status ReAssignVarAddr(uint64_t session_id,
                       const std::string &var_name,
                       const GeTensorDesc &tensor_desc,
                       void **var_device) {
  uint8_t *var_logic = nullptr;
  Status ret = VarManager::Instance(session_id)->GetVarAddr(var_name, tensor_desc, &var_logic);
  if (ret != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Get][VarAddr] failed, var name:%s, session_id:%lu, ret:%u",
           var_name.c_str(), session_id, ret);
    return INTERNAL_ERROR;
  }

  uint8_t *var_addr = VarManager::Instance(session_id)->GetVarMemoryAddr(var_logic, RT_MEMORY_HBM);
  if (var_addr == nullptr) {
    REPORT_CALL_ERROR("E19999", "Get variable memory addr failed, mem_type:%d, var_name:%s, session_id:%lu,",
                      RT_MEMORY_HBM, var_name.c_str(), session_id);
    GELOGE(INTERNAL_ERROR, "[Get][VarMemoryAddr] failed, mem_type:%d, var_name:%s, session_id:%lu",
           RT_MEMORY_HBM, var_name.c_str(), session_id);
    return INTERNAL_ERROR;
  }
  *var_device = var_addr;

  GELOGI("var_logic:%p, var_addr:%p", var_logic, var_addr);

  return SUCCESS;
}

Status TransVarData(const NodePtr &var, const VarTransRoad &trans_road, uint64_t session_id) {
  // do not need to do anything if only all reshape/reformat node on the trans_road
  GE_CHECK_NOTNULL(var);
  bool need_trans = false;
  for (auto &road : trans_road) {
    if (road.node_type != RESHAPE && road.node_type != REFORMAT) {
      need_trans = true;
      break;
    }
  }
  if (!need_trans) {
    return SUCCESS;
  }

  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_data;
  if (trans_road.empty()) {
    REPORT_INNER_ERROR("E19999", "Param trans_road is empty, session_id:%lu, check invalid", session_id);
    GELOGE(INTERNAL_ERROR, "[Check][Param] trans_road is empty, session_id:%lu", session_id);
    return INTERNAL_ERROR;
  }
  const GeTensorDesc &input_desc = trans_road.begin()->input;
  auto ret = CopyVarFromDevice(session_id, var, var_data, input_desc);
  if (ret != SUCCESS) {
    return ret;
  }

  formats::TransResult trans_result{};
  ret = TransVarOnHost(var_data.get(), trans_road, trans_result);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][TransVarOnHost] failed, session_id:%lu, ret:%u", session_id, ret);
    return ret;
  }

  void *var_device = nullptr;

  /// It is a temporary solution to use the last GeTensorDesc to assign variable memory because the variable manager
  /// depends on TensorDesc and it is difficult to be modified. The correct solution is to assign memory based on the
  /// size of the converted variable. To complete the final solution, the dependency of the variable manager on
  /// TensorDesc needs to be removed. This change is large and needs to be performed step by step.
  ret = ReAssignVarAddr(session_id, var->GetName(), trans_road.rbegin()->output, &var_device);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][ReAssignVarAddr] failed, session id:%lu, op:%s, ret:%u",
           session_id, var->GetName().c_str(), ret);
    return ret;
  }

  // sync new data to device
  ret = CopyVarToDevice(var, trans_result, var_device);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][CopyVarToDevice] failed, var:%s, ret:%u", var->GetName().c_str(), ret);
    return ret;
  }

  return SUCCESS;
}

Status TransTensor(uint8_t *var_data, const NodePtr &var_src, const NodePtr &var_dst, formats::TransResult &result) {
  GE_CHECK_NOTNULL(var_src);
  GE_CHECK_NOTNULL(var_src->GetOpDesc());
  GE_CHECK_NOTNULL(var_dst);
  GE_CHECK_NOTNULL(var_dst->GetOpDesc());
  auto src_data_shape_size = var_src->GetOpDesc()->GetOutputDesc(0).GetShape().GetShapeSize();
  auto src_data_datatype = var_src->GetOpDesc()->GetOutputDesc(0).GetDataType();
  auto dst_data_datatype = var_dst->GetOpDesc()->GetOutputDesc(0).GetDataType();
  GE_IF_BOOL_EXEC(
      src_data_datatype != dst_data_datatype,
      auto ret = formats::TransDataType(
          {var_data, static_cast<size_t>(src_data_shape_size), src_data_datatype, dst_data_datatype}, result);
          if (ret != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Trans data type from %s to %s failed, data size %ld, ret:%u",
                              TypeUtils::DataTypeToSerialString(src_data_datatype).c_str(),
                              TypeUtils::DataTypeToSerialString(dst_data_datatype).c_str(),
                              src_data_shape_size, ret);
            GELOGE(INTERNAL_ERROR, "[Trans][DataType] from %s to %s failed, data size %ld, ret:%u",
                   TypeUtils::DataTypeToSerialString(src_data_datatype).c_str(),
                   TypeUtils::DataTypeToSerialString(dst_data_datatype).c_str(),
                   src_data_shape_size, ret);
            return ret;
          });
  return SUCCESS;
}

Status CopyTensorFromSrcVarNode(const NodePtr &var_src,
                                const NodePtr &var_dst,
                                uint64_t session_id,
                                uint32_t device_id) {
  /// after FE fusion pass, input num of applymomentum op was changed, 0th input is var_fp32, 6th input is
  /// var_fp16(new).
  /// unlink edges between var_fp32 and "dst_node" (need fp16) of var_fp32, add edge between var_fp16 and dst_node.
  /// need copy value from var_fp32 to var_fp16.
  /// [opdesc of var_src and var_dst are checked before passed in, no need to check if they are nullptr]
  GE_IF_BOOL_EXEC(var_src == nullptr || var_dst == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param var_src or var_dst is nullptr, session_id:%lu, device_id:%u, "
                                     "check invalid", session_id, device_id);
                  GELOGE(FAILED, "[Check][Param] Param var_src or var_dst is nullptr, session_id:%lu, device_id:%u",
                         session_id, device_id);
                  return FAILED);
  // src_node output_desc (fp32)
  GeTensorDesc output_desc = var_src->GetOpDesc()->GetOutputDesc(0);
  auto src_data_type = output_desc.GetDataType();
  auto src_shape = output_desc.GetShape();
  auto src_format = output_desc.GetFormat();
  GELOGI("src_node %s, src_format %s, src_shape %s, src_type %s", var_src->GetName().c_str(),
         TypeUtils::FormatToSerialString(src_format).c_str(), formats::ShapeToString(src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(src_data_type).c_str());
  // dst_node output_desc (fp16)
  GeTensorDesc dst_tensor_desc = var_dst->GetOpDesc()->GetOutputDesc(0);
  auto data_type = dst_tensor_desc.GetDataType();
  auto data_shape = dst_tensor_desc.GetShape();
  auto data_format = dst_tensor_desc.GetFormat();
  GELOGI("dst_node %s, src_format %s, src_shape %s, src_type %s", var_dst->GetName().c_str(),
         TypeUtils::FormatToSerialString(data_format).c_str(), formats::ShapeToString(data_shape).c_str(),
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  // Sync var data from device
  std::unique_ptr<uint8_t[]> var_src_data;
  RtContextSwitchGuard switch_context(RT_CTX_NORMAL_MODE, device_id);
  // copy from src_node
  auto ret = CopyVarFromDevice(session_id, var_src, var_src_data, output_desc);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(FAILED, "[Call][CopyVarFromDevice] failed, session id:%lu, var_src:%s",
                         session_id, var_src->GetName().c_str());
                  return ret);
  // trans dtype
  formats::TransResult trans_result{};
  ret = TransTensor(var_src_data.get(), var_src, var_dst, trans_result);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "[Trans][Tensor] failed, var_src:%s, var_dst:%s",
                         var_src->GetName().c_str(), var_dst->GetName().c_str());
                  return ret);
  // reset src value.
  void *var_device = nullptr;
  ret = ReAssignVarAddr(session_id, var_dst->GetName(), dst_tensor_desc, &var_device);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(INTERNAL_ERROR, "[Call][ReAssignVarAddr] failed, session id:%lu, var_dst:%s",
                         session_id, var_dst->GetName().c_str());
                  return ret);
  // copy to device
  ret = CopyVarToDevice(var_dst, trans_result, var_device);
  GE_IF_BOOL_EXEC(ret != SUCCESS,
                  GELOGE(ret, "[Call][CopyVarToDevice] failed, var_dst:%s, ret:%u",
                         var_dst->GetName().c_str(), ret);
                  return ret);
  return SUCCESS;
}
} // namespace
Status TransVarDataUtils::TransAllVarData(const vector<NodePtr> &variable_nodes,
                                          uint64_t session_id,
                                          rtContext_t context,
                                          uint32_t graph_id,
                                          uint32_t thread_num) {
  ThreadPool executor(thread_num);
  std::vector<std::future<Status>> vector_future;
  for (auto &node : variable_nodes) {
    if (node == nullptr) {
      continue;
    }

    if (node->GetType() != VARIABLE) {
      continue;
    }

    std::future<Status> f = executor.commit(
        [](const ge::NodePtr &node, uint64_t session_id, rtContext_t ctx, uint32_t graph_id,
           const struct error_message::Context &error_context) -> Status {
          ErrorManager::GetInstance().SetErrorContext(error_context);
          rtError_t rt_ret = rtCtxSetCurrent(ctx);
          if (rt_ret != RT_ERROR_NONE) {
            REPORT_CALL_ERROR("E19999", "Call rtCtxSetCurrent failed, session_id:%lu, graph_id:%u, ret:0x%X,",
                              session_id, graph_id, rt_ret);
            GELOGE(RT_FAILED, "[Call][RtCtxSetCurrent] failed, session_id:%lu, graph_id:%u, ret:0x%X,",
                   session_id, graph_id, rt_ret);
            return RT_ERROR_TO_GE_STATUS(rt_ret);
          }
          uint32_t allocated_graph_id = 0;
          Status ret = VarManager::Instance(session_id)->GetAllocatedGraphId(node->GetName(), allocated_graph_id);
          if (ret != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "Get allocated GraphId failed, session_id:%lu, graph_id:%u, ret:0x%X,",
                              session_id, graph_id, ret);
            GELOGE(INTERNAL_ERROR, "[Get][AllocatedGraphId] failed, node:%s, graph_id:%u.",
                   node->GetName().c_str(), graph_id);
            return INTERNAL_ERROR;
          }
          uint32_t changed_graph_id = 0;
          ret = VarManager::Instance(session_id)->GetChangedGraphId(node->GetName(), changed_graph_id);
          bool call_trans_var =
              (ret == SUCCESS && changed_graph_id == graph_id && changed_graph_id != allocated_graph_id);
          if (call_trans_var) {
            GELOGI("VarManager::GetChangedGraphId() success, node:%s, graph_id:%u.", node->GetName().c_str(), graph_id);
            VarTransRoad *trans_road = VarManager::Instance(session_id)->GetTransRoad(node->GetName());
            if (trans_road == nullptr) {
              GELOGI("The variable %s does not have any trans road", node->GetName().c_str());
              return SUCCESS;
            }
            ret = TransVarData(node, *trans_road, session_id);
            if (ret != SUCCESS) {
              GELOGE(INTERNAL_ERROR, "[Trans][VarData] failed, node:%s, graph_id:%u, session_id:%lu.",
                     node->GetName().c_str(), graph_id, session_id);
              return INTERNAL_ERROR;
            }
            VarManager::Instance(session_id)->RemoveChangedGraphId(node->GetName());
          }
          return SUCCESS;
        },
        node, session_id, context, graph_id, ErrorManager::GetInstance().GetErrorManagerContext());
    if (!f.valid()) {
      GELOGE(FAILED, "[Check][Param] Future is invalid, session id:%lu, graph id:%u", session_id, graph_id);
      return FAILED;
    }
    vector_future.push_back(std::move(f));
  }

  Status ret_status;
  for (size_t i = 0; i < vector_future.size(); ++i) {
    ret_status = vector_future[i].get();
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "[Check][Param] trans %zu vardata failed", i);
      return ret_status;
    }
  }

  return SUCCESS;
}

Status TransVarDataUtils::CopyVarData(const ComputeGraphPtr &compute_graph, uint64_t session_id, uint32_t device_id) {
  GELOGD("CopyVarData start: session_id:%lu.", session_id);
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, session_id:%lu, device_id:%u, check invalid",
                       session_id, device_id);
    GELOGE(FAILED, "[Check][Param] compute_graph is nullptr, session_id:%lu, device_id:%u",
           session_id, device_id);
    return FAILED;
  }

  string cp_from_node;
  bool copy_value = false;
  for (auto &node : compute_graph->GetAllNodes()) {
    GE_IF_BOOL_EXEC(node->GetOpDesc() == nullptr || node->GetOpDesc()->GetType() != VARIABLE, continue);
    GE_IF_BOOL_EXEC(ge::AttrUtils::GetStr(node->GetOpDesc(), "_copy_from_var_node", cp_from_node),
                    GELOGI("Get original type of cp_from_node"));
    if (cp_from_node.length() != 0) {
      (void) ge::AttrUtils::GetBool(node->GetOpDesc(), "_copy_value", copy_value);  // no need to check value
      if (!copy_value) {
        auto src_node = compute_graph->FindNode(cp_from_node);
        GE_CHECK_NOTNULL(src_node);
        GELOGI("current_var_node__: [%s] copy_from_var_node__: [%s].", node->GetName().c_str(),
               src_node->GetName().c_str());
        auto ret = CopyTensorFromSrcVarNode(src_node, node, session_id, device_id);
        GE_IF_BOOL_EXEC(ret != SUCCESS,
                        GELOGE(FAILED, "[Copy][Tensor] failed, src_node:%s, node:%s, session_id:%lu, device_id:%u",
                               src_node->GetName().c_str(), node->GetName().c_str(), session_id, device_id);
                        return FAILED);
        // only copy once
        (void) ge::AttrUtils::SetBool(node->GetOpDesc(), "_copy_value", true);  // no need to check value
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
