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

#include "hybrid/node_executor/hccl/hccl_node_executor.h"

#include "common/ge/plugin_manager.h"
#include "common/math/math_util.h"
#include "external/graph/attr_value.h"
#include "external/graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/util/hcom_util.h"
#include "graph/utils/type_utils.h"
#include "hccl/hcom.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "runtime/event.h"

namespace ge {
namespace {
constexpr size_t kVarTableDims = 2;
constexpr size_t kVarTableRowCnt = 3;
constexpr size_t kVarTableIdxAddr = 1;
constexpr size_t kVarTableIdxLen = 2;
// input anchor nums according to IR
constexpr size_t kAllToAllVInputNums = 5;
constexpr size_t kGatherAllToAllVInputNums = 4;

const std::set<std::string> kRdmaReadTypes = { HCOMREMOTEREAD, HCOMREMOTEREFREAD };
const std::set<std::string> kRdmaWriteTypes = { HCOMREMOTEWRITE, HCOMREMOTESCATTERWRITE };
const std::set<std::string> kRdmaScatterTypes = { HCOMREMOTEREFREAD, HCOMREMOTESCATTERWRITE };
const std::set<std::string> kAllToAllTypes = {HCOMALLTOALLV, HCOMGATHERALLTOALLV};
}  // namespace
namespace hybrid {

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::HCCL, HcclNodeExecutor);

Status HcclNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] HcclNodeTask::ExecuteAsync in.", context.GetNodeName());
  if (context.handle_ == nullptr) {
    REPORT_INNER_ERROR("E19999", " %s invalid, hccl handle is nullptr!", context.GetNodeName());
    GELOGE(FAILED, "[Check][Param:context] %s hccl handle is nullptr!", context.GetNodeName());
    return FAILED;
  }
  auto HcomExecEnqueueOperation = (HcclResult(*)(HcomOperation, std::function<void(HcclResult status)>))dlsym(
      context.handle_, "HcomExecEnqueueOperation");
  if (HcomExecEnqueueOperation == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecEnqueueOperation] failed for %s hcom unknown node function.",
           context.GetNodeName());
    if (dlclose(context.handle_) != 0) {
      GELOGW("Failed to close handle %s", dlerror());
    }
    return FAILED;
  }

  vector<void *> inputs;
  for (int i = 0; i < context.NumInputs(); ++i) {
    TensorValue *tv = context.MutableInput(i);
    GE_CHECK_NOTNULL(tv);
    inputs.emplace_back(tv->MutableData());
  }

  vector<void *> outputs;
  for (int i = 0; i < context.NumOutputs(); ++i) {
    TensorValue *tv = context.MutableOutput(i);
    GE_CHECK_NOTNULL(tv);
    outputs.emplace_back(tv->MutableData());
  }

  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  HcomOperation op_info;
  op_info.hcclType = op_desc->GetType();
  op_info.inputPtr = inputs.empty() ? nullptr : inputs[0];
  op_info.outputPtr = outputs.empty() ? nullptr : outputs[0];
  auto input_desc = node_item.MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  ge::DataType src_data_type = input_desc->GetDataType();
  auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERROR("E19999", "%s inputdesc0 datatype:%s not support.",
                       op_desc->GetName().c_str(),
                       TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType]%s inputdesc0 datatype:%s not support.",
           op_desc->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }
  op_info.dataType = iter->second;
  HcclReduceOp op_type = HCCL_REDUCE_SUM;
  std::set<std::string> hccl_types = { HCOMALLREDUCE, HCOMREDUCESCATTER, HVDCALLBACKALLREDUCE, HCOMREDUCE };
  if (hccl_types.count(op_desc->GetType()) > 0) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclOperationType(op_desc, op_type),
                      "[Get][HcclOperationType] failed for %s type:%s", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str());
    op_info.opType = op_type;
  }
  int64_t root_id = 0;
  if (op_desc->GetType() == HCOMBROADCAST) {
    GE_CHK_STATUS_RET(HcomOmeUtil::GetHcclRootId(op_desc, root_id),
                      "[Get][HcclRootId] failed for %s type:%s", op_desc->GetName().c_str(),
                      op_desc->GetType().c_str());
  }
  op_info.root = root_id;
  auto callback = [op_desc, done_callback](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "call HcomExecEnqueueOperation failed for node %s, ret: 0x%X",
                        op_desc->GetName().c_str(), status);
      GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueOperation] failed for node %s, ret: 0x%X",
             op_desc->GetName().c_str(), status);
    }

    done_callback();
    GELOGI("node %s hccl callback success.", op_desc->GetName().c_str());
  };
  int32_t count = 0;
  GE_CHK_STATUS_RET(HcomOmeUtil::GetHcomCount(op_desc, static_cast<HcclDataType>(op_info.dataType),
                                              op_desc->GetType() == HCOMALLGATHER, count),
                    "[Get][HcomCount] failed for %s type:%s", op_desc->GetName().c_str(),
                    op_desc->GetType().c_str());
  GELOGI("[%s] HcclNodeTask::ExecuteAsync hccl_type %s, count %d, data_type %d, op_type %d, root %d.",
         context.GetNodeName(), op_info.hcclType.c_str(), count, op_info.dataType, op_info.opType, op_info.root);
  op_info.count = count;

  HcclResult hccl_ret = HcomExecEnqueueOperation(op_info, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call HcomExecEnqueueOperation failed for node:%s(%s), ret: 0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), hccl_ret);
    GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueOperation] failed for node:%s(%s), ret: 0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), hccl_ret);
    return HCCL_E_INTERNAL;
  }

  GELOGI("[%s] HcclNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status RdmaNodeTask::UpdateArgs(TaskContext &context) { return SUCCESS; }

Status RdmaNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] RdmaNodeTask::Init in.", context.GetNodeName());
  const NodeItem &node_item = context.GetNodeItem();
  auto op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  auto remote_idx = op_desc->GetInputIndexByName("remote");
  auto in_data_anchor = node_item.node->GetInDataAnchor(remote_idx);
  GE_CHECK_NOTNULL(in_data_anchor);
  auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(out_data_anchor);
  auto peer_node = out_data_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(peer_node->GetOpDesc());

  remote_index_ = {peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx()};
  if (kRdmaReadTypes.count(node_item.node->GetType()) > 0) {
    local_index_ = 0;
  } else {
    local_index_ = op_desc->GetInputIndexByName("local");
  }
  int32_t offset_idx = node_item.op_desc->GetInputIndexByName("local_offset");
  if ((offset_idx != -1) && (node_item.op_desc->GetInputDescPtr(offset_idx) != nullptr)) {
    skip_flag_ = true;
    GE_CHECK_NOTNULL(node_item.node->GetInDataAnchor(offset_idx));
    GE_CHECK_NOTNULL(node_item.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor());
    GE_CHECK_NOTNULL(node_item.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode());
    GE_CHECK_NOTNULL(node_item.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc());
    offset_index_ = {
        node_item.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetOwnerNode()->GetOpDesc()->GetId(),
        node_item.node->GetInDataAnchor(offset_idx)->GetPeerOutAnchor()->GetIdx() };
  }
  return SUCCESS;
}

Status RdmaNodeTask::SetAddrInfo(TaskContext &context, RuntimeInferenceContext *ctx, uint64_t *data, int64_t row_num,
                                 vector<HcomRemoteAccessAddrInfo> &addr_infos) {
  TensorValue *tv = nullptr;
  if (kRdmaReadTypes.count(context.GetNodeItem().NodeType()) > 0) {
    tv = context.MutableOutput(local_index_);
  } else {
    tv = context.MutableInput(local_index_);
  }
  GE_CHECK_NOTNULL(tv);
  addr_infos.resize(row_num);
  if (skip_flag_) {
    int32_t offset_idx = context.GetNodeItem().op_desc->GetInputIndexByName("local_offset");
    GE_CHECK_NOTNULL(context.GetNodeItem().op_desc->GetInputDescPtr(offset_idx));
    auto data_type = context.GetNodeItem().op_desc->GetInputDesc(offset_idx).GetDataType();

    Tensor offset_tensor;
    GE_CHK_STATUS_RET(ctx->GetTensor(offset_index_.first, offset_index_.second, offset_tensor))
    if (static_cast<int64_t>(offset_tensor.GetSize() / GetSizeByDataType(data_type)) != row_num) {
      REPORT_INNER_ERROR("E19999", "num of offset and remote addr mismatch, check invalid"
                         "offset size=%zu, remote_addr size=%ld, dtype=%s", offset_tensor.GetSize(), row_num,
                         TypeUtils::DataTypeToSerialString(data_type).c_str());
      GELOGE(PARAM_INVALID, "[Check][Size]num of offset and remote addr mismatch,"
             "offset size=%zu, remote_addr size=%ld, dtype=%s",
             offset_tensor.GetSize(), row_num, TypeUtils::DataTypeToSerialString(data_type).c_str());
      return PARAM_INVALID;
    }

    auto addr_offset = reinterpret_cast<uint64_t *>(offset_tensor.GetData());
    GE_CHECK_NOTNULL(addr_offset);
    auto base_addr = reinterpret_cast<float *>(tv->MutableData());
    GE_CHECK_NOTNULL(base_addr);

    for (auto idx = 0; idx < row_num; idx++) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt)
      auto line_idx = idx * kVarTableRowCnt;
      addr_infos[idx] = { static_cast<uint32_t>(data[line_idx]),
                          data[line_idx + kVarTableIdxAddr],
                          reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(base_addr + addr_offset[idx])),
                          data[line_idx + kVarTableIdxLen] };
    }
  } else {
    auto local_addr = reinterpret_cast<uint64_t>(reinterpret_cast<uintptr_t>(tv->MutableData()));
    auto device_len = tv->GetSize() / row_num;
    if (device_len <= 0 || device_len > data[kVarTableIdxLen]) {
      REPORT_INNER_ERROR("E19999", "Local embedding length is out of range, expect %ld, but %ld exactly.",
                         data[kVarTableIdxLen], device_len);
      GELOGE(FAILED, "[Check][Size]Local embedding length is out of range, expect %ld, but %ld exactly.",
             data[kVarTableIdxLen], device_len);
      return FAILED;
    }

    for (auto idx = 0; idx < row_num; ++idx) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt)
      auto line_idx = idx * kVarTableRowCnt;
      addr_infos[idx] = { static_cast<uint32_t>(data[line_idx]), data[line_idx + kVarTableIdxAddr], local_addr,
                          device_len };
      local_addr += device_len;
    }
  }

  return SUCCESS;
}

Status RdmaNodeTask::ExtractTensor(TaskContext &context, vector<HcomRemoteAccessAddrInfo> &addr_infos) {
  RuntimeInferenceContext *ctx = nullptr;
  GE_CHK_STATUS_RET(
      RuntimeInferenceContext::GetContext(std::to_string(context.GetExecutionContext()->context_id), &ctx));

  ge::Tensor remote_tensor;
  GE_CHK_STATUS_RET(ctx->GetTensor(remote_index_.first, remote_index_.second, remote_tensor));
  auto data = reinterpret_cast<uint64_t *>(remote_tensor.GetData());
  if (data == nullptr) {
    if (kRdmaScatterTypes.count(context.GetNodeItem().NodeType()) > 0) {
      GELOGD("data is null, no need to do rdma read/write, node=%s", context.GetNodeName());
      return SUCCESS;
    } else {
      REPORT_INNER_ERROR("E19999", "Tensor data is nullptr. and kRdmaScatterTypes not contain %s",
                         context.GetNodeItem().NodeType().c_str());
      GELOGE(FAILED, "[Find][NodeType]Tensor data is nullptr. and kRdmaScatterTypes not contain %s",
             context.GetNodeItem().NodeType().c_str());
      return FAILED;
    }
  }
  auto dims = remote_tensor.GetTensorDesc().GetShape().GetDims();
  if (dims.size() != kVarTableDims && dims.back() != kVarTableRowCnt) {
    REPORT_INNER_ERROR("E19999",
                       "Variable table shape check failed, number of shape dims:%zu not equal expect:%zu"
                       "and shape dims back:%zu not equal expect:%zu, node:%s(%s)",
                       dims.size(), kVarTableDims, dims.back(), kVarTableRowCnt, context.GetNodeName(),
                       context.GetNodeItem().NodeType().c_str());
    GELOGE(PARAM_INVALID,
           "[Check][Param]Variable table shape check failed,"
           "number of shape dims:%zu not equal expect:%zu and shape dims back:%zu not equal expect:%zu, node:%s(%s)",
           dims.size(), kVarTableDims, dims.back(), kVarTableRowCnt, context.GetNodeName(),
           context.GetNodeItem().NodeType().c_str());
    return PARAM_INVALID;
  }

  if (context.GetNodeItem().NodeType() == HCOMREMOTEREAD) {
    size_t remote_size = 0;
    for (auto idx = 0; idx < dims.front(); ++idx) {
      FMK_INT64_MULCHECK(idx, kVarTableRowCnt);
      auto line_idx = idx * kVarTableRowCnt;
      remote_size += data[line_idx + kVarTableIdxLen];
    }
    auto allocator = NpuMemoryAllocator::GetAllocator();
    GE_CHECK_NOTNULL(allocator);
    AllocationAttr attr;
    attr.SetMemType(RDMA_HBM);
    for (auto i = 0; i < context.NumOutputs(); ++i) {
      GELOGD("Allocate rdma memory for node %s, size: %zu", context.GetNodeName(), remote_size);
      auto tensor_buffer = TensorBuffer::Create(allocator, remote_size, &attr);
      GE_CHK_STATUS_RET(context.SetOutput(i, TensorValue(std::shared_ptr<TensorBuffer>(tensor_buffer.release()))));
    }
  } else if (context.GetNodeItem().NodeType() == HCOMREMOTEREFREAD) {
    AllocationAttr attr;
    attr.SetMemType(RDMA_HBM);
    GE_CHK_STATUS_RET(context.AllocateOutputs(&attr))
  }

  auto row_num = dims.front();
  return SetAddrInfo(context, ctx, data, row_num, addr_infos);
}

Status RdmaNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] RdmaNodeTask::ExecuteAsync in.", context.GetNodeName());
  auto HcomExecEnqueueRemoteAccess =
      (HcclResult(*)(const string &, const vector<HcomRemoteAccessAddrInfo> &,
                     std::function<void(HcclResult status)>))dlsym(context.handle_, "HcomExecEnqueueRemoteAccess");
  if (HcomExecEnqueueRemoteAccess == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecEnqueueRemoteAccess] failed for node:%s(%s) hcom unknown node function.",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
    if (dlclose(context.handle_) != 0) {
      GELOGW("Failed to close handle %s", dlerror());
    }
    return FAILED;
  }
  vector<HcomRemoteAccessAddrInfo> addr_infos;
  GE_CHK_STATUS_RET(ExtractTensor(context, addr_infos));
  if (addr_infos.empty()) {
    done_callback();
    return SUCCESS;
  }

  rtEvent_t evt = nullptr;
  if (context.GetExecutionContext()->hccl_stream != nullptr) {
    GE_CHK_RT_RET(rtEventCreateWithFlag(&evt, RT_EVENT_WITH_FLAG));
    GE_CHK_RT_RET(rtStreamWaitEvent(context.GetExecutionContext()->hccl_stream, evt));
  }
  TaskContext *p_ctx = &context;
  auto callback = [p_ctx, done_callback, evt](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "[Call][HcomExcutorInitialize] failed for node:%s(%s), ret: 0x%X",
             p_ctx->GetNodeName(), p_ctx->GetNodeItem().NodeType().c_str(), status);
      p_ctx->SetStatus(FAILED);
    }
    done_callback();
    if (evt != nullptr) {
      GE_CHK_RT_RET(rtEventRecord(evt, nullptr));
      GE_CHK_RT_RET(rtEventDestroy(evt));
    }
    GELOGI("rdma callback success.");
    return SUCCESS;
  };

  HcclResult hccl_ret = HcomExecEnqueueRemoteAccess(context.GetNodeItem().NodeType(), addr_infos, callback);
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(HCCL_E_INTERNAL, "[Call][HcomExecEnqueueRemoteAccess] failed for node:%s(%s), ret: 0x%X",
           context.GetNodeName(), context.GetNodeItem().NodeType().c_str(), hccl_ret);
    return HCCL_E_INTERNAL;
  }

  GELOGI("[%s] RdmaNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status BuildAllToAllVparams(TaskContext &context, HcomAllToAllVParams &params) {
  void **input_addrs[kAllToAllVInputNums] = {&params.sendbuf, &params.sendcounts, &params.sdispls, &params.recvcounts,
                                             &params.rdispls};
  for (size_t i = 0; i < kAllToAllVInputNums; ++i) {
    auto addr = context.MutableInput(i);
    GE_CHECK_NOTNULL(addr);
    *input_addrs[i] = addr->MutableData();
  }
  auto recv_tv = context.MutableOutput(0);
  GE_CHECK_NOTNULL(recv_tv);
  params.recvbuf = recv_tv->MutableData();

  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = node_item.GetOpDesc();
  auto input_desc = node_item.MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  ge::DataType src_data_type = input_desc->GetDataType();
  auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(src_data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERROR("E19999", "%s alltoallv datatype:%s not support.", op_desc->GetName().c_str(),
                       TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType]%s alltoallv datatype:%s not support.", op_desc->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return PARAM_INVALID;
  }
  params.sendtype = iter->second;
  params.recvtype = iter->second;
  params.group = nullptr;

  return SUCCESS;
}

Status BuildGatherAllToAllParams(TaskContext &context, HcomGatherAllToAllVParams &params) {
  void **input_addrs[kGatherAllToAllVInputNums] = {&params.addrInfo, &params.addrInfoCountPerRank, &params.recvcounts,
                                                   &params.rdispls};
  for (size_t i = 0; i < kGatherAllToAllVInputNums; ++i) {
    auto addr = context.MutableInput(i);
    GE_CHECK_NOTNULL(addr);
    *input_addrs[i] = addr->MutableData();
  }
  auto recv_tv = context.MutableOutput(0);
  GE_CHECK_NOTNULL(recv_tv);
  params.recvbuf = recv_tv->MutableData();
  auto gathered_tv = context.MutableOutput(1);
  GE_CHECK_NOTNULL(gathered_tv);
  params.gatheredbuf = gathered_tv->MutableData();

  const NodeItem &node_item = context.GetNodeItem();
  const OpDescPtr op_desc = node_item.GetOpDesc();

  ge::DataType data_type = ge::DT_FLOAT;
  (void)ge::AttrUtils::GetDataType(op_desc, HCOM_ATTR_DATA_TYPE, data_type);
  auto iter = kConstOpHcclDataType.find(static_cast<int64_t>(data_type));
  if (iter == kConstOpHcclDataType.end()) {
    REPORT_INNER_ERROR("E19999", "%s received datatype:%s not support.", op_desc->GetName().c_str(),
                       TypeUtils::DataTypeToSerialString(data_type).c_str());
    GELOGE(PARAM_INVALID, "[Find][DataType]%s received datatype:%s not support.", op_desc->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return PARAM_INVALID;
  }
  params.recvtype = iter->second;

  int64_t addr_len = 0;
  (void)ge::AttrUtils::GetInt(op_desc, "addr_length", addr_len);
  params.addrLength = static_cast<int>(addr_len);
  params.group = nullptr;

  return SUCCESS;
}

Status AllToAllNodeTask::ExecuteAsync(TaskContext &context, std::function<void()> done_callback) {
  GELOGI("[%s] AllToAllNodeTask::ExecuteAsync in.", context.GetNodeName());

  TaskContext *p_ctx = &context;
  auto callback = [p_ctx, done_callback](HcclResult status) {
    if (status != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "[%s] AllToAllNodeTask execute failed.", p_ctx->GetNodeName());
      p_ctx->SetStatus(FAILED);
    }
    done_callback();
    GELOGI("[%s] AllToAllNodeTask callback successfully.", p_ctx->GetNodeName());
  };

  if (context.GetNodeItem().NodeType() == HCOMALLTOALLV) {
    auto HcomExecEnqueueAllToAllV = (HcclResult(*)(HcomAllToAllVParams, std::function<void(HcclResult status)>))dlsym(
        context.handle_, "HcomExecEnqueueAllToAllV");
    if (HcomExecEnqueueAllToAllV == nullptr) {
      GELOGE(FAILED, "Failed to invoke function [HcomExecEnqueueAllToAllV] for node:%s.",context.GetNodeName());
      return FAILED;
    }
    HcomAllToAllVParams params;
    GE_CHK_STATUS_RET(BuildAllToAllVparams(context, params));
    HcclResult hccl_ret = HcomExecEnqueueAllToAllV(params, callback);
    if (hccl_ret != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "AllToAllV teak enqueue failed for node [%s].", context.GetNodeName());
      return HCCL_E_INTERNAL;
    }
  } else {
    auto HcomExecEnqueueGatherAllToAllV =
        (HcclResult(*)(HcomGatherAllToAllVParams, std::function<void(HcclResult status)>))dlsym(
            context.handle_, "HcomExecEnqueueGatherAllToAllV");
    if (HcomExecEnqueueGatherAllToAllV == nullptr) {
      GELOGE(FAILED, "Failed to invoke function [HcomExecEnqueueGatherAllToAllV] for node:%s.", context.GetNodeName());
      return FAILED;
    }
    HcomGatherAllToAllVParams params;
    GE_CHK_STATUS_RET(BuildGatherAllToAllParams(context, params));
    HcclResult hccl_ret = HcomExecEnqueueGatherAllToAllV(params, callback);
    if (hccl_ret != HCCL_SUCCESS) {
      GELOGE(HCCL_E_INTERNAL, "GatherAllToAllV teak enqueue failed for node [%s].", context.GetNodeName());
      return HCCL_E_INTERNAL;
    }
  }
  GELOGI("[%s] AllToAllNodeTask::ExecuteAsync success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeTask::UpdateArgs(TaskContext &context) { return SUCCESS; }

Status HcclNodeTask::Init(TaskContext &context) {
  GELOGI("[%s] HcclNodeExecutor::Init success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::PrepareTask(NodeTask &task, TaskContext &context) const {
  GELOGI("[%s] HcclNodeExecutor::PrepareTask in.", context.GetNodeName());

  GE_CHK_STATUS_RET(task.Init(context), "[Invoke][Init]hccl node %s(%s) load hccl so failed.",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  // allocate output mem, output mem or remote read will be calculated when node execute.
  if (kRdmaReadTypes.count(context.GetNodeItem().NodeType()) == 0) {
    GE_CHK_STATUS_RET(context.AllocateOutputs(),
                      "[Invoke][AllocateOutputs]hccl node %s(%s) task allocate output failed.",
                      context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  }

  GE_CHK_STATUS_RET(task.UpdateArgs(context), "[Update][Args] failed for hccl node %s(%s).",
                    context.GetNodeName(), context.GetNodeItem().NodeType().c_str());
  GELOGI("[%s] HcclNodeExecutor::PrepareTask success.", context.GetNodeName());
  return SUCCESS;
}

Status HcclNodeExecutor::LoadTask(const HybridModel &model, const NodePtr &node, shared_ptr<NodeTask> &task) const {
  GELOGI("[%s] HcclNodeExecutor::LoadTask in.", node->GetName().c_str());
  GE_CHECK_NOTNULL(node);
  if ((kRdmaReadTypes.count(node->GetType()) > 0) || (kRdmaWriteTypes.count(node->GetType()) > 0)) {
    task = MakeShared<RdmaNodeTask>();
  } else if (kAllToAllTypes.count(node->GetType()) > 0) {
    task = MakeShared<AllToAllNodeTask>();
  } else {
    task = MakeShared<HcclNodeTask>();
  }
  GE_CHECK_NOTNULL(task);
  GELOGI("[%s] HcclNodeExecutor::LoadTask success.", node->GetName().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::ExecuteTask(NodeTask &task, TaskContext &context,
                                     const std::function<void()> &callback) const {
  context.handle_ = handle_;
  GE_CHK_STATUS_RET(task.ExecuteAsync(context, callback),
                    "[Invoke][ExecuteAsync] failed to execute task. node:%s(%s)",
                    context.GetNodeItem().NodeName().c_str(), context.GetNodeItem().NodeType().c_str());
  return SUCCESS;
}

Status HcclNodeExecutor::Initialize() {
  std::string file_name = "libhcom_graph_adaptor.so";
  std::string path = PluginManager::GetPath();
  path.append(file_name);
  string canonical_path = RealPath(path.c_str());
  if (canonical_path.empty()) {
    GELOGW("failed to get realpath of %s", path.c_str());
    return FAILED;
  }

  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonical_path.c_str());
  handle_ = dlopen(canonical_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (handle_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "Open SoFile %s failed, error:%s! ", canonical_path.c_str(), dlerror());
    GELOGE(GE_PLGMGR_SO_NOT_EXIST, "[Open][SoFile] %s failed, error:%s! ", canonical_path.c_str(), dlerror());
    return FAILED;
  }
  auto HcomExecInitialize = (HcclResult(*)())dlsym(handle_, "HcomExecInitialize");
  if (HcomExecInitialize == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecInitialize] Failed for hcom unknown node function.");
    return FAILED;
  }
  HcclResult hccl_ret = HcomExecInitialize();
  if (hccl_ret == HCCL_E_PTR) {
    GELOGI("Hccl comm is null, hcom executor initialize is not required.");
  } else if (hccl_ret == HCCL_SUCCESS) {
    GELOGI("Hcom executor initialize success.");
  } else {
    GELOGE(FAILED, "[Call][HcomExecInitialize] failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  return SUCCESS;
}

Status HcclNodeExecutor::Finalize() {
  auto HcomExecFinalize = (HcclResult(*)())dlsym(handle_, "HcomExecFinalize");
  if (HcomExecFinalize == nullptr) {
    GELOGE(FAILED, "[Invoke][HcomExecFinalize] failed for hcom unknown node function.");
    return FAILED;
  }
  HcclResult hccl_ret = HcomExecFinalize();
  if (hccl_ret != HCCL_SUCCESS) {
    GELOGE(FAILED, "[Call][HcomExecFinalize] failed, ret: 0x%X", hccl_ret);
    return FAILED;
  }
  // dlclose file handle
  if (dlclose(handle_) != 0) {
    GELOGW("Failed to close handle %s", dlerror());
  }
  GELOGI("Hcom executor finalize success.");
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
