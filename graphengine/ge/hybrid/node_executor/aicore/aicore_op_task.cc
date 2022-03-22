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

#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/common/taskdown_common.h"
#include "framework/common/debug/log.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "graph/load/model_manager/tbe_handle_store.h"
#include "external/graph/types.h"
#include "single_op/task/build_task_utils.h"
#include "single_op/task/tbe_task_builder.h"

using optiling::utils::OpRunInfo;

namespace ge {
namespace hybrid {
namespace {
constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
constexpr char const *kAttrOpParamSize = "op_para_size";
constexpr char const *kAttrAtomicOpParamSize = "atomic_op_para_size";
const string kAtomicOpType = "DynamicAtomicAddrClean";
std::atomic<std::uint64_t> log_id(0);
}  // namespace

TbeHandleHolder::TbeHandleHolder(void *bin_handle)
    : bin_handle_(bin_handle) {}

TbeHandleHolder::~TbeHandleHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

bool TbeHandleRegistry::AddHandle(std::unique_ptr<TbeHandleHolder> &&holder) {
  auto ret = registered_handles_.emplace(std::move(holder));
  return ret.second;
}

Status AiCoreOpTask::Init(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  op_type_ = op_desc.GetType();
  log_name_ = op_desc.GetName() + "_tvmbin";
  log_id_ = log_id++;
  auto op_desc_ptr = MakeShared<OpDesc>(op_desc);
  GE_CHECK_NOTNULL(op_desc_ptr);
  auto task_info = BuildTaskUtils::GetTaskInfo(op_desc_ptr);
  GELOGI("[TASK_INFO] %lu/%s %s.", log_id_, log_name_.c_str(), task_info.c_str());
  GE_CHK_STATUS_RET_NOLOG(InitWithTaskDef(op_desc, task_def));
  GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(op_desc));

  GE_CHECK_LE(op_desc.GetOutputsSize(), static_cast<size_t>(INT_MAX));
  int outputs_size = static_cast<int>(op_desc.GetOutputsSize());

  for (int i = 0; i < outputs_size; ++i) {
    const GeTensorDescPtr tensor_desc = op_desc.MutableOutputDesc(i);
    if (tensor_desc == nullptr) {
      GELOGW("Op: %s, Index: %d, Tensor Desc is null", op_desc.GetName().c_str(), i);
      continue;
    }

    int32_t calc_type = 0;
    bool ret = ge::AttrUtils::GetInt(tensor_desc, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
    if (ret && (calc_type == static_cast<int32_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY))) {
      output_indices_to_skip_.push_back(i);
    }
  }
  return SUCCESS;
}

Status AiCoreOpTask::RegisterTbeHandle(const OpDesc &op_desc) {
  rtError_t rt_ret = rtQueryFunctionRegistered(stub_name_.c_str());
  if (rt_ret != RT_ERROR_NONE) {
    auto op_desc_ptr = MakeShared<OpDesc>(op_desc);
    GE_CHECK_NOTNULL(op_desc_ptr);
    auto tbe_kernel = op_desc_ptr->TryGetExtAttr(GetKeyForTbeKernel(), TBEKernelPtr());
    if (tbe_kernel == nullptr) {
      GELOGE(INTERNAL_ERROR, "TBE: %s can't find tvm bin file!", op_desc_ptr->GetName().c_str());
      return INTERNAL_ERROR;
    }
    TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
    void *bin_handle = nullptr;
    if (!kernel_store.FindTBEHandle(stub_name_.c_str(), bin_handle)) {
      GELOGI("TBE: can't find the binfile_key[%s] in HandleMap", stub_name_.c_str());
      rtDevBinary_t binary;
      std::string json_string;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, GetKeyForTvmMagic(), json_string),
                      GELOGI("Get original type of session_graph_id."));
      if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICPU") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICPU;
      } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;
      } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
        binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
      } else {
        GELOGE(PARAM_INVALID, "[Check][JsonStr]Attr:%s in op:%s(%s), value:%s check invalid",
               TVM_ATTR_NAME_MAGIC.c_str(), op_desc_ptr->GetName().c_str(),
               op_desc_ptr->GetType().c_str(), json_string.c_str());
        REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%s check invalid",
                           TVM_ATTR_NAME_MAGIC.c_str(), op_desc_ptr->GetName().c_str(),
                           op_desc_ptr->GetType().c_str(), json_string.c_str());
        return PARAM_INVALID;
      }
      binary.version = 0;
      binary.data = tbe_kernel->GetBinData();
      binary.length = tbe_kernel->GetBinDataSize();
      GELOGI("TBE: binary.length: %lu", binary.length);
      GE_CHK_RT_RET(rtDevBinaryRegister(&binary, &bin_handle));
      std::string meta_data;
      GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, GetKeyForTvmMetaData(), meta_data),
                      GELOGI("Get original type of json_string"));
      GELOGI("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
      GE_IF_BOOL_EXEC(!meta_data.empty(),
                      GE_CHK_RT_RET(rtMetadataRegister(bin_handle, meta_data.c_str())));
      kernel_store.StoreTBEHandle(stub_name_.c_str(), bin_handle, tbe_kernel);
    } else {
      GELOGI("TBE: find the binfile_key[%s] in HandleMap", stub_name_.c_str());
      kernel_store.ReferTBEHandle(stub_name_.c_str());
    }
    std::string kernel_name;
    GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_ptr, GetKeyForKernelName(op_desc), kernel_name),
                    GELOGI("Get original type of kernel_name"));
    GELOGI("TBE: binfile_key=%s, kernel_name=%s", stub_name_.c_str(), kernel_name.c_str());
    auto stub_func = KernelBinRegistry::GetInstance().GetUnique(stub_name_);
    GE_CHK_RT_RET(rtFunctionRegister(bin_handle, stub_func, stub_name_.c_str(), kernel_name.c_str(), 0));
  }
  return SUCCESS;
}

Status AiCoreOpTask::RegisterKernelHandle(const OpDesc &op_desc) {
  TbeHandleRegistry &registry = TbeHandleRegistry::GetInstance();
  auto tbe_kernel = op_desc.TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Invoke][TryGetExtAttr]TBE: %s can't find tvm bin file!",
           op_desc.GetName().c_str());
    REPORT_CALL_ERROR("E19999", "TBE: %s can't find tvm bin file.", op_desc.GetName().c_str());
    return INTERNAL_ERROR;
  }

  void *bin_handle = nullptr;
  GELOGD("Start to register kernel for node: [%s].", op_desc.GetName().c_str());
  rtDevBinary_t binary;
  std::string json_string;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(&op_desc, TVM_ATTR_NAME_MAGIC, json_string),
                  GELOGI("Get original type of session_graph_id."));
  if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICPU") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AICPU;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else {
    GELOGE(PARAM_INVALID, "[Check][JsonStr]Attr:%s in op:%s(%s), value:%s check invalid",
           TVM_ATTR_NAME_MAGIC.c_str(), op_desc.GetName().c_str(),
           op_desc.GetType().c_str(), json_string.c_str());
    REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%s check invalid",
                       TVM_ATTR_NAME_MAGIC.c_str(), op_desc.GetName().c_str(),
                       op_desc.GetType().c_str(), json_string.c_str());
    return PARAM_INVALID;
  }
  binary.version = 0;
  binary.data = tbe_kernel->GetBinData();
  binary.length = tbe_kernel->GetBinDataSize();
  GELOGI("TBE: binary.length: %lu", binary.length);
  GE_CHK_RT_RET(rtRegisterAllKernel(&binary, &bin_handle));
  handle_ = bin_handle;
  auto holder = std::unique_ptr<TbeHandleHolder>(new (std::nothrow) TbeHandleHolder(handle_));
  if (holder == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION,
           "[Create][TbeHandleHolder] failed, node name = %s", op_desc.GetName().c_str());
    REPORT_CALL_ERROR("E19999", "create TbeHandleHolder failed, node name = %s.",
                      op_desc.GetName().c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  if (!registry.AddHandle(std::move(holder))) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Add][Handle] failed. node name = %s", op_desc.GetName().c_str());
    REPORT_CALL_ERROR("E19999", "AddHandle failed, node name = %s.", op_desc.GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status AiCoreOpTask::InitWithKernelDef(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  stub_name_ = is_single_op_ ? to_string(log_id_) + kernel_def.stub_func() : kernel_def.stub_func();
  GE_CHK_STATUS_RET(RegisterTbeHandle(op_desc));
  GE_CHK_RT_RET(rtGetFunctionByName(stub_name_.c_str(), &stub_func_));
  args_size_ = kernel_def.args_size();
  block_dim_ = kernel_def.block_dim();
  // malloc args memory
  args_.reset(new(std::nothrow) uint8_t[args_size_]);
  GE_CHECK_NOTNULL(args_);
  if (kernel_def.args().size() < args_size_) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]args size:%zu of kernel_def is smaller than args_size_:%u, op:%s op_type:%s",
           kernel_def.args().size(), args_size_, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "args size:%zu of kernel_def is smaller than args_size_:%u op:%s op_type:%s.",
                       kernel_def.args().size(), args_size_, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }
  errno_t err = memcpy_s(args_.get(), args_size_, kernel_def.args().data(), args_size_);
  if (err != EOK) {
    GELOGE(INTERNAL_ERROR, "[Update][Date]AiCoreTask memcpy args failed, op:%s op_type:%s.",
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "AiCoreTask memcpy args failed, op:%s op_type:%s.",
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Invalid args_offset,"
           "size:%zu is smaller than size of uint16_t, op:%s op_type:%s",
           context.args_offset().size(), op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "Invalid args_offset, size:%zu is smaller than size of uint16_t, op:%s op_type:%s",
                       context.args_offset().size(), op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto *args_offset_buffer = reinterpret_cast<const uint16_t *>(context.args_offset().data());
  offset_ = *args_offset_buffer;
  if (offset_ > args_size_) {
    GELOGE(INTERNAL_ERROR, "[Check][Offset][%s] Arg offset out of range. offset = %u,"
           "arg size = %u , op:%s op_type:%s", GetName().c_str(), offset_, args_size_,
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Arg offset out of range. offset = %u, arg size = %u"
                       "op:%s op_type:%s", GetName().c_str(), offset_, args_size_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  arg_base_ = reinterpret_cast<uintptr_t *>(args_.get() + offset_);
  max_arg_count_ = (args_size_ - offset_) / sizeof(void *);
  GELOGD("[%s] Done setting kernel args successfully. stub_func = %s, block_dim = %d,"
         "arg base = %p, arg size = %u",
         op_desc.GetName().c_str(),  stub_name_.c_str(),
         block_dim_, arg_base_, args_size_);
  return SUCCESS;
}

Status AiCoreOpTask::InitWithKernelDefWithHandle(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  const domi::KernelDefWithHandle &kernel_with_handle = task_def.kernel_with_handle();
  const domi::KernelContext &context = kernel_with_handle.context();

  GE_CHK_STATUS_RET(RegisterKernelHandle(op_desc));
  original_kernel_key_ = kernel_with_handle.original_kernel_key() + "_";
  node_info_ = kernel_with_handle.node_info() + "/";
  args_size_ = kernel_with_handle.args_size();
  block_dim_ = kernel_with_handle.block_dim();
  // malloc args memory
  args_.reset(new(std::nothrow) uint8_t[args_size_]);
  GE_CHECK_NOTNULL(args_);
  if (kernel_with_handle.args().size() < args_size_) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]args size:%zu of kernel_def is smaller than args_size_:%u. op:%s op_type:%s",
           kernel_with_handle.args().size(), args_size_, op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "args size:%zu of kernel_def is smaller than args_size_:%u. op:%s op_type:%s",
                       kernel_with_handle.args().size(), args_size_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }
  errno_t err = memcpy_s(args_.get(), args_size_, kernel_with_handle.args().data(), args_size_);

  if (err != EOK) {
    GELOGE(INTERNAL_ERROR, "[Update][Date]AiCoreTask memcpy args failed. op:%s op_type:%s",
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_CALL_ERROR("E19999", "AiCoreTask memcpy args failed. op:%s op_type:%s",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() < sizeof(uint16_t)) {
    GELOGE(INTERNAL_ERROR, "[Check][Size]Invalid args_offset, size:%zu is smaller"
           "than size of uint16_t. op:%s op_type:%s", context.args_offset().size(),
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "Invalid args_offset, size:%zu is smaller"
                       "than size of uint16_t. op:%s op_type:%s", context.args_offset().size(),
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  const auto *args_offset_buffer = reinterpret_cast<const uint16_t *>(context.args_offset().data());
  offset_ = *args_offset_buffer;
  if (offset_ > args_size_) {
    GELOGE(INTERNAL_ERROR, "[Check][Offset][%s] Arg offset out of range. offset = %u, arg size = %u"
           "op:%s op_type:%s", GetName().c_str(), offset_, args_size_,
           op_desc.GetName().c_str(), op_desc.GetType().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Arg offset out of range. offset = %u, arg size = %u"
                       "op:%s op_type:%s", GetName().c_str(), offset_, args_size_,
                       op_desc.GetName().c_str(), op_desc.GetType().c_str());
    return INTERNAL_ERROR;
  }

  arg_base_ = reinterpret_cast<uintptr_t *>(args_.get() + offset_);
  max_arg_count_ = (args_size_ - offset_) / sizeof(void *);
  return SUCCESS;
}

Status AiCoreOpTask::InitWithTaskDef(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  
  auto rt_ret = ValidateTaskDef(task_def);
  if (rt_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "op:%s(op_type:%s) failed to validate task def:%s",
                      op_desc.GetName().c_str(), op_desc.GetType().c_str(), task_def.DebugString().c_str());
    GELOGE(rt_ret, "[Invoke][ValidateTaskDef]failed for op:%s(op_type:%s) to validate task def:%s",
           op_desc.GetName().c_str(), op_desc.GetType().c_str(), task_def.DebugString().c_str());
    return rt_ret;
  }
 
  if (task_def.type() != RT_MODEL_TASK_ALL_KERNEL) {
    GE_CHK_STATUS_RET(InitWithKernelDef(op_desc, task_def));
  } else {
    GE_CHK_STATUS_RET(InitWithKernelDefWithHandle(op_desc, task_def));
  }
  return SUCCESS;
}

Status AiCoreOpTask::ValidateTaskDef(const domi::TaskDef &task_def) {
  auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
  if (task_type != RT_MODEL_TASK_KERNEL && task_type != RT_MODEL_TASK_ALL_KERNEL) {
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid task type (%d) in AiCore CreateTask.", static_cast<int>(task_type));
    return INTERNAL_ERROR;
  }
  const auto &context = task_type == RT_MODEL_TASK_KERNEL ? task_def.kernel().context() :
                                                            task_def.kernel_with_handle().context();
  auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type != ccKernelType::TE) {
    GELOGE(INTERNAL_ERROR,
           "[Check][TaskType]Invalid kernel type(%d) in AiCore TaskDef.", static_cast<int>(kernel_type));
    REPORT_INNER_ERROR("E19999", "Invalid kernel type(%d) in AiCore TaskDef.",
                       static_cast<int>(kernel_type));
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status AiCoreOpTask::PrepareWithShape(TaskContext &context) {
  if (is_dynamic_) {
    return UpdateTilingInfo(context);
  }
  return SUCCESS;
}

Status AiCoreOpTask::UpdateTilingInfo(TaskContext &context) {
  auto node = context.GetNodeItem().node;
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  GELOGD("[%s] Start to update tiling info for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  OpRunInfo tiling_info(-1, true, 0);

  auto execution_context = context.GetExecutionContext();

  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] Start");
  GE_CHK_STATUS_RET(CalcTilingInfo(node, tiling_info));
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CalcTilingInfo] End");

  // update op args by tiling info
  block_dim_ = tiling_info.GetBlockDim();
  clear_atomic_ = tiling_info.GetClearAtomic();

  tiling_data_ = tiling_info.GetAllTilingData().str();
  tiling_key_ = tiling_info.GetTilingKey();
  GELOGD("Successfully getting [tiling_key] : %u", tiling_key_);
  if (tiling_data_.empty()) {
    GELOGD("[%s] Tiling data is empty.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (tiling_buffer_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "[Check][Buffer] %s tiling_buffer is nullptr while tiling_data is not empty!",
           op_desc->GetName().c_str());
    REPORT_INNER_ERROR("E19999",  "%s tiling_buffer is nullptr while tiling_data is not empty.",
                       op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  if (tiling_data_.size() > tiling_buffer_->GetSize()) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] Tiling data size now (%zu)"
           "shouldn't larger than we alloc before (%zu). op:%s op_type:%s",
           stub_name_.c_str(), tiling_data_.size(), tiling_buffer_->GetSize(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] Tiling data size now (%zu)"
                       "shouldn't larger than we alloc before (%zu). op:%s op_type:%s",
                       stub_name_.c_str(), tiling_data_.size(), tiling_buffer_->GetSize(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CopyTilingInfo] Start");
  GE_CHK_RT_RET(rtMemcpyAsync(tiling_buffer_->GetData(), tiling_buffer_->GetSize(), tiling_data_.c_str(),
                              tiling_data_.size(), RT_MEMCPY_HOST_TO_DEVICE_EX, context.GetStream()));
  RECORD_EXECUTION_EVENT(execution_context, context.GetNodeName(), "[CopyTilingInfo] End");

  GELOGD("[%s] Done updating tiling info for task: [%s]", node->GetName().c_str(), stub_name_.c_str());
  return SUCCESS;
}

Status AiCoreOpTask::CalcTilingInfo(const NodePtr &node, OpRunInfo &tiling_info) {
  GELOGD("[%s] Start to invoke OpParaCalculate.", node->GetName().c_str());
  GE_CHK_STATUS_RET(optiling::OpParaCalculateV2(*node, tiling_info),
                    "[Invoke][OpParaCalculate]Failed calc tiling data of node %s.",
                    node->GetName().c_str());
  // Only non atomic task need update workspace
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> workspaces;
  tiling_info.GetAllWorkspaces(workspaces);
  op_desc->SetWorkspaceBytes(workspaces);
  GELOGD("[%s] Done invoking OpParaCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AiCoreOpTask::UpdateArgs(TaskContext &task_context) {
  size_t expected_arg_count = task_context.NumInputs() + task_context.NumOutputs() + task_context.NumWorkspaces() -
                              output_indices_to_skip_.size();
  if (tiling_buffer_ != nullptr) {
    ++expected_arg_count;
  }
  if (expected_arg_count > max_arg_count_) {
    GELOGD("Need to reset size of args_ from %u to %zu.", max_arg_count_, expected_arg_count);
    auto length = expected_arg_count * sizeof(uintptr_t) + offset_;
    std::unique_ptr<uint8_t[]> new_args(new(std::nothrow) uint8_t[length]);
    GE_CHECK_NOTNULL(new_args);
    if (memcpy_s(new_args.get(), length, args_.get(), offset_) != EOK) {
      GELOGE(ACL_ERROR_GE_MEMORY_OPERATE_FAILED, "[Update][new_args]failed, dst length is %zu, src length is %u.",
             length, offset_);
      REPORT_INNER_ERROR("E19999", "update kernel args failed of %s.", task_context.GetNodeName());
      return ACL_ERROR_GE_MEMORY_OPERATE_FAILED;
    }
    args_ = std::move(new_args);
    max_arg_count_ = static_cast<uint32_t>(expected_arg_count);
    args_size_ = static_cast<uint32_t>(length);
    arg_base_  = reinterpret_cast<uintptr_t *>(args_.get() + offset_);
  }

  int index = 0;
  for (int i = 0; i < task_context.NumInputs(); ++i) {
    const auto input = task_context.GetInput(i);
    GE_CHECK_NOTNULL(input);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(input->GetData());
  }

  for (int i = 0; i < task_context.NumOutputs(); ++i) {
    const auto output = task_context.GetOutput(i);
    GE_CHECK_NOTNULL(output);
    if (find(output_indices_to_skip_.begin(), output_indices_to_skip_.end(), i) !=
        output_indices_to_skip_.end()) {
      GELOGD("Node:%s output[%d] is an optional, the address don't need to be saved.",
             task_context.GetNodeName(), i);
      continue;
    }
    arg_base_[index++] = reinterpret_cast<uintptr_t>(output->GetData());
  }

  int workspace_num = static_cast<int>(task_context.NumWorkspaces());
  for (int i = 0; i < workspace_num; ++i) {
    const auto workspace = task_context.MutableWorkspace(i);
    GE_CHECK_NOTNULL(workspace);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(workspace);
  }

  if (tiling_buffer_ != nullptr) {
    arg_base_[index++] = reinterpret_cast<uintptr_t>(tiling_buffer_->GetData());
  }

  if (task_context.IsTraceEnabled()) {
    for (int i = 0; i < index; ++i) {
      GELOGD("[%s] Arg[%d] = %lu", stub_name_.c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}

Status AiCoreOpTask::LaunchKernel(rtStream_t stream) {
  if (handle_ != nullptr) {
    std::string dev_func = original_kernel_key_ + std::to_string(tiling_key_);
    std::string kernel_info = node_info_ + std::to_string(tiling_key_);
    GELOGD("AiCoreOpTask rtKernelLaunchWithHandle Start (dev_func = %s, block_dim = %u).",
           dev_func.c_str(), block_dim_);
    GE_CHK_RT_RET(rtKernelLaunchWithHandle(handle_, dev_func.c_str(), block_dim_, args_.get(),
                                           args_size_, nullptr, stream, kernel_info.c_str()));
    GELOGD("AiCoreOpTask rtKernelLaunchWithHandle End (dev_func = %s, block_dim = %u).",
           dev_func.c_str(), block_dim_);
  } else {
    GELOGD("AiCoreOpTask LaunchKernel Start (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);
    GE_CHK_RT_RET(rtKernelLaunch(stub_func_, block_dim_, args_.get(), args_size_, nullptr, stream));
    GELOGD("AiCoreOpTask LaunchKernel End (task = %s, block_dim = %u).", stub_name_.c_str(), block_dim_);
  }
  GELOGI("[TASK_INFO] %lu/%s", log_id_, log_name_.c_str());
  return SUCCESS;
}

Status AiCoreOpTask::InitTilingInfo(const OpDesc &op_desc) {
  (void) AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, is_dynamic_);
  if (!is_dynamic_) {
    GELOGD("[%s] Dynamic shape is not supported.", op_desc.GetName().c_str());
    return SUCCESS;
  }

  GELOGD("Start alloc tiling data of node %s.", op_desc.GetName().c_str());
  int64_t max_size = -1;
  (void) AttrUtils::GetInt(op_desc, GetKeyForOpParamSize(), max_size);
  GELOGD("Got op param size by key: %s, ret = %ld", GetKeyForOpParamSize().c_str(), max_size);
  if (max_size < 0) {
    GELOGE(PARAM_INVALID, "[Check][Size][%s] Invalid op_param_size: %ld.", op_desc.GetName().c_str(), max_size);
    REPORT_INNER_ERROR("E19999", "[%s] Invalid op_param_size: %ld.", op_desc.GetName().c_str(), max_size);
    return PARAM_INVALID;
  }

  auto allocator = NpuMemoryAllocator::GetAllocator();
  GE_CHECK_NOTNULL(allocator);
  if (max_size > 0) {
    tiling_buffer_ = TensorBuffer::Create(allocator, static_cast<size_t>(max_size));
    GE_CHECK_NOTNULL(tiling_buffer_);
    GELOGD("[%s] Done allocating tiling buffer, size=%ld.", op_desc.GetName().c_str(), max_size);
  } else {
    GELOGD("op_param_size is 0, no need to create tiling buffer.");
  }

  return SUCCESS;
}

bool AiCoreOpTask::IsDynamicShapeSupported() {
  return is_dynamic_;
}

const std::string &AiCoreOpTask::GetName() const {
  return stub_name_;
}

const std::string &AiCoreOpTask::GetOpType() const {
  return op_type_;
}

std::string AiCoreOpTask::GetKeyForOpParamSize() const {
  return kAttrOpParamSize;
}

std::string AiCoreOpTask::GetKeyForTbeKernel() const {
  return OP_EXTATTR_NAME_TBE_KERNEL;
}

std::string AiCoreOpTask::GetKeyForTvmMagic() const {
  return TVM_ATTR_NAME_MAGIC;
}

std::string AiCoreOpTask::GetKeyForTvmMetaData() const {
  return TVM_ATTR_NAME_METADATA;
}

std::string AiCoreOpTask::GetKeyForKernelName(const OpDesc &op_desc) const {
  return op_desc.GetName() + "_kernelname";
}

Status AtomicAddrCleanOpTask::Init(const OpDesc &op_desc, const domi::TaskDef &task_def) {
  GE_CHK_STATUS_RET_NOLOG(AiCoreOpTask::Init(op_desc, task_def));
  return InitAtomicAddrCleanIndices(op_desc);
}

Status AtomicAddrCleanOpTask::InitAtomicAddrCleanIndices(const OpDesc &op_desc) {
  GELOGD("[%s] Start to setup AtomicAddrClean task.", op_desc.GetName().c_str());
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  map<string, map<int64_t, int64_t>> workspace_info; // op_name, ws_index, ws_offset
  workspace_info = op_desc.TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  if (atomic_output_indices.empty() && workspace_info.empty()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size][%s] ATOMIC_ATTR_OUTPUT_INDEX and EXT_ATTR_ATOMIC_WORKSPACE_INFO is empty. check invalid",
           op_desc.GetName().c_str());
    REPORT_INNER_ERROR("E19999", "[%s] ATOMIC_ATTR_OUTPUT_INDEX and EXT_ATTR_ATOMIC_WORKSPACE_INFO"
                       "is empty. check invalid", op_desc.GetName().c_str());
    return INTERNAL_ERROR;
  }

  for (auto output_index : atomic_output_indices) {
    GELOGD("[%s] Adding output index [%ld]", op_desc.GetName().c_str(), output_index);
    GE_CHECK_GE(output_index, 0);
    GE_CHECK_LE(output_index, INT32_MAX);
    atomic_output_indices_.emplace_back(static_cast<int>(output_index));
  }

  for (auto &iter : workspace_info) {
    for (auto &info_iter : iter.second) {
      auto workspace_index = info_iter.first;
      GELOGD("[%s] Adding workspace index [%ld]", op_desc.GetName().c_str(), workspace_index);
      GE_CHECK_GE(workspace_index, 0);
      GE_CHECK_LE(workspace_index, INT32_MAX);
      atomic_workspace_indices_.emplace_back(static_cast<int>(workspace_index));
    }
  }

  size_t arg_count = atomic_workspace_indices_.size() + atomic_output_indices_.size();
  if (tiling_buffer_ != nullptr) {
    arg_count += 1;
  }

  if (arg_count > max_arg_count_) {
    GELOGE(INTERNAL_ERROR, "[Check][arg_count][%s] Invalid arg memory, max arg count = %u,"
           "but expect = %zu", GetName().c_str(), max_arg_count_, arg_count);
    REPORT_INNER_ERROR("E19999", "[%s] Invalid arg memory, max arg count = %u, but expect = %zu",
                       GetName().c_str(), max_arg_count_, arg_count);
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

std::string AtomicAddrCleanOpTask::GetKeyForOpParamSize() const {
  return kAttrAtomicOpParamSize;
}

std::string AtomicAddrCleanOpTask::GetKeyForTbeKernel() const {
  return EXT_ATTR_ATOMIC_TBE_KERNEL;
}

std::string AtomicAddrCleanOpTask::GetKeyForTvmMagic() const {
  return ATOMIC_ATTR_TVM_MAGIC;
}

std::string AtomicAddrCleanOpTask::GetKeyForTvmMetaData() const {
  return ATOMIC_ATTR_TVM_METADATA;
}

std::string AtomicAddrCleanOpTask::GetKeyForKernelName(const OpDesc &op_desc) const {
  return op_desc.GetName() + "_atomic_kernelname";
}

const std::string &AtomicAddrCleanOpTask::GetOpType() const {
  return kAtomicOpType;
}

Status AtomicAddrCleanOpTask::CalcTilingInfo(const NodePtr &node, OpRunInfo &tiling_info) {
  GELOGD("[%s] Start to invoke OpAtomicCalculate.", node->GetName().c_str());
  GE_CHK_STATUS_RET(optiling::OpAtomicCalculateV2(*node, tiling_info),
                    "[Invoke][OpAtomicCalculate]Failed calc tiling data of node %s.",
                    node->GetName().c_str());
  GELOGD("[%s] Done invoking OpAtomicCalculate successfully.", node->GetName().c_str());
  return SUCCESS;
}

Status AtomicAddrCleanOpTask::UpdateArgs(TaskContext &task_context) {
  // refresh atomic output addr
  int index = 0;
  for (auto atomic_output_index : atomic_output_indices_) {
    const auto output_tensor = task_context.GetOutput(atomic_output_index);
    GE_CHECK_NOTNULL(output_tensor);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(output_tensor->GetData());
  }

  // refresh atomic workspace addr
  for (auto atomic_ws_index : atomic_workspace_indices_) {
    const auto workspace_tensor = task_context.GetOutput(atomic_ws_index);
    GE_CHECK_NOTNULL(workspace_tensor);
    arg_base_[index++] = reinterpret_cast<uintptr_t>(workspace_tensor->GetData());
  }

  if (tiling_buffer_ != nullptr) {
    arg_base_[index++] = reinterpret_cast<uintptr_t>(tiling_buffer_->GetData());
  } else {
    GELOGD("[%s] Not a dynamic op", GetName().c_str());
  }

  if (task_context.IsTraceEnabled()) {
    for (int i = 0; i < index; ++i) {
      GELOGD("[%s] Arg[%d] = %lu", GetName().c_str(), i, arg_base_[i]);
    }
  }

  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge
