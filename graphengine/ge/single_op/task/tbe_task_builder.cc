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

#include "single_op/task/tbe_task_builder.h"

#include <mutex>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "runtime/rt.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace {
constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
constexpr char const *kAttrOpParamSize = "op_para_size";
constexpr char const *kAttrAtomicOpParamSize = "atomic_op_para_size";
std::mutex g_reg_mutex;
}  // namespace

KernelHolder::KernelHolder(const char *stub_func, std::shared_ptr<ge::OpKernelBin> kernel_bin)
    : stub_func_(stub_func), bin_handle_(nullptr), kernel_bin_(std::move(kernel_bin)) {}

KernelHolder::~KernelHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

HandleHolder::HandleHolder(void *bin_handle)
    : bin_handle_(bin_handle) {}

HandleHolder::~HandleHolder() {
  if (bin_handle_ != nullptr) {
    GE_CHK_RT(rtDevBinaryUnRegister(bin_handle_));
  }
}

const char *KernelBinRegistry::GetUnique(const string &stub_func) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_func);
  if (it != unique_stubs_.end()) {
    return it->c_str();
  } else {
    it = unique_stubs_.insert(unique_stubs_.end(), stub_func);
    return it->c_str();
  }
}

const char *KernelBinRegistry::GetStubFunc(const std::string &stub_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = registered_bins_.find(stub_name);
  if (iter != registered_bins_.end()) {
    return iter->second->stub_func_;
  }

  return nullptr;
}

bool KernelBinRegistry::AddKernel(const std::string &stub_name, std::unique_ptr<KernelHolder> &&holder) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ret = registered_bins_.emplace(stub_name, std::move(holder));
  return ret.second;
}

bool HandleRegistry::AddHandle(std::unique_ptr<HandleHolder> &&holder) {
  auto ret = registered_handles_.emplace(std::move(holder));
  return ret.second;
}

TbeTaskBuilder::TbeTaskBuilder(const std::string &model_name, const NodePtr &node, const domi::TaskDef &task_def)
    : node_(node),
      op_desc_(node->GetOpDesc()),
      task_def_(task_def),
      kernel_def_(task_def.kernel()),
      kernel_def_with_handle_(task_def.kernel_with_handle()),
      model_name_(model_name) {}

TBEKernelPtr TbeTaskBuilder::GetTbeKernel(const OpDescPtr &op_desc) const {
  return op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
}

void TbeTaskBuilder::GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const {
  (void)AttrUtils::GetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
}

Status TbeTaskBuilder::DoRegisterBinary(const OpKernelBin &kernel_bin, void **bin_handle,
                                        const SingleOpModelParam &param) const {
  rtDevBinary_t binary;
  binary.version = 0;
  binary.data = kernel_bin.GetBinData();
  binary.length = kernel_bin.GetBinDataSize();
  GE_CHK_STATUS_RET_NOLOG(GetMagic(binary.magic));
  Status ret = 0;
  if (task_def_.type() == RT_MODEL_TASK_ALL_KERNEL) {
    ret = rtRegisterAllKernel(&binary, bin_handle);
  } else {
    ret = rtDevBinaryRegister(&binary, bin_handle);
  }
  if (ret != RT_ERROR_NONE) {
    GELOGE(ret, "[DoRegister][Binary] failed, bin key = %s, core_type = %ld, rt ret = %d", stub_name_.c_str(),
        param.core_type, static_cast<int>(ret));
    REPORT_CALL_ERROR("E19999", "DoRegisterBinary failed, bin key = %s, core_type = %ld, rt ret = %d", 
        stub_name_.c_str(), param.core_type, static_cast<int>(ret));
    return ret;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterMeta(void *bin_handle) {
  std::string meta_data;
  (void)AttrUtils::GetStr(op_desc_, GetKeyForTvmMetaData(), meta_data);
  GELOGI("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
  if (!meta_data.empty()) {
    auto rt_ret = rtMetadataRegister(bin_handle, meta_data.c_str());
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "[Invoke][rtMetadataRegister] failed. bin key = %s, meta_data = %s, rt ret = %d", 
          stub_name_.c_str(), meta_data.c_str(), static_cast<int>(rt_ret));
      REPORT_CALL_ERROR("E19999", "rtMetadataRegister failed, bin key = %s, meta_data = %s, rt ret = %d", 
          stub_name_.c_str(), meta_data.c_str(), static_cast<int>(rt_ret));
      return rt_ret;
    }
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterFunction(void *bin_handle, const char *stub_name, const char *kernel_name) {
  auto rt_ret = rtFunctionRegister(bin_handle, stub_name, stub_name, kernel_name, FUNC_MODE_NORMAL);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Invoke][rtFunctionRegister] failed. bin key = %s, kernel name = %s, rt ret = %d", 
        stub_name, kernel_name, static_cast<int>(rt_ret));
    REPORT_CALL_ERROR("E19999", "rtFunctionRegister failed. bin key = %s, kernel name = %s, rt ret = %d", 
        stub_name, kernel_name, static_cast<int>(rt_ret));
    return rt_ret;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::DoRegisterKernel(const ge::OpKernelBin &tbe_kernel, const char *bin_file_key, void **bin_handle,
                                        const SingleOpModelParam &param) {
  void *handle = nullptr;
  auto ret = DoRegisterBinary(tbe_kernel, &handle, param);
  if (ret != SUCCESS) {
    return ret;
  }
  if (task_def_.type() == RT_MODEL_TASK_ALL_KERNEL) {
    *bin_handle = handle;
    return SUCCESS;
  }

  ret = DoRegisterMeta(handle);
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  std::string kernel_name;
  GetKernelName(op_desc_, kernel_name);
  ret = DoRegisterFunction(handle, bin_file_key, kernel_name.c_str());
  if (ret != SUCCESS) {
    GE_CHK_RT(rtDevBinaryUnRegister(handle));
    return ret;
  }

  GELOGI("Register function succeeded: kernel_name = %s", kernel_name.c_str());
  *bin_handle = handle;
  return SUCCESS;
}

Status TbeTaskBuilder::RegisterKernel(TbeOpTask &task, const SingleOpModelParam &param) {
  KernelBinRegistry &registry = KernelBinRegistry::GetInstance();
  // check if already registered
  const char *stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func != nullptr) {
    task.SetStubFunc(stub_name_, stub_func);
    return SUCCESS;
  }

  // to avoid repeat register
  std::lock_guard<std::mutex> lock(g_reg_mutex);
  // check again
  stub_func = registry.GetStubFunc(stub_name_);
  if (stub_func == nullptr) {
    stub_func = registry.GetUnique(stub_name_);
    GELOGI("RegisterKernel begin, stub_func = %s", stub_func);

    auto tbe_kernel = GetTbeKernel(op_desc_);
    if (tbe_kernel == nullptr) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TbeKernel] fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
          op_desc_->GetName().c_str());
      REPORT_CALL_ERROR("E19999", "GetTbeKernel fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
          op_desc_->GetName().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    auto holder = std::unique_ptr<KernelHolder>(new (std::nothrow) KernelHolder(stub_func, tbe_kernel));
    if (holder == nullptr) {
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][KernelHodler] failed.");
      REPORT_INNER_ERROR("E19999", "Create KernelHodler failed.");
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }

    void *bin_handle = nullptr;
    auto ret = DoRegisterKernel(*tbe_kernel, stub_func, &bin_handle, param);
    if (ret != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Register][Kernel] failed. stub name = %s", stub_name_.c_str());
      REPORT_CALL_ERROR("E19999", "DoRegisterKernel failed, stub name = %s", stub_name_.c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    holder->SetBinHandle(bin_handle);
    if (!registry.AddKernel(stub_name_, std::move(holder))) {
      // should not happen. only one thread can reach here
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Add][Kernel] failed. stub name = %s", stub_name_.c_str());
      REPORT_CALL_ERROR("E19999", "AddKernel failed. stub name = %s", stub_name_.c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
  }

  task.SetStubFunc(stub_name_, stub_func);
  return SUCCESS;
}

Status TbeTaskBuilder::RegisterKernelWithHandle(TbeOpTask &task, const SingleOpModelParam &param) {
  GELOGD("RegisterKernelWithHandle begin.");
  HandleRegistry &registry = HandleRegistry::GetInstance();
  auto tbe_kernel = GetTbeKernel(op_desc_);
  if (tbe_kernel == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TbeKernel] fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
        op_desc_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "GetTbeKernel fail for OP EXT ATTR NAME TBE_KERNEL not found. op = %s",
        op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  void *bin_handle = nullptr;
  auto ret = DoRegisterKernel(*tbe_kernel, nullptr, &bin_handle, param);
  if (ret != SUCCESS) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Register][Kernel] failed. node name = %s", op_desc_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "DoRegisterKernel failed, node name = %s", op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  handle_ = bin_handle;
  auto holder = std::unique_ptr<HandleHolder>(new (std::nothrow) HandleHolder(handle_));
  if (holder == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][HandleHolder] failed.");
    REPORT_INNER_ERROR("E19999", "Create HandleHolder failed.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  if (!registry.AddHandle(std::move(holder))) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Add][Handle] failed. node name = %s", op_desc_->GetName().c_str());
    REPORT_CALL_ERROR("E19999", "AddHandle failed, node name = %s", op_desc_->GetName().c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status TbeTaskBuilder::GetSmDesc(void **sm_desc, const SingleOpModelParam &param) const {
  const std::string &sm_desc_str = kernel_def_.sm_desc();
  if (sm_desc_str.empty()) {
    *sm_desc = nullptr;
  } else {
    GELOGD("To process sm desc, size = %zu", sm_desc_str.size());
    char *sm_control = const_cast<char *>(sm_desc_str.data());
    auto *l2_ctrl_info = reinterpret_cast<rtL2Ctrl_t *>(sm_control);
    uint64_t gen_base_addr = param.base_addr;
    // There is no weight for te op now. Update L2_mirror_addr by data memory base.
    uint64_t data_base_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(param.mem_base)) - gen_base_addr;
    for (auto &data_index : l2_ctrl_info->data) {
      if (data_index.L2_mirror_addr != 0) {
        data_index.L2_mirror_addr += data_base_addr;
      }
    }

    auto rt_ret = rtMemAllocManaged(sm_desc, sm_desc_str.size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(rt_ret, "[Invoke][rtMemAllocManaged] failed, ret: %d.", static_cast<int>(rt_ret));
      REPORT_CALL_ERROR("E19999", "rtMemAllocManaged failed, ret: %d.", static_cast<int>(rt_ret));
      return rt_ret;
    }

    rt_ret = rtMemcpy(*sm_desc, sm_desc_str.size(), sm_desc_str.data(), sm_desc_str.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      (void)rtMemFreeManaged(*sm_desc);
      GELOGE(rt_ret, "[Update][Param:sm_desc] fail for rtMemcpy return: %d.", static_cast<int>(rt_ret));
      REPORT_INNER_ERROR("E19999", "rtMemcpy failed, ret:%d.", static_cast<int>(rt_ret));
      return rt_ret;
    }
  }

  return SUCCESS;
}

Status TbeTaskBuilder::InitKernelArgs(void *arg_addr, size_t arg_size, const SingleOpModelParam &param) {
  // copy args
  std::vector<void *> tensor_device_addr_vec = BuildTaskUtils::GetKernelArgs(op_desc_, param);
  void *src_addr = reinterpret_cast<void *>(tensor_device_addr_vec.data());
  uint64_t src_len = sizeof(void *) * tensor_device_addr_vec.size();
  GE_CHK_RT_RET(rtMemcpy(arg_addr, arg_size, src_addr, src_len, RT_MEMCPY_HOST_TO_HOST));
  return SUCCESS;
}

Status TbeTaskBuilder::SetKernelArgs(TbeOpTask &task, const SingleOpModelParam &param, const OpDescPtr &op_desc) {
  auto task_type = static_cast<rtModelTaskType_t>(task_def_.type());
  bool is_task_all_kernel = (task_type == RT_MODEL_TASK_ALL_KERNEL);
  size_t arg_size = 0;
  std::unique_ptr<uint8_t[]> args = nullptr;
  if (is_task_all_kernel) {
    GELOGD("SetKernelArgs of %s in branch of RT_MODEL_TASK_ALL_KERNEL.", op_desc->GetName().c_str());
    arg_size = kernel_def_with_handle_.args_size();
    args = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[arg_size]);
    GE_CHECK_NOTNULL(args);
    GE_CHK_RT_RET(rtMemcpy(args.get(), arg_size, kernel_def_with_handle_.args().data(), arg_size,
                           RT_MEMCPY_HOST_TO_HOST))
  } else {
    GELOGD("SetKernelArgs of %s in branch of RT_MODEL_TASK_KERNEL.", op_desc->GetName().c_str());
    arg_size = kernel_def_.args_size();
    args = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[arg_size]);
    GE_CHECK_NOTNULL(args);
    GE_CHK_RT_RET(rtMemcpy(args.get(), arg_size, kernel_def_.args().data(), arg_size, RT_MEMCPY_HOST_TO_HOST))
  }

  const domi::KernelContext &context = task_type == RT_MODEL_TASK_ALL_KERNEL ?
                                       kernel_def_with_handle_.context() : kernel_def_.context();
  const auto *args_offset_tmp = reinterpret_cast<const uint16_t *>(context.args_offset().data());
  uint16_t offset = *args_offset_tmp;
  GE_CHK_STATUS_RET_NOLOG(InitKernelArgs(args.get() + offset, arg_size - offset, param));

  if (is_task_all_kernel) {
    task.SetKernelWithHandleArgs(std::move(args), arg_size, kernel_def_with_handle_.block_dim(), op_desc,
                                 kernel_def_with_handle_);
  } else {
    task.SetKernelArgs(std::move(args), arg_size, kernel_def_.block_dim(), op_desc);
  }

  bool is_dynamic = false;
  (void)AttrUtils::GetBool(op_desc_, kAttrSupportDynamicShape, is_dynamic);
  if (is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(InitTilingInfo(task));
    if (!param.graph_is_dynamic && task.tiling_buffer_ != nullptr) {
      GELOGD("Need to update run info when graph is static with dynamic node: %s.", op_desc->GetName().c_str());
      task.UpdateRunInfo();
      GE_CHK_RT_RET(rtMemcpy(task.tiling_buffer_, task.max_tiling_size_, task.tiling_data_.data(),
                             task.tiling_data_.size(), RT_MEMCPY_HOST_TO_DEVICE));
    }
  }
  return SUCCESS;
}

Status TbeTaskBuilder::BuildTask(TbeOpTask &task, const SingleOpModelParam &param) {
  GELOGD("Build tbe task begin");
  auto ret = SetKernelArgs(task, param, op_desc_);
  if (ret != SUCCESS) {
    return ret;
  }

  auto task_type = static_cast<rtModelTaskType_t>(task_def_.type());
  if (task_type == RT_MODEL_TASK_ALL_KERNEL) {
    stub_name_ = model_name_ + "/" + node_->GetName() + "_tvmbin";
    ret = RegisterKernelWithHandle(task, param);
  } else {
    const domi::KernelDef &kernel_def = task_def_.kernel();
    stub_name_ = model_name_ + "/" + kernel_def.stub_func() + "_tvmbin";
    ret = RegisterKernel(task, param);
  }

  task.SetHandle(handle_);
  if (ret != SUCCESS) {
    return ret;
  }

  auto task_info = BuildTaskUtils::GetTaskInfo(op_desc_);
  GELOGI("[TASK_INFO] %s %s", stub_name_.c_str(), task_info.c_str());

  if (task_type != RT_MODEL_TASK_ALL_KERNEL) {
    void *stub_func = nullptr;
    auto rt_ret = rtGetFunctionByName(stub_name_.c_str(), &stub_func);
    if (rt_ret != SUCCESS) {
      GELOGE(rt_ret, "[Get][FunctionByName] failed. stub_name:%s.", stub_name_.c_str());
      REPORT_CALL_ERROR("E19999", "rtGetFunctionByName failed, stub_name:%s.", stub_name_.c_str());
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    task.SetStubFunc(stub_name_, stub_func);
  }
  GE_CHK_STATUS_RET(task.SetArgIndex(), "[Set][ArgTable] failed.");
  task.input_num_ = op_desc_->GetInputsSize();
  task.output_num_ = op_desc_->GetOutputsSize();

  return SUCCESS;
}

Status TbeTaskBuilder::InitTilingInfo(TbeOpTask &task) {
  GELOGD("Start alloc tiling data of node %s.", op_desc_->GetName().c_str());
  int64_t max_size = -1;
  (void)AttrUtils::GetInt(op_desc_, GetKeyForOpParamSize(), max_size);
  GELOGD("Got op param size by key: %s, ret = %ld", GetKeyForOpParamSize().c_str(), max_size);
  if (max_size < 0) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Get][Int] %s Invalid op_param_size: %ld.", 
        op_desc_->GetName().c_str(), max_size);
    REPORT_CALL_ERROR("E19999", "AttrUtils::GetInt failed, %s Invalid op_param_size: %ld.", 
        op_desc_->GetName().c_str(), max_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  void *tiling_buffer = nullptr;
  if (max_size > 0) {
    GE_CHK_RT_RET(rtMalloc(&tiling_buffer, static_cast<uint64_t>(max_size), RT_MEMORY_HBM));
    GE_CHECK_NOTNULL(tiling_buffer);
    GELOGD("[%s] Done allocating tiling buffer, size=%ld.", op_desc_->GetName().c_str(), max_size);
  }

  task.EnableDynamicSupport(node_, tiling_buffer, static_cast<uint32_t>(max_size));
  return SUCCESS;
}

Status TbeTaskBuilder::GetMagic(uint32_t &magic) const {
  std::string json_string;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(op_desc_, TVM_ATTR_NAME_MAGIC, json_string),
                  GELOGD("Get original type of session_graph_id."));
  if (json_string == "RT_DEV_BINARY_MAGIC_ELF") {
    magic = RT_DEV_BINARY_MAGIC_ELF;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AIVEC") {
    magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  } else if (json_string == "RT_DEV_BINARY_MAGIC_ELF_AICUBE") {
    magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE;
  } else {
    REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), value:%s check invalid",
                       TVM_ATTR_NAME_MAGIC.c_str(), op_desc_->GetName().c_str(),
                       op_desc_->GetType().c_str(), json_string.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s in op:%s(%s), value:%s check invalid",
           TVM_ATTR_NAME_MAGIC.c_str(), op_desc_->GetName().c_str(),
           op_desc_->GetType().c_str(), json_string.c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

std::string TbeTaskBuilder::GetKeyForOpParamSize() const {
  return kAttrOpParamSize;
}

std::string TbeTaskBuilder::GetKeyForTvmMetaData() const {
  return TVM_ATTR_NAME_METADATA;
}

Status AtomicAddrCleanTaskBuilder::InitKernelArgs(void *args_addr, size_t arg_size, const SingleOpModelParam &param) {
  return SUCCESS;
}

std::string AtomicAddrCleanTaskBuilder::GetKeyForOpParamSize() const {
  return kAttrAtomicOpParamSize;
}

std::string AtomicAddrCleanTaskBuilder::GetKeyForTvmMetaData() const {
  return ATOMIC_ATTR_TVM_METADATA;
}

void AtomicAddrCleanTaskBuilder::GetKernelName(const OpDescPtr &op_desc, std::string &kernel_name) const {
  (void)AttrUtils::GetStr(op_desc, op_desc->GetName() + "_atomic_kernelname", kernel_name);
}

TBEKernelPtr AtomicAddrCleanTaskBuilder::GetTbeKernel(const OpDescPtr &op_desc)  const {
  return op_desc->TryGetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, TBEKernelPtr());
}

}  // namespace ge
