/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_mod.h"

#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "runtime/mem.h"
#include "runtime/rt.h"
#include "include/common/utils/convert_utils.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_util.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/device/executor/ai_cpu_dynamic_kernel.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/device/executor/host_dynamic_kernel.h"

using AicpuTaskInfoPtr = std::shared_ptr<mindspore::ge::model_runner::AicpuTaskInfo>;
using AicpuDynamicKernel = mindspore::device::ascend::AiCpuDynamicKernel;
using HostDynamicKernel = mindspore::device::ascend::HostDynamicKernel;

namespace mindspore {
namespace kernel {
AicpuOpKernelMod::AicpuOpKernelMod() {}

AicpuOpKernelMod::~AicpuOpKernelMod() {
  args_.clear();
  input_list_.clear();
  output_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
  ext_info_.clear();
}

void AicpuOpKernelMod::SetInputList(const std::vector<int64_t> &input_list) { input_list_ = input_list; }
void AicpuOpKernelMod::SetOutputList(const std::vector<int64_t> &output_list) { output_list_ = output_list; }
void AicpuOpKernelMod::SetNodeDef(const std::string &node_def) { (void)node_def_str_.assign(node_def); }
void AicpuOpKernelMod::SetExtInfo(const std::string &ext_info) { ext_info_ = ext_info; }
void AicpuOpKernelMod::SetNodeName(const std::string &node_name) { node_name_ = node_name; }
void AicpuOpKernelMod::SetCustSo(const std::string &cust_so) {
  node_so_ = cust_so;
  cust_kernel_ = true;
}
void AicpuOpKernelMod::SetAnfNode(const mindspore::AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  anf_node_ = anf_node;
}

void AicpuOpKernelMod::CreateCpuKernelInfo(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &outputs) {
  MS_LOG(INFO) << "CreateCpuKernelInfoOffline start";

  if (!cust_kernel_) {
    if (kCpuKernelOps.find(node_name_) != kCpuKernelOps.end()) {
      node_so_ = kLibCpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else if (kCacheKernelOps.find(node_name_) != kCacheKernelOps.end()) {
      node_so_ = kLibAicpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else {
      if (node_so_ != kLibCpuKernelSoName) {
        node_so_ = kLibAicpuKernelSoName;
      }
    }
  } else if (kCpuKernelBaseOps.find(node_name_) == kCpuKernelBaseOps.end()) {
    node_name_ = kCpuRunApi;
  }

  if (node_name_ == kTopK) {
    node_name_ = kTopKV2;
  }

  if (node_name_ == kStack) {
    node_name_ = kPack;
  }

  // InputOutputAddr
  vector<void *> io_addrs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(io_addrs),
                       [](const AddressPtr &input) -> void * { return input->addr; });
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(io_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  auto io_addrs_num = io_addrs.size();
  // calculate paramLen: AicpuParamHead.len + ioAddrsSize + notifyId.len + customizedAttr.len
  auto param_len = sizeof(AicpuParamHead);

  // get input and output addrs size, no need to check overflow
  auto io_addrs_size = io_addrs_num * sizeof(uint64_t);
  // refresh paramLen, no need to check overflow
  param_len += io_addrs_size;

  auto node_def_len = node_def_str_.length();
  param_len += node_def_len;
  param_len += sizeof(uint32_t);

  AicpuParamHead aicpu_param_head{};
  aicpu_param_head.length = param_len;
  aicpu_param_head.ioAddrNum = io_addrs_num;

  if (ext_info_.empty()) {
    MS_LOG(INFO) << "Static Shape Kernel";
    aicpu_param_head.extInfoLength = 0;
    aicpu_param_head.extInfoAddr = 0;
  } else {
    MS_LOG(INFO) << "Dynamic Kernel Ext Info size:" << ext_info_.size();
    aicpu_param_head.extInfoLength = SizeToUint(ext_info_.size());
    aicpu_param_head.extInfoAddr = reinterpret_cast<uint64_t>(ext_info_addr_dev_);
  }

  args_.clear();
  (void)args_.append(reinterpret_cast<const char *>(&aicpu_param_head), sizeof(AicpuParamHead));
  // TaskArgs append ioAddrs
  if (io_addrs_size != 0) {
    (void)args_.append(reinterpret_cast<const char *>(io_addrs.data()), io_addrs_size);
  }

  // size for node_def
  args_.append(reinterpret_cast<const char *>(&node_def_len), sizeof(uint32_t));

  // When it's aicpu customized ops, taskArgs should append customized attr
  if (node_def_len != 0) {
    (void)args_.append(reinterpret_cast<const char *>(node_def_str_.data()), node_def_len);
  }

  MS_LOG(INFO) << "CreateCpuKernelInfoOffline end";
}

bool AicpuOpKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  if (stream_ptr == nullptr) {
    MS_LOG(ERROR) << "stream_ptr should not be nullptr.";
    return false;
  }
  if (stream_ == nullptr) {
    stream_ = stream_ptr;
  }
  CreateCpuKernelInfo(inputs, outputs);
  if (node_name_ == kTopK) {
    node_name_ = kTopKV2;
  }
  if (node_name_ == kStack) {
    node_name_ = kPack;
  }
  auto flag = RT_KERNEL_DEFAULT;
  if (cust_kernel_) {
    flag = RT_KERNEL_CUSTOM_AICPU;
  }
  MS_LOG(INFO) << "Aicpu launch, node_so_:" << node_so_ << ", node name:" << node_name_
               << ", args_size:" << args_.length();
  // cppcheck-suppress unreadVariable
  auto lock = device::KernelRuntime::LockRuntime();
  rtArgsEx_t argsInfo = {};
  argsInfo.args = args_.data();
  argsInfo.argsSize = static_cast<uint32_t>(args_.length());
  if (rtCpuKernelLaunchWithFlagV2(reinterpret_cast<const void *>(node_so_.c_str()),
                                  reinterpret_cast<const void *>(node_name_.c_str()), 1, &argsInfo, nullptr, stream_,
                                  flag) != RT_ERROR_NONE) {
    MS_LOG(ERROR) << "Aicpu op launch failed!";

    return false;
  }
  return true;
}

std::vector<TaskInfoPtr> AicpuOpKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &,
                                                   const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  MS_LOG(INFO) << "AicpuOpKernelMod GenTask start";

  stream_id_ = stream_id;
  if (!cust_kernel_) {
    if (kCpuKernelOps.find(node_name_) != kCpuKernelOps.end()) {
      node_so_ = kLibCpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else if (kCacheKernelOps.find(node_name_) != kCacheKernelOps.end()) {
      node_so_ = kLibAicpuKernelSoName;
      node_name_ = kCpuRunApi;
    } else {
      if (node_so_ != kLibCpuKernelSoName) {
        node_so_ = kLibAicpuKernelSoName;
      }
    }
  } else {
    if (kCpuKernelBaseOps.find(node_name_) == kCpuKernelBaseOps.end()) {
      node_name_ = kCpuRunApi;
    }
  }
  std::vector<void *> input_data_addrs;
  (void)std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(input_data_addrs),
                       [](const AddressPtr &input) -> void * { return input->addr; });

  std::vector<void *> output_data_addrs;
  (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_data_addrs),
                       [](const AddressPtr &output) -> void * { return output->addr; });

  if (node_name_ == kTopK) {
    node_name_ = kTopKV2;
  }

  if (node_name_ == kStack) {
    node_name_ = kPack;
  }

  AicpuTaskInfoPtr task_info_ptr = std::make_shared<mindspore::ge::model_runner::AicpuTaskInfo>(
    unique_name_, stream_id, node_so_, node_name_, node_def_str_, ext_info_, input_data_addrs, output_data_addrs,
    NeedDump(), cust_kernel_);

  MS_LOG(INFO) << "AicpuOpKernelMod GenTask end";
  return {task_info_ptr};
}

device::DynamicKernelPtr AicpuOpKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) {
  KernelLaunchInfo kernel_launch_info;
  device::KernelRuntime::GenLaunchArgs(*this, cnode_ptr, &kernel_launch_info);

  CreateCpuKernelInfo(kernel_launch_info.inputs_, kernel_launch_info.outputs_);
  return std::make_shared<AicpuDynamicKernel>(stream_ptr, cnode_ptr, args_, ext_info_, node_so_, node_name_,
                                              cust_kernel_);
}
}  // namespace kernel
}  // namespace mindspore
