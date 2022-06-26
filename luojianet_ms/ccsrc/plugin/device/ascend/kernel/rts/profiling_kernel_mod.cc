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

#include "plugin/device/ascend/kernel/rts/profiling_kernel_mod.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ge_runtime/task_info.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/hal/device/executor/rts/profiling_rts_dynamic_kernel.h"

using ProfilerTraceTaskInfo = luojianet_ms::ge::model_runner::ProfilerTraceTaskInfo;
using luojianet_ms::device::ascend::ProfilingRtsDynamicKernel;
using luojianet_ms::device::ascend::ProfilingUtils;

namespace luojianet_ms {
namespace kernel {
bool ProfilingKernelMod::Init(const AnfNodePtr &anf_node) {
  MS_LOG(INFO) << "[profiling] init profiling kernel mod";
  auto primitive = common::AnfAlgo::GetCNodePrimitive(anf_node);

  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr notify_ptr = primitive->GetAttr(ProfilingUtils::kNotify);
  MS_EXCEPTION_IF_NULL(notify_ptr);

  ValuePtr log_id_ptr = primitive->GetAttr(ProfilingUtils::kProfilerTraceId);
  MS_EXCEPTION_IF_NULL(log_id_ptr);

  ValuePtr flags_ptr = primitive->GetAttr(ProfilingUtils::kFlags);
  MS_EXCEPTION_IF_NULL(flags_ptr);

  notify_ = GetValue<bool>(notify_ptr);
  log_id_ = GetValue<uint64_t>(log_id_ptr);
  flags_ = GetValue<uint32_t>(flags_ptr);
  MS_LOG(INFO) << "[profiling] profiling kernel notify_:" << notify_ << ", log_id_:" << log_id_
               << ", flags_:" << flags_;
  return true;
}

bool ProfilingKernelMod::Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                                const std::vector<AddressPtr> &, void *) {
  return true;
}

std::vector<TaskInfoPtr> ProfilingKernelMod::GenTask(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs, uint32_t stream_id) {
  MS_LOG(INFO) << "gen task inputs size:" << inputs.size() << ", workspace size:" << workspace.size()
               << ", outputs size:" << outputs.size();
  stream_id_ = stream_id;
  std::shared_ptr<ProfilerTraceTaskInfo> task_info_ptr =
    std::make_shared<ProfilerTraceTaskInfo>(unique_name_, stream_id, log_id_, notify_, flags_);
  return {task_info_ptr};
}

device::DynamicKernelPtr ProfilingKernelMod::GenDynamicKernel(const CNodePtr &cnode_ptr, void *stream_ptr) {
  return std::make_shared<ProfilingRtsDynamicKernel>(stream_ptr, cnode_ptr, log_id_, notify_, flags_);
}
}  // namespace kernel
}  // namespace luojianet_ms
