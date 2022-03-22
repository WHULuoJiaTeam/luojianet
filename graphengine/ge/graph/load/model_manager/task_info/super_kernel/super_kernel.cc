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

#include "graph/load/model_manager/task_info/super_kernel/super_kernel.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace skt {
Status SuperKernel::Launch(rtStream_t stream, uint32_t dump_flag) {
  const void *func_stub_ = this->GetFuncStub();

  const void *args[] = {this->GetNavTablePtr(),
                        reinterpret_cast<const void *>(static_cast<uintptr_t>(this->GetNavTableSize()))};

  rtError_t rt_ret = rtMalloc(reinterpret_cast<void **>(&device_args_addr_), sizeof(args), RT_MEMORY_HBM);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%lu, ret:0x%X", sizeof(args), rt_ret);
                  GELOGE(RT_FAILED, "[Call][RtMalloc] failied, size:%lu, ret:0x%X", sizeof(args), rt_ret);
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)
  rt_ret = rtMemcpy(reinterpret_cast<void *>(device_args_addr_), sizeof(args), reinterpret_cast<void *>(args),
                    sizeof(args), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%lu, ret:0x%X", sizeof(args), rt_ret);
                  GELOGE(RT_FAILED, "[Call][RtMemcpy] failied, size:%lu, ret:0x%X", sizeof(args), rt_ret);
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)
  rt_ret = rtKernelLaunchWithFlag((void *const)func_stub_, block_dim_, device_args_addr_, sizeof(args), NULL, stream,
                                  dump_flag);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchWithFlag failed, dump_flag:%u, ret:0x%X",
                                    dump_flag, rt_ret);
                  GELOGE(RT_FAILED, "[Call][RtKernelLaunchWithFlag] failied. error: 0x%X", rt_ret);
                  return RT_ERROR_TO_GE_STATUS(rt_ret);)
  return SUCCESS;
}
}  // namespace skt
}  // namespace ge
