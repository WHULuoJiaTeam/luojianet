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

#ifndef GE_COMMON_DUMP_OPDEBUG_REGISTER_H_
#define GE_COMMON_DUMP_OPDEBUG_REGISTER_H_

#include <map>
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "graph/load/model_manager/data_dumper.h"

namespace ge {
class OpdebugRegister {
 public:
  OpdebugRegister() = default;
  ~OpdebugRegister();

  Status RegisterDebugForModel(rtModel_t model_handle, uint32_t op_debug_mode, DataDumper &data_dumper);
  void UnregisterDebugForModel(rtModel_t model_handle);

  Status RegisterDebugForStream(rtStream_t stream, uint32_t op_debug_mode, DataDumper &data_dumper);
  void UnregisterDebugForStream(rtStream_t stream);

 private:
  Status MallocMemForOpdebug();

  void *op_debug_addr_ = nullptr;
  void *p2p_debug_addr_ = nullptr;
};
}  // namespace ge
#endif  // GE_COMMON_DUMP_OPDEBUG_REGISTER_H_
