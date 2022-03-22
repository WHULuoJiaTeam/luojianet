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

#ifndef INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_
#define INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_

#include <vector>

#include "graph/node.h"
#include "register/op_tiling_registry.h"
#include "runtime/rt_ffts_plus.h"
#include "external/ge/ge_api_error_codes.h"

namespace ge {
struct AutoThreadSubTaskFlush {
  int32_t device_id{0};
  void *args_base{nullptr};
  std::vector<optiling::utils::OpRunInfo> op_run_info;

  uintptr_t aic_non_tail_task_start_pc{0U};
  uintptr_t aic_tail_task_start_pc{0U};
  uint32_t aic_icache_prefetch_cnt{0U};

  uintptr_t aiv_non_tail_task_start_pc{0U};
  uintptr_t aiv_tail_task_start_pc{0U};
  uint32_t aiv_icache_prefetch_cnt{0U};

  // AICPU task Addrs.
  std::vector<uintptr_t> input_addr_base;
  std::vector<uintptr_t> output_addr_base;
  void *extinfo_base{nullptr};
};

struct AutoThreadParam {
  uint16_t thread_dim{0U};  // thread dim after Pre-Thread
  uint32_t input_output_num{0U};  // input + output
  std::vector<uint64_t> task_addr_offset; // input + output + workspace

  // AICPU task Params.
  uint32_t args_size{0U}; // size for args_base
  uint32_t extinfo_size{0U}; // size for extinfo_base
};

class FFTSPlusTaskUpdate {
 public:
  FFTSPlusTaskUpdate() = default;
  virtual ~FFTSPlusTaskUpdate() = default;

  virtual Status GetAutoThreadParam(const NodePtr &node, const std::vector<optiling::utils::OpRunInfo> &op_run_info,
                                    AutoThreadParam &auto_thread_param) {
    return SUCCESS;
  }

  virtual Status UpdateSubTaskAndCache(const NodePtr &node, const AutoThreadSubTaskFlush &sub_task_flush,
                                       rtFftsPlusTaskInfo_t &ffts_plus_task_info) {
    return SUCCESS;
  }
};
} // namespace ge
#endif // INC_REGISTER_FFTS_PLUS_TASK_UPDATE_H_
