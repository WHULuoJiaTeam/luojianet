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

#ifndef GE_COMMON_DUMP_DUMP_OP_H_
#define GE_COMMON_DUMP_DUMP_OP_H_

#include <string>

#include "framework/common/ge_inner_error_codes.h"
#include "common/properties_manager.h"
#include "proto/op_mapping.pb.h"
#include "runtime/stream.h"

namespace ge {
class DumpOp {
 public:
  DumpOp() = default;
  ~DumpOp();

  void SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc, vector<uintptr_t> input_addrs,
                   vector<uintptr_t> output_addrs, rtStream_t stream);
  Status LaunchDumpOp();
  void SetLoopAddr(void *global_step, void *loop_per_iter, void *loop_cond);
  void SetDynamicModelInfo(const string &dynamic_model_name, const string &dynamic_om_name, uint32_t dynamic_model_id);

 private:
  Status ExecutorDumpOp(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);
  Status DumpOutput(toolkit::aicpu::dump::Task &task);
  Status DumpInput(toolkit::aicpu::dump::Task &task);
  Status SetDumpModelName(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info);

  DumpProperties dump_properties_;
  OpDescPtr op_desc_;
  std::vector<uintptr_t> input_addrs_;
  std::vector<uintptr_t> output_addrs_;

  void *proto_dev_mem_ = nullptr;
  void *proto_size_dev_mem_ = nullptr;
  rtStream_t stream_;
  uintptr_t global_step_;
  uintptr_t loop_per_iter_;
  uintptr_t loop_cond_;

  std::string dynamic_model_name_;
  std::string dynamic_om_name_;
  std::uint32_t dynamic_model_id_;
};
}  // namespace ge

#endif  // GE_COMMON_DUMP_DUMP_OP_H_
