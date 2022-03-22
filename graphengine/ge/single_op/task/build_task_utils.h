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

#ifndef GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_
#define GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_

#include <vector>
#include <sstream>

#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"
#include "hybrid/node_executor/task_context.h"

namespace ge {
class BuildTaskUtils {
 public:
  static constexpr int kAddressIndexOutput = 1;
  static constexpr int kAddressIndexWorkspace = 2;

  static std::vector<std::vector<void *>> GetAddresses(const OpDescPtr &op_desc,
                                                       const SingleOpModelParam &param,
                                                       bool keep_workspace = true);
  static std::vector<void *> JoinAddresses(const std::vector<std::vector<void *>> &addresses);
  static std::vector<void *> GetKernelArgs(const OpDescPtr &op_desc, const SingleOpModelParam &param);
  static std::string InnerGetTaskInfo(const OpDescPtr &op_desc,
                                      const std::vector<const void *> &input_addrs,
                                      const std::vector<const void *> &output_addrs);
  static std::string GetTaskInfo(const OpDescPtr &op_desc);
  static std::string GetTaskInfo(const OpDescPtr &op_desc,
                                 const std::vector<DataBuffer> &inputs,
                                 const std::vector<DataBuffer> &outputs);
  static std::string GetTaskInfo(const hybrid::TaskContext& task_context);
  template<typename T>
  static std::string VectorToString(const std::vector<T> &values) {
    std::stringstream ss;
    ss << '[';
    auto size = values.size();
    for (size_t i = 0; i < size; ++i) {
      ss << values[i];
      if (i != size - 1) {
        ss << ", ";
      }
    }
    ss << ']';
    return ss.str();
  }
};
}  // namespace ge
#endif  // GE_SINGLE_OP_TASK_BUILD_TASK_UTILS_H_
