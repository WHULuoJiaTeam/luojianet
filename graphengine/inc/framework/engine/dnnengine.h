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

#ifndef INC_FRAMEWORK_ENGINE_DNNENGINE_H_
#define INC_FRAMEWORK_ENGINE_DNNENGINE_H_

#include <map>
#include <string>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "graph/types.h"

namespace ge {
enum PriorityEnum {
  COST_0 = 0,
  COST_1,
  COST_2,
  COST_3,
  COST_9 = 9,
  COST_10 = 10,
};

struct DNNEngineAttribute {
  std::string engine_name;
  std::vector<std::string> mem_type;
  uint32_t compute_cost;
  enum RuntimeType runtime_type;  // HOST, DEVICE
  // If engine input format must be specific, set this attribute, else set FORMAT_RESERVED
  Format engine_input_format;
  Format engine_output_format;
  bool atomic_engine_flag;
};

class GE_FUNC_VISIBILITY DNNEngine {
 public:
  DNNEngine() = default;
  explicit DNNEngine(const DNNEngineAttribute &attrs) {
    engine_attribute_ = attrs;
  }
  virtual ~DNNEngine() = default;
  Status Initialize(const std::map<std::string, std::string> &options) {
    return SUCCESS;
  }
  Status Finalize() {
    return SUCCESS;
  }
  void GetAttributes(DNNEngineAttribute &attr) const {
    attr = engine_attribute_;
  }
  bool IsAtomic() const {
    return engine_attribute_.atomic_engine_flag;
  }

 protected:
  DNNEngineAttribute engine_attribute_;
};
}  // namespace ge

#endif  // INC_FRAMEWORK_ENGINE_DNNENGINE_H_
