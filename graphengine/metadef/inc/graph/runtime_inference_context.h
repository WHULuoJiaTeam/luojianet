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

#ifndef INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
#define INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_

#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include "external/graph/ge_error_codes.h"
#include "external/graph/tensor.h"
#include "ge_attr_value.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY RuntimeInferenceContext {
 public:
  graphStatus SetTensor(int64_t node_id, int32_t output_id, GeTensorPtr tensor);
  graphStatus GetTensor(int64_t node_id, int32_t output_id, GeTensorPtr &tensor);
  void Release();

 private:
  std::map<int64_t, std::vector<GeTensorPtr>> ge_tensors_;
  std::mutex mu_;
};
} // namespace ge

#endif // INC_GRAPH_RUNTIME_INFERENCE_CONTEXT_H_
