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

#include "frontend/operator/ops.h"
#include <string>
#include "pipeline/jit/parse/python_adapter.h"

namespace luojianet_ms {
// namespace to support primitive operators
namespace prim {
ValuePtr GetPythonOps(const std::string &op_name, const std::string &module_name, bool use_signature) {
  py::object obj = parse::python_adapter::GetPyFn(module_name, op_name);
  ValuePtr node = nullptr;
  bool succ = parse::ConvertData(obj, &node, use_signature);
  if (!succ) {
    MS_LOG(EXCEPTION) << "Get Python op " << op_name << " from " << module_name << " fail.";
  }
  return node;
}
}  // namespace prim
}  // namespace luojianet_ms
