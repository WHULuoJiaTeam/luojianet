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

#ifndef LUOJIANET_MS_LUOJIANET_MS_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_
#define LUOJIANET_MS_LUOJIANET_MS_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_

#include <string>
#include <vector>
#include <memory>
#include "backend/session/kernel_graph.h"

namespace luojianet_ms {
namespace session {
class SingleKernelGraph {
 public:
  SingleKernelGraph() = default;
  ~SingleKernelGraph() = default;

  static std::shared_ptr<session::KernelGraph> ConstructKernelGraphBasedOnSingleOp(
    const std::string &op_name, const std::vector<TypeId> &input_dtypes, const std::vector<ShapeVector> &input_shapes,
    const std::vector<TypeId> &output_dtypes, const std::vector<std::vector<size_t>> &output_shapes);
};
}  // namespace session
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_LUOJIANET_MS_CCSRC_BACKEND_SESSION_SINGLE_KERNEL_GRAPH_H_
