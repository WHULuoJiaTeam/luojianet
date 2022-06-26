/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
#include <memory>
#include <string>
#include "common/graph_kernel/core/graph_kernel_splitter.h"

namespace mindspore::graphkernel {
class GraphKernelSplitterWithPy : public GraphKernelSplitter {
 public:
  GraphKernelSplitterWithPy() = default;
  ~GraphKernelSplitterWithPy() = default;
  std::shared_ptr<SplitSchemer> GetSplitSchema(const std::string &processor) override;
};
using GraphKernelSplitterWithPyPtr = std::shared_ptr<GraphKernelSplitterWithPy>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_ADAPTER_GRAPH_KERNEL_SPLITTER_WITH_PY_H_
