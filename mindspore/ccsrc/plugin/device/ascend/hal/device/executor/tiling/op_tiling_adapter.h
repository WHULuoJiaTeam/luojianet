/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_ADAPTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_ADAPTER_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include "kernel/oplib/opinfo.h"
#include "register/op_tiling.h"
#include "external/graph/operator.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace device {
namespace tiling {
class OpTilingCalculateAdapter {
 public:
  OpTilingCalculateAdapter() = default;
  ~OpTilingCalculateAdapter() = default;

  ::ge::Operator AnfNodeToGeNodeAdapter(const CNodePtr &node, ::ge::ComputeGraphPtr *ge_graph,
                                        const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map,
                                        const std::string &op_compile_info);

 private:
  void ConvertInputShapeAndType(const CNodePtr &node, ::ge::OpDescPtr *op_desc);
  void ConvertOutputShapeAndType(const CNodePtr &node, ::ge::OpDescPtr *op_desc);
  void ConvertCompileInfo(const CNodePtr &node, ::ge::OpDescPtr *op_desc);
  void ConvertAttrs(const CNodePtr &node, ::ge::OpDescPtr *op_desc);
  std::vector<std::tuple<std::size_t, ::ge::NodePtr>> ConvertDepends(
    const CNodePtr &node, const std::map<uint32_t, tensor::TensorPtr> &depend_tensor_map, ::ge::OpDescPtr *op_desc,
    ::ge::ComputeGraphPtr *ge_graph);
  ::ge::NodePtr NewConstantOp(const CNodePtr &node, const std::string &name, const tensor::TensorPtr &tensor_data,
                              ::ge::ComputeGraphPtr *ge_graph, size_t index);
  void AddEdge(const ::ge::NodePtr &ge_node, const std::vector<std::tuple<std::size_t, ::ge::NodePtr>> &constant_ops);
  std::string GetRealOpType(const std::string &op_type);
  std::string GetInputName(const CNodePtr &node, size_t index);
  std::string GetOutputName(const CNodePtr &node, size_t index);
  void InitOpIoName(const CNodePtr &node);
  std::string op_name_;
  std::string op_compile_info_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
}  // namespace tiling
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_TILING_OP_TILING_ADAPTER_H_
