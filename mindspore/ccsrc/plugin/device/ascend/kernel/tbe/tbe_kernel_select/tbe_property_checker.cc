/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_select/tbe_property_checker.h"
#include <map>
#include <string>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "utils/trace_base.h"

namespace mindspore {
namespace kernel {
using Tensor = mindspore::tensor::Tensor;
using TensorPtr = mindspore::tensor::TensorPtr;
using AbstractTensor = mindspore::abstract::AbstractTensor;
using AbstractTensorPtr = mindspore::abstract::AbstractTensorPtr;
using CheckSupportFun = bool (*)(const CNodePtr &cnode);

constexpr char kAttrSorted[] = "sorted";
constexpr char kAttrStrides[] = "strides";
constexpr char kAttrShrinkAxisMask[] = "shrink_axis_mask";

static bool CheckStridedSlice(const CNodePtr &cnode) {
  // check stride[-1] != 1
  if (common::AnfAlgo::HasNodeAttr(kAttrStrides, cnode)) {
    auto strides = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrStrides);
    if (!strides.empty() && strides[strides.size() - 1] != 1) {
      return false;
    }
  } else {
    auto inputs = cnode->inputs();
    const size_t kInputNum = 5;
    if (inputs.size() == kInputNum + 1) {
      auto input_node = inputs[kInputNum];
      MS_EXCEPTION_IF_NULL(input_node);
      auto value_node = input_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto value = value_node->value();
      if (value->isa<tensor::Tensor>()) {
        auto tensor = value->cast<tensor::TensorPtr>();
        TypePtr data_type = tensor->Dtype();
        MS_EXCEPTION_IF_NULL(data_type);
        TypeId type_id = data_type->type_id();
        auto element_size = tensor->data().size();
        if (type_id == kNumberTypeInt32) {
          auto *data = reinterpret_cast<int *>(tensor->data_c());
          if ((data[element_size - 1]) != 1) {
            return false;
          }
        } else if (type_id == kNumberTypeInt64) {
          auto *data = reinterpret_cast<int64_t *>(tensor->data_c());
          if ((data[element_size - 1]) != 1) {
            return false;
          }
        } else {
          MS_EXCEPTION(TypeError) << "The strides of StridedSliceGrad must be int.";
        }
      } else {
        MS_EXCEPTION(ValueError) << "The strides of StridedSliceGrad must be a constant." << inputs.size();
      }
    }
  }

  // check reduction on the last dimension
  if (GetCNodeFuncName(cnode) == kStridedSliceOpName && common::AnfAlgo::HasNodeAttr(kAttrShrinkAxisMask, cnode)) {
    auto shrink_axis_mask = static_cast<int>(common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrShrinkAxisMask));
    AnfNodePtr input = cnode->input(1);
    int input_dims = 0;
    MS_EXCEPTION_IF_NULL(input);
    if (input->isa<ValueNode>()) {
      ValuePtr input_value = input->cast<ValueNodePtr>()->value();
      MS_EXCEPTION_IF_NULL(input_value);
      if (!input_value->isa<Tensor>()) {
        MS_LOG(EXCEPTION) << "For 'StrideSlice', the first input value should be a tensor, but got "
                          << input_value->ToString() << trace::DumpSourceLines(cnode);
      }
      input_dims = SizeToInt(input_value->cast<TensorPtr>()->shape().size());
    } else if (input->isa<CNode>() || input->isa<Parameter>()) {
      AbstractBasePtr input_abstract = input->abstract();
      MS_EXCEPTION_IF_NULL(input_abstract);
      if (!input_abstract->isa<AbstractTensor>()) {
        MS_LOG(EXCEPTION) << "For 'StrideSlice', the first input value should be a tensor, but got "
                          << input_abstract->ToString() << trace::DumpSourceLines(cnode);
      }
      input_dims = SizeToInt(input_abstract->cast<AbstractTensorPtr>()->shape()->shape().size());
    } else {
      MS_LOG(EXCEPTION) << "For 'StrideSlice', the first input node should be a 'ValueNode' or a 'CNode', but got "
                        << input->ToString() << trace::DumpSourceLines(cnode);
    }
    const int base_number = 2;
    if (shrink_axis_mask >= std::pow<int, int>(base_number, input_dims - 1) && input_dims > 1) {
      return false;
    }
  }
  return true;
}

static bool CheckTopK(const CNodePtr &cnode) {
  if (common::AnfAlgo::HasNodeAttr(kAttrSorted, cnode)) {
    auto sorted = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrSorted);
    return sorted;
  }
  MS_LOG(EXCEPTION) << "For 'TopK', it should be have attribute 'sorted'." << trace::DumpSourceLines(cnode);
}

bool TbePropertyChecker::CheckTbeProperties(const mindspore::CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  static std::map<std::string, CheckSupportFun> tbe_property_checker = {
    {kStridedSliceOpName, CheckStridedSlice}, {kStridedSliceGradOpName, CheckStridedSlice}, {kTopKOpName, CheckTopK}};
  auto cnode_type = common::AnfAlgo::GetCNodeName(cnode);
  auto find_iter = tbe_property_checker.find(cnode_type);
  if (find_iter != tbe_property_checker.end()) {
    return find_iter->second(cnode);
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
