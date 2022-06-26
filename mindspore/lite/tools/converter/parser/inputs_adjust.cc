/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/inputs_adjust.h"
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "nnacl/op_base.h"

namespace mindspore::lite {
namespace {
constexpr int kBuildInputFlagTwo = 2;
constexpr int kBuildInputFlagThree = 3;
constexpr int kBuildInputFlagFour = 4;
constexpr int kBuildInputFlagFive = 5;
}  // namespace
STATUS InputAdjust::AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                                   const std::string &attr_name, int flag) {
  MS_ASSERT(cnode != nullptr);
  MS_CHECK_TRUE_MSG(func_graph != nullptr, RET_ERROR, "func_graph is nullptr");
  if (!opt::CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  MS_CHECK_TRUE_MSG(cnode->input(0) != nullptr, RET_ERROR, "cnode->input(0) is nullptr");
  auto primitive_c = GetValueNode<PrimitiveCPtr>(cnode->input(0));
  MS_CHECK_TRUE_MSG(primitive_c != nullptr, RET_ERROR, "primitive_c is nullptr");
  auto value_ptr = primitive_c->GetAttr(attr_name);
  if (value_ptr == nullptr) {
    MS_LOG(DEBUG) << "there is no attr :" << attr_name;
    return lite::RET_NO_CHANGE;
  }
  if (static_cast<int>(cnode->size()) > input_num) {
    primitive_c->EraseAttr(attr_name);
    MS_LOG(DEBUG) << "input num has been meet, which is " << cnode->size();
    return lite::RET_OK;
  } else if (static_cast<int>(cnode->size()) < input_num) {
    MS_LOG(ERROR) << "input num is invalid.";
    return lite::RET_ERROR;
  }
  AnfNodePtr param_node = nullptr;
  switch (flag) {
    case 1: {
      MS_CHECK_TRUE_MSG(!opt::CastToInt(value_ptr).empty(), RET_ERROR, "value is empty");
      auto value_data = opt::CastToInt(value_ptr).front();
      param_node =
        opt::BuildIntValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagTwo: {
      auto value_data = opt::CastToInt(value_ptr);
      param_node =
        opt::BuildIntVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagThree: {
      auto value_data = opt::CastToVec2DInt(value_ptr);
      param_node =
        opt::BuildIntVec2DParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagFour: {
      auto value_data = GetValue<float>(value_ptr);
      param_node =
        opt::BuildFloatValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagFive: {
      auto value_data = opt::CastToFloat(value_ptr);
      param_node =
        opt::BuildFloatVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    default: {
      MS_LOG(ERROR) << "Error attr flag";
      return lite::RET_ERROR;
    }
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto tr = manager->Transact();
  tr.AddEdge(cnode, param_node);
  tr.Commit();
  return lite::RET_OK;
}

bool InputAdjust::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  STATUS status = lite::RET_OK;
  for (auto &node : node_list) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimTranspose)) {
      MS_LOG(INFO) << "Adjust Transpose";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "perm", kBuildInputFlagTwo);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimReshape)) {
      MS_LOG(INFO) << "Adjust Reshape";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "shape", kBuildInputFlagTwo);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimGather)) {
      MS_LOG(INFO) << "Adjust Gather";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexThree, "axis", 1);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimCast)) {
      MS_LOG(INFO) << "Adjust Cast";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "to", 1);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimTopKFusion)) {
      MS_LOG(INFO) << "Adjust TopKFusion";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "k", 1);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimTileFusion)) {
      MS_LOG(INFO) << "Adjust TileFusion";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "multiples", kBuildInputFlagTwo);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimReduceFusion)) {
      MS_LOG(INFO) << "Adjust ReduceFusion";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "axes", kBuildInputFlagTwo);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimPad) || opt::CheckPrimitiveType(node, prim::kPrimPadFusion)) {
      MS_LOG(INFO) << "Adjust PadFusion";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "paddings", kBuildInputFlagThree);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimPowFusion)) {
      MS_LOG(INFO) << "Adjust PowFuison";
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "power", kBuildInputFlagFour);
    } else if (opt::CheckPrimitiveType(node, prim::kPrimResize)) {
      MS_LOG(INFO) << "Adjust Resize";
      // only one of zoom_factor and scale is valid.
      status = AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "zoom_factor", 1);
      status = (status != lite::RET_NO_CHANGE)
                 ? status
                 : AddAttrToInput(func_graph, cnode, opt::kInputIndexTwo, "scale", kBuildInputFlagFive);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
