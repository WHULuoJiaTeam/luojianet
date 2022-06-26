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
#include "tools/converter/import/cast_op_adjust.h"
#include "tools/anf_exporter/fetch_content.h"
#include "mindspore/lite/include/errorcode.h"

namespace mindspore::lite {
constexpr size_t kCastInputNum = 3;

bool GetInOutDataTypeValue(const CNodePtr &cast_cnode, int *output_type_value, int *input_type_value) {
  MS_ASSERT(cast_cnode != nullptr);
  auto input_node = cast_cnode->input(1);
  MS_CHECK_TRUE_RET(input_node != nullptr, false);
  auto abstract_base = input_node->abstract();
  MS_CHECK_TRUE_RET(abstract_base != nullptr, false);
  if (!utils::isa<abstract::AbstractTensorPtr>(abstract_base)) {
    MS_LOG(ERROR) << "Abstract of parameter should be abstract tensor, " << input_node->fullname_with_scope();
    return false;
  }
  auto abstract_tensor = utils::cast<abstract::AbstractTensorPtr>(abstract_base);
  MS_CHECK_TRUE_RET(abstract_tensor != nullptr && abstract_tensor->element() != nullptr, false);
  auto input_type = abstract_tensor->element()->GetTypeTrack();
  MS_CHECK_TRUE_RET(input_type != nullptr, false);
  *input_type_value = input_type->type_id();

  DataInfo data_info;
  auto output_type_node = cast_cnode->input(opt::kInputIndexTwo);
  if (utils::isa<ParameterPtr>(output_type_node)) {
    if (FetchDataFromParameterNode(cast_cnode, opt::kInputIndexTwo, converter::kFmkTypeMs, &data_info, true) !=
        lite::RET_OK) {
      MS_LOG(ERROR) << "Fetch data from parameter node failed.";
      return false;
    }
  } else if (utils::isa<ValueNodePtr>(output_type_node)) {
    if (FetchDataFromValueNode(cast_cnode, opt::kInputIndexTwo, converter::kFmkTypeMs, false, &data_info, true) !=
        lite::RET_OK) {
      MS_LOG(ERROR) << "Fetch data from value node failed.";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "The output data type is not a const.";
    return false;
  }
  switch (data_info.data_type_) {
    case kNumberTypeInt:
    case kNumberTypeInt32:
      if (data_info.data_.size() < sizeof(int32_t)) {
        MS_LOG(ERROR) << "Data and datatype of data-info not match.";
        return false;
      }
      *output_type_value = reinterpret_cast<int *>(data_info.data_.data())[0];
      break;
    case kNumberTypeInt64:
      if (data_info.data_.size() < sizeof(int64_t)) {
        MS_LOG(ERROR) << "Data and datatype of data-info not match.";
        return false;
      }
      *output_type_value = reinterpret_cast<int64_t *>(data_info.data_.data())[0];
      break;
    default:
      MS_LOG(ERROR) << "Unsupported output data type.";
      return false;
  }
  return true;
}

bool CastOpAdjust::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node) || !opt::CheckPrimitiveType(node, prim::kPrimCast)) {
      continue;
    }
    auto cast_cnode = node->cast<CNodePtr>();
    MS_ASSERT(cast_node != nullptr);
    MS_CHECK_TRUE_RET(cast_cnode->inputs().size() == kCastInputNum, false);
    int input_type_value;
    int output_type_value;
    if (!GetInOutDataTypeValue(cast_cnode, &output_type_value, &input_type_value)) {
      MS_LOG(ERROR) << "Get output type failed.";
      return false;
    }
    bool int_input = input_type_value == kNumberTypeInt || input_type_value == kNumberTypeInt32;
    bool int_output = output_type_value == kNumberTypeInt || output_type_value == kNumberTypeInt32 ||
                      output_type_value == kNumberTypeInt64;
    if ((int_input && int_output) || (input_type_value == output_type_value) ||
        (input_type_value == kNumberTypeFloat32 && output_type_value == kNumberTypeFloat16) ||
        (input_type_value == kNumberTypeFloat16 && output_type_value == kNumberTypeFloat32)) {
      auto ret = manager->Replace(node, cast_cnode->input(1));
      if (!ret) {
        MS_LOG(ERROR) << "Replace node to its input failed.";
        return false;
      }
    }
    if (output_type_value == kNumberTypeInt64) {
      auto parameter = opt::BuildIntValueParameterNode(func_graph, kNumberTypeInt32,
                                                       cast_cnode->input(opt::kInputIndexTwo)->fullname_with_scope());
      if (parameter == nullptr) {
        MS_LOG(ERROR) << "Create parameter failed.";
        return false;
      }
      auto ret = manager->Replace(node, parameter);
      if (!ret) {
        MS_LOG(ERROR) << "Replace node to parameter failed.";
        return false;
      }
    }
  }
  return true;
}
}  // namespace mindspore::lite
