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

#include "graph/preprocess/multi_batch_options.h"

#include "framework/common/debug/ge_log.h"
#include "framework/omg/omg_inner_types.h"
#include "framework/common/util.h"
#include "framework/common/string_util.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/ge_context.h"
#include "common/local_context.h"
#include "framework/common/types.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/omg_util.h"

namespace ge {
namespace multibatch {
constexpr int kDecimal = 10;
constexpr uint8_t kMaxShapesCount = 100;
constexpr uint8_t kMinShapesCount = 2;
const int kDynmaicDims = -1;
const int kDynamicImgSizeDynamciDimsNum = 2;
const size_t kNumOfGetnextNode = 1;
const int kDivisionConst = 2;
const char *const kSubstrOfGetNextNosinkName = "IteratorGetNext";
const char *const kShapeDataName = "ascend_mbatch_shape_data";
const char *const kGetNextName = "IteratorV2";

inline bool IsGetNextType(const NodePtr &node) {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                  GELOGW("Get original type failed."); return false);
  return (original_type == kGetNextName);
}

void ParseDynamicSize(string dynamic_size, vector<vector<int64_t>> &shapes) {
  std::vector<std::string> shape_strs = ge::StringUtils::Split(dynamic_size, ';');
  for (const auto &shape_str : shape_strs) {
    if (shape_str.empty()) {
      continue;
    }
    std::vector<int64_t> shape;
    std::vector<std::string> dims = ge::StringUtils::Split(shape_str, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shape.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
    }
    if (!shape.empty()) {
      shapes.emplace_back(shape);
    }
  }
}

Status DistinguishGetNextAndData(ComputeGraphPtr &graph, vector<NodePtr> &data_nodes,
                                 vector<NodePtr> &getnext_nosink_nodes, vector<NodePtr> &getnext_sink_nodes) {
  GELOGD("Start distinguish getnext and data node.");
  for (NodePtr &input_node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op_desc = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (op_desc->GetType() == DATA && op_desc->GetName() != kShapeDataName) {
      if (op_desc->GetName().find(kSubstrOfGetNextNosinkName) == string::npos) {
        data_nodes.emplace_back(input_node);
        GELOGD("Name of data node is %s.", op_desc->GetName().c_str());
      } else {
        getnext_nosink_nodes.emplace_back(input_node);
        GELOGD("Name of getnext nosink is %s.", op_desc->GetName().c_str());
      }
    }
    if (IsGetNextType(input_node)) {
      GELOGD("Name of getnext sink is %s.", op_desc->GetName().c_str());
      getnext_sink_nodes.emplace_back(input_node);
    }
  }
  GELOGI("Data count is %zu, getnext nosink count is %zu, getnext sink count is %zu.", data_nodes.size(),
         getnext_nosink_nodes.size(), getnext_sink_nodes.size());
  GetLocalOmgContext().data_nodes = data_nodes;
  GetLocalOmgContext().getnext_nosink_nodes = getnext_nosink_nodes;
  return SUCCESS;
}

Status CheckSequenceOfData(ComputeGraphPtr &graph, const vector<NodePtr> &data_nodes) {
  GELOGD("Start check input sequence from data nodes and input shape.");
  if (data_nodes.size() != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERROR("E19999", "Count:%zu of data_nodes in graph:%s should be equal to "
                       "input_shape count:%zu from option, check invalid",
                       data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Count:%zu of data_nodes in graph:%s should be equal to "
           "input_shape count:%zu from option",
           data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < data_nodes.size(); ++i) {
    auto data_node = data_nodes.at(i);
    GE_CHECK_NOTNULL(data_node);
    GE_CHECK_NOTNULL(data_node->GetOpDesc());
    auto output_shape = data_node->GetOpDesc()->GetOutputDesc(0).GetShape().GetDims();
    auto dynamic_dims = GetLocalOmgContext().user_input_dims.at(i).second;
    GELOGD("The %zu data node is %s, node shape is %s, dynamic dim is %s.", i, data_node->GetName().c_str(),
           formats::JoinToString(output_shape).c_str(), formats::JoinToString(dynamic_dims).c_str());
    if (output_shape.empty() && dynamic_dims.size() == 1 && dynamic_dims.at(0) == 0) {
      GELOGI("No need to check sequence for constant.");
      continue;
    }
    if (dynamic_dims.size() != output_shape.size()) {
      REPORT_INNER_ERROR("E19999", "The output shape of %s is %s, the input shape from options of %s is %s, graph:%s,"
                         "check invalid", data_node->GetName().c_str(),
                         formats::JoinToString(output_shape).c_str(),
                         GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
                         formats::JoinToString(dynamic_dims).c_str(), graph->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] The output shape of %s is %s, "
             "the input shape from options of %s is %s, graph:%s",
             data_node->GetName().c_str(), formats::JoinToString(output_shape).c_str(),
             GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
             formats::JoinToString(dynamic_dims).c_str(), graph->GetName().c_str());
      return PARAM_INVALID;
    }
    for (size_t j = 0; j < dynamic_dims.size(); ++j) {
      if (dynamic_dims.at(j) != kDynmaicDims && dynamic_dims.at(j) != output_shape.at(j)) {
        REPORT_INNER_ERROR("E19999", "Value of input shape %s from option and output shape %s of data op:%s "
                           "should be equal to %d, index:%zu, graph:%s, check invalid",
                           formats::JoinToString(dynamic_dims).c_str(),
                           formats::JoinToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
                           j, graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Value of input shape %s from option and output shape %s of data op:%s "
               "should be equal to %d, index:%zu, graph:%s",
               formats::JoinToString(dynamic_dims).c_str(), formats::JoinToString(output_shape).c_str(),
               data_node->GetName().c_str(), kDynmaicDims, j, graph->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

Status CheckSequenceOfGetnext(ComputeGraphPtr &graph, const vector<NodePtr> &getnext_sink_node) {
  GELOGD("Start check input sequence from getnext sink nodes and input shape.");
  if (getnext_sink_node.size() != kNumOfGetnextNode) {
    REPORT_INNER_ERROR("E19999", "Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
                       "num of getnext node:%zu, check invalid",
                       graph->GetName().c_str(), getnext_sink_node.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
           "num of getnext node:%zu", graph->GetName().c_str(), getnext_sink_node.size());
    return PARAM_INVALID;
  }
  auto data_node = getnext_sink_node.at(0);
  GE_CHECK_NOTNULL(data_node);
  auto op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  size_t data_count = data_node->GetAllOutDataAnchors().size() / kDivisionConst;
  if (data_count != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERROR("E19999", "Output desc count of %s is %zu, should be equal to count of input shape:%zu, "
                       "graph:%s, check invalid", op_desc->GetName().c_str(), data_count,
                       GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Output desc count of %s is %zu, "
           "should be equal to count of input shape:%zu, graph:%s", op_desc->GetName().c_str(),
           data_count, GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < data_count; ++i) {
    auto output_shape = data_node->GetOpDesc()->GetOutputDesc(i).GetShape().GetDims();
    auto dynamic_dims = GetLocalOmgContext().user_input_dims.at(i).second;
    GELOGD("The %zu getnext node is %s, node shape is %s, dynamic dim is %s.", i, data_node->GetName().c_str(),
           formats::JoinToString(output_shape).c_str(), formats::JoinToString(dynamic_dims).c_str());
    if (output_shape.empty() && dynamic_dims.size() == 1 && dynamic_dims.at(0) == 0) {
      GELOGI("No need to check sequence for constant.");
      continue;
    }
    if (dynamic_dims.size() != output_shape.size()) {
      REPORT_INNER_ERROR("E19999", "The %zu output_shape of %s is %s not equal to the input_shape:%s "
                         "from options of %s, graph:%s, check invalid", i,
                         data_node->GetName().c_str(), formats::JoinToString(output_shape).c_str(),
                         formats::JoinToString(dynamic_dims).c_str(),
                         GetLocalOmgContext().user_input_dims.at(i).first.c_str(),
                         graph->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] The %zu output_shape of %s is %s not equal to the input_shape:%s "
             "from options of %s, graph:%s", i, data_node->GetName().c_str(),
             formats::JoinToString(output_shape).c_str(), formats::JoinToString(dynamic_dims).c_str(),
             GetLocalOmgContext().user_input_dims.at(i).first.c_str(), graph->GetName().c_str());
      return PARAM_INVALID;
    }
    for (size_t j = 0; j < dynamic_dims.size(); ++j) {
      if (dynamic_dims.at(j) != kDynmaicDims && dynamic_dims.at(j) != output_shape.at(j)) {
        REPORT_INNER_ERROR("E19999", "Value of input shape %s from option and output shape %s of data op:%s "
                           "should be equal to %d, index:%zu, graph:%s, check invalid",
                           formats::JoinToString(dynamic_dims).c_str(),
                           formats::JoinToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
                           j, graph->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Check][Param] Value of input shape %s from option and output shape %s of data op:%s "
               "should be equal to %d, index:%zu, graph:%s", formats::JoinToString(dynamic_dims).c_str(),
               formats::JoinToString(output_shape).c_str(), data_node->GetName().c_str(), kDynmaicDims,
               j, graph->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

Status CheckSequenceOfOptions(ComputeGraphPtr &graph, vector<NodePtr> &data_nodes,
                              vector<NodePtr> &getnext_nosink_nodes, vector<NodePtr> &getnext_sink_nodes) {
  if (GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGI("No need to CheckSequenceOfOptions.");
    return SUCCESS;
  }

  if (DistinguishGetNextAndData(graph, data_nodes, getnext_nosink_nodes, getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Call][DistinguishGetNextAndData] failed.");
    return PARAM_INVALID;
  }

  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    GELOGD("Users want data nodes to be dynamic.");
    if (CheckSequenceOfData(graph, data_nodes) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][Sequence] Of Data nodes failed.");
      return PARAM_INVALID;
    }
  } else {
    GELOGD("Users want getnext nodes to be dynamic.");
    if (!getnext_nosink_nodes.empty()) {
      if (CheckSequenceOfData(graph, getnext_nosink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Check][Sequence] of getnext nosink nodes failed.");
        return PARAM_INVALID;
      }
    } else {
      if (CheckSequenceOfGetnext(graph, getnext_sink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Check][Sequence] of getnext sink nodes failed.");
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status UpdateNameOfData(ComputeGraphPtr &graph, const vector<NodePtr> &data_nodes) {
  GELOGD("Update first value of input shape by data nodes.");
  if (data_nodes.size() != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERROR("E19999", "Count:%zu of data_nodes in graph:%s should be equal to "
                       "input_shape count:%zu from option, check invalid",
                       data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Count:%zu of data_nodes in graph:%s should be equal to "
           "input_shape count:%zu from option",
           data_nodes.size(), graph->GetName().c_str(), GetLocalOmgContext().user_input_dims.size());
    return PARAM_INVALID;
  }
  for (size_t i = 0; i < data_nodes.size(); ++i) {
    GELOGD("The %zu data name is %s.", i, data_nodes.at(i)->GetOpDesc()->GetName().c_str());
    GetLocalOmgContext().user_input_dims.at(i).first = data_nodes.at(i)->GetOpDesc()->GetName();
  }
  return SUCCESS;
}

Status UpdateNameOfGetnext(ComputeGraphPtr &graph, const vector<NodePtr> &getnext_sink_nodes) {
  GELOGD("Update first value of input shape by getnext sink nodes.");
  if (getnext_sink_nodes.size() != kNumOfGetnextNode) {
    REPORT_INNER_ERROR("E19999", "Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
                       "num of getnext node:%zu, check invalid",
                       graph->GetName().c_str(), getnext_sink_nodes.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Not support dynamic dims when a graph with multi getnext nodes, graph:%s, "
           "num of getnext node:%zu", graph->GetName().c_str(), getnext_sink_nodes.size());
    return PARAM_INVALID;
  }
  auto input_node = getnext_sink_nodes.at(0);
  GE_CHECK_NOTNULL(input_node);
  auto op_desc = input_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  // user want getnext dynamic, just getnext or data+getnext_sink
  size_t data_count = input_node->GetAllOutDataAnchors().size() / kDivisionConst;
  if (data_count != GetLocalOmgContext().user_input_dims.size()) {
    REPORT_INNER_ERROR("E19999", "Output desc count of %s is %zu, should be equal to count of input shape:%zu, "
                       "graph:%s, check invalid", op_desc->GetName().c_str(), data_count,
                       GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param]Output desc count of %s is %zu, "
           "should be equal to count of input shape:%zu, graph:%s", op_desc->GetName().c_str(), data_count,
           GetLocalOmgContext().user_input_dims.size(), graph->GetName().c_str());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < data_count; ++i) {
    string data_name = op_desc->GetName() + "_" + std::to_string(i);
    GELOGD("Data just from getnext sink is %s.", data_name.c_str());
    GetLocalOmgContext().user_input_dims.at(i).first = data_name;
  }
  return SUCCESS;
}

// need to distinguish online and offline, offline no need to update the name of input_shape
Status UpdateNameOfInputShape(ComputeGraphPtr &graph, const vector<NodePtr> &data_nodes,
                              const vector<NodePtr> &getnext_nosink_nodes, const vector<NodePtr> &getnext_sink_nodes) {
  if (GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGI("No need to update first value of input shape when offline infer.");
    return SUCCESS;
  }

  if (GetLocalOmgContext().dynamic_node_type == DATA) {
    GELOGD("Users want data nodes to be dynamic.");
    if (UpdateNameOfData(graph, data_nodes) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Call][UpdateNameOfData] update first value of input shape of data nodes failed.");
      return PARAM_INVALID;
    }
  } else {
    GELOGD("Users want getnext nodes to be dynamic.");
    if (!getnext_nosink_nodes.empty()) {
      if (UpdateNameOfData(graph, getnext_nosink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID,
               "[Call][UpdateNameOfData] update first value of input shape of getnext nosink nodes failed.");
        return PARAM_INVALID;
      }
    } else {
      if (UpdateNameOfGetnext(graph, getnext_sink_nodes) != SUCCESS) {
        GELOGE(PARAM_INVALID,
               "[Call][UpdateNameOfGetnext] update first value of input shape of getnext sink nodes failed.");
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status DeleteIdentityInsertByAdapter(ComputeGraphPtr &graph) {
  GELOGD("Start delete identity node inserted by adapter.");
  for (NodePtr &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (IsGetNextType(node)) {
      for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
        GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
        for (auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
          GE_IF_BOOL_EXEC(peer_in_anchor == nullptr, continue);
          auto dst_node = peer_in_anchor->GetOwnerNode();
          GE_IF_BOOL_EXEC(dst_node == nullptr, continue);
          if (dst_node->GetType() == IDENTITY && dst_node->GetOutDataNodes().empty()) {
            GELOGI("Need to remove %s.", dst_node->GetName().c_str());
            if (GraphUtils::RemoveNodeWithoutRelink(graph, dst_node) != GRAPH_SUCCESS) {
              REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) from graph:%s failed",
                                dst_node->GetName().c_str(), dst_node->GetType().c_str(), graph->GetName().c_str());
              GELOGE(FAILED, "[Remove][Node] %s(%s) from graph:%s failed",
                     dst_node->GetName().c_str(), dst_node->GetType().c_str(), graph->GetName().c_str());
              return FAILED;
            }
          }
        }
      }
    }
  }
  return SUCCESS;
}

Status CheckNegativeCountOfOptions(const std::vector<std::vector<int64_t>> &shapes) {
  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    size_t negative_count = 0;
    for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
      for (size_t j = 0; j < GetLocalOmgContext().user_input_dims.at(i).second.size(); ++j) {
        if (GetLocalOmgContext().user_input_dims.at(i).second.at(j) == kDynmaicDims) {
          negative_count++;
        }
      }
    }
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (shapes.at(i).size() != negative_count) {
        REPORT_INNER_ERROR("E19999", "gear num of dynamic_dims is %zu should be equal to num:%zu from option, "
                           "check invalid", shapes.at(i).size(), negative_count);
        GELOGE(PARAM_INVALID, "[Check][Param] gear num of dynamic_dims is %zu should be equal to num:%zu from option",
               shapes.at(i).size(), negative_count);
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Init Dynamic Param from Options.
/// @param [out] std::vector<std::vector<int64_t>> &shapes: Result for Params.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
bool InitDynamicParams(vector<vector<int64_t>> &shapes) {
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    GELOGD("Found dynamic batch option, value %s", GetLocalOmgContext().dynamic_batch_size.c_str());
    std::vector<std::string> dims = ge::StringUtils::Split(GetLocalOmgContext().dynamic_batch_size, ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shapes.emplace_back(std::vector<int64_t>({std::strtol(dim.c_str(), nullptr, kDecimal)}));
      GELOGI("Found dynamic batch, shape %s", formats::JoinToString(*shapes.rbegin()).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    GELOGD("Found dynamic image size option, value %s", GetLocalOmgContext().dynamic_image_size.c_str());
    ParseDynamicSize(GetLocalOmgContext().dynamic_image_size, shapes);

    for (const auto &shape : shapes) {
      GELOGI("Found dynamic image size, shape %s", formats::JoinToString(shape).c_str());
    }
  }

  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    GELOGD("Found dynamic dims option, value %s", GetLocalOmgContext().dynamic_dims.c_str());
    ParseDynamicSize(GetLocalOmgContext().dynamic_dims, shapes);

    for (const auto &shape : shapes) {
      GELOGI("Found dynamic dims, shape %s", formats::JoinToString(shape).c_str());
    }
  }

  return !shapes.empty();
}

///
/// @ingroup ge
/// @brief parse each data's own dynamic dims.
/// @param [out] map<string, vector<vector<int64_t>>> &data_to_dynamic_info: key:data_name. value:dynamic dims.
/// @return true: Configed for Multi batch / false: Not configed for Multi batch.
///
Status ParserDataToDynamicInfo(const vector<vector<int64_t>> &shapes,
                               vector<pair<string, vector<int64_t>>> &data_name_and_shape,
                               map<string, vector<vector<int64_t>> > &data_to_dynamic_info) {
  size_t cur_data_index = 0;
  for (size_t index = 0; index < data_name_and_shape.size(); ++index) {
    auto &cur_item = data_name_and_shape[index];
    auto &data_name = cur_item.first;
    auto &data_shape = cur_item.second;
    auto dynamic_dims_num = std::count_if(data_shape.begin(), data_shape.end(),
                                          [&data_shape](int64_t dim){ return dim < 0; });
    GELOGI("Train_Dynamic dynamic_dims_num of %s is %zu", data_name.c_str(), dynamic_dims_num);
    vector<vector<int64_t> > dynamic_info;
    for (auto &dynamic_gear_info : shapes) {
      GELOGI("Train_Dynamic dynamic_gear_info is %s", formats::JoinToString(dynamic_gear_info).c_str());
      vector<int64_t> one_gear;
      if (dynamic_gear_info.size() == static_cast<size_t>(dynamic_dims_num)) {
        one_gear = dynamic_gear_info;
      } else if (dynamic_gear_info.size() > static_cast<size_t>(dynamic_dims_num)) {
        auto tmp_index = cur_data_index;
        for (size_t i = 0; i < static_cast<size_t>(dynamic_dims_num); ++i) {
          if (tmp_index >= dynamic_gear_info.size()) {
            ErrorManager::GetInstance().ATCReportErrMessage(
                "E10045", {"name", "shape"}, {data_name, formats::JoinToString(data_shape)});
            GELOGE(PARAM_INVALID, "[Check][Param] Data:%s shape:%s make dynamic dims overflow", data_name.c_str(),
                   formats::JoinToString(data_shape).c_str());
            return FAILED;
          }
          one_gear.push_back(dynamic_gear_info[tmp_index++]);
        }
      } else {
        ErrorManager::GetInstance().ATCReportErrMessage(
            "E10046", {"name", "shape"}, {data_name, formats::JoinToString(data_shape)});
        GELOGE(PARAM_INVALID, "[Check][Param] Dynamic dims num of data: %s shape: %s "
               "can not be more than one gear dynamic info size",
               data_name.c_str(), formats::JoinToString(data_shape).c_str());
        return FAILED;
      }
      GELOGI("Train_Dynamic one_gear is %s.", formats::JoinToString(one_gear).c_str());
      dynamic_info.push_back(one_gear);
    }
    cur_data_index += dynamic_dims_num;
    data_to_dynamic_info[data_name] = dynamic_info;
  }
  return SUCCESS;
}


///
/// @ingroup ge
/// @brief Check Dynamic Param is invalid.
/// @param [in] const vector<vector<int64_t>> &shapes: Params for check.
/// @return SUCCESS: valid / PARAM_INVALID: invalid.
///
Status CheckDynamicParams(const vector<vector<int64_t>> &shapes) {
  if (shapes.size() < kMinShapesCount) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10035", {"shapesize", "minshapesize"}, {std::to_string(shapes.size()), std::to_string(kMinShapesCount - 1)});
    GELOGE(PARAM_INVALID,
           "[Check][Param] Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
           "value size [%zu] must be greater than [%d].",
           shapes.size(), kMinShapesCount - 1);
    return PARAM_INVALID;
  }
  if (shapes.size() > kMaxShapesCount) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10036", {"shapesize", "maxshapesize"}, {std::to_string(shapes.size()), std::to_string(kMaxShapesCount + 1)});
    GELOGE(PARAM_INVALID,
           "[Check][Param] Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
           "value size [%zu] must be less than [%d].",
           shapes.size(), kMaxShapesCount + 1);
    return PARAM_INVALID;
  }
  std::set<std::vector<int64_t>> shapes_set;
  size_t shape_size = shapes.at(0).size();
  for (auto &shape : shapes) {
    if (shape_size != shape.size()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10037", {"shapesize1", "shapesize2"},
                                                      {std::to_string(shape_size), std::to_string(shape.size())});
      GELOGE(PARAM_INVALID,
             "[Check][Param] Input parameter[--dynamic_batch_size, --dynamic_image_size or --dynamic_dims]'s "
             "value size must be same, first group's size is %zu and another's is %zu.",
             shape_size, shape.size());
      return PARAM_INVALID;
    }
    for (auto dim : shape) {
      if (dim <= 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10038", {"dim"}, {std::to_string(dim)});
        GELOGE(PARAM_INVALID, "[Check][Param] Invalid dim %ld, all dims must be greater than 0", dim);
        return PARAM_INVALID;
      }
    }
    shapes_set.insert(shape);
  }
  if (shapes_set.size() != shapes.size()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10039");
    GELOGE(PARAM_INVALID, "[Check][Param] Input parameter[--dynamic_batch_size, "
           "--dynamic_image_size or --dynamic_dims] exist duplicate shapes.");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get GeShape from configed shape.
/// @param [in] const std::vector<int64_t> &batch_shape: Configed shape.
/// @param [out] GeShape &data_shape: GeShape for configed shape.
/// @return SUCCESS / PARAM_INVALID
///
Status CalcShape(const std::vector<int64_t> &batch_shape, GeShape &data_shape) {
  size_t batch_shape_index = 0;
  for (size_t i = 0; i < data_shape.GetDimNum(); ++i) {
    if (data_shape.GetDim(i) < 0) {
      if (batch_shape_index >= batch_shape.size()) {
        REPORT_INNER_ERROR("E19999", "the batch shape count %zu, does not match the data shape %s",
                           batch_shape.size(), data_shape.ToString().c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] Failed to calc tensor shape, the batch shape count %zu, "
               "does not match the data shape %s", batch_shape.size(), data_shape.ToString().c_str());
        return PARAM_INVALID;
      }
      data_shape.SetDim(i, batch_shape[batch_shape_index++]);
    }
  }
  GELOGI("CalcShape size of batch_shape is %zu, batch_shape_index is %zu.", batch_shape.size(), batch_shape_index);
  if (batch_shape_index != batch_shape.size()) {
    REPORT_INNER_ERROR("E19999", "the batch shape count %zu, does not match the data shape %s",
                       batch_shape.size(), data_shape.ToString().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Failed to calc tensor shape, the batch shape count %zu, "
           "does not match the data shape %s", batch_shape.size(), data_shape.ToString().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Set mbatch_dynamic_type on node.
/// @param [in] const OpDescPtr &op_desc: Node for set attribute.
/// @return 0: SUCCESS / others: INTERNAL_ERROR
///
Status StampDynamicType(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc);
  int32_t dynamic_type = static_cast<int32_t>(FIXED);
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_BATCH);
  }
  if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_IMAGE);
  }
  if (!GetLocalOmgContext().dynamic_dims.empty()) {
    dynamic_type = static_cast<int32_t>(DYNAMIC_DIMS);
  }
  if (!AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_DYNAMIC_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           ATTR_DYNAMIC_TYPE.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Check dynamic batch Shape.
/// @param [in] const vector<int64_t> &shape: data_shape to be checked.
/// @param [in] const string &data_name: cur data name.
/// @return 0: true/false
///
bool CheckDynamicBatchShape(const vector<int64_t> &shape, const string &data_name) {
  if (shape[0] == kDynmaicDims) {
    for (size_t i = 1; i < shape.size(); ++i) {
      if (shape[i] < 1) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10018", {"index", "shape"},
                                                        {std::to_string(i), std::to_string(shape[i])});
        GELOGE(ge::PARAM_INVALID, "[Check][Param] Only batch N can be -1 when set --dynamic_batch_size, "
               "current data: %s shape[%zu] is %ld", data_name.c_str(), i, shape[i]);
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

///
/// @ingroup ge
/// @brief Check Dynamic image size shape.
/// @param [in] unordered_map<string, vector<int64_t>> &shape_map: map of data_name and data_shape.
/// @param [in]  const std::string &input_format: format of input.
/// @return 0: true/false
///
bool CheckDynamicImageSizeShape(const vector<int64_t> &shape, const string &data_name,
                                const std::string &input_format) {
  int64_t height = 0;
  int64_t width = 0;
  if (input_format == "NCHW") {
    height = shape[NCHW_DIM_H];
    width = shape[NCHW_DIM_W];
  }

  if (input_format == "NHWC") {
    height = shape[NHWC_DIM_H];
    width = shape[NHWC_DIM_W];
  }

  if (height == kDynmaicDims && width == kDynmaicDims &&
      std::count(shape.begin(), shape.end(), kDynmaicDims) == kDynamicImgSizeDynamciDimsNum) {
    return true;
  } else {
    ErrorManager::GetInstance().ATCReportErrMessage("E10019");
    GELOGE(ge::PARAM_INVALID, "[Check][Param] --input_shape's shape is invalid, only height and width can be -1 "
           "when set --dynamic_image_size.");
    return false;
  }
}
}  // namespace multibatch
}  // namespace ge
