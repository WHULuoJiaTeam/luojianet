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

#include "graph/preprocess/insert_op/ge_aipp_op.h"
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "graph/preprocess/insert_op/base_insert_op.h"
#include "common/dynamic_aipp.h"
#include "common/ge/ge_util.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/optimize/common/params.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "proto/insert_op.pb.h"
#include "common/local_context.h"

#define SAVE_AIPP_ATTR(KEY, SAVE_TYPE)                                                       \
  do {                                                                                       \
    (void)aipp_attrs.SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY())); \
  } while (0)

#define SAVE_AIPP_ATTR_LIST(KEY, SAVE_TYPE)                                                     \
  do {                                                                                          \
    if (aipp_params_->KEY##_size() > 0) {                                                       \
      (void)aipp_attrs.SetAttr(#KEY, GeAttrValue::CreateFrom<SAVE_TYPE>(aipp_params_->KEY(0))); \
    }                                                                                           \
  } while (0)

namespace {
const int32_t DEFAULT_MATRIX_R0C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R0C1_YUV2RGB = 0;
const int32_t DEFAULT_MATRIX_R0C2_YUV2RGB = 409;
const int32_t DEFAULT_MATRIX_R1C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R1C1_YUV2RGB = -100;
const int32_t DEFAULT_MATRIX_R1C2_YUV2RGB = -208;
const int32_t DEFAULT_MATRIX_R2C0_YUV2RGB = 298;
const int32_t DEFAULT_MATRIX_R2C1_YUV2RGB = 516;
const int32_t DEFAULT_MATRIX_R2C2_YUV2RGB = 0;
const int32_t DEFAULT_MATRIX_R0C0_RGB2YUV = 66;
const int32_t DEFAULT_MATRIX_R0C1_RGB2YUV = 129;
const int32_t DEFAULT_MATRIX_R0C2_RGB2YUV = 25;
const int32_t DEFAULT_MATRIX_R1C0_RGB2YUV = -38;
const int32_t DEFAULT_MATRIX_R1C1_RGB2YUV = -74;
const int32_t DEFAULT_MATRIX_R1C2_RGB2YUV = 112;
const int32_t DEFAULT_MATRIX_R2C0_RGB2YUV = 112;
const int32_t DEFAULT_MATRIX_R2C1_RGB2YUV = -94;
const int32_t DEFAULT_MATRIX_R2C2_RGB2YUV = -18;
const int32_t DEFAULT_OUTPUT_BIAS_0 = 16;
const int32_t DEFAULT_OUTPUT_BIAS_1 = 128;
const int32_t DEFAULT_OUTPUT_BIAS_2 = 128;
const int32_t DEFAULT_INPUT_BIAS_0 = 16;
const int32_t DEFAULT_INPUT_BIAS_1 = 128;
const int32_t DEFAULT_INPUT_BIAS_2 = 128;
const float DEFAULT_VAR_RECI_CHN = 1.0;
}  // namespace

namespace ge {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
const char *const kAippConfigPath = "aipp_config_path";
const char *const kCurrentAippIndex = "current_aipp_index";
const char *const kDynamicAippData = "ascend_dynamic_aipp_data";
const uint64_t kMinTransferShape = 3;
const int kAippImageInputIndex = 0;
const int kAippParamsInputIndex = 1;
const int kAippDataOutputIndex = 0;
const int64_t kDynamicDim = -1;

// the `format` must one NCHW or NHWC
Status GetDataDimN(const ge::NodePtr &data_node, ge::Format format, int64_t &batch) {
  auto output_desc = NodeUtils::GetOutputDesc(*data_node, 0);
  auto shape = output_desc.GetShape().GetDims();
  if (shape.size() == kMinTransferShape) {
    batch = 1;
    return SUCCESS;
  }
  if (shape.size() == DIM_DEFAULT_SIZE) {
    switch (format) {
      case FORMAT_NCHW:
        batch = shape[NCHW_DIM_N];
        return SUCCESS;
      case FORMAT_NHWC:
        batch = shape[NHWC_DIM_N];
        return SUCCESS;
      default:
        REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                           std::vector<std::string>({
                               data_node->GetName() + " format",
                               TypeUtils::FormatToSerialString(format),
                               "only format " + TypeUtils::FormatToSerialString(FORMAT_NCHW) + " and " +
                               TypeUtils::FormatToSerialString(FORMAT_NHWC) +
                               " supported which dynamic aipp is linked"}));
        GELOGE(PARAM_INVALID, "[Check][Param] Not support data format:%s, node:%s",
               TypeUtils::FormatToSerialString(format).c_str(), data_node->GetName().c_str());
        return PARAM_INVALID;
    }
  }
  string errormsg = "its shape size must be in range[3,4] which dynamic aipp is linked, "
                    "maybe this input is not suitable for dynamic aipp";
  ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                  {data_node->GetName() + " shape size",
                                                  to_string(shape.size()), errormsg});
  GELOGE(PARAM_INVALID, "[Check][Param] The shape size of this node [%s] "
         "which linked dynamic aipp must be in range[3, 4], but is %zu",
         data_node->GetName().c_str(), shape.size());
  return PARAM_INVALID;
}

// the batch_count must be more than 0
int64_t CalcMaxSize(int64_t batch_count) {
  batch_count--;
  if (batch_count > 0) {
    if (INT64_MAX / batch_count < static_cast<int64_t>(sizeof(kAippDynamicBatchPara))) {
      return -1;
    }
  }

  int64_t size = batch_count * sizeof(kAippDynamicBatchPara);
  if (INT64_MAX - static_cast<int64_t>(sizeof(kAippDynamicPara)) < size) {
    return -1;
  }
  return size + sizeof(kAippDynamicPara);
}

Format GetAndCheckFormat() {
  switch (GetLocalOmgContext().format) {
    case domi::DOMI_TENSOR_NCHW:
      return FORMAT_NCHW;
    case domi::DOMI_TENSOR_NHWC:
      return FORMAT_NHWC;
    default:
      GELOGE(PARAM_INVALID, "[Check][Param] Unexpected format found %d",
             static_cast<int>(GetLocalOmgContext().format));
      return FORMAT_ND;
  }
}
}  // namespace

Status AippOp::Init(domi::AippOpParams *aipp_params) {
  aipp_params_ = new (std::nothrow) domi::AippOpParams();
  if (aipp_params_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "New AippOpParams failed");
    GELOGE(FAILED, "[New][AippOpParams] failed");
    return FAILED;
  }
  aipp_params_->CopyFrom(*aipp_params);
  return SUCCESS;
}

AippOp::~AippOp() {
  if (aipp_params_ != nullptr) {
    delete aipp_params_;
    aipp_params_ = nullptr;
  }
}

Status AippOp::InsertAippToGraph(ComputeGraphPtr &graph, std::string &aippConfigPath, const uint32_t index) {
  GE_CHECK_NOTNULL(graph);
  NodePtr target_input = nullptr;
  std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> target_edges;

  if (this->ConvertRelatedInputNameToRank() != SUCCESS) {
    GELOGE(FAILED, "[Call][ConvertRelatedInputNameToRank] failed.");
    return FAILED;
  }
  GE_CHK_STATUS_RET(this->GetTargetPosition(graph, target_input, target_edges), "[Get][TargetPosition] failed");

  std::map<OutDataAnchorPtr, NodePtr> out_anchors_to_aipp;
  for (auto &out_in_anchors : target_edges) {
    auto iter = out_anchors_to_aipp.find(out_in_anchors.first);
    if (iter == out_anchors_to_aipp.end()) {
      auto aipp = CreateAipp(out_in_anchors.first, aippConfigPath, index);
      GE_CHECK_NOTNULL(aipp);
      out_anchors_to_aipp[out_in_anchors.first] = aipp;

      auto ret = GraphUtils::InsertNodeBetweenDataAnchors(out_in_anchors.first, out_in_anchors.second, aipp);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Insert aipp:%s(%s) node between op:%s(%s) and op:%s:%s failed",
                          aipp->GetName().c_str(), aipp->GetType().c_str(),
                          out_in_anchors.first->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.first->GetOwnerNode()->GetType().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Insert][Node] %s(%s) between op:%s(%s) and op:%s:%s failed",
               aipp->GetName().c_str(), aipp->GetType().c_str(),
               out_in_anchors.first->GetOwnerNode()->GetName().c_str(),
               out_in_anchors.first->GetOwnerNode()->GetType().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        return INTERNAL_ERROR;
      }

      // add aipp data if needed
      if (GetAippMode() == domi::AippOpParams::dynamic) {
        ret = CreateAippData(aipp);
        if (ret != SUCCESS) {
          GELOGE(INTERNAL_ERROR, "[Create][AippData] for aipp %s data %s failed", aipp->GetName().c_str(),
                 out_in_anchors.first->GetOwnerNode()->GetName().c_str());
          return INTERNAL_ERROR;
        }
      }
      GELOGI("Create aipp %s and insert it to the graph", aipp->GetName().c_str());
    } else {
      out_in_anchors.second->UnlinkAll();
      auto &aipp = iter->second;
      auto ret = out_in_anchors.second->LinkFrom(aipp->GetOutDataAnchor(0));
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "link aipp:%s(%s) to peer op:%s(%s) failed",
                          aipp->GetName().c_str(), aipp->GetType().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetName().c_str(),
                          out_in_anchors.second->GetOwnerNode()->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Call][LinkFrom] Failed to link aipp %s to the peer node %s", aipp->GetName().c_str(),
               out_in_anchors.second->GetOwnerNode()->GetName().c_str());
        return INTERNAL_ERROR;
      }
    }
  }

  return SUCCESS;
}
NodePtr AippOp::CreateAipp(const OutDataAnchorPtr &out_anchor,
                           const std::string &aippConfigPath, const uint32_t &index) {
  const auto &node = out_anchor->GetOwnerNode();
  std::string current_name = node->GetName() + "_" + std::to_string(out_anchor->GetIdx()) + "_huawei_aipp";
  auto aipp_opdesc_ptr = MakeShared<OpDesc>(current_name, AIPP);
  if (aipp_opdesc_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed, name %s", current_name.c_str());
    return nullptr;
  }

  // Update attributes
  if (AddAippAttrbutes(aipp_opdesc_ptr, aippConfigPath, index) != SUCCESS) {
    return nullptr;
  }

  // Update input desc, the output desc will be flushed when InferShape
  auto node_desc = out_anchor->GetOwnerNode()->GetOpDesc();
  if (node_desc == nullptr) {
    return nullptr;
  }
  auto opdesc_src_data = node_desc->GetOutputDesc(out_anchor->GetIdx());
  if (opdesc_src_data.GetDataType() != DT_FLOAT) {
    GELOGW("The datatype of data node %s is not FP32", node_desc->GetName().c_str());
    opdesc_src_data.SetDataType(DT_FLOAT);
  }

  // We must get the TensorDesc from the output anchor on the Data node,
  // and update the TensorDesc to the input anchor on the Aipp node.
  // Because the InferShape function for the Aipp node needs the input tensor format,
  // but the InferFormat process before InferShape can not infer the format
  // if the tensor on the Aipp has an unknown shape
  if (aipp_opdesc_ptr->UpdateInputDesc(kAippImageInputIndex, opdesc_src_data) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Update the output desc from node:%s(%s) to aipp:%s(%s) failed",
                      node_desc->GetName().c_str(), node_desc->GetType().c_str(),
                      aipp_opdesc_ptr->GetName().c_str(), aipp_opdesc_ptr->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Call][UpdateInputDesc] Failed to update the output desc from node %s to aipp %s",
           node_desc->GetName().c_str(), aipp_opdesc_ptr->GetName().c_str());
    return nullptr;
  }

  return node->GetOwnerComputeGraph()->AddNode(aipp_opdesc_ptr);
}

Status AippOp::AddAippAttrbutes(const OpDescPtr &op_desc, const std::string &aipp_cfg_path, const uint32_t &index) {
  GeAttrValue::NAMED_ATTRS aipp_attr;
  ConvertParamToAttr(aipp_attr);
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr),
                         INTERNAL_ERROR, "[Set][NamedAttrs] %s for aipp node:%s failed", ATTR_NAME_AIPP.c_str(),
                         op_desc->GetName().c_str());

  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(op_desc, kAippConfigPath, aipp_cfg_path),
                         INTERNAL_ERROR, "[Set][Attr] config file path for aipp node:%s failed",
                         op_desc->GetName().c_str());

  std::vector<std::string> empty_names;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, empty_names),
                         INTERNAL_ERROR, "[Set][Attr] %s for aipp node:%s failed",
                         ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str(), op_desc->GetName().c_str());

  GE_CHK_BOOL_RET_STATUS(AttrUtils::SetInt(op_desc, kCurrentAippIndex, index),
                         INTERNAL_ERROR, "[Set][Attr] %s for aipp node:%s failed", kCurrentAippIndex,
                         op_desc->GetName().c_str());
  // add input/output desc
  GeTensorDesc tensor;
  GE_CHK_GRAPH_STATUS_RET(op_desc->AddInputDesc("images", tensor),
                          "[Add][InputDesc] images for aipp node:%s failed", op_desc->GetName().c_str());

  if (GetAippMode() == domi::AippOpParams::dynamic) {
    GE_CHK_GRAPH_STATUS_RET(op_desc->AddOptionalInputDesc("params", tensor),
                            "[Call][AddOptionalInputDesc] Failed to add params for aipp node:%s",
                            op_desc->GetName().c_str());
  }
  GE_CHK_GRAPH_STATUS_RET(op_desc->AddOutputDesc("features", tensor),
                          "[Add][OutputDesc] features for aipp node:%s failed", op_desc->GetName().c_str());

  return SUCCESS;
}

domi::AippOpParams::AippMode AippOp::GetAippMode() { return aipp_params_->aipp_mode(); }

NodePtr AippOp::FindDataByIndex(const ComputeGraphPtr &graph, int rank) {
  int64_t data_index = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      continue;
    }

    // For functional multi batch, Skip Data for index.
    if (node->GetOpDesc()->HasAttr(ATTR_INSERT_BY_MBATCH)) {
      continue;
    }

    // There is no `index` attribute on the `Data` node when compile in inference scene
    // so we can only use the order of all `Data` nodes to infer the data index
    if (data_index++ != rank) {
      continue;
    }
    return node;
  }
  string error_msg = "Can not find the data node by aipp parameter related_input_rank " + to_string(rank);
  GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
  return nullptr;
}
Status AippOp::GetAndCheckTarget(const ComputeGraphPtr &graph, int rank, NodePtr &target,
                                 std::set<uint32_t> &edge_indexes) {
  auto data_node = FindDataByIndex(graph, rank);
  if (data_node == nullptr) {
    GELOGE(PARAM_INVALID, "[Call][FindDataByIndex] Get target input node for rank %d failed", rank);
    return PARAM_INVALID;
  }
  data_node_linked_aipp = data_node;
  auto data_opdesc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_opdesc);
  string set_dt_str;
  if (ge::AttrUtils::GetStr(data_opdesc, ATTR_ATC_USER_DEFINE_DATATYPE, set_dt_str)) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10034", {"opname"}, {data_opdesc->GetName()});
    GELOGE(INTERNAL_ERROR,
           "[Get][Attr] This input op [%s] is linked to aipp, can not be set to fp16, "
           "please check your atc parameter --insert_op_conf, --input_fp16_nodes.",
           data_opdesc->GetName().c_str());
    return PARAM_INVALID;
  }

  // add dynamic or static attr memsage to data
  if (GetAippMode() == domi::AippOpParams::static_) {
    (void)AttrUtils::SetStr(data_opdesc, ATTR_DATA_RELATED_AIPP_MODE, "static_aipp");
  } else if (GetAippMode() == domi::AippOpParams::dynamic) {
    (void)AttrUtils::SetStr(data_opdesc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp");
  }

  // In scenario AIPP+CONV2D+POOLING, keep the aipp info to Data, since AIPP disappear after subgraph optimize
  GeAttrValue::NAMED_ATTRS aipp_attr;
  ConvertParamToAttr(aipp_attr);
  if (!AttrUtils::SetNamedAttrs(data_opdesc, ATTR_NAME_AIPP, aipp_attr)) {
    REPORT_INNER_ERROR("E19999", "Set Attr:%s for op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                       data_opdesc->GetName().c_str(), data_opdesc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s for op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
           data_opdesc->GetName().c_str(), data_opdesc->GetType().c_str());
    return INTERNAL_ERROR;
  }

  if (aipp_params_->input_edge_idx_size() > 0) {
    for (auto edge_index : aipp_params_->input_edge_idx()) {
      edge_indexes.insert(edge_index);
    }
  }

  if (!edge_indexes.empty() && (*edge_indexes.rbegin() >= data_node->GetOutDataNodes().size())) {
    string error_msg = "The aipp parameter input_edge_idx[" + std::to_string(*edge_indexes.rbegin()) +
        "] should be smaller than the target input[" +
        std::to_string(data_node->GetOutDataNodes().size()) +"]'s outnodes.";
    GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
    return PARAM_INVALID;
  }
  target = data_node;

  return GetStaticTargetNode(graph, data_node, target);
}

Status AippOp::GetStaticTargetNode(const ComputeGraphPtr &graph, NodePtr &data_node, NodePtr &target) {
  if (GetAippMode() != domi::AippOpParams::static_) {
    return SUCCESS;
  }

  std::string related_node_name;
  if (AttrUtils::GetStr(data_node->GetOpDesc(), kMbatchSwitchnName, related_node_name)) {
    if (related_node_name.empty()) {
      REPORT_INNER_ERROR("E19999", "The data node %s has switchn node flag, but the value of attr:%s is empty, "
                         "check invalid", data_node->GetName().c_str(),
                         kMbatchSwitchnName);
      GELOGE(INTERNAL_ERROR, "[Check][Param] The data node %s has switchn node flag, but the value of attr:%s is empty",
             data_node->GetName().c_str(), kMbatchSwitchnName);
      return INTERNAL_ERROR;
    }
    auto switchn = graph->FindNode(related_node_name);
    if (switchn == nullptr) {
      REPORT_INNER_ERROR("E19999", "The data node %s has switchn node %s, but can not find it on the graph, "
                         "check invalid", data_node->GetName().c_str(), related_node_name.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] The data node %s has switchn node %s, but can not find it on the graph",
             data_node->GetName().c_str(), related_node_name.c_str());
      return INTERNAL_ERROR;
    }
    target = switchn;
    GELOGI("Multi-batch/image size and static aipp for data %s, "
           "the aipp node will be insert after %s instead of origin data node",
           data_node->GetName().c_str(), switchn->GetName().c_str());

    return SUCCESS;
  }

  const auto out_anchor = data_node->GetOutDataAnchor(0);
  for (const auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
    if (in_anchor == nullptr) {
      continue;
    }

    const auto &case_node = in_anchor->GetOwnerNode();
    if (case_node->GetType() == CASE) {
      target = case_node;
      return SUCCESS;
    }
  }

  return SUCCESS;
}
Status AippOp::ConvertRelatedInputNameToRank() {
  GE_CHECK_NOTNULL(aipp_params_);

  string related_input_name = aipp_params_->related_input_name();
  if (related_input_name.empty()) {
    return SUCCESS;
  }

  std::vector<std::string> data_top_names = domi::GetContext().data_top_names;
  GELOGI("Convert name to rank start: data size[%zu]", data_top_names.size());
  uint32_t index = 0;
  bool convert_flag = false;
  for (const auto &data_top_name : data_top_names) {
    if (related_input_name == data_top_name) {
      aipp_params_->set_related_input_rank(index);
      convert_flag = true;
      GELOGI("AippOp: rank: %u, top name: %s.", index, data_top_name.c_str());
      break;
    }
    index++;
  }
  if (!convert_flag) {
    string error_msg = "Top name " + related_input_name + "convert rank failed, Please"
                       " ensure top name in aipp config is the top name of data node.";
    GELOGE(PARAM_INVALID, "[Check][InputParam]%s", error_msg.c_str());
    REPORT_INPUT_ERROR("E10052", std::vector<std::string>({"reason"}), std::vector<std::string>({error_msg}));
    return PARAM_INVALID;
  }

  return SUCCESS;
}


Status AippOp::GetTargetPosition(ComputeGraphPtr graph, NodePtr &target_input,
                                 std::vector<std::pair<OutDataAnchorPtr, InDataAnchorPtr>> &target_edges) {
  GE_CHECK_NOTNULL(graph);
  GE_CHECK_NOTNULL(aipp_params_);

  std::set<uint32_t> edge_indexes;
  const uint32_t related_input_rank = aipp_params_->related_input_rank();
  auto ret = GetAndCheckTarget(graph, related_input_rank, target_input, edge_indexes);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][TargetInputNode] for rank %u failed", related_input_rank);
    return ret;
  }

  target_edges.clear();
  if (target_input->GetType() != CASE) {
    for (OutDataAnchorPtr &src_out : target_input->GetAllOutDataAnchors()) {
      auto dst_ins = src_out->GetPeerInDataAnchors();
      for (uint32_t i = 0; i < dst_ins.size(); ++i) {
        auto dst_in = dst_ins.at(i);
        if (edge_indexes.empty() || edge_indexes.count(i) > 0) {
          target_edges.emplace_back(src_out, dst_in);
        }
      }
    }
  } else {
    const auto &func_desc = target_input->GetOpDesc();
    for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
      const auto &subgraph = graph->GetSubgraph(name);
      if (subgraph == nullptr) {
        REPORT_INNER_ERROR("E19999", "Subgraph:%s of op:%s(%s) not find in graph:%s, check invalid",
                           name.c_str(), func_desc->GetName().c_str(), func_desc->GetType().c_str(),
                           graph->GetName().c_str());
        GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][Subgraph] failed, Subgraph:%s of op:%s(%s) not find in graph:%s",
               name.c_str(), func_desc->GetName().c_str(), func_desc->GetType().c_str(), graph->GetName().c_str());
        return GE_GRAPH_EMPTY_SUBGRAPH;
      }

      auto data_node = FindDataByIndex(subgraph, related_input_rank);
      if (data_node == nullptr) {
        GELOGE(PARAM_INVALID, "[Get][TargetInputNode] for rank %d failed", related_input_rank);
        return PARAM_INVALID;
      }

      for (OutDataAnchorPtr &src_out : data_node->GetAllOutDataAnchors()) {
        auto dst_ins = src_out->GetPeerInDataAnchors();
        for (uint32_t i = 0; i < dst_ins.size(); ++i) {
          auto dst_in = dst_ins.at(i);
          if (edge_indexes.empty() || edge_indexes.count(i) > 0) {
            target_edges.emplace_back(src_out, dst_in);
          }
        }
      }
    }
  }

  return SUCCESS;
}

Status AippOp::SetDefaultParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::static_) {
    if (aipp_params_->csc_switch()) {
      SetCscDefaultValue();
    }

    SetDtcDefaultValue();

    GELOGI("parse aipp params:input_format:%s, csc_switch:%d.",
           domi::AippOpParams::InputFormat_Name(aipp_params_->input_format()).c_str(), aipp_params_->csc_switch());

    GELOGI("parse aipp params:mean_chn_0:%d, mean_chn_1:%d, mean_chn_2:%d, mean_chn_3:%d.", aipp_params_->mean_chn_0(),
           aipp_params_->mean_chn_1(), aipp_params_->mean_chn_2(), aipp_params_->mean_chn_3());

    GELOGI("parse aipp params:min_chn_0:%f, min_chn_1:%f, min_chn_2:%f.", aipp_params_->min_chn_0(),
           aipp_params_->min_chn_1(), aipp_params_->min_chn_2());

    GE_IF_BOOL_EXEC(!aipp_params_->crop(), aipp_params_->set_load_start_pos_h(0); aipp_params_->set_load_start_pos_w(0);
                    aipp_params_->set_crop_size_h(0); aipp_params_->set_crop_size_w(0););

    GE_IF_BOOL_EXEC(!aipp_params_->resize(), aipp_params_->set_resize_output_h(0);
                    aipp_params_->set_resize_output_w(0););

    GE_IF_BOOL_EXEC(!aipp_params_->padding(), aipp_params_->set_left_padding_size(0);
                    aipp_params_->set_right_padding_size(0); aipp_params_->set_top_padding_size(0);
                    aipp_params_->set_bottom_padding_size(0););
  }

  return SUCCESS;
}

Status AippOp::ValidateParams() {
  GE_CHECK_NOTNULL(aipp_params_);
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->aipp_mode() != domi::AippOpParams::undefined, PARAM_INVALID,
                          "When insert AIPP op, aipp_mode must be configured as static or dynamic ");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_0_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_1_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_2_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_2 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->var_reci_chn_3_size() <= 1, PARAM_INVALID,
                          "The parameter var_reci_chn_3 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r0c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r0c2 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r1c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r1c2 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c0_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c1_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->matrix_r2c2_size() <= 1, PARAM_INVALID,
                          "The parameter matrix_r2c2 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_0_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_1_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->output_bias_2_size() <= 1, PARAM_INVALID,
                          "The parameter output_bias_2 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_0_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_0 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_1_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_1 can not be configed repeatedly");
  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_bias_2_size() <= 1, PARAM_INVALID,
                          "The parameter input_bias_2 can not be configed repeatedly");

  GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_edge_idx_size() <= 1, PARAM_INVALID,
                          "The parameter input_edge_idx can not be configed repeatedly");

  const domi::AippOpParams::AippMode aipp_mode = aipp_params_->aipp_mode();
  if (aipp_mode == domi::AippOpParams::dynamic) {
    GE_CHK_LOG_AND_ERRORMSG(
        aipp_params_->max_src_image_size() > 0, PARAM_INVALID,
        "For dynamic AIPP params, max_src_image_size must be set which number should be greater than 0");
  } else {
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->input_format() != domi::AippOpParams::UNDEFINED, PARAM_INVALID,
                            "Input format of AIPP conf is undefined");

    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->src_image_size_w() >= 0, PARAM_INVALID,
                            "Src_image_size_w must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->src_image_size_h() >= 0, PARAM_INVALID,
                            "Src_image_size_h must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->load_start_pos_w() >= 0, PARAM_INVALID,
                            "Load_start_pos_w must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->load_start_pos_h() >= 0, PARAM_INVALID,
                            "Load_start_pos_h must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->crop_size_w() >= 0, PARAM_INVALID,
                            "Crop_size_w must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->resize_output_w() >= 0, PARAM_INVALID,
                            "Resize_output_w must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->resize_output_h() >= 0, PARAM_INVALID,
                            "Resize_output_h must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->left_padding_size() >= 0, PARAM_INVALID,
                            "Left_padding_size must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->right_padding_size() >= 0, PARAM_INVALID,
                            "Right_padding_size must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->top_padding_size() >= 0, PARAM_INVALID,
                            "Top_padding_size must not be configed smaller than 0");
    GE_CHK_LOG_AND_ERRORMSG(aipp_params_->bottom_padding_size() >= 0, PARAM_INVALID,
                            "Bottom_padding_size must not be configed smaller than 0");
  }

  return SUCCESS;
}

void AippOp::SetCscDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  if (aipp_params_->input_format() == domi::AippOpParams::YUV420SP_U8) {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(DEFAULT_MATRIX_R2C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(DEFAULT_MATRIX_R2C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(DEFAULT_MATRIX_R2C2_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(DEFAULT_MATRIX_R1C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(DEFAULT_MATRIX_R1C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(DEFAULT_MATRIX_R1C2_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(DEFAULT_MATRIX_R0C0_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(DEFAULT_MATRIX_R0C1_YUV2RGB));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(DEFAULT_MATRIX_R0C2_YUV2RGB));
  } else {
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c0_size() > 0, aipp_params_->add_matrix_r0c0(DEFAULT_MATRIX_R0C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c1_size() > 0, aipp_params_->add_matrix_r0c1(DEFAULT_MATRIX_R0C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r0c2_size() > 0, aipp_params_->add_matrix_r0c2(DEFAULT_MATRIX_R0C2_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c0_size() > 0, aipp_params_->add_matrix_r1c0(DEFAULT_MATRIX_R1C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c1_size() > 0, aipp_params_->add_matrix_r1c1(DEFAULT_MATRIX_R1C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r1c2_size() > 0, aipp_params_->add_matrix_r1c2(DEFAULT_MATRIX_R1C2_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c0_size() > 0, aipp_params_->add_matrix_r2c0(DEFAULT_MATRIX_R2C0_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c1_size() > 0, aipp_params_->add_matrix_r2c1(DEFAULT_MATRIX_R2C1_RGB2YUV));
    CHECK_FALSE_EXEC(aipp_params_->matrix_r2c2_size() > 0, aipp_params_->add_matrix_r2c2(DEFAULT_MATRIX_R2C2_RGB2YUV));
  }
  CHECK_FALSE_EXEC(aipp_params_->input_bias_0_size() > 0, aipp_params_->add_input_bias_0(DEFAULT_INPUT_BIAS_0));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_1_size() > 0, aipp_params_->add_input_bias_1(DEFAULT_INPUT_BIAS_1));
  CHECK_FALSE_EXEC(aipp_params_->input_bias_2_size() > 0, aipp_params_->add_input_bias_2(DEFAULT_INPUT_BIAS_2));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_0_size() > 0, aipp_params_->add_output_bias_0(DEFAULT_OUTPUT_BIAS_0));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_1_size() > 0, aipp_params_->add_output_bias_1(DEFAULT_OUTPUT_BIAS_1));
  CHECK_FALSE_EXEC(aipp_params_->output_bias_2_size() > 0, aipp_params_->add_output_bias_2(DEFAULT_OUTPUT_BIAS_2));
}

void AippOp::SetDtcDefaultValue() {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_0_size() > 0, aipp_params_->add_var_reci_chn_0(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_0 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_0_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_1_size() > 0, aipp_params_->add_var_reci_chn_1(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_1 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_1_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_2_size() > 0, aipp_params_->add_var_reci_chn_2(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_2 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_2_size());
  CHECK_FALSE_EXEC(aipp_params_->var_reci_chn_3_size() > 0, aipp_params_->add_var_reci_chn_3(DEFAULT_VAR_RECI_CHN));
  GELOGD("var_reci_chn_3 is %f, size is %u.", DEFAULT_VAR_RECI_CHN, aipp_params_->var_reci_chn_3_size());
}

Status AippOp::GenerateOpDesc(OpDescPtr op_desc) {
  GE_CHECK_NOTNULL(op_desc);

  static std::atomic_long atomic_op_idx(0);
  auto op_idx = atomic_op_idx.fetch_add(1);
  op_desc->SetName(std::string("aipp_node").append(std::to_string(op_idx)));
  op_desc->SetType(AIPP);

  // Add two InputDesc, add the second after the first one is added successfully.
  if ((op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS) ||
      (op_desc->AddInputDesc(GeTensorDesc()) != GRAPH_SUCCESS)) {
    REPORT_CALL_ERROR("E19999", "Add input desc into op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Add][InputDesc] into op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  if (op_desc->AddOutputDesc(GeTensorDesc()) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc into op:%s(%s) failed",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Add][OutputDesc] into op:%s(%s) failed",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }
  GeAttrValue::NAMED_ATTRS aipp_attrs;
  ConvertParamToAttr(aipp_attrs);

  GE_IF_BOOL_EXEC(!AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attrs),
                  REPORT_INNER_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_AIPP.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  return FAILED);

  return SUCCESS;
}

void AippOp::ConvertParamToAttr(GeAttrValue::NAMED_ATTRS &aipp_attrs) {
  GE_CHECK_NOTNULL_JUST_RETURN(aipp_params_);
  SAVE_AIPP_ATTR(aipp_mode, GeAttrValue::INT);
  SAVE_AIPP_ATTR(related_input_rank, GeAttrValue::INT);

  if (aipp_params_->aipp_mode() == domi::AippOpParams::static_) {
    SAVE_AIPP_ATTR(input_format, GeAttrValue::INT);
    SAVE_AIPP_ATTR(csc_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(crop, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(resize, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(load_start_pos_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(load_start_pos_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(crop_size_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(crop_size_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(resize, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(resize_output_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(resize_output_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(padding, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(left_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(right_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(top_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(bottom_padding_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(src_image_size_w, GeAttrValue::INT);
    SAVE_AIPP_ATTR(src_image_size_h, GeAttrValue::INT);
    SAVE_AIPP_ATTR(cpadding_value, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(rbuv_swap_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(ax_swap_switch, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(single_line_mode, GeAttrValue::BOOL);
    SAVE_AIPP_ATTR(mean_chn_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR(mean_chn_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR(mean_chn_2, GeAttrValue::INT);
    SAVE_AIPP_ATTR(mean_chn_3, GeAttrValue::INT);
    SAVE_AIPP_ATTR(min_chn_0, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(min_chn_1, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(min_chn_2, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR(min_chn_3, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_0, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_1, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_2, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(var_reci_chn_3, GeAttrValue::FLOAT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r0c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r1c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(matrix_r2c2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(output_bias_2, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_0, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_1, GeAttrValue::INT);
    SAVE_AIPP_ATTR_LIST(input_bias_2, GeAttrValue::INT);
  } else {
    SAVE_AIPP_ATTR(max_src_image_size, GeAttrValue::INT);
    SAVE_AIPP_ATTR(support_rotation, GeAttrValue::BOOL);
  }
}
Status AippOp::CreateAippData(const NodePtr &aipp_node) {
  GELOGD("Enter add aipp data node process.");
  // get previous node, it should be DATA
  auto data_node = aipp_node->GetInDataNodes().at(kAippImageInputIndex);
  auto data_op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);

  auto ori_data_format = GetAndCheckFormat();
  if (ori_data_format != FORMAT_NCHW && ori_data_format != FORMAT_NHWC) {
    string format_str = TypeUtils::FormatToSerialString(ori_data_format);
    GELOGE(PARAM_INVALID, "[Check][Param] when dynamic aipp, input_format must be NCHW or NHWC, but [%s] format is %s",
           data_node->GetName().c_str(), format_str.c_str());
    string reason = "format must be NCHW or NHWC in dynamic aipp process";
    ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
                                                    {data_node->GetName(), "format " + format_str, reason});
    return PARAM_INVALID;
  }

  // dynamic aipp shape HWC is not fixed, need to be set -1
  int64_t data_shape_n = 0;
  // dynamic batch or HW, need acquire N from ATTR_MBATCH_ORIGIN_INPUT_DIMS
  if (data_op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
    vector<int64_t> origin_input_dims;
    (void)AttrUtils::GetListInt(data_op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
    if (!origin_input_dims.empty()) {
      data_shape_n = origin_input_dims[0];
    }
  } else {
    data_shape_n = data_op_desc->MutableInputDesc(0)->GetShape().GetDim(0);
  }
  vector<int64_t> dynamic_aipp_linked_data_shape{data_shape_n, kDynamicDim, kDynamicDim, kDynamicDim};
  (void)AttrUtils::SetListInt(data_op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_linked_data_shape);

  int64_t batch_count = -1;
  if (GetDataDimN(data_node, ori_data_format, batch_count) != ge::SUCCESS) {
    string error_msg = "Get data_node dims and transfer to nchw_dims failed!";
    GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
    return PARAM_INVALID;
  }
  if (batch_count <= 0) {
    string error_msg = "Batch count[" + std::to_string(batch_count) + "] is invalid, it must positive.";
    GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
    return PARAM_INVALID;
  }

  int64_t max_dynamic_aipp_size = CalcMaxSize(batch_count);
  if (max_dynamic_aipp_size < 0) {
    string error_msg = "The dynamic aipp size is not positive";
    GE_ERRORLOG_AND_ERRORMSG(PARAM_INVALID, error_msg.c_str());
    return PARAM_INVALID;
  }

  GELOGI("Add aipp input data, batch count is %ld, max_dynamic_aipp_size is %ld", batch_count, max_dynamic_aipp_size);
  return AddNodeToGraph(aipp_node, max_dynamic_aipp_size);
}

Status AippOp::AddAttrToAippData(const OpDescPtr &aipp_data_op_desc) {
  // Add dynamic aipp config to aipp_data
  GeAttrValue::NAMED_ATTRS aipp_attr;
  ConvertParamToAttr(aipp_attr);
  (void)AttrUtils::SetNamedAttrs(aipp_data_op_desc, ATTR_NAME_AIPP, aipp_attr);
  (void)AttrUtils::SetStr(aipp_data_op_desc, ATTR_DATA_RELATED_AIPP_MODE, "dynamic_aipp_conf");

  // add node name attr to data linked aipp_data, it can be queried by acl.
  GE_CHECK_NOTNULL(data_node_linked_aipp);
  auto data_op_desc = data_node_linked_aipp->GetOpDesc();
  GE_CHECK_NOTNULL(data_op_desc);
  (void)AttrUtils::SetStr(data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, aipp_data_op_desc->GetName());
  (void)AttrUtils::SetStr(aipp_data_op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, data_op_desc->GetName());
  return SUCCESS;
}

Status AippOp::AddNodeToGraph(const NodePtr &aipp_node, int64_t max_dynamic_aipp_size) {
  std::vector<int64_t> input_shape_dim(1, max_dynamic_aipp_size);
  GeShape input_shape(input_shape_dim);
  // construct input tensor
  GeTensorDesc input_tensor(input_shape, FORMAT_ND, DT_UINT8);
  TensorUtils::SetReuseInput(input_tensor, false);
  TensorUtils::SetSize(input_tensor, max_dynamic_aipp_size);
  GE_CHECK_NOTNULL(aipp_node);
  const ComputeGraphPtr &graph = aipp_node->GetOwnerComputeGraph();
  string node_name;
  // First aippdata name should be definite.
  if (graph->FindFirstNodeMatchType(AIPPDATA) == nullptr) {
    GELOGI("Current graph has no aippdata node, so the name of it must be definite.");
    node_name = kDynamicAippData;
  } else {
    node_name = string(kDynamicAippData) + "_" + aipp_node->GetName();
  }
  GELOGI("Current add aippdata node name is %s", node_name.c_str());

  // new add aipp_data ops for dynamic aipp param input
  OpDescPtr op_desc_ptr_data = MakeShared<OpDesc>(node_name, AIPPDATA);
  GE_CHECK_NOTNULL(op_desc_ptr_data);

  if (AddAttrToAippData(op_desc_ptr_data) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  auto stat1 = op_desc_ptr_data->AddInputDesc(input_tensor);

  GeShape output_shape(input_shape_dim);
  // construct output tensor
  GeTensorDesc output_tensor(output_shape, FORMAT_ND, DT_UINT8);
  TensorUtils::SetReuseInput(output_tensor, false);
  TensorUtils::SetSize(output_tensor, max_dynamic_aipp_size);
  auto stat2 = op_desc_ptr_data->AddOutputDesc(output_tensor);

  NodePtr aipp_data_node_ptr = graph->AddNode(op_desc_ptr_data);
  GE_CHECK_NOTNULL(aipp_data_node_ptr);

  // add node desc for aipp node
  auto stat3 = aipp_node->GetOpDesc()->UpdateInputDesc(kAippParamsInputIndex, output_tensor);
  if (stat1 != GRAPH_SUCCESS || stat2 != GRAPH_SUCCESS || stat3 != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add and Update InputDesc to op:%s(%s) failed, index:%d",
                      aipp_node->GetName().c_str(), aipp_node->GetType().c_str(), kAippParamsInputIndex);
    GELOGE(INTERNAL_ERROR, "[Update][InputDesc] to op:%s(%s) failed, index:%d",
           aipp_node->GetName().c_str(), aipp_node->GetType().c_str(), kAippParamsInputIndex);
    return INTERNAL_ERROR;
  }
  // aipp_node should have two input data but now tbe only one input
  if (GraphUtils::AddEdge(aipp_data_node_ptr->GetOutDataAnchor(kAippDataOutputIndex),
                          aipp_node->GetInDataAnchor(kAippParamsInputIndex)) != GRAPH_SUCCESS) {
    REPORT_INNER_ERROR("E19999", "Add edge between op:%s(%s)(out_index:%u) and op:%s(%s)(in_index:%u) failed",
                       aipp_data_node_ptr->GetName().c_str(), aipp_data_node_ptr->GetType().c_str(),
                       kAippDataOutputIndex, aipp_node->GetName().c_str(), aipp_node->GetType().c_str(),
                       kAippParamsInputIndex);
    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(out_index:%u) and op:%s(%s)(in_index:%u) failed",
           aipp_data_node_ptr->GetName().c_str(), aipp_data_node_ptr->GetType().c_str(),
           kAippDataOutputIndex, aipp_node->GetName().c_str(), aipp_node->GetType().c_str(),
           kAippParamsInputIndex);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}
}  // namespace ge
