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

#include "graph/model_serialize.h"
#include <google/protobuf/text_format.h>

#include <queue>
#include <iostream>

#include "graph/debug/ge_attr_define.h"
#include "proto/ge_ir.pb.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/ge_tensor_impl.h"
#include "graph/compute_graph_impl.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "proto/ge_ir.pb.h"
#include "graph/utils/graph_utils.h"
#include "debug/ge_op_types.h"
#include "utils/mem_utils.h"

namespace ge {
bool ModelSerializeImp::ParseNodeIndex(const std::string &node_index, std::string &node_name, int32_t &index) const {
  const auto sep = node_index.rfind(":");
  if (sep == std::string::npos) {
    GELOGW("[Parse][CheckParam] Separator \":\" is not found in node_index.");
    return false;
  }
  node_name = node_index.substr(0UL, sep);
  const auto index_str = node_index.substr(sep + 1UL);
  index = static_cast<int32_t>(std::strtol(index_str.c_str(), nullptr, 10));
  return true;
}

bool ModelSerializeImp::SerializeEdge(const NodePtr &node, proto::OpDef *op_def_proto) const {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return false, "[Check][Param] node is null.");
  GE_CHK_BOOL_EXEC(op_def_proto != nullptr, REPORT_INNER_ERROR("E19999", "param op_def_proto is null, check invalid.");
                   return false, "[Check][Param] op_def_proto is null.");

  op_def_proto->clear_input();
  // Inputs
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    if (in_data_anchor != nullptr) {
      const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode()) {
        op_def_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":" +
                                std::to_string(peer_out_anchor->GetIdx()));
      } else {
        op_def_proto->add_input("");
      }
    }
  }
  // Control edge
  const auto in_control_anchor = node->GetInControlAnchor();
  if (in_control_anchor != nullptr) {
    const auto peer_out_anchors = in_control_anchor->GetPeerOutControlAnchors();
    for (const auto &peer_out_anchor : peer_out_anchors) {
      if (peer_out_anchor != nullptr && peer_out_anchor->GetOwnerNode()) {
        op_def_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":-1");
      }
    }
  }
  return true;
}

void ModelSerializeImp::FixOpDefSubgraphInstanceName(const ConstOpDescPtr &op_desc) const {
  const size_t op_def_subgraph_name_size = op_desc->impl_->meta_data_.GetSubgraphNames().size();
  const size_t op_desc_subgraph_name_size = op_desc->GetSubgraphInstanceNames().size();
  if (op_def_subgraph_name_size == op_desc_subgraph_name_size) {
    return;
  }

  if (op_def_subgraph_name_size == 0UL) {
    for (const std::string &name : op_desc->GetSubgraphInstanceNames()) {
      op_desc->impl_->meta_data_.AddSubGraphName(name);
    }
  }
}

bool ModelSerializeImp::SerializeOpDesc(const ConstOpDescPtr &op_desc, proto::OpDef *op_def_proto,
                                        const bool is_dump) {
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_INNER_ERROR("E19999", "param op_desc is nullptr. check invalid.");
                   return false, "[Check][Param] op_desc is null.");
  GE_CHK_BOOL_EXEC(op_def_proto != nullptr, REPORT_INNER_ERROR("E19999", "param op_def_proto is null, check invalid.");
                   return false, "[Check][Param] op_def_proto is null.");
  GE_CHK_BOOL_EXEC(op_desc->impl_ != nullptr,
                   REPORT_INNER_ERROR("E19999", "param op_desc impl is null, check invalid.");
                   return false, "[Check][Param] op_desc impl is null.");

  FixOpDefSubgraphInstanceName(op_desc);
  op_desc->impl_->SerializeMetaDataToOpDef(op_def_proto);
  // Delete unnecessary attr
  op_def_proto->clear_input_desc();
  op_def_proto->clear_output_desc();
  // Input descs
  if (op_desc->GetAllInputsSize() > 0UL) {
    const auto size = static_cast<uint32_t>(op_desc->GetAllInputsSize());
    for (uint32_t i = 0U; i < size; i++) {
      const auto tensor_desc = op_desc->GetInputDescPtrDfault(i);
      if (tensor_desc != nullptr && tensor_desc->impl_ != nullptr) {
        GeTensorSerializeUtils::GeTensorDescAsProto(*tensor_desc, op_def_proto->add_input_desc());
      }
    }
  }
  // Output descs
  if (op_desc->GetOutputsSize() > 0UL) {
    const auto size = static_cast<uint32_t>(op_desc->GetOutputsSize());
    for (uint32_t i = 0U; i < size; i++) {
      const auto tensor_desc = op_desc->GetOutputDescPtr(i);
      if ((tensor_desc != nullptr) && (tensor_desc->impl_ != nullptr)) {
        GeTensorSerializeUtils::GeTensorDescAsProto(*tensor_desc, op_def_proto->add_output_desc());
      }
    }
  }

  op_def_proto->set_id(op_desc->GetId());
  OpDescToAttrDef(op_desc, op_def_proto, is_dump);

  return true;
}

void ModelSerializeImp::OpDescToAttrDef(const ConstOpDescPtr &op_desc, proto::OpDef *op_def_proto,
                                        const bool is_dump) const {
  proto::AttrDef key_in;
  proto::AttrDef value_in;
  auto op_desc_attr = op_def_proto->mutable_attr();
  if (op_desc == nullptr || op_desc->impl_ == nullptr) {
    GELOGE(FAILED, "[Check][Param] op desc or impl is nullptr.");
    return;
  }
  if (!op_desc->impl_->input_name_idx_.empty()) {
    for (auto &item : op_desc->impl_->input_name_idx_) {
      key_in.mutable_list()->add_s(item.first);
      value_in.mutable_list()->add_i(static_cast<int64_t>(item.second));
    }
    (void) op_desc_attr->insert({"_input_name_key", key_in});
    (void) op_desc_attr->insert({"_input_name_value", value_in});
  }
  proto::AttrDef key_out;
  proto::AttrDef value_out;
  if (!op_desc->impl_->output_name_idx_.empty()) {
    for (auto &item : op_desc->impl_->output_name_idx_) {
      key_out.mutable_list()->add_s(item.first);
      value_out.mutable_list()->add_i(static_cast<int64_t>(item.second));
    }
    (void) op_desc_attr->insert({"_output_name_key", key_out});
    (void) op_desc_attr->insert({"_output_name_value", value_out});
  }
  proto::AttrDef opt_input;
  if (!op_desc->impl_->optional_input_names_.empty()) {
    for (auto &item : op_desc->impl_->optional_input_names_) {
      opt_input.mutable_list()->add_s(item);
    }
    (*op_desc_attr)["_opt_input"] = opt_input;
  }

  if (!SerializeAllAttrsFromAnyMap(op_desc->GetAllAttrs(), op_desc_attr)) {
    GELOGE(GRAPH_FAILED, "OpDesc [%s] attr serialize failed.", op_desc->GetName().c_str());
    return;
  }

  if (is_dump) {
    (void) op_desc_attr->erase(ATTR_NAME_FRAMEWORK_NODE_DEF);
    (void) op_desc_attr->erase(ATTR_NAME_FRAMEWORK_OP_DEF);
    (void) op_desc_attr->erase(ATTR_NAME_FRAMEWORK_FUNC_DEF);
    GE_IF_BOOL_EXEC(((op_def_proto->type() == CONSTANT) || (op_def_proto->type() == CONSTANTOP)),
                    (void) op_desc_attr->erase(ATTR_NAME_WEIGHTS));
  }
}

bool ModelSerializeImp::SerializeNode(const NodePtr &node, proto::OpDef *op_def_proto, const bool is_dump) {
  if ((node == nullptr) || (op_def_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or op_def_proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] param node or op_def_proto is nullptr, check invalid.");
    return false;
  }
  if (!SerializeOpDesc(node->GetOpDesc(), op_def_proto, is_dump)) {
    GELOGE(GRAPH_FAILED, "[Serialize][OpDesc] failed, node:%s", node->GetName().c_str());
    return false;
  }
  if (SerializeEdge(node, op_def_proto)) {
    return true;
  } else {
    return false;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::SerializeGraph(const ConstComputeGraphPtr &graph,
                                                                                      proto::GraphDef *graph_proto,
                                                                                      const bool is_dump) {
  if ((graph == nullptr) || (graph_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param graph or graph_proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] param graph or graph_proto is nullptr, check invalid.");
    return false;
  }
  graph_proto->set_name(graph->GetName());
  // Inputs
  for (const auto &input : graph->GetInputNodes()) {
    if (input != nullptr) {
      graph_proto->add_input(input->GetName() + ":0");
    }
  }
  // Outputs
  for (const auto &output : graph->GetGraphOutNodesInfo()) {
    if (output.first != nullptr) {
      graph_proto->add_output(output.first->GetName() + ":" + std::to_string(output.second));
      GELOGI("Add output to graph proto, node name:%s, index:%d", output.first->GetName().c_str(), output.second);
    }
  }
  // ComputeGraph中的属性序列化
  if (!SerializeAllAttrsFromAnyMap(graph->GetAllAttrs(), graph_proto->mutable_attr())) {
    GELOGE(GRAPH_FAILED, "ComputeGraph [%s] serialize attr failed.", graph->GetName().c_str());
    return false;
  }

  for (const auto &node : graph->GetDirectNode()) {
    if (!SerializeNode(node, graph_proto->add_op(), is_dump)) {
      if (node->GetOpDesc() != nullptr) {
        REPORT_CALL_ERROR("E19999", "op desc of node:%s is nullptr.", node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Get][OpDesc] Serialize Node %s failed as node opdesc is null", node->GetName().c_str());
      }
      return false;
    }
  }
  return true;
}

bool ModelSerializeImp::SerializeModel(const Model &model, proto::ModelDef *model_proto, const bool is_dump) {
  if (model_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param model_proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] param model_proto is nullptr, check Invalid");
    return false;
  }
  model_proto->set_name(model.GetName());
  model_proto->set_custom_version(model.GetPlatformVersion());
  model_proto->set_version(model.GetVersion());

  // Model属性序列化
  if (!SerializeAllAttrsFromAnyMap(model.GetAllAttrs(), model_proto->mutable_attr())) {
    GELOGE(GRAPH_FAILED, "Model [%s] serialize attr failed.", model.GetName().c_str());
    return false;
  }

  auto &graph = model.graph_;
  const auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "get compute graph from graph failed as graph is invalid.");
    GELOGE(GRAPH_FAILED, "[Get][ComputeGraph] return nullptr");
    return false;
  }
  if (!SerializeGraph(compute_graph, model_proto->add_graph(), is_dump)) {
    GELOGE(GRAPH_FAILED, "[Serialize][Graph] failed");
    return false;
  }

  for (const auto subgraph : compute_graph->GetAllSubgraphs()) {
    if (!SerializeGraph(subgraph, model_proto->add_graph(), is_dump)) {
      GELOGE(GRAPH_FAILED, "[Serialize][Subgraph] failed");
      return false;
    }
  }

  return true;
}

void ModelSerializeImp::AttrDefToOpDesc(OpDescPtr &op_desc,
                                        std::vector<std::string> &key_in,
                                        std::vector<std::string> &key_out,
                                        std::vector<uint32_t> &value_in,
                                        std::vector<uint32_t> &value_out,
                                        std::vector<std::string> &opt_input) const {
  if ((op_desc == nullptr) || (op_desc->impl_ == nullptr)) {
    GELOGE(FAILED, "[Serialize][Opdesc] op desc or impl is nullptr.");
    return;
  }
  if (!key_in.empty()) {
    if (key_in.size() != value_in.size()) {
      GELOGW("[ParseAttrDef][CheckParam] Input key and value vector size is different. key_size=%zu, value_size=%zu.",
             key_in.size(), value_in.size());
    } else {
      for (size_t i = 0UL; i < key_in.size(); ++i) {
        (void) op_desc->impl_->input_name_idx_.insert(std::pair<std::string, uint32_t>(key_in.at(i), value_in.at(i)));
      }
    }
  }
  if (!key_out.empty()) {
    if (key_out.size() != value_out.size()) {
      GELOGW("[ParseAttrDef][CheckParam] Output key and value vector size is different. key_size=%zu, value_size=%zu.",
             key_out.size(), value_out.size());
    } else {
      for (size_t i = 0UL; i < key_out.size(); ++i) {
        (void)op_desc->impl_->output_name_idx_.insert(std::pair<std::string, uint32_t>(key_out.at(i), value_out.at(i)));
      }
    }
  }
  if (!opt_input.empty()) {
    for (const auto &i : opt_input) {
      (void) op_desc->impl_->optional_input_names_.insert(i);
    }
  }
}

bool ModelSerializeImp::UnserializeOpDesc(OpDescPtr &op_desc, proto::OpDef &op_def_proto) {
  std::vector<std::string> opt_input;
  std::vector<std::string> key_in;
  std::vector<uint32_t> value_in;
  std::vector<std::string> key_out;
  std::vector<uint32_t> value_out;

  ExtractMetaDataAttr(op_def_proto, opt_input, key_in, value_in, key_out, value_out);

  op_desc = ComGraphMakeShared<OpDesc>(op_def_proto);
  GE_CHK_BOOL_EXEC(op_desc != nullptr, REPORT_CALL_ERROR("E19999", "create OpDesc failed.");
                   return false, "[Create][OpDesc] op_desc is nullptr.");
  GE_CHK_BOOL_EXEC(op_desc->impl_ != nullptr, REPORT_CALL_ERROR("E19999", "create OpDesc impl failed.");
                   return false, "[Create][OpDesc] op_desc impl is nullptr.");
  // Input tensor
  for (auto &input_desc : *op_def_proto.mutable_input_desc()) {
    const std::shared_ptr<GeTensorDesc> temp_value =
        std::shared_ptr<GeTensorDesc>(new (std::nothrow) GeTensorDesc(protobuf_owner_, &input_desc));
    GE_CHK_BOOL_EXEC(temp_value != nullptr, REPORT_CALL_ERROR("E19999", "create GeTensorDesc failed.");
                     return false, "[Create][GeTensorDesc] temp_value is nullptr.");
    op_desc->impl_->inputs_desc_.push_back(temp_value);
  }
  // Output tensor
  for (auto &output_desc : *op_def_proto.mutable_output_desc()) {
    const std::shared_ptr<GeTensorDesc> temp_value =
        std::shared_ptr<GeTensorDesc>(new (std::nothrow) GeTensorDesc(protobuf_owner_, &output_desc));
    GE_CHK_BOOL_EXEC(temp_value != nullptr, REPORT_CALL_ERROR("E19999", "create GeTensorDesc failed.");
                     return false, "[Create][GeTensorDesc] temp_value is nullptr.");
    op_desc->impl_->outputs_desc_.push_back(temp_value);
  }

  op_desc->SetId(op_def_proto.id());
  uint32_t graph_index = 0U;
  for (const std::string &name : op_def_proto.subgraph_name()) {
    (void) op_desc->AddSubgraphName(name);
    (void) op_desc->SetSubgraphInstanceName(graph_index++, name);
  }

  // insert name index by key and value
  AttrDefToOpDesc(op_desc, key_in, key_out, value_in, value_out, opt_input);

  if (!DeserializeAllAttrsToAttrHolder(op_def_proto.attr(), op_desc.get())) {
    GELOGE(GRAPH_FAILED, "Opdesc [%s] attr deserialize failed", op_def_proto.name().c_str());
    return false;
  }

  return true;
}
void ModelSerializeImp::ExtractMetaDataAttr(proto::OpDef &op_def_proto, std::vector<std::string> &opt_input,
                                            std::vector<std::string> &key_in, std::vector<uint32_t> &value_in,
                                            std::vector<std::string> &key_out, std::vector<uint32_t> &value_out) const {
  if (op_def_proto.attr().count("_opt_input") > 0UL) {
    const auto &name_list = op_def_proto.attr().at("_opt_input").list();
    for (const auto &item_s : name_list.s()) {
      opt_input.push_back(item_s);
    }
    (void) op_def_proto.mutable_attr()->erase("_opt_input");
  }
  if (op_def_proto.attr().count("_input_name_key") > 0UL) {
    const auto &output_name_key_list = op_def_proto.attr().at("_input_name_key").list();
    for (const auto &item_s : output_name_key_list.s()) {
      key_in.push_back(item_s);
    }
    (void) op_def_proto.mutable_attr()->erase("_input_name_key");
  }
  if (op_def_proto.attr().count("_input_name_value") > 0UL) {
    const auto &input_name_value_list = op_def_proto.attr().at("_input_name_value").list();
    for (const auto &item_i : input_name_value_list.i()) {
      value_in.push_back(static_cast<uint32_t>(item_i));
    }
    (void) op_def_proto.mutable_attr()->erase("_input_name_value");
  }
  if (op_def_proto.attr().count("_output_name_key") > 0UL) {
    const auto &output_name_key_list = op_def_proto.attr().at("_output_name_key").list();
    for (const auto &item_s : output_name_key_list.s()) {
      key_out.push_back(item_s);
    }
    (void) op_def_proto.mutable_attr()->erase("_output_name_key");
  }
  if (op_def_proto.attr().count("_output_name_value") > 0UL) {
    const auto &output_name_value_list = op_def_proto.attr().at("_output_name_value").list();
    for (const auto &item_i : output_name_value_list.i()) {
      value_out.push_back(static_cast<uint32_t>(item_i));
    }
    (void) op_def_proto.mutable_attr()->erase("_output_name_value");
  }
}

bool ModelSerializeImp::UnserializeNode(ComputeGraphPtr &graph, proto::OpDef &op_def_proto) {
  GE_RT_FALSE_CHECK_NOTNULL(graph);
  OpDescPtr op_desc = nullptr;
  if (!UnserializeOpDesc(op_desc, op_def_proto)) {
    GELOGE(ge::INTERNAL_ERROR, "[Unserialize][OpDesc] error.");
    return false;
  }

  const NodePtr node = graph->AddNode(op_desc, op_desc->GetId());
  GE_CHK_BOOL_EXEC(node != nullptr,
                   REPORT_CALL_ERROR("E19999", "add node to graph:%s failed", graph->GetName().c_str());
                   return false, "[Add][Node] to graph:%s failed.", graph->GetName().c_str());

  // Inputs
  int32_t dst_index = 0;
  for (const auto &input : op_def_proto.input()) {
    std::string node_name;
    int32_t index = 0;
    if (ParseNodeIndex(input, node_name, index)) {
      node_input_node_names_.push_back(NodeNameNodeReq{node_name, index, node, dst_index, op_def_proto.name()});
    }
    if (index >= 0) {
      dst_index++;
    }
  }
  node_map_[op_def_proto.name()] = node;
  return true;
}

bool ModelSerializeImp::HandleNodeNameRef() {
  // Edges
  for (auto &item : node_input_node_names_) {
    const auto src_node_it = node_map_.find(item.src_node_name);
    if (src_node_it == node_map_.end()) {
      REPORT_INNER_ERROR("E19999", "cannot find edge node %s", item.src_node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] cannot find edge node %s", item.src_node_name.c_str());
      return false;
    }
    GE_IF_BOOL_EXEC(src_node_it->second == nullptr || item.dst_node == nullptr, continue);
    if (item.src_out_index >= 0) {
      const auto src_anchor = src_node_it->second->GetOutDataAnchor(item.src_out_index);
      const auto dst_anchor = item.dst_node->GetInDataAnchor(item.dst_in_index);
      if (src_anchor == nullptr || dst_anchor == nullptr) {
        REPORT_CALL_ERROR("E19999", "get Anchor failed %s:%d, %s:%d ", item.src_node_name.c_str(), item.src_out_index,
                          item.dst_node_name.c_str(), item.dst_in_index);
        GELOGE(GRAPH_FAILED, "[Get][Anchor] failed %s:%d, %s:%d ", item.src_node_name.c_str(), item.src_out_index,
               item.dst_node_name.c_str(), item.dst_in_index);
        return false;
      }
      GE_CHK_BOOL_ONLY_LOG((src_anchor->LinkTo(dst_anchor) == GRAPH_SUCCESS), " linkTo failed.");
    } else {
      // Control edge
      const auto src_anchor = src_node_it->second->GetOutControlAnchor();
      const auto dst_anchor = item.dst_node->GetInControlAnchor();
      if (src_anchor != nullptr && dst_anchor != nullptr) {
        GE_CHK_BOOL_ONLY_LOG((src_anchor->LinkTo(dst_anchor) == GRAPH_SUCCESS), " linkTo failed.");
      }
    }
  }
  // Graph input
  for (auto &item : graph_input_node_names_) {
    const auto node_it = node_map_.find(item.node_name);
    if (node_it == node_map_.end()) {
      REPORT_INNER_ERROR("E19999", "cannot find graph input node %s", item.node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] cannot find graph input node %s", item.node_name.c_str());
      return false;
    }
    GE_IF_BOOL_EXEC(item.graph == nullptr, continue);
    const auto ret = item.graph->AddInputNode(node_it->second);
    if (ret == nullptr) {
      return false;
    }
  }
  // Graph output
  for (auto &item : graph_output_node_names_) {
    const auto node_it = node_map_.find(item.node_name);
    if (node_it == node_map_.end()) {
      REPORT_INNER_ERROR("E19999", "cannot find graph output node %s", item.node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] cannot find graph output node %s", item.node_name.c_str());
      return false;
    }

    GE_IF_BOOL_EXEC(item.graph == nullptr, continue);
    const auto ret = item.graph->AddOutputNodeByIndex(node_it->second, item.index);
    GELOGI("node name:%s, item.index:%d", node_it->second->GetName().c_str(), item.index);
    if (ret == nullptr) {
      REPORT_CALL_ERROR("E19999", "add output node to graph:%s failed.", item.graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputNode] to graph:%s failed.", item.graph->GetName().c_str());
      return false;
    }
  }
  node_input_node_names_.clear();
  graph_input_node_names_.clear();
  graph_output_node_names_.clear();
  node_map_.clear();
  return true;
}

bool ModelSerializeImp::RebuildOwnership(ComputeGraphPtr &compute_graph,
                                         std::map<std::string, ComputeGraphPtr> &subgraphs) const {
  std::queue<ComputeGraphPtr> all_graphs;
  all_graphs.emplace(compute_graph);
  while (!all_graphs.empty()) {
    const ComputeGraphPtr graph = all_graphs.front();
    all_graphs.pop();

    for (const NodePtr &node : graph->GetDirectNode()) {
      const OpDescPtr op_desc = node->GetOpDesc();
      for (const std::string &name : op_desc->GetSubgraphInstanceNames()) {
        const auto it = subgraphs.find(name);
        if (it == subgraphs.end()) {
          REPORT_INNER_ERROR("E19999", "Node:%s, Subgraph:%s not found, num:%zu.",
                             op_desc->GetName().c_str(), name.c_str(), subgraphs.size());
          GELOGE(GRAPH_FAILED, "[Check][Param] Node:%s, Subgraph:%s not found, num:%zu.",
                 op_desc->GetName().c_str(), name.c_str(), subgraphs.size());
          return false;
        }

        ComputeGraphPtr &subgraph = it->second;
        subgraph->SetParentGraph(graph);
        subgraph->SetParentNode(node);
        (void) compute_graph->AddSubgraph(subgraph->GetName(), subgraph);
        all_graphs.emplace(subgraph);
      }
    }
  }

  return true;
}

bool ModelSerializeImp::UnserializeModel(Model &model, proto::ModelDef &model_proto) {
  model.name_ = model_proto.name();
  model.version_ = model_proto.version();
  model.platform_version_ = model_proto.custom_version();

  // Model属性反序列化
  if (!DeserializeAllAttrsToAttrHolder(model_proto.attr(), &model)) {
    GELOGE(GRAPH_FAILED, "Model [%s] deserialize attr failed.", model.GetName().c_str());
    return false;
  }

  auto &graphs_proto = *model_proto.mutable_graph();
  if (!graphs_proto.empty()) {
    auto &graph_proto = graphs_proto[0];
    ComputeGraphPtr compute_graph_ptr;
    if (UnserializeGraphWithoutEdge(compute_graph_ptr, graph_proto)) {
      model.graph_ = GraphUtils::CreateGraphFromComputeGraph(compute_graph_ptr);
    }

    // 0 is main graph, following is subgraph.
    std::map<std::string, ComputeGraphPtr> subgraphs;
    for (auto idx = 1; idx < graphs_proto.size(); ++idx) {
      ComputeGraphPtr subgraph;
      ModelSerializeImp impl;
      if (!impl.UnserializeGraphWithoutEdge(subgraph, graphs_proto[idx])) {
        GELOGE(GRAPH_FAILED, "[Call][UnserializeGraphWithoutEdge] failed");
        return false;
      }

      if (!impl.HandleNodeNameRef()) {
        GELOGE(GRAPH_FAILED, "[Call][HandleNodeNameRef] failed");
        return false;
      }

      subgraphs[subgraph->GetName()] = subgraph;
    }

    if (!subgraphs.empty()) {
      if (!RebuildOwnership(compute_graph_ptr, subgraphs)) {
        GELOGE(GRAPH_FAILED, "[Rebuild][GraphOwnerShip] failed");
        return false;
      }
    }
  }

  if (!HandleNodeNameRef()) {
    GELOGE(GRAPH_FAILED, "[Call][HandleNodeNameRef] failed");
    return false;
  }
  return true;
}

bool ModelSerializeImp::UnserializeGraphWithoutEdge(ComputeGraphPtr &graph, proto::GraphDef &graph_proto) {
  graph = ComGraphMakeShared<ComputeGraph>(graph_proto.name());
  if (graph == nullptr || graph->impl_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "create ComputeGraph failed.");
    GELOGE(GRAPH_FAILED, "[Create][ComputeGraph] ComputeGraph make shared failed");
    return false;
  }

  // Inputs
  for (const auto &input : graph_proto.input()) {
    std::string node_name;
    int32_t index;
    if (ParseNodeIndex(input, node_name, index)) {
      graph_input_node_names_.push_back(NodeNameGraphReq{node_name, index, graph});
    }
  }
  // Outputs
  for (const auto &output : graph_proto.output()) {
    std::string node_name;
    int32_t index;
    if (ParseNodeIndex(output, node_name, index)) {
      graph_output_node_names_.push_back(NodeNameGraphReq{node_name, index, graph});
    }
  }
  // ComputeGraph 属性反序列化
  if (!DeserializeAllAttrsToAttrHolder(graph_proto.attr(), graph.get())) {
    GELOGE(GRAPH_FAILED, "ComputeGraph [%s] deserialize attr failed.", graph->GetName().c_str());
    return false;
  }

  for (auto &op_def_proto : *graph_proto.mutable_op()) {
    if (!UnserializeNode(graph, op_def_proto)) {
      GELOGE(GRAPH_FAILED, "[Unserialize][Node] failed");
      return false;
    }
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::UnserializeGraph(ComputeGraphPtr &graph,
                                                                                        proto::GraphDef &graph_proto) {
  if (!UnserializeGraphWithoutEdge(graph, graph_proto)) {
    GELOGW("[Deserialize][Graph] Deserialize graph without edges failed");
  }
  if (!HandleNodeNameRef()) {
    GELOGE(GRAPH_FAILED, "[Call][HandleNodeNameRef] Link Anchor or set graph input or output fail");
    return false;
  }
  return true;
}

static bool ReadProtoFromBinaryFile(const uint8_t *const data, const size_t len, google::protobuf::Message *proto) {
  GE_CHK_BOOL_EXEC(data != nullptr, REPORT_INNER_ERROR("E19999", "param data is nullptr, check invalid.");
                   return false, "[Check][Param] data is null.");
  GE_CHK_BOOL_EXEC(proto != nullptr, REPORT_INNER_ERROR("E19999", "param proto is nullptr, check invalid.");
                   return false, "[Check][Param] proto is null.");

  google::protobuf::io::CodedInputStream coded_stream(data, static_cast<int32_t>(len));
  // 2048M -1
  coded_stream.SetTotalBytesLimit(INT32_MAX, -1);
  if (!proto->ParseFromCodedStream(&coded_stream)) {
    REPORT_CALL_ERROR("E19999", "Read proto from BinaryFile failed, len %zu", len);
    GELOGE(GRAPH_FAILED, "[Read][Proto] from BinaryFile failed, len %zu", len);
    return false;
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::SerializeAllAttrsFromAnyMap(
    const std::map<std::string, AnyValue> &attr_map,
    google::protobuf::Map<std::string, ::ge::proto::AttrDef> * mutable_attr) {
  if (mutable_attr == nullptr) {
    GELOGE(GRAPH_FAILED, "mutable_attr is nullptr.");
    return false;
  }

  for (const auto &attr : attr_map) {
    const AnyValue attr_value = attr.second;
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr_value.GetValueTypeId());
    if (serializer == nullptr) {
      GELOGE(GRAPH_FAILED, "Get serialized failed,name:[%s] value type:%u.",
             attr.first.c_str(), attr_value.GetValueType());
      return false;
    }
    proto::AttrDef attr_def;
    if (serializer->Serialize(attr_value, attr_def) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Attr serialized failed, name:[%s].", attr.first.c_str());
      return false;
    }
    (*mutable_attr)[attr.first] = attr_def;
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool ModelSerializeImp::DeserializeAllAttrsToAttrHolder(
    const google::protobuf::Map<std::string, ::ge::proto::AttrDef> &proto_attr_map, AttrHolder *attr_holder) {
  if (attr_holder == nullptr) {
    return false;
  }
  for (const auto &iter : proto_attr_map) {
    // skip not set attribute
    if ((iter.second.value_case() == proto::AttrDef::VALUE_NOT_SET) ||
        ((iter.second.value_case() == proto::AttrDef::kList) &&
            (iter.second.list().val_type() == ge::proto::AttrDef::ListValue::VT_LIST_NONE))) {
      continue;
    }

    const auto deserializer =
        AttrSerializerRegistry::GetInstance().GetDeserializer(iter.second.value_case());
    if (deserializer == nullptr) {
      GELOGE(GRAPH_FAILED, "Get deserialize failed, attr type:[%d].", static_cast<int32_t>(iter.second.value_case()));
      return false;
    }
    AnyValue attr_value;
    if (deserializer->Deserialize(iter.second, attr_value) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Attr deserialized failed, name:[%s].", iter.first.c_str());
      return false;
    }

    if (attr_holder->SetAttr(iter.first, attr_value) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "Set attr [%s] failed.", iter.first.c_str());
      return false;
    }
  }
  return true;
}

Buffer ModelSerialize::SerializeModel(const Model &model, const bool is_dump) {
  proto::ModelDef model_def;
  ModelSerializeImp model_imp;
  if (!model_imp.SerializeModel(model, &model_def, is_dump)) {
    return Buffer();
  }
#if !defined(__ANDROID__) && !defined(ANDROID)
  Buffer buffer(model_def.ByteSizeLong());
#else
  Buffer buffer(model_def.ByteSize());
#endif
  GE_CHK_BOOL_ONLY_LOG(buffer.GetSize() != 0UL, "get size failed");
  GE_CHK_BOOL_ONLY_LOG((buffer.GetData() != nullptr), "get size failed");
  const auto ret = model_def.SerializeToArray(buffer.GetData(), static_cast<int32_t>(buffer.GetSize()));
  if (!ret) {
    GELOGW("[Serialize][Model] Serialize to array failed");
  }
  return buffer;
}

bool ModelSerialize::UnserializeModel(const uint8_t *data, const size_t len, Model &model) {
  if (data == nullptr) {
    REPORT_INNER_ERROR("E19999", "param data is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] data is nullptr");
    return false;
  }

  std::shared_ptr<proto::ModelDef> model_proto_ptr;
  model_proto_ptr = ComGraphMakeShared<proto::ModelDef>();
  if (model_proto_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "create ModelDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][ModelDef] proto::ModelDef make shared failed");
    return false;
  }

  auto &model_proto = *model_proto_ptr;
  if (!ReadProtoFromBinaryFile(data, len, &model_proto)) {
    GELOGE(GRAPH_FAILED, "[Read][Proto] from binaryfile failed.");
    return false;
  }

  ModelSerializeImp model_imp;
  model_imp.SetProtobufOwner(model_proto_ptr);
  if (!model_imp.UnserializeModel(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "[Unserialize][Model] failed");
    return false;
  }
  return model.IsValid();
}

bool ModelSerialize::UnserializeModel(ge::proto::ModelDef &model_def, Model &model) {
  const std::shared_ptr<proto::ModelDef> model_def_ptr = ComGraphMakeShared<proto::ModelDef>(model_def);
  GE_CHK_BOOL_EXEC(model_def_ptr != nullptr, REPORT_CALL_ERROR("E19999", "create ModelDef failed.");
                   return false, "[Create][ModelDef] mode_def make shared failed");

  ModelSerializeImp model_imp;
  model_imp.SetProtobufOwner(model_def_ptr);
  if (!model_imp.UnserializeModel(model, *model_def_ptr)) {
    GELOGE(GRAPH_FAILED, "[Unserialize][Model] fail");
    return false;
  }
  return model.IsValid();
}
}  // namespace ge
