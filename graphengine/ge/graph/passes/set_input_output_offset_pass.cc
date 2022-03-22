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

#include "graph/passes/set_input_output_offset_pass.h"

#include "runtime/mem.h"

namespace ge {
Status SetInputOutputOffsetPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  for (auto &node : graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    vector<int> connect_input;
    (void)AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_INPUT, connect_input);
    if (!connect_input.empty()) {
      Status ret = SetInputOffset(node, connect_input);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Set][InputOffset] for node:%s failed.", node->GetName().c_str());
        return ret;
      }
    }
    vector<int> connect_output;
    (void)AttrUtils::GetListInt(node->GetOpDesc(), ATTR_NAME_NODE_CONNECT_OUTPUT, connect_output);
    if (!connect_output.empty()) {
      Status ret = SetOutputOffset(node, connect_output);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Set][OutputOffset] for node:%s failed.", node->GetName().c_str());
        return ret;
      }
    }
  }
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetInputOffsetForFusion(const std::vector<int64_t> &memory_type,
                                                         const ge::NodePtr &node) {
  GELOGI("Start to SetInputOffsetForFusion for %s", node->GetName().c_str());
  auto op_desc = node->GetOpDesc();
  for (size_t i = 0; i < memory_type.size(); ++i) {
    if (memory_type.at(i) != RT_MEMORY_L1) {
      std::vector<int64_t> input_offset_of_node;
      input_offset_of_node = op_desc->GetInputOffset();
      if (input_offset_of_node.size() < i) {
        REPORT_INNER_ERROR("E19999", "Input offsets size:%zu of node:%s(%s) < index:%zu, check invalid",
                           input_offset_of_node.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
        GELOGE(PARAM_INVALID, "[Check][Param] Input offsets size:%zu of node:%s(%s) < index:%zu",
               input_offset_of_node.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), i);
        return PARAM_INVALID;
      }
      int64_t input_offset = input_offset_of_node.at(i);
      GELOGI("input_offset of %s is %ld.", node->GetName().c_str(), input_offset);
      auto in_anchor = node->GetInDataAnchor(i);
      GE_IF_BOOL_EXEC(in_anchor == nullptr, continue);
      auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
      int out_index = peer_out_anchor->GetIdx();
      auto data_op_desc = peer_out_anchor->GetOwnerNode()->GetOpDesc();
      GE_CHECK_NOTNULL(data_op_desc);
      int64_t out_offset = data_op_desc->GetOutputOffset().at(out_index);
      GELOGI("output_offset of %s is %ld.", peer_out_anchor->GetOwnerNode()->GetName().c_str(), out_offset);
      vector<int64_t> zero_copy_basic_offset;
      vector<int64_t> zero_copy_relative_offset;

      (void)ge::AttrUtils::GetListInt(data_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
      (void)ge::AttrUtils::GetListInt(data_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);
      zero_copy_basic_offset.emplace_back(out_offset);
      int64_t relative_offset = input_offset - out_offset;
      zero_copy_relative_offset.emplace_back(relative_offset);
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(data_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset),
                       REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                         ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                                         data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
                       GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                              data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
                       return FAILED);
      GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(data_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET,
                                                 zero_copy_relative_offset),
                       REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                         ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                                         data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
                       GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                              data_op_desc->GetName().c_str(), data_op_desc->GetType().c_str());
                       return FAILED);
    }
  }
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetInputOffsetForHcom(const ge::NodePtr &node, const vector<int> &connect_input) {
  GELOGI("Start SetInputOffsetForHcom for %s.", node->GetName().c_str());

  auto op_desc = node->GetOpDesc();
  vector<int64_t> input_offset_of_node;
  input_offset_of_node = node->GetOpDesc()->GetInputOffset();
  for (size_t input_index = 0; input_index < connect_input.size(); ++input_index) {
    int connect_input_index = connect_input.at(input_index);
    int64_t input_offset = input_offset_of_node.at(connect_input_index);
    NodePtr in_data = node->GetInDataNodes().at(connect_input_index);
    auto in_op_desc = in_data->GetOpDesc();
    GE_CHECK_NOTNULL(in_op_desc);
    if (in_op_desc->GetType() == DATA) {
      int64_t output_offset = in_op_desc->GetOutputOffset().at(0);
      if (output_offset == input_offset) {
        continue;
      } else {
        vector<int64_t> zero_copy_basic_offset;
        vector<int64_t> zero_copy_relative_offset;
        (void)ge::AttrUtils::GetListInt(in_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
        (void)ge::AttrUtils::GetListInt(in_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);
        GELOGI("input offset from %s to %s is %ld to %ld.", in_op_desc->GetName().c_str(), op_desc->GetName().c_str(),
               output_offset, input_offset);
        int64_t relative_offset = input_offset - output_offset;
        zero_copy_basic_offset.emplace_back(output_offset);
        zero_copy_relative_offset.emplace_back(relative_offset);
        GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(in_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset),
                         REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                           ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                                           in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                         GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                                in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                         return FAILED);
        GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(in_op_desc,
                                                   ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset),
                         REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                           ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                                           in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                         GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                                in_op_desc->GetName().c_str(), in_op_desc->GetType().c_str());
                         return FAILED);
      }
    }
  }
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetInputOffset(const NodePtr &node, const vector<int> &connect_input) {
  GELOGD("Start to SetInputOffset for %s.", node->GetName().c_str());
  std::vector<int64_t> memory_type;
  auto op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, memory_type);
  if (!memory_type.empty()) {
    Status ret = SetInputOffsetForFusion(memory_type, node);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Set][InputOffset] For Fusion failed, node:%s.", node->GetName().c_str());
      return ret;
    }
  }
  // Data->Hcom
  bool is_input_continuous = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
  if (is_input_continuous) {
    Status ret = SetInputOffsetForHcom(node, connect_input);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Set][InputOffset] For Hcom failed, node:%s.", node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetOutputOffsetForConcat(const NodePtr &node) {
  GELOGI("Start SetOutputOffsetForConcat for %s.", node->GetName().c_str());
  auto op_desc = node->GetOpDesc();
  std::vector<int64_t> output_offset_of_concat;
  output_offset_of_concat = op_desc->GetOutputOffset();
  // phony_concat has one output
  GE_IF_BOOL_EXEC(output_offset_of_concat.size() != 1,
                  REPORT_INNER_ERROR("E19999", "Output offsets size:%zu of node:%s(%s) not equal to 1, check invalid",
                                     output_offset_of_concat.size(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  GELOGE(PARAM_INVALID, "[Check][Param] Output offsets size:%zu of node:%s(%s) not equal to 1.",
                         output_offset_of_concat.size(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
                  return PARAM_INVALID);
  NodePtr net_output = node->GetOutDataNodes().at(0);
  auto out_op_desc = net_output->GetOpDesc();
  GE_CHECK_NOTNULL(out_op_desc);
  vector<int64_t> zero_copy_basic_offset;
  vector<int64_t> zero_copy_relative_offset;
  (void)ge::AttrUtils::GetListInt(out_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
  (void)ge::AttrUtils::GetListInt(out_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);

  int64_t basic_offset = output_offset_of_concat.at(0);
  GELOGI("output_offset of %s is %ld.", op_desc->GetName().c_str(), basic_offset);
  for (InDataAnchorPtr &in_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC(peer_out_anchor == nullptr, continue);
    NodePtr in_node = peer_out_anchor->GetOwnerNode();
    auto out_index = peer_out_anchor->GetIdx();
    std::vector<int64_t> output_offset_of_in_node;
    GE_CHECK_NOTNULL(in_node->GetOpDesc());
    output_offset_of_in_node = in_node->GetOpDesc()->GetOutputOffset();
    GELOGI("input offset from %s to %s is %ld.", in_node->GetName().c_str(), op_desc->GetName().c_str(),
           output_offset_of_in_node.at(out_index));
    int64_t relative_offset = output_offset_of_in_node.at(out_index) - basic_offset;
    zero_copy_basic_offset.emplace_back(basic_offset);
    zero_copy_relative_offset.emplace_back(relative_offset);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(out_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                                     out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                          out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(out_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                     ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                                     out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                          out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   return FAILED);
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetOutputOffsetForHcom(const NodePtr &node, const vector<int> &connect_output) {
  GELOGI("Start SetOutputOffsetForHcom, %s connect with %zu output.", node->GetName().c_str(), connect_output.size());
  vector<int64_t> output_offset_of_node;
  output_offset_of_node = node->GetOpDesc()->GetOutputOffset();
  int connect_output_index = connect_output.at(0);
  int64_t basic_offset = output_offset_of_node.at(connect_output_index);
  GELOGI("basic_offset of %s is %ld.", node->GetName().c_str(), basic_offset);

  NodePtr net_output = node->GetOutDataNodes().at(connect_output_index);
  auto out_op_desc = net_output->GetOpDesc();
  GE_CHECK_NOTNULL(out_op_desc);
  vector<int64_t> zero_copy_basic_offset;
  vector<int64_t> zero_copy_relative_offset;
  (void)ge::AttrUtils::GetListInt(out_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset);
  (void)ge::AttrUtils::GetListInt(out_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset);

  for (auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_IF_BOOL_EXEC(out_anchor == nullptr, continue);
    for (auto &in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_IF_BOOL_EXEC(in_anchor == nullptr, continue);
      if (in_anchor->GetOwnerNode()->GetType() == NETOUTPUT && out_anchor->GetIdx() != connect_output_index) {
        continue;
      } else {
        NodePtr out_node = in_anchor->GetOwnerNode();
        auto in_index = in_anchor->GetIdx();
        std::vector<int64_t> input_offset_of_out_node;
        GE_CHECK_NOTNULL(out_node->GetOpDesc());
        input_offset_of_out_node = out_node->GetOpDesc()->GetInputOffset();
        GELOGI("input offset from %s to %s is %ld.", node->GetName().c_str(), out_node->GetName().c_str(),
               input_offset_of_out_node.at(in_index));
        int64_t relative_offset = input_offset_of_out_node.at(in_index) - basic_offset;
        zero_copy_basic_offset.emplace_back(basic_offset);
        zero_copy_relative_offset.emplace_back(relative_offset);
      }
    }
  }

  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(out_op_desc, ATTR_ZERO_COPY_BASIC_OFFSET, zero_copy_basic_offset),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                                     out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_BASIC_OFFSET.c_str(),
                          out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   return FAILED);
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListInt(out_op_desc, ATTR_ZERO_COPY_RELATIVE_OFFSET, zero_copy_relative_offset),
                   REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed",
                                     ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                                     out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_ZERO_COPY_RELATIVE_OFFSET.c_str(),
                          out_op_desc->GetName().c_str(), out_op_desc->GetType().c_str());
                   return FAILED);
  return SUCCESS;
}

Status SetInputOutputOffsetPass::SetOutputOffset(const NodePtr &node, const vector<int> &connect_output) {
  GELOGD("Start SetOutputOffset of %s.", node->GetName().c_str());
  bool attr_no_task = false;
  bool get_attr_no_task = ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOTASK, attr_no_task);
  if (get_attr_no_task && attr_no_task) {
    bool is_input_continuous = false;
    (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, is_input_continuous);
    bool buffer_fusion = CheckBufferFusion(node);
    // A/B/C -> Phony_concat -> Netoutput : input_continuous
    if (is_input_continuous || buffer_fusion) {
      Status ret = SetOutputOffsetForConcat(node);
      if (ret != SUCCESS) {
        GELOGE(ret, "[Set][OutputOffset] For Concat failed, node:%s.", node->GetName().c_str());
        return ret;
      }
    }
  }
  // allreduce->netoutput : output_continuous
  bool is_output_continuous = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_CONTINUOUS_OUTPUT, is_output_continuous);
  if (is_output_continuous) {
    Status ret = SetOutputOffsetForHcom(node, connect_output);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Set][OutputOffset] For Hcom failed, node:%s.", node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

bool SetInputOutputOffsetPass::CheckBufferFusion(const NodePtr &node) {
  for (auto &in_node : node->GetInDataNodes()) {
    GE_CHECK_NOTNULL(in_node);
    auto op_desc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if (!op_desc->HasAttr(ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION)) {
      GELOGI("The node: %s not have ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION.", node->GetName().c_str());
      return false;
    }
  }
  return true;
}
}  // namespace ge