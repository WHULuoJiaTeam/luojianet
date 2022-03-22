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

#include <algorithm>
#include <set>
#include <vector>

#include "ge_util.h"
#include "graph/types.h"
#include "graph/debug/graph_debug.h"

namespace {
const std::string TAB = "    ";
std::set<std::string> control_anchor;
std::vector<std::string> types = {
    "DT_FLOAT", "DT_FLOAT16", "DT_INT8",          "DT_INT32",          "DT_UINT8",    "",
    "DT_INT16", "DT_UINT16",  "DT_UINT32",        "DT_INT64",          "DT_UINT64",   "DT_DOUBLE",
    "DT_BOOL",  "DT_DUAL",    "DT_DUAL_SUB_INT8", "DT_DUAL_SUB_UINT8", "DT_UNDEFINED"};

std::vector<std::string> formats = {"FORMAT_NCHW",
                                    "FORMAT_NHWC",
                                    "FORMAT_ND",
                                    "FORMAT_NC1HWC0",
                                    "FORMAT_FRACTAL_Z",
                                    "FORMAT_NC1C0HWPAD",
                                    "FORMAT_NHWC1C0",
                                    "FORMAT_FSR_NCHW",
                                    "FORMAT_FRACTAL_DECONV",
                                    "FORMAT_C1HWNC0",
                                    "FORMAT_FRACTAL_DECONV_TRANSPOSE",
                                    "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS",
                                    "FORMAT_NC1HWC0_C04",
                                    "FORMAT_FRACTAL_Z_C04",
                                    "FORMAT_CHWN",
                                    "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS",
                                    "FORMAT_HWCN",
                                    "FORMAT_NC1KHKWHWC0",
                                    "FORMAT_BN_WEIGHT",
                                    "FORMAT_FILTER_HWCK",
                                    "FORMAT_HASHTABLE_LOOKUP_LOOKUPS",
                                    "FORMAT_HASHTABLE_LOOKUP_KEYS",
                                    "FORMAT_HASHTABLE_LOOKUP_VALUE",
                                    "FORMAT_HASHTABLE_LOOKUP_OUTPUT",
                                    "FORMAT_HASHTABLE_LOOKUP_HITS",
                                    "FORMAT_RESERVED"};

std::vector<std::string> data_nodes = {"Const", "Data"};

inline string StrFmt(const std::string &str) {
  return " \"" + str + "\" ";
}

inline string InputAnchorPort(const std::string &name) {
  return "__input__" + name;
}

inline string OutputAnchorPort(const std::string &name) {
  return "__output__" + name;
}
}

namespace ge {
void GraphDebugPrinter::DumpNodeToDot(const NodePtr node, std::ostringstream &out_) {
  if (node == nullptr) {
    GELOGI("Some nodes are null.");
    return;
  }

  bool in_control = false;
  const auto name = node->GetName();
  out_ << TAB << StrFmt(name);
  auto input_cnt = std::max(static_cast<size_t>(1), node->GetAllInDataAnchors().size());
  const auto output_cnt = std::max(static_cast<size_t>(1), node->GetAllOutDataAnchors().size());
  if (control_anchor.find(node->GetName()) != control_anchor.end()) {
    input_cnt++;
    in_control = true;
  }
  const auto max_col = input_cnt * output_cnt;
  out_ << "[\n";
  if (find(data_nodes.begin(), data_nodes.end(), node->GetType()) != data_nodes.end()) {
    out_ << TAB << TAB << "shape=plaintext, color=goldenrod\n";
  } else {
    out_ << TAB << TAB << "shape=plaintext, color=deepskyblue\n";
  }
  out_ << TAB << TAB << "label=<\n";
  out_ << TAB << TAB << R"(<table border="0" cellborder="1" align="center")"
       << ">" << std::endl;

  const auto input_anchors = node->GetAllInDataAnchors();
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return);
  if (!input_anchors.empty()) {
    out_ << TAB << TAB << "<tr>";
  }
  for (const auto &anchor : input_anchors) {
    const std::string anchor_text = op_desc->GetInputNameByIndex(static_cast<uint32_t>(anchor->GetIdx()));

    out_ << "<td port = " << StrFmt(InputAnchorPort(anchor_text)) << " colspan='" << output_cnt << "'>"
         << anchor_text << "</td>";
  }
  if (in_control) {
    const std::string anchor_text = "ctrl";
    out_ << "<td port = " << StrFmt(InputAnchorPort(anchor_text)) << " colspan='" << output_cnt << "'>"
         << anchor_text << "</td>";
  }
  if (!input_anchors.empty()) {
    out_ << "</tr>\n";
  }
  // Node type
  out_ << TAB << TAB << "<tr><td colspan='" << max_col << "'>"
       << "<b>" << node->GetType() << "</b></td></tr>\n";
  // Output
  const auto output_anchors = node->GetAllOutDataAnchors();
  if (!output_anchors.empty()) {
    out_ << TAB << TAB << "<tr>";
  }
  for (const auto &anchor : output_anchors) {
    const std::string anchor_text = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(anchor->GetIdx()));

    out_ << "<td port = " << StrFmt(OutputAnchorPort(anchor_text)) << " colspan='" << input_cnt << "'>"
         << anchor_text << "</td>";
  }

  if (!output_anchors.empty()) {
    out_ << "</tr>\n";
  }
  out_ << TAB << TAB << "</table>\n" << TAB << ">];\n";
}

void GraphDebugPrinter::DumpEdgeToDot(const NodePtr node, std::ostringstream &out_, const uint32_t flag) {
  if (node == nullptr) {
    GELOGI("Some nodes are null.");
    return;
  }
  const auto all_out_anchor = node->GetAllOutDataAnchors();
  const auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return);
  for (const auto &anchor : all_out_anchor) {
    const auto src_anchor = anchor;
    const auto src_node_name = node->GetName();
    const auto src_anchor_index = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(src_anchor->GetIdx()));
    const auto des_anchors = anchor->GetPeerAnchors();
    for (const auto &peer_in_anchor : des_anchors) {
      const auto in_data_anchor = Anchor::DynamicAnchorCast<InDataAnchor>(peer_in_anchor);
      std::string dst_node_name;
      out_ << TAB << StrFmt(src_node_name);
      out_ << ":" << OutputAnchorPort(src_anchor_index);
      const auto op = peer_in_anchor->GetOwnerNode()->GetOpDesc();
      if (op == nullptr) {
        continue;
      }
      if (in_data_anchor != nullptr) {
        dst_node_name = in_data_anchor->GetOwnerNode()->GetName();
        const std::string des_anchor_index = op->GetInputNameByIndex(static_cast<uint32_t>(in_data_anchor->GetIdx()));
        out_ << " -> " << StrFmt(dst_node_name);
        out_ << ":" << InputAnchorPort(des_anchor_index);
        out_ << "[";
      }
      const auto in_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(peer_in_anchor);
      if (in_control_anchor != nullptr) {
        dst_node_name = in_control_anchor->GetOwnerNode()->GetName();
        const std::string des_anchor_index = "ctrl";
        out_ << " -> " << StrFmt(dst_node_name);
        out_ << ":" << InputAnchorPort(des_anchor_index);
        out_ << "[";
        out_ << " style=dashed ";
      }
      if ((flag != static_cast<uint32_t>(DotFileFlag::DOT_NOT_SHOW_EDGE_LABEL)) && in_data_anchor) {
        const auto src_ops = src_anchor->GetOwnerNode()->GetOpDesc();
        GE_CHECK_NOTNULL_EXEC(src_ops, return);
        GE_CHECK_NOTNULL_EXEC(src_ops->GetOutputDescPtr(static_cast<uint32_t>(src_anchor->GetIdx())), return);
        const std::string label = GetSrcOpStr(src_ops, src_anchor);
        out_ << "label=" << StrFmt(label);
      }
      out_ << "]" << std::endl;
    }
  }
}

graphStatus GraphDebugPrinter::DumpGraphDotFile(const Graph &graph, const std::string &output_dot_file_name,
                                                const uint32_t flag) {
  const auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGI("Compute graph is NULL .");
    return GRAPH_SUCCESS;
  }
  return DumpGraphDotFile(compute_graph, output_dot_file_name, flag);
}

graphStatus GraphDebugPrinter::DumpGraphDotFile(const ComputeGraphPtr graph, const std::string &output_dot_file_name,
                                                const uint32_t flag) {
  if (graph == nullptr) {
    GELOGI("graph is null.");
    return GRAPH_SUCCESS;
  }
  std::ostringstream out_;
  out_ << "digraph G{\n";
  out_ << TAB << R"(ratio=compress;size="8, 100")" << std::endl;
  out_ << TAB << R"(node[fontname="Consolas"])" << std::endl;
  out_ << TAB << R"(edge[fontsize = "8" fontname = "Consolas" color="dimgray" ])" << std::endl;
  const auto all_nodes = graph->GetAllNodes();
  for (const auto &node : all_nodes) {
    for (const auto &temp : node->GetAllOutDataAnchors()) {
      for (const auto &peer : temp->GetPeerAnchors()) {
        const auto temp_control_anchor = Anchor::DynamicAnchorCast<InControlAnchor>(peer);
        if (temp_control_anchor) {
          (void)control_anchor.insert(peer->GetOwnerNode()->GetName());
        }
      }
    }
  }
  for (const auto &node : all_nodes) {
    DumpNodeToDot(node, out_);
  }
  for (const auto &node : all_nodes) {
    DumpEdgeToDot(node, out_, flag);
  }
  out_ << "}";
  std::ofstream output_file(output_dot_file_name);
  if (output_file.is_open()) {
    output_file << out_.str();
  } else {
    GELOGW("[DumpGraph][OpenFile] Open file %s failed.", output_dot_file_name.c_str());
  }
  return GRAPH_SUCCESS;
}

std::string GraphDebugPrinter::GetSrcOpStr(const OpDescPtr &src_ops, const OutDataAnchorPtr &src_anchor) {
  std::string label;
  const auto src_shape = src_ops->GetOutputDesc(static_cast<uint32_t>(src_anchor->GetIdx())).GetShape();
  const auto dim = src_shape.GetDims();
  std::ostringstream tensor_info;
  if (!dim.empty()) {
    for (size_t i = 0UL; i < dim.size(); i++) {
      if (i != (dim.size() - 1UL)) {
        tensor_info << dim[i] << "x";
      } else {
        tensor_info << dim[i];
      }
    }
  } else {
    tensor_info << "?";
  }
  const auto src_tensor_desc = src_ops->GetOutputDescPtr(static_cast<uint32_t>(src_anchor->GetIdx()));
  const auto format = src_tensor_desc->GetFormat();
  const auto datatype = src_tensor_desc->GetDataType();
  tensor_info << " : " << formats[format] << " : " << types[datatype];
  label = tensor_info.str();
  return label;
}
}  // namespace ge
