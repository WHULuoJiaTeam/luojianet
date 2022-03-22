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

#ifndef COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
#define COMMON_GRAPH_UTILS_GE_IR_UTILS_H_

#include <google/protobuf/map.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/stubs/port.h>

#include <graph/anchor.h>
#include <graph/debug/ge_log.h>
#include <graph/debug/ge_util.h>
#include <graph/detail/attributes_holder.h>
#include <graph/ge_tensor.h>
#include <graph/graph.h>
#include <graph/model.h>
#include <graph/node.h>
#include <graph/utils/graph_utils.h>
#include <graph/utils/type_utils.h>
#include <graph/types.h>

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "proto/ge_ir.pb.h"
#include "proto/onnx/ge_onnx.pb.h"

namespace ge {
///
///  @ingroup ge_ir_utils
///  @brief check, if not equal, log with tag
///  @param [in] const left_value, right_value reference, log_info_tag
///  @return bool
///
template <typename T>
bool IsEqual(const T &l_value, const T &r_value, const std::string &log_info_tag) {
  if ((l_value == r_value)) {
    return true;
  } else {
    GELOGD("Check not equal with %s", log_info_tag.c_str());
    return false;
  }
}

class OnnxUtils {
 public:
  enum DumpLevel { NO_DUMP = 0, DUMP_ALL = 1, DUMP_WITH_OUT_DATA = 2, DUMP_WITH_OUT_DESC = 3, DUMP_LEVEL_END = 4 };

  static bool ConvertGeModelToModelProto(const ge::Model &model, ge::onnx::ModelProto &model_proto);

  static bool ConvertModelProtoToGeModel(const ge::onnx::ModelProto &model_proto, ge::Model &model);

 private:
  // Part 1: from IR convert to ONNX Protobuf
  static void AddAttrProto(ge::onnx::NodeProto *const node_proto, const ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const void *const data);

  static void AddAttrProto(ge::onnx::NodeProto *const node_proto, const ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name,
                           const ::google::protobuf::RepeatedField<::google::protobuf::int64> data);

  static void AddAttrProto(ge::onnx::NodeProto *const node_proto,
                           const ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedField<bool> data);

  static void AddAttrProto(ge::onnx::NodeProto *const node_proto, const ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedField<float> data);

  static void AddAttrProto(ge::onnx::NodeProto *const node_proto,
                           const ge::onnx::AttributeProto_AttributeType type,
                           const std::string &name, const ::google::protobuf::RepeatedPtrField<::std::string> data);

  static void AddListAttrProto(const std::string &attr_name, const ::ge::proto::AttrDef &attr_def,
                               const std::string &prefix, const std::string &suffix, onnx::NodeProto *node_proto);

  static void AddAttrProtoFromNodeMembers(const NodePtr &node, ge::onnx::NodeProto *const node_proto);

  static void AddAttrProtoFromAttribute(const std::pair<const std::string, ge::GeAttrValue> &string_attr_value,
                                        ge::onnx::NodeProto *const node_proto);

  static void AddAttrProtoForOpInDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForOpOutDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForOpInAndOutDesc(ge::onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static void AddAttrProtoForAttrsFromAttrMap(const ::google::protobuf::Map<std::string,
                                              ge::proto::AttrDef> &attr_map,
                                              ge::onnx::NodeProto *const node_proto,
                                              const std::string &prefix = "",
                                              const std::string &suffix = "");

  static ge::onnx::TensorProto_DataType EncodeDataType(const ge::DataType data_type);

  static void EncodeNodeLinkForNetronVisual(const NodePtr &node, ge::onnx::NodeProto *const node_proto);

  static bool EncodeNodeLink(const NodePtr &node, ge::onnx::NodeProto *const node_proto);

  static bool EncodeNodeDesc(const NodePtr &node, ge::onnx::NodeProto *const node_proto);

  static bool EncodeNode(const NodePtr &node, ge::onnx::NodeProto *const node_proto);

  static void EncodeTypeProtoTensorType(const NodePtr &node, ge::onnx::TypeProto_Tensor *const tensor_type);

  static void EncodeValueInfo(const NodePtr &node, ge::onnx::ValueInfoProto *const value_info_proto);

  static bool EncodeGraph(const ConstComputeGraphPtr &graph, ge::onnx::GraphProto *const graph_proto);

  /// Part 2: from ONNX Protobuf convert to IR
  /// Describes node's link relationships
  class NodeLinkInfo {
   public:
    NodeLinkInfo() = default;
    ~NodeLinkInfo() = default;
    NodeLinkInfo(std::string src_name,
                 int32_t src_out_index,
                 NodePtr dst_node,
                 int32_t dst_in_index,
                 std::string dst_name) :
        src_node_name_(std::move(src_name)),
        src_out_index_(src_out_index),
        dst_node_(std::move(dst_node)),
        dst_in_index_(dst_in_index),
        dst_node_name_(std::move(dst_name)){}

    std::string GetSrcNodeName() const { return src_node_name_; };
    int32_t GetSrcOutIndex() const { return src_out_index_; };
    NodePtr GetDstNode() const { return dst_node_; };
    int32_t GetDstInIndex() const { return dst_in_index_; };
    std::string GetDstNodeName() const { return dst_node_name_; };

   private:
    std::string src_node_name_;
    int32_t src_out_index_;
    NodePtr dst_node_;
    int32_t dst_in_index_;
    std::string dst_node_name_;
  };

  // Parse node name and index
  static bool ParseNameAndIndex(const std::string &node_name_index, std::string &node_name, int32_t &idx);

  static ge::DataType DecodeDataType(const ge::onnx::TensorProto_DataType data_type);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<std::string> &strings);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<int64_t> &ints);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, int64_t &value);

  static void DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::string &value);

  static void DecodeNodeAttributeForOpOutDesc(const ge::onnx::AttributeProto &attr_proto,
                                              const std::string &attr_name_for_output_desc,
                                              const int32_t index, const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInDesc(const ge::onnx::AttributeProto &attr_proto,
                                             const std::string &attr_name_for_input_desc,
                                             const int32_t idx,
                                             const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpInAndOutDesc(const ge::onnx::AttributeProto &attr_proto,
                                                   const std::string &attr_name_for_input_output_desc,
                                                   const int32_t idx,
                                                   const OpDescPtr &op_desc);

  static void DecodeNodeAttributeForOpDesc(const ge::onnx::AttributeProto &attr_proto, OpDescPtr &op_desc);

  static bool DecodeNodeLinkImp(const NodeLinkInfo &item, const NodePtr &node_ptr);

  static bool DecodeNodeLink(const std::vector<ge::onnx::NodeProto> &node_proto_vector,
                             const std::map<std::string, NodePtr> &node_map);

  static bool DecodeNodeDesc(const ge::onnx::NodeProto *const node_proto, OpDescPtr &op_desc);

  static bool DecodeGraph(const int32_t recursion_depth,
                          const ge::onnx::GraphProto &graph_proto, ComputeGraphPtr &graph);

  static void AddShapeFormatAndDtypeIntoProto(const bool is_input, onnx::NodeProto *const node_proto,
                                              const ge::ConstGeTensorDescPtr &desc, const uint32_t idx);

  static void AddAllAttr(onnx::NodeProto *const node_proto, const ConstGeTensorDescPtr &op_desc,
                         const char_t *const prefix, const uint32_t idx);

  static void AddCommonAttrIntoProto(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc);

  static bool AddInputAndOutputNodesForGraph(const onnx::GraphProto &graph_proto,
                                             ComputeGraphPtr &graph,
                                             const std::map<std::string, NodePtr> &node_map);
};
}  // namespace ge

#endif  // COMMON_GRAPH_UTILS_GE_IR_UTILS_H_
