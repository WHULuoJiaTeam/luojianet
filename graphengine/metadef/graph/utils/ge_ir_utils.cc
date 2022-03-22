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

#include "graph/utils/ge_ir_utils.h"
#include <utility>
#include "framework/common/debug/ge_log.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/ge_tensor_impl.h"
#include "graph/node_impl.h"
#include "graph/op_desc_impl.h"
#include "mmpa/mmpa_api.h"

namespace {
const ge::char_t *const kControlAnchorIndex = ":-1";
const ge::char_t *const kNodeTypeForSubgraph = "subgraph";
const ge::char_t *const kPrefixForInputDesc = "input_desc_attr_";
const ge::char_t *const kPrefixForOutputDesc = "output_desc_attr_";
const ge::char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
const int8_t kMaxRecursiveDepth = 10;
const int32_t kDecimalBase = 10;
ge::char_t kDumpGeGraphEnv[MMPA_MAX_PATH] = {};
const int64_t kDumpLevel =
    (mmGetEnv(kDumpGEGraph, &(kDumpGeGraphEnv[0U]), static_cast<UINT32>(MMPA_MAX_PATH)) == EN_OK) ?
    std::strtol(&(kDumpGeGraphEnv[0U]), nullptr, kDecimalBase) : ge::OnnxUtils::NO_DUMP;
const uint64_t kInputPrefixLength = 5U;
const uint64_t kOutputPrefixLength = 6U;
}  // namespace

namespace ge {
// Part 1: from IR convert to ONNX Protobuf
namespace{
const std::map<ge::DataType, onnx::TensorProto_DataType> kGeDataTypeToOnnxMap = {
    {DT_INT64, onnx::TensorProto_DataType_INT64},   {DT_UINT64, onnx::TensorProto_DataType_UINT64},
    {DT_FLOAT, onnx::TensorProto_DataType_FLOAT},   {DT_INT32, onnx::TensorProto_DataType_INT32},
    {DT_UINT32, onnx::TensorProto_DataType_UINT32}, {DT_INT8, onnx::TensorProto_DataType_INT8},
    {DT_UINT8, onnx::TensorProto_DataType_UINT8},   {DT_INT16, onnx::TensorProto_DataType_INT16},
    {DT_UINT16, onnx::TensorProto_DataType_UINT16}, {DT_FLOAT16, onnx::TensorProto_DataType_FLOAT16},
    {DT_DOUBLE, onnx::TensorProto_DataType_DOUBLE}, {DT_BOOL, onnx::TensorProto_DataType_BOOL},
};
}

struct AttrNameComp {
  inline bool operator()(const onnx::AttributeProto &lsh, const onnx::AttributeProto &rsh) const {
    return lsh.name() < rsh.name();
  }
};

onnx::TensorProto_DataType OnnxUtils::EncodeDataType(const DataType data_type) {
  const auto it = kGeDataTypeToOnnxMap.find(data_type);
  if (it != kGeDataTypeToOnnxMap.end()) {
    return it->second;
  } else {
    GELOGW("[Encode][DataType] Datatype %u not support", data_type);
    return onnx::TensorProto_DataType_UNDEFINED;
  }
}

void OnnxUtils::AddAttrProtoFromAttribute(const std::pair<const std::string, ge::GeAttrValue> &string_attr_value,
                                          onnx::NodeProto *const node_proto) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Node proto is nullptr.");
    return;
  }
  const auto attr = node_proto->add_attribute();
  if (attr == nullptr) {
    REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
    return;
  }
  const auto attr_name = string_attr_value.first;
  attr->set_name(attr_name);
  const auto attr_value = string_attr_value.second;
  const auto value_type = attr_value.GetValueType();
  switch (value_type) {
    case GeAttrValue::VT_FLOAT: {
      float32_t data_f = 0.0F;
      (void)attr_value.GetValue(data_f);
      attr->set_f(data_f);
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    }
    case GeAttrValue::VT_LIST_FLOAT: {
      std::vector<float> data_fs = {};
      (void)attr_value.GetValue(data_fs);
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for (auto &v : data_fs) {
        attr->add_floats(v);
      }
      break;
    }
    case GeAttrValue::VT_INT: {
      int64_t data_i = 0;
      (void)attr_value.GetValue(data_i);
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(data_i);
      break;
    }
    case GeAttrValue::VT_LIST_INT: {
      std::vector<int64_t> data_is = {};
      (void)attr_value.GetValue(data_is);
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for (auto &v : data_is) {
        attr->add_ints(v);
      }
      break;
    }
    case GeAttrValue::VT_STRING: {
      std::string data_s;
      (void)attr_value.GetValue(data_s);
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(data_s);
      break;
    }
    case GeAttrValue::VT_LIST_STRING: {
      std::vector<std::string> data_ss = {};
      (void)attr_value.GetValue(data_ss);
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for (auto &v : data_ss) {
        attr->add_strings(v);
      }
      break;
    }
    default:
      GELOGW("[Add][Attr] ValueType %u is not supported", value_type);
      break;
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *const node_proto, const onnx::AttributeProto_AttributeType type,
                             const std::string &name, const void *const data) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr.");
    GELOGE(FAILED, "[Check][Param] Node_proto is nullptr.");
    return;
  }
  const auto attr = node_proto->add_attribute();
  if (attr == nullptr) {
    REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
    return;
  }
  attr->set_name(name);
  switch (type) {
    case onnx::AttributeProto_AttributeType_FLOAT:
      attr->set_f((*(static_cast<const float32_t *const>(data))));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;

    case onnx::AttributeProto_AttributeType_FLOATS:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for (auto &v : (*(static_cast<const std::vector<float> *const>(data)))) {
        attr->add_floats(v);
      }
      break;

    case onnx::AttributeProto_AttributeType_INT:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i((*(static_cast<const int64_t *const>(data))));
      break;

    case onnx::AttributeProto_AttributeType_INTS:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for (auto &v : *(static_cast<const std::vector<int64_t> *const>(data))) {
        attr->add_ints(v);
      }
      break;

    case onnx::AttributeProto_AttributeType_STRING:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s((*(static_cast<const std::string *const>(data))));
      break;

    case onnx::AttributeProto_AttributeType_STRINGS:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for (auto &v : *(static_cast<const std::vector<std::string> *const>(data))) {
        attr->add_strings(v);
      }
      break;

    default:
      GELOGW("[Add][Attr] AttributeType %u is not supported", type);
      break;
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *const node_proto, const onnx::AttributeProto_AttributeType type,
                             const std::string &name,
                             const ::google::protobuf::RepeatedField<::google::protobuf::int64> data) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr.");
    GELOGE(FAILED, "[Check][Param] Node_proto is nullptr.");
    return;
  }
  if (!data.empty()) {
    const auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
      GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_ints(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *const node_proto, const onnx::AttributeProto_AttributeType type,
                             const std::string &name, const ::google::protobuf::RepeatedField<bool> data) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr.");
    GELOGE(FAILED, "[Check][Param] Node proto is nullptr.");
    return;
  }
  if (!data.empty()) {
    const auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
      GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_ints(static_cast<int64_t>(v));
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *const node_proto, const onnx::AttributeProto_AttributeType type,
                             const std::string &name, const ::google::protobuf::RepeatedField<float> data) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr.");
    GELOGE(FAILED, "[Check][Param] Node_proto is nullptr.");
    return;
  }
  if (!data.empty()) {
    const auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
      GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_floats(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAttrProto(onnx::NodeProto *const node_proto, const onnx::AttributeProto_AttributeType type,
                             const std::string &name, const ::google::protobuf::RepeatedPtrField<::std::string> data) {
  if (node_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_proto is nullptr.");
    GELOGE(FAILED, "[Check][Param] Node proto is nullptr.");
    return;
  }
  if (!data.empty()) {
    const auto attr = node_proto->add_attribute();
    if (attr == nullptr) {
      REPORT_INNER_ERROR("E19999", "add attr to node proto return nullptr.");
      GELOGE(GRAPH_FAILED, "[Check][Param] attr is nullptr.");
      return;
    }
    attr->set_name(name);
    for (auto &v : data) {
      attr->add_strings(v);
    }
    attr->set_type(type);
  }
}

void OnnxUtils::AddAllAttr(onnx::NodeProto *const node_proto, const ConstGeTensorDescPtr &op_desc,
                           const char_t *const prefix, const uint32_t idx) {
  const std::map<std::string, AnyValue> attr_maps = op_desc->GetAllAttrs();
  google::protobuf::Map<std::string, ge::proto::AttrDef> tensor_desc_map;
  (void)ModelSerializeImp::SerializeAllAttrsFromAnyMap(attr_maps, &tensor_desc_map);
  const std::string suffix = ":" + std::to_string(idx);
  AddAttrProtoForAttrsFromAttrMap(tensor_desc_map, node_proto, prefix, suffix);
}

void OnnxUtils::AddShapeFormatAndDtypeIntoProto(const bool is_input, onnx::NodeProto *const node_proto,
                                                const ge::ConstGeTensorDescPtr &desc, const uint32_t idx) {
  const std::string prefix = is_input ? "input_desc_" : "output_desc_";
  const auto data_type = ge::TypeUtils::DataTypeToSerialString(desc->GetDataType());
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
               prefix + "dtype:" + std::to_string(idx), &data_type);
  const auto data_type_origin = ge::TypeUtils::DataTypeToSerialString(desc->GetOriginDataType());
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
               prefix + "origin_dtype:" + std::to_string(idx), &data_type_origin);
  const auto dims = desc->GetShape().GetDims();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
               prefix + "shape:" + std::to_string(idx), &dims);
  const auto dims_origin = desc->GetOriginShape().GetDims();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
               prefix + "origin_shape:" + std::to_string(idx), &dims_origin);
  const auto layout = ge::TypeUtils::FormatToSerialString(desc->GetFormat());
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
               prefix + "layout:" + std::to_string(idx), &layout);
  const auto layout_origin = ge::TypeUtils::FormatToSerialString(desc->GetOriginFormat());
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
               prefix + "origin_layout:" + std::to_string(idx), &layout_origin);
}

void OnnxUtils::AddAttrProtoForOpInDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc) {
  const auto size_in = op_desc->GetAllInputsSize();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "input_desc_nums", &size_in);
  if (size_in > 0U) {
    for (uint32_t i = 0U; i < size_in; i++) {
      const auto input_desc = op_desc->GetInputDescPtrDfault(i);
      if (input_desc == nullptr || input_desc->impl_ == nullptr) {
        GELOGW("[Add][InAttr] Input desc of input %u is nullptr", i);
        continue;
      }
      AddShapeFormatAndDtypeIntoProto(true, node_proto, input_desc, i);
      auto &tensor_descriptor = input_desc->impl_->ext_meta_;
      const auto size = static_cast<int64_t>(tensor_descriptor.GetSize());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_size:" + std::to_string(i), &size);
      const auto weight_size = static_cast<int64_t>(tensor_descriptor.GetWeightSize());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_weight_size:" + std::to_string(i), &weight_size);
      const auto reuse_input_int = static_cast<int64_t>(tensor_descriptor.GetReuseInput());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_reuse_input:" + std::to_string(i), &reuse_input_int);
      const auto output_tensor_int = static_cast<int64_t>(tensor_descriptor.GetOutputTensor());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_output_tensor:" + std::to_string(i), &output_tensor_int);
      const auto device_type = tensor_descriptor.GetDeviceTypeStr();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   "input_desc_device_type:" + std::to_string(i), &device_type);
      const auto input_tensor_int = static_cast<int64_t>(tensor_descriptor.GetInputTensor());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_input_tensor:" + std::to_string(i), &input_tensor_int);
      const auto real_dim_cnt = static_cast<int64_t>(tensor_descriptor.GetRealDimCnt());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_real_dim_cnt:" + std::to_string(i), &real_dim_cnt);
      const auto data_offset = static_cast<int64_t>(tensor_descriptor.GetDataOffset());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_data_offset:" + std::to_string(i), &data_offset);
      const auto cmps_size = static_cast<int64_t>(tensor_descriptor.GetCmpsSize());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_cmps_size:" + std::to_string(i), &cmps_size);
      const auto cmps_tab = tensor_descriptor.GetCmpsTab();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   "input_desc_cmps_tab:" + std::to_string(i), &cmps_tab);
      const auto cmps_tab_offset = static_cast<int64_t>(tensor_descriptor.GetCmpsTabOffset());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "input_desc_cmps_tab_offset:" + std::to_string(i), &cmps_tab_offset);
      AddAllAttr(node_proto, input_desc, kPrefixForInputDesc, i);
    }
  }
}

void OnnxUtils::AddAttrProtoForOpOutDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc) {
  // Output describes
  const auto size_out = static_cast<int64_t>(op_desc->GetOutputsSize());
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "output_desc_nums", &size_out);
  if (size_out > 0) {
    for (uint32_t i = 0U; i < static_cast<uint32_t>(size_out); i++) {
      const auto output_desc = op_desc->GetOutputDescPtr(i);
      if (output_desc == nullptr || output_desc->impl_ == nullptr) {
        GELOGW("[Add][OutAttr] Output desc of output %u is nullptr", i);
        continue;
      }
      AddShapeFormatAndDtypeIntoProto(false, node_proto, output_desc, i);
      auto &tensor_descriptor = output_desc->impl_->ext_meta_;
      const auto size = static_cast<int64_t>(tensor_descriptor.GetSize());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "output_desc_size:" + std::to_string(i),
                   &size);
      const auto weight_size = static_cast<int64_t>(tensor_descriptor.GetWeightSize());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "output_desc_weight_size:" + std::to_string(i), &weight_size);
      const auto device_type = tensor_descriptor.GetDeviceTypeStr();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   "output_desc_device_type:" + std::to_string(i), &device_type);
      const auto real_dim_cnt = static_cast<int64_t>(tensor_descriptor.GetRealDimCnt());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT,
                   "output_desc_real_dim_cnt:" + std::to_string(i), &real_dim_cnt);
      AddAllAttr(node_proto, output_desc, kPrefixForOutputDesc, i);
    }
  }
}

void OnnxUtils::AddAttrProtoForOpInAndOutDesc(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc) {
  if ((node_proto == nullptr) || (op_desc == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node_proto or op_desc is nullptr");
    GELOGE(GRAPH_FAILED, "[Check][Param] node_proto or op_desc is nullptr");
    return;
  }
  AddAttrProtoForOpInDesc(node_proto, op_desc);
  AddAttrProtoForOpOutDesc(node_proto, op_desc);
}

void OnnxUtils::AddAttrProtoForAttrsFromAttrMap(
    const ::google::protobuf::Map<std::string, ::ge::proto::AttrDef> &attr_map, onnx::NodeProto *const node_proto,
    const std::string& prefix, const std::string& suffix) {
  for (const auto &item : attr_map) {
    const auto attr_name = item.first;
    const auto attr_def = item.second;
    const auto attr_type = attr_def.value_case();
    if (attr_type == ge::proto::AttrDef::kT) {
      const auto &tensor_def = attr_def.t();
      const auto &tensor_desc = tensor_def.desc();
      const auto data_type = ge::proto::DataType_Name(tensor_desc.dtype());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   prefix + attr_name + "_desc_dtype" + suffix, &data_type);
      const auto dims = tensor_desc.shape().dim();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
                   prefix + attr_name + "_desc_shape" + suffix, dims);
      const auto layout = tensor_desc.layout();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                   prefix + attr_name + "_desc_layout" + suffix, &layout);
      const auto device_type = tensor_desc.device_type();
      AddAttrProto(node_proto, ge::onnx::AttributeProto_AttributeType_STRING,
                   prefix + attr_name + "_desc_device_type" + suffix, &device_type);
      if (kDumpLevel == DUMP_ALL) {
        const auto data = tensor_def.data();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING,
                     prefix + attr_name + "_data" + suffix, &data);
      }
    }
    if (attr_type == ge::proto::AttrDef::kS) {
      if (kDumpLevel == DUMP_ALL) {
        const auto str_value = attr_def.s();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRING, prefix + attr_name + suffix, &str_value);
      }
    }
    if (attr_type == ge::proto::AttrDef::kI) {
      const auto int_value = attr_def.i();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, prefix + attr_name + suffix, &int_value);
    }
    if (attr_type == ge::proto::AttrDef::kF) {
      const auto float_value = attr_def.f();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_FLOAT, prefix + attr_name + suffix, &float_value);
    }
    if (attr_type == ge::proto::AttrDef::kB) {
      const auto int_value = static_cast<int64_t>(attr_def.b());
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, prefix + attr_name + suffix, &int_value);
    }
    if (attr_type == ge::proto::AttrDef::kList) {
      AddListAttrProto(attr_name, attr_def, prefix, suffix, node_proto);
    }
    if (attr_type == ge::proto::AttrDef::kListListInt) {
      const auto &list_value = attr_def.list_list_int();
      const auto &list_ints = list_value.list_list_i();
      int64_t list_index = 0;
      for (const auto &one_ints : list_ints) {
        const auto &ints = one_ints.list_i();
        AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS,
                     prefix + attr_name + suffix + "_" + std::to_string(list_index++), ints);
      }
    }
  }
}

void OnnxUtils::AddListAttrProto(const std::string &attr_name,
                                 const ::ge::proto::AttrDef &attr_def, const std::string &prefix,
                                 const std::string &suffix, onnx::NodeProto *node_proto) {
  const auto &list_value = attr_def.list();
  auto list_value_type = list_value.val_type();
  if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_STRING) {
    if (kDumpLevel == DUMP_ALL) {
      const auto &strings = list_value.s();
      AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, prefix + attr_name + suffix, strings);
    }
  }
  if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_FLOAT) {
    const auto &floats = list_value.f();
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_FLOATS, prefix + attr_name + suffix, floats);
  }
  if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_INT) {
    const auto &ints = list_value.i();
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, prefix + attr_name + suffix, ints);
  }
  if (list_value_type == ge::proto::AttrDef_ListValue_ListValueType::AttrDef_ListValue_ListValueType_VT_LIST_BOOL) {
    const auto &bools = list_value.b();
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, prefix + attr_name + suffix, bools);
  }
}

void OnnxUtils::AddCommonAttrIntoProto(onnx::NodeProto *const node_proto, const OpDescPtr &op_desc) {
  const auto meta_data = op_desc->impl_->meta_data_;
  const auto id = meta_data.GetId();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "id", &id);
  const auto stream_id = meta_data.GetStreamId();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INT, "stream_id", &stream_id);
  const auto &input_name = meta_data.GetInputNames();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "input_name", &input_name);
  const auto &src_name = meta_data.GetSrcNames();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "src_name", &src_name);
  const auto src_index = meta_data.GetSrcIndexes();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "src_index", &src_index);
  const auto &dst_name = meta_data.GetDstNames();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "dst_name", &dst_name);
  const auto dst_index = meta_data.GetDstIndexes();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "dst_index", &dst_index);
  const auto input_i = meta_data.GetInputOffsets();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "input_i", &input_i);
  const auto output_i = meta_data.GetOutputOffsets();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "output_i", &output_i);
  const auto workspace = op_desc->impl_->GetWorkspace();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "workspace", &workspace);
  const auto workspace_bytes = op_desc->impl_->GetWorkspaceBytes();
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "workspace_bytes", &workspace_bytes);
  const auto &is_input_const = meta_data.GetIsInputConsts();
  vector<int64_t> int_const(is_input_const.size());
  for (size_t idx = 0UL; idx < is_input_const.size(); ++idx) {
    int_const[idx] = static_cast<int64_t>(is_input_const[idx]);
  }
  AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "is_input_const", &int_const);
  google::protobuf::Map<std::string, ::ge::proto::AttrDef> op_def_attr_map;
  const std::map<string, AnyValue> attr_maps = op_desc->GetAllAttrs();
  (void)ModelSerializeImp::SerializeAllAttrsFromAnyMap(attr_maps, &op_def_attr_map);
  AddAttrProtoForAttrsFromAttrMap(op_def_attr_map, node_proto);
}

void OnnxUtils::AddAttrProtoFromNodeMembers(const NodePtr &node, onnx::NodeProto *const node_proto) {
  if ((node == nullptr) || (node->impl_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr.");
    GELOGE(GRAPH_FAILED, "[Check][Param] node is nullptr");
    return;
  }
  // 1.Attributes added from node's methods
  const auto send_list = node->impl_->send_event_id_list_;
  if (!send_list.empty()) {
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "send_event_id_list", &send_list);
  }
  const auto recv_list = node->impl_->recv_event_id_list_;
  if (!recv_list.empty()) {
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "recv_event_id_list", &recv_list);
  }
  const auto op_desc = node->impl_->op_;
  if (op_desc != nullptr && op_desc->impl_ != nullptr) {
    // for input_name_idx_ in opdesc
    const auto input_name_2_indexs = op_desc->GetAllInputName();
    ::google::protobuf::RepeatedPtrField<::std::string> input_names;
    ::google::protobuf::RepeatedField<::google::protobuf::int64> input_indexes;
    for (const auto &input_name_2_index: input_name_2_indexs) {
      std::string input_name = input_name_2_index.first;
      input_names.Add(std::move(input_name));
      input_indexes.Add(static_cast<int64_t>(input_name_2_index.second));
    }
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_STRINGS, "_input_name_key", input_names);
    AddAttrProto(node_proto, onnx::AttributeProto_AttributeType_INTS, "_input_name_value", input_indexes);
    // 2.Attributes added from node's op_(message OpDef)
    // Input and out describes
    AddAttrProtoForOpInAndOutDesc(node_proto, op_desc);
    // Others
    AddCommonAttrIntoProto(node_proto, op_desc);
  } else {
    REPORT_INNER_ERROR("E19999", "Opdesc is nullptr, node:%s", node->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] Opdesc is nullptr");
    return;
  }
}

bool OnnxUtils::EncodeNodeDesc(const NodePtr &node, onnx::NodeProto *const node_proto) {
  if ((node == nullptr) || (node->impl_ == nullptr) || (node_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or node_proto is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeOpDesc: Input Para Node Invalid");
    return false;
  }

  // 2.Encode std::map<std::string, GeAttrValue> attrs_ to AttributeProto
  for (auto &node_attr : node->impl_->attrs_) {
    AddAttrProtoFromAttribute(node_attr, node_proto);
  }
  // 3.Encode ge::Node members to AttributeProto
  AddAttrProtoFromNodeMembers(node, node_proto);

  // 4. Sort node attributes by name.
  std::sort(node_proto->mutable_attribute()->begin(), node_proto->mutable_attribute()->end(), AttrNameComp());
  return true;
}

void OnnxUtils::EncodeNodeLinkForNetronVisual(const NodePtr &node, onnx::NodeProto *const node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or node_proto is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeNodeLinkForNetronVisual: Input Para Node Invalid");
    return;
  }
  const auto &node_name = node->GetName();
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    if ((out_data_anchor != nullptr) && (!out_data_anchor->GetPeerInDataAnchors().empty())) {
      node_proto->add_output(node_name + ":" + std::to_string(out_data_anchor->GetIdx()));
    }
  }
  const auto out_control_anchor = node->GetOutControlAnchor();
  if ((out_control_anchor != nullptr) && (!out_control_anchor->GetPeerInControlAnchors().empty())) {
    node_proto->add_output(node_name + kControlAnchorIndex);
  }
}

bool OnnxUtils::EncodeNodeLink(const NodePtr &node, onnx::NodeProto *const node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or node_proto is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeNodeLink: Input Para Node Invalid");
    return false;
  }
  node_proto->clear_input();
  // 1. Add input by in data edge
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if ((peer_out_anchor != nullptr) && (peer_out_anchor->GetOwnerNode() != nullptr)) {
      node_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + ":" +
                            std::to_string(peer_out_anchor->GetIdx()));
    } else {
      // Add "" input
      node_proto->add_input("");
    }
  }

  // 2. Add input by in control edge
  const auto in_control_anchor = node->GetInControlAnchor();
  if (in_control_anchor != nullptr) {
    const auto peer_out_anchors = in_control_anchor->GetPeerOutControlAnchors();
    for (const auto &peer_out_anchor : peer_out_anchors) {
      if (peer_out_anchor->GetOwnerNode()) {
        node_proto->add_input(peer_out_anchor->GetOwnerNode()->GetName() + kControlAnchorIndex);
      }
    }
  } else {
    REPORT_INNER_ERROR("E19999", "In control anchor of node(%s) is nullptr", node->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] In control anchor of node(%s) is nullptr", node->GetName().c_str());
    return false;
  }

  // 3. Add output for Netron visual support
  EncodeNodeLinkForNetronVisual(node, node_proto);
  return true;
}

bool OnnxUtils::EncodeNode(const NodePtr &node, onnx::NodeProto *const node_proto) {
  if ((node == nullptr) || (node_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or node_proto is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeNode: Input Para Node Invalid");
    return false;
  }
  // 1. Encode name and type
  node_proto->set_name(node->GetName());
  /// Netron believes that some operators, such as the activation operator of softplus, only have one input,
  /// while the link relation of control anchor may exist in ge, resulting in two inputs. Therefore, "ge:" prefix
  /// is added to correctly display the link relation at the expense of some color features
  node_proto->set_op_type("ge:" + node->GetType());

  if (kDumpLevel != DUMP_WITH_OUT_DESC) {
    // 2.for attr
    if (!EncodeNodeDesc(node, node_proto)) {
      GELOGE(GRAPH_FAILED, "[Encode][NodeDesc] failed, node:%s", node->GetName().c_str());
      return false;
    }
  }
  // 3.for link info
  return EncodeNodeLink(node, node_proto);
}

void OnnxUtils::EncodeTypeProtoTensorType(const NodePtr &node, onnx::TypeProto_Tensor *const tensor_type) {
  if ((node == nullptr) || (tensor_type == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or tensor type is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeTypeProtoTensorType: Input Para Node or tensor_type Invalid");
    return;
  }
  const auto &op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGW("[Encode][Tensor] op_desc is empty, name %s, type %s", node->GetName().c_str(), node->GetType().c_str());
    return;
  }
  for (size_t i = 0U; i < op_desc->GetOutputsSize(); ++i) {
    const ConstGeTensorDescPtr &ge_tensor = op_desc->GetOutputDescPtr(static_cast<uint32_t>(i));
    if (ge_tensor == nullptr) {
      GELOGW("[Encode][Tensor] Output desc %zu of node %s is nullptr", i, node->GetName().c_str());
      continue;
    }
    const auto ge_data_type = ge_tensor->GetDataType();
    const auto onnx_data_type = EncodeDataType(ge_data_type);
    tensor_type->set_elem_type(onnx_data_type);
    onnx::TensorShapeProto *const shape = tensor_type->mutable_shape();
    if (shape == nullptr) {
      GELOGW("[Encode][Tensor] Shape is nullptr");
      continue;
    }
    for (const auto d : ge_tensor->GetShape().GetDims()) {
      const auto dim = shape->add_dim();
      dim->set_dim_value(d);
    }
  }
}

void OnnxUtils::EncodeValueInfo(const NodePtr &node, onnx::ValueInfoProto *const value_info_proto) {
  if ((node == nullptr) || (value_info_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param node or value info proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeValueInfo: Input Param Node or value_info_proto Invalid");
    return;
  }
  value_info_proto->set_name(node->GetName());
  onnx::TypeProto *const t = value_info_proto->mutable_type();
  onnx::TypeProto_Tensor *const tensor_type = t->mutable_tensor_type();
  EncodeTypeProtoTensorType(node, tensor_type);
}

bool OnnxUtils::EncodeGraph(const ConstComputeGraphPtr &graph, onnx::GraphProto *const graph_proto) {
  if ((graph == nullptr) || (graph_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param graph or graph proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] EncodeGraph: Input para Invalid");
    return false;
  }
  graph_proto->set_name(graph->GetName());
  // 1. Add graph inputs
  for (const auto &input : graph->GetInputNodes()) {
    const auto value_info_proto = graph_proto->add_input();
    EncodeValueInfo(input, value_info_proto);
  }
  // 2. Add graph outputs
  for (const auto &output : graph->GetOutputNodes()) {
    const auto value_info_proto = graph_proto->add_output();
    EncodeValueInfo(output, value_info_proto);
  }
  // 3. Add nodes
  for (const auto &node : graph->GetDirectNode()) {
    if (!EncodeNode(node, graph_proto->add_node())) {
      GELOGW("[Encode][Graph] Encode node %s failed", node->GetName().c_str());
      continue;
    }
  }
  return true;
}

bool OnnxUtils::ConvertGeModelToModelProto(const ge::Model &model, onnx::ModelProto &model_proto) {
  model_proto.set_model_version(static_cast<int64_t>(model.GetVersion()));
  model_proto.set_ir_version(onnx::IR_VERSION);
  model_proto.set_producer_name(model.GetName());
  auto &graph = model.graph_;
  const auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "GetComputeGraph for model return nullptr.");
    GELOGE(GRAPH_FAILED, "[Invoke][GetComputeGraph] return nullptr");
    return false;
  }
  const auto graph_proto = model_proto.mutable_graph();
  if (graph_proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "mutable_graph return nullptr, graph:%s",  compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Invoke][MutableGraph] return nullptr, graph:%s", compute_graph->GetName().c_str());
    return false;
  }
  if (!EncodeGraph(compute_graph, graph_proto)) {
    GELOGE(GRAPH_FAILED, "[Invoke][EncodeGraph] fail, graph:%s", compute_graph->GetName().c_str());
    return false;
  }

  // For subgraphs: a subgraph is represented by a node
  for (const auto &sub_compute_graph : compute_graph->GetAllSubgraphs()) {
    if (sub_compute_graph == nullptr) {
      GELOGW("[Convert][GeModel] Graph %s subgraph is nullptr, skip EncodeGraph", compute_graph->GetName().c_str());
      continue;
    }
    const auto node_proto = graph_proto->add_node();
    if (node_proto == nullptr) {
      GELOGW("[Convert][GeModel] Add node failed");
      continue;
    }
    node_proto->set_name(sub_compute_graph->GetName());
    node_proto->set_op_type(kNodeTypeForSubgraph);
    const auto attr = node_proto->add_attribute();
    attr->set_name("graph");
    attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
    const auto sub_graph_proto = attr->mutable_g();
    if (sub_graph_proto == nullptr) {
      GELOGW("[Convert][GeModel] Sub graph proto is nullptr");
      continue;
    }
    if (!EncodeGraph(sub_compute_graph, sub_graph_proto)) {
      GELOGW("[Convert][GeModel] Encode sub graph %s failed", sub_compute_graph->GetName().c_str());
      continue;
    }
  }
  return true;
}

// Part 2: from ONNX Protobuf convert to IR
static std::map<onnx::TensorProto_DataType, ge::DataType> onnxDataTypeToGeMap = {
    {onnx::TensorProto_DataType_INT64, DT_INT64},   {onnx::TensorProto_DataType_UINT64, DT_UINT64},
    {onnx::TensorProto_DataType_FLOAT, DT_FLOAT},   {onnx::TensorProto_DataType_INT32, DT_INT32},
    {onnx::TensorProto_DataType_UINT32, DT_UINT32}, {onnx::TensorProto_DataType_INT8, DT_INT8},
    {onnx::TensorProto_DataType_UINT8, DT_UINT8},   {onnx::TensorProto_DataType_INT16, DT_INT16},
    {onnx::TensorProto_DataType_UINT16, DT_UINT16}, {onnx::TensorProto_DataType_FLOAT16, DT_FLOAT16},
    {onnx::TensorProto_DataType_DOUBLE, DT_DOUBLE}, {onnx::TensorProto_DataType_BOOL, DT_BOOL},
};

ge::DataType OnnxUtils::DecodeDataType(const onnx::TensorProto_DataType data_type) {
  const auto it = onnxDataTypeToGeMap.find(data_type);
  if (it != onnxDataTypeToGeMap.end()) {
    return it->second;
  } else {
    GELOGW("[Decode][DataType] Datatype %u not support", data_type);
    return ge::DT_UNDEFINED;
  }
}

bool OnnxUtils::ParseNameAndIndex(const std::string &node_name_index, std::string &node_name, int32_t &idx) {
  const auto sep = node_name_index.rfind(':');
  if (sep == std::string::npos) {
    return false;
  }
  node_name = node_name_index.substr(0U, sep);
  const auto index_str = node_name_index.substr(sep + 1U);
  idx = static_cast<int32_t>(std::strtol(index_str.c_str(), nullptr, kDecimalBase));
  return true;
}

bool OnnxUtils::DecodeNodeLinkImp(const NodeLinkInfo &item, const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node_ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] DecodeNodeLinkImp: node_ptr is nullptr");
    return false;
  }
  // Data edge
  if (item.GetSrcOutIndex() >= 0) {
    const auto src_anchor = node_ptr->GetOutDataAnchor(item.GetSrcOutIndex());
    const auto dst_anchor = item.GetDstNode()->GetInDataAnchor(item.GetDstInIndex());
    if ((src_anchor == nullptr) || (dst_anchor == nullptr)) {
      REPORT_INNER_ERROR("E19999", "Get DataAnchor failed, %s:%d, %s:%d ", item.GetSrcNodeName().c_str(),
                         item.GetSrcOutIndex(), item.GetDstNodeName().c_str(), item.GetDstInIndex());
      GELOGE(GRAPH_FAILED, "[Get][DataAnchor] failed, %s:%d, %s:%d ", item.GetSrcNodeName().c_str(),
             item.GetSrcOutIndex(), item.GetDstNodeName().c_str(), item.GetDstInIndex());
      return false;
    }
    if (src_anchor->LinkTo(dst_anchor) != GRAPH_SUCCESS) {
      REPORT_INNER_ERROR("E19999", "src anchor link to dst anchor failed.");
      GELOGE(GRAPH_FAILED, "[Invoke][LinkTo] Data Anchor: src anchor link to dst anchor failed");
      return false;
    }
    // Control edge
  } else {
    const auto src_anchor = node_ptr->GetOutControlAnchor();
    const auto dst_anchor = item.GetDstNode()->GetInControlAnchor();
    if ((src_anchor == nullptr) || (dst_anchor == nullptr)) {
      REPORT_INNER_ERROR("E19999", "Get ControlAnchor failed, %s:%d, %s:%d ", item.GetSrcNodeName().c_str(),
                         item.GetSrcOutIndex(), item.GetDstNodeName().c_str(), item.GetDstInIndex());
      GELOGE(GRAPH_FAILED, "[Get][ControlAnchor] failed, %s:%d, %s:%d ", item.GetSrcNodeName().c_str(),
             item.GetSrcOutIndex(), item.GetDstNodeName().c_str(), item.GetDstInIndex());
      return false;
    }
    if (src_anchor->LinkTo(dst_anchor) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "src anchor(%s) link to dst anchor(%s) failed.",
                            src_anchor->GetOwnerNode()->GetName().c_str(),
                            dst_anchor->GetOwnerNode()->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Invoke][LinkTo] Control Anchor: src anchor(%s) link to dst anchor(%s) failed",
             src_anchor->GetOwnerNode()->GetName().c_str(), dst_anchor->GetOwnerNode()->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool OnnxUtils::DecodeNodeLink(const std::vector<onnx::NodeProto> &node_proto_vector,
                               const std::map<std::string, NodePtr> &node_map) {
  for (const auto &node_proto : node_proto_vector) {
    const auto &node_name = node_proto.name();
    const auto dst_node = node_map.find(node_name);
    if ((dst_node == node_map.end()) || (dst_node->second == nullptr)) {
      REPORT_INNER_ERROR("E19999", "destination node: %s find failed or is nullptr", node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] destination node: %s find failed or is nullptr", node_name.c_str());
      return false;
    }
    int32_t dst_index = 0;
    for (const auto &input : node_proto.input()) {
      std::string input_node_name;
      int32_t index = 0;
      if (ParseNameAndIndex(input, input_node_name, index)) {
        const auto item = NodeLinkInfo{input_node_name, index, dst_node->second, dst_index, node_proto.name()};
        const auto src_node = node_map.find(input_node_name);
        if (src_node == node_map.end()) {
          REPORT_INNER_ERROR("E19999", "find src node: %s failed", input_node_name.c_str());
          GELOGE(GRAPH_FAILED, "[Check][Param] find src node: %s failed", input_node_name.c_str());
          return false;
        }
        const auto node_ptr = src_node->second;
        if (node_ptr == nullptr) {
          REPORT_INNER_ERROR("E19999", "src node: %s is nullptr", input_node_name.c_str());
          GELOGE(GRAPH_FAILED, "[Check][Param] src node: %s is nullptr", input_node_name.c_str());
          return false;
        }
        if (!DecodeNodeLinkImp(item, node_ptr)) {
          GELOGE(GRAPH_FAILED, "[Invoke][DecodeNodeLinkImp] failed, node: %s", input_node_name.c_str());
          return false;
        }
      }
      if (index >= 0) {
        dst_index++;
      }
    }
  }
  return true;
}

void OnnxUtils::DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<std::string> &strings) {
  if (attr_proto.type() != ge::onnx::AttributeProto_AttributeType_STRINGS) {
    REPORT_INNER_ERROR("E19999", "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  for (int32_t i = 0; i < attr_proto.strings_size(); i++) {
    strings.push_back(attr_proto.strings(i));
  }
}

void OnnxUtils::DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::string &value) {
  if (attr_proto.type() != ge::onnx::AttributeProto_AttributeType_STRING) {
    REPORT_INNER_ERROR("E19999", "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  value = attr_proto.s();
}

void OnnxUtils::DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, std::vector<int64_t> &ints) {
  if (attr_proto.type() != ge::onnx::AttributeProto_AttributeType_INTS) {
    REPORT_INNER_ERROR("E19999", "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  for (int32_t i = 0; i < attr_proto.ints_size(); i++) {
    ints.push_back(attr_proto.ints(i));
  }
}

void OnnxUtils::DecodeAttribute(const ge::onnx::AttributeProto &attr_proto, int64_t &value) {
  if (attr_proto.type() != ge::onnx::AttributeProto_AttributeType_INT) {
    REPORT_INNER_ERROR("E19999", "Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attribute %s call wrong decode attribute function", attr_proto.name().c_str());
    return;
  }
  value = attr_proto.i();
}

void OnnxUtils::DecodeNodeAttributeForOpInDesc(const onnx::AttributeProto &attr_proto,
                                               const std::string &attr_name_for_input_desc,
                                               const int32_t idx,
                                               const OpDescPtr &op_desc) {
  const auto tensor_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(idx));
  if (tensor_desc == nullptr || tensor_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "MutableInputDesc index:%d return nullptr, op:%s, attr:%s",
                       idx, op_desc->GetName().c_str(), attr_name_for_input_desc.c_str());
    GELOGE(GRAPH_FAILED, "[Invoke][MutableInputDesc] index:%d return nullptr, op name %s, attr name %s",
           idx, op_desc->GetName().c_str(), attr_name_for_input_desc.c_str());
    return;
  }
  if (attr_name_for_input_desc == "input_desc_dtype") {
    const auto data_type = TypeUtils::SerialStringToDataType(attr_proto.s());
    tensor_desc->SetDataType(data_type);
  } else if (attr_name_for_input_desc == "input_desc_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    const GeShape ge_shape(ints);
    tensor_desc->SetShape(ge_shape);
  } else if (attr_name_for_input_desc == "input_desc_layout") {
    const auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    tensor_desc->SetFormat(data_format);
  } else if (attr_name_for_input_desc == "input_desc_origin_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    const GeShape ge_shape(ints);
    tensor_desc->SetOriginShape(ge_shape);
  } else if (attr_name_for_input_desc == "input_desc_origin_layout") {
    const auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    tensor_desc->SetOriginFormat(data_format);
  } else if (attr_name_for_input_desc == "input_desc_size") {
    int64_t input_size = 0;
    DecodeAttribute(attr_proto, input_size);
    tensor_desc->impl_->ext_meta_.SetSize(input_size);
  } else if (attr_name_for_input_desc == "input_desc_data_offset") {
    int64_t offset = 0;
    DecodeAttribute(attr_proto, offset);
    tensor_desc->impl_->ext_meta_.SetDataOffset(offset);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpOutDesc(const onnx::AttributeProto &attr_proto,
                                                const std::string &attr_name_for_output_desc,
                                                const int32_t index, const OpDescPtr &op_desc) {
  const auto tensor_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  if (tensor_desc == nullptr || tensor_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "MutableOutputDesc index:%d return nullptr, op:%s, attr:%s",
                       index, op_desc->GetName().c_str(), attr_name_for_output_desc.c_str());
    GELOGE(GRAPH_FAILED, "[Invoke][MutableOutputDesc] index:%d return nullptr, op name %s, attr name %s",
           index, op_desc->GetName().c_str(), attr_name_for_output_desc.c_str());
    return;
  }
  if (attr_name_for_output_desc == "output_desc_dtype") {
    const auto data_type = TypeUtils::SerialStringToDataType(attr_proto.s());
    tensor_desc->SetDataType(data_type);
  } else if (attr_name_for_output_desc == "output_desc_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    const GeShape ge_shape(ints);
    tensor_desc->SetShape(ge_shape);
  } else if (attr_name_for_output_desc == "output_desc_layout") {
    const auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    tensor_desc->SetFormat(data_format);
  } else if (attr_name_for_output_desc == "output_desc_origin_shape") {
    std::vector<std::int64_t> ints;
    DecodeAttribute(attr_proto, ints);
    const GeShape ge_shape(ints);
    tensor_desc->SetOriginShape(ge_shape);
  } else if (attr_name_for_output_desc == "output_desc_origin_layout") {
    const auto data_format = TypeUtils::SerialStringToFormat(attr_proto.s());
    tensor_desc->SetOriginFormat(data_format);
  } else if (attr_name_for_output_desc == "output_desc_size") {
    int64_t output_size = 0;
    DecodeAttribute(attr_proto, output_size);
    tensor_desc->impl_->ext_meta_.SetSize(output_size);
  } else if (attr_name_for_output_desc == "output_desc_data_offset") {
    int64_t offset = 0;
    DecodeAttribute(attr_proto, offset);
    tensor_desc->impl_->ext_meta_.SetDataOffset(offset);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpInAndOutDesc(const onnx::AttributeProto &attr_proto,
                                                     const std::string &attr_name_for_input_output_desc,
                                                     const int32_t idx,
                                                     const OpDescPtr &op_desc) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] op_desc is nullptr");
    return;
  }
  if (attr_name_for_input_output_desc.substr(0U, kInputPrefixLength) == "input") {
    DecodeNodeAttributeForOpInDesc(attr_proto, attr_name_for_input_output_desc, idx, op_desc);
  } else if (attr_name_for_input_output_desc.substr(0U, kOutputPrefixLength) == "output") {
    DecodeNodeAttributeForOpOutDesc(attr_proto, attr_name_for_input_output_desc, idx, op_desc);
  } else {
    return;
  }
}

void OnnxUtils::DecodeNodeAttributeForOpDesc(const onnx::AttributeProto &attr_proto, OpDescPtr &op_desc) {
  if (op_desc == nullptr || op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] DecodeNodeAttributeForOpDesc: op_desc is nullptr");
    return;
  }
  const auto &attr_name = attr_proto.name();
  std::string attr_name_for_input_output_desc;
  int32_t index = 0;
  if (!ParseNameAndIndex(attr_name, attr_name_for_input_output_desc, index)) {
    if (attr_name == "id") {
      op_desc->SetId(attr_proto.i());
    } else if (attr_name == "stream_id") {
      op_desc->SetStreamId(attr_proto.i());
    } else if (attr_name == "src_name") {
      std::vector<std::string> strings;
      DecodeAttribute(attr_proto, strings);
      op_desc->SetSrcName(strings);
    } else if (attr_name == "dst_name") {
      std::vector<std::string> strings;
      DecodeAttribute(attr_proto, strings);
      op_desc->SetDstName(strings);
    } else if (attr_name == "src_index") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetSrcIndex(ints);
    } else if (attr_name == "dst_index") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetDstIndex(ints);
    } else if (attr_name == "fusion_scope") {
      int64_t val = 0;
      DecodeAttribute(attr_proto, val);
      AnyValue av;
      (void)av.SetValue(val);
      (void)op_desc->SetAttr(attr_proto.name(), av);
    } else if (attr_name == "input_i") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetInputOffset(ints);
    } else if (attr_name == "output_i") {
      std::vector<std::int64_t> ints;
      DecodeAttribute(attr_proto, ints);
      op_desc->SetOutputOffset(ints);
    } else {
      return;
    }
    // Update input and output desc
  } else {
    DecodeNodeAttributeForOpInAndOutDesc(attr_proto, attr_name_for_input_output_desc, index, op_desc);
  }
}

bool OnnxUtils::DecodeNodeDesc(const onnx::NodeProto *const node_proto, OpDescPtr &op_desc) {
  if ((op_desc == nullptr) || (node_proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param op_desc or node_proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Op_desc is nullptr or node_proto is nullptr");
    return false;
  }
  // 1. Decode node_proto name and type
  op_desc->SetName(node_proto->name());
  const auto &node_type_with_ge_prefix = node_proto->op_type();
  const auto sep = node_type_with_ge_prefix.find(':');
  if (sep == std::string::npos) {
    return false;
  }
  const auto node_type = node_type_with_ge_prefix.substr(sep + 1U);
  op_desc->SetType(node_type);
  // 2. Add empty input and output desc
  for (const auto &attr : node_proto->attribute()) {
    if (attr.name() == "input_desc_nums") {
      const auto size_in = attr.i();
      for (int64_t i = 0; i < size_in; i++) {
        const GeTensorDesc ge_tensor_desc;
        GE_CHK_BOOL_EXEC(op_desc->AddInputDesc(ge_tensor_desc) == GRAPH_SUCCESS, continue, "Add inputdesc failed.");
      }
    }
    if (attr.name() == "output_desc_nums") {
      const auto size_out = attr.i();
      for (int64_t i = 0; i < size_out; i++) {
        const GeTensorDesc ge_tensor_desc;
        GE_CHK_BOOL_EXEC(op_desc->AddOutputDesc(ge_tensor_desc) == GRAPH_SUCCESS, continue, "Add outputdesc failed.");
      }
    }
  }
  // 3.Decode node_proto attributes
  for (decltype(node_proto->attribute_size()) i = 0; i < node_proto->attribute_size(); i++) {
    DecodeNodeAttributeForOpDesc(node_proto->attribute(i), op_desc);
  }
  return true;
}

bool OnnxUtils::AddInputAndOutputNodesForGraph(const onnx::GraphProto &graph_proto,
                                               ComputeGraphPtr &graph,
                                               const std::map<std::string, NodePtr> &node_map) {
  // Add inputs nodes for graph
  for (const auto &input : graph_proto.input()) {
    const auto &input_node_name = input.name();
    const auto input_node_item = node_map.find(input_node_name);
    if (input_node_item == node_map.end()) {
      REPORT_INNER_ERROR("E19999", "cannot find graph's input node %s in node_", input_node_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] cannot find graph's input node %s in node_", input_node_name.c_str());
      return false;
    }
    const auto ret = graph->AddInputNode(input_node_item->second);
    GE_CHK_BOOL_EXEC(ret != nullptr, continue,
                     "[Add][InputNode] %s failed, graph:%s", input_node_name.c_str(), graph->GetName().c_str());
  }
  // Add outputs nodes for graph
  for (const auto &output : graph_proto.output()) {
    const auto &output_name = output.name();
    const auto output_node_item = node_map.find(output_name);
    if (output_node_item == node_map.end()) {
      REPORT_INNER_ERROR("E19999", "cannot find graph's output node %s in node_", output_name.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] cannot find graph's output node %s in node_", output_name.c_str());
      return false;
    }
    const auto ret = graph->AddOutputNode(output_node_item->second);
    if (ret == nullptr) {
      GELOGW("[Decode][Graph] Add output node %s failed", output_name.c_str());
      continue;
    }
  }
  return true;
}

bool OnnxUtils::DecodeGraph(const int32_t recursion_depth,
                            const onnx::GraphProto &graph_proto, ComputeGraphPtr &graph) {
  if (recursion_depth > kMaxRecursiveDepth) {
    REPORT_INNER_ERROR("E19999", "param recursion_depth:%d is bigger than kMaxRecursiveDepth:%d",
                       recursion_depth, kMaxRecursiveDepth);
    GELOGE(GRAPH_FAILED, "[Check][Param] DecodeGraph: recursion depth is too large, abort");
    return false;
  }

  graph = ComGraphMakeShared<ge::ComputeGraph>(graph_proto.name());
  GE_CHK_BOOL_EXEC(graph != nullptr,
                   REPORT_CALL_ERROR("E19999", "create ComputeGraph failed.");
                   return false, "[Create][ComputeGraph]ComputeGraph make shared failed");
  /// 1. Decode all nodes first, node should include input
  /// and output nodes and nodes which represent sub graphs
  std::map<std::string, NodePtr> node_map;
  std::vector<onnx::NodeProto> node_proto_vector;
  for (const auto &node_proto : graph_proto.node()) {
    // a. nodes represent sub graphs
    if (node_proto.op_type() == kNodeTypeForSubgraph) {
      ComputeGraphPtr compute_graph;
      // in this case, node only have one attr, whose type is AttributeProto_AttributeType_GRAPH
      const auto &node_attr = node_proto.attribute(0);
      if ((node_attr.type() == onnx::AttributeProto_AttributeType_GRAPH) &&
          (DecodeGraph(recursion_depth + 1, node_attr.g(), compute_graph))) {
        (void)graph->AddSubGraph(compute_graph);
      } else {
        REPORT_INNER_ERROR("E19999", "Decode sub graph %s failed with node type:%d", node_proto.name().c_str(),
                           node_attr.type());
        GELOGE(GRAPH_FAILED, "[Check][Param] Decode sub graph %s failed with node type:%d", node_proto.name().c_str(),
               node_attr.type());
        return false;
      }
      // b. direct nodes in graph
    } else {
      node_proto_vector.push_back(node_proto);
      OpDescPtr op_desc = ComGraphMakeShared<OpDesc>();
      // b.1 For node desc
      if (!DecodeNodeDesc(&node_proto, op_desc)) {
        GELOGE(GRAPH_FAILED, "[Decode][NodeDesc] %s failed ", node_proto.name().c_str());
        return false;
      }
      auto node = graph->AddNode(op_desc);
      (void)node_map.insert(std::make_pair(node_proto.name(), node));
    }
  }
  /// We get all nodes in graph here
  /// b.2 For node link
  if (!DecodeNodeLink(node_proto_vector, node_map)) {
    GELOGE(GRAPH_FAILED, "[Decode][NodeLink] failed");
    return false;
  }

  return AddInputAndOutputNodesForGraph(graph_proto, graph, node_map);
}

bool OnnxUtils::ConvertModelProtoToGeModel(const onnx::ModelProto &model_proto, ge::Model &model) {
  model.name_ = model_proto.producer_name();
  model.version_ = static_cast<uint32_t>(model_proto.model_version());

  auto &graph_proto = model_proto.graph();
  ComputeGraphPtr compute_graph;
  // 0 means recursion depth, father call
  if (!DecodeGraph(0, graph_proto, compute_graph)) {
    GELOGE(GRAPH_FAILED, "[Decode][Graph] from graph_proto:%s failed", model.name_.c_str());
    return false;
  }
  model.graph_ = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  return true;
}
}  // namespace ge
