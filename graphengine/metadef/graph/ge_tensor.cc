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

#include "graph/ge_tensor.h"

#include <cstring>
#include <map>
#include <securec.h>
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "graph/ge_tensor_impl.h"
#include "graph/ge_attr_value.h"
#include "graph/model_serialize.h"
#include "graph/small_vector.h"
#include "graph/detail/model_serialize_imp.h"
#include "proto/ge_ir.pb.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/mem_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace{
template <typename T>
class IntegerChecker {
public:
  template <typename T1>
  static bool Compat(T1 const v) {
    static_assert((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t)),
                  "IntegerChecker can only check integers less than 64 bits");
    if (v >= static_cast<T1>(0)) {
      return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
    }
    return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
  }
};

static const size_t PAIR_ELEMENT_SIZE = 2UL;
static const size_t PAIR_ELEMENT_KEY = 0UL;
static const size_t PAIR_ELEMENT_VALUE = 1UL;
const char_t * const kKeyDataTypeSelfDefined = "__tensor_desc_data_type__";
const std::map<DataType, ::ge::proto::DataType> kDataTypeMap = {
    {DT_UNDEFINED, proto::DT_UNDEFINED},
    {DT_FLOAT, proto::DT_FLOAT},
    {DT_FLOAT16, proto::DT_FLOAT16},
    {DT_INT8, proto::DT_INT8},
    {DT_UINT8, proto::DT_UINT8},
    {DT_INT16, proto::DT_INT16},
    {DT_UINT16, proto::DT_UINT16},
    {DT_INT32, proto::DT_INT32},
    {DT_INT64, proto::DT_INT64},
    {DT_UINT32, proto::DT_UINT32},
    {DT_UINT64, proto::DT_UINT64},
    {DT_BOOL, proto::DT_BOOL},
    {DT_DOUBLE, proto::DT_DOUBLE},
    {DT_DUAL, proto::DT_DUAL},
    {DT_DUAL_SUB_INT8, proto::DT_DUAL_SUB_INT8},
    {DT_DUAL_SUB_UINT8, proto::DT_DUAL_SUB_UINT8},
    {DT_COMPLEX64, proto::DT_COMPLEX64},
    {DT_COMPLEX128, proto::DT_COMPLEX128},
    {DT_QINT8, proto::DT_QINT8},
    {DT_QINT16, proto::DT_QINT16},
    {DT_QINT32, proto::DT_QINT32},
    {DT_QUINT8, proto::DT_QUINT8},
    {DT_QUINT16, proto::DT_QUINT16},
    {DT_RESOURCE, proto::DT_RESOURCE},
    {DT_STRING_REF, proto::DT_STRING_REF},
    {DT_STRING, proto::DT_STRING},
    {DT_VARIANT, proto::DT_VARIANT},
    {DT_BF16, proto::DT_BF16},
    {DT_INT4, proto::DT_INT4},
    {DT_UINT1, proto::DT_UINT1},
    {DT_INT2, proto::DT_INT2},
    {DT_UINT2, proto::DT_UINT2}
};

const std::map<DataType, int> kDataTypeSelfDefinedMap = {
    {DT_DUAL, 13},  {DT_DUAL_SUB_INT8, 14}, {DT_DUAL_SUB_UINT8, 15}, {DT_COMPLEX64, 16}, {DT_COMPLEX128, 17},
    {DT_QINT8, 18}, {DT_QINT16, 19},        {DT_QINT32, 20},         {DT_QUINT8, 21},    {DT_QUINT16, 22},
};

const std::map<DeviceType, std::string> kDeviceToStrMap = {
    {NPU, "NPU"}, {CPU, "CPU"},
};

const std::map<std::string, DeviceType> kStrToDeviceMap = {
    {"NPU", NPU}, {"CPU", CPU}
};

const std::string TENSOR_UTILS_SIZE = "size";
const std::string TENSOR_UTILS_WEIGHT_SIZE = "weight_size";
const std::string TENSOR_UTILS_REUSE_INPUT = "reuse_input";
const std::string TENSOR_UTILS_OUTPUT_TENSOR = "output_tensor";
const std::string TENSOR_UTILS_DEVICE_TYPE = "device_type";
const std::string TENSOR_UTILS_INPUT_TENSOR = "input_tensor";
const std::string TENSOR_UTILS_REAL_DIM_CNT = "real_dim_cnt";
const std::string TENSOR_UTILS_REUSE_INPUT_INDEX = "reuse_input_index";
const std::string TENSOR_UTILS_DATA_OFFSET = "data_offset";
const std::string TENSOR_UTILS_CMPS_SIZE = "cmps_size";
const std::string TENSOR_UTILS_CMPS_TAB = "cmps_tab";
const std::string TENSOR_UTILS_CMPS_TAB_OFFSET = "cmps_tab_offset";
const std::string TENSOR_UTILS_CMPSINFO = "cmps_info";
const std::string TENSOR_UTILS_ALLOFFSET_QUANTIZE_INFO = "alloffset_quantize_info";
const std::string TENSOR_UTILS_RC = "rc";
const std::string TENSOR_UTILS_ORIGIN_SHAPE = "origin_shape";
const std::string TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED = "origin_shape_initialized";
const std::string TENSOR_UTILS_ORIGIN_FORMAT = "origin_format";
const std::string TENSOR_UTILS_ORIGIN_DATA_TYPE = "origin_data_type";
const std::string TENSOR_UTILS_SHAPE_RANGE = "shape_range";
const std::string TENSOR_UTILS_ORIGIN_SHAPE_RANGE = "origin_shape_range";
const std::string TENSOR_UTILS_VALUE_RANGE = "value_range";
const std::string TENSOR_UTILS_REF_PORT_INDEX = "ref_port_index";
const std::string TENSOR_UTILS_PLACEMENT = "placement";
}

void GeTensorSerializeUtils::GeShapeAsProto(const GeShape &shape, proto::ShapeDef *proto) {
  if (proto != nullptr) {
    proto->clear_dim();
    for (auto dim : shape.GetDims()) {
      proto->add_dim(dim);
    }
  }
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDescImpl &desc, proto::TensorDescriptor *proto) {
  if (proto != nullptr) {
    // serialize extension tensor meta data
    proto->set_size(desc.ext_meta_.GetSize());
    proto->set_weight_size(desc.ext_meta_.GetWeightSize());
    proto->set_reuse_input(desc.ext_meta_.GetReuseInput());
    proto->set_output_tensor(desc.ext_meta_.GetOutputTensor());
    if (kDeviceToStrMap.find(desc.ext_meta_.GetDeviceType()) != kDeviceToStrMap.end()) {
      proto->set_device_type(kDeviceToStrMap.at(desc.ext_meta_.GetDeviceType()));
    }
    proto->set_input_tensor(desc.ext_meta_.GetInputTensor());
    proto->set_real_dim_cnt(static_cast<int64_t>(desc.ext_meta_.GetRealDimCnt()));
    proto->set_reuse_input_index(static_cast<int64_t>(desc.ext_meta_.GetReuseInputIndex()));
    proto->set_data_offset(desc.ext_meta_.GetDataOffset());
    proto->set_cmps_size(desc.ext_meta_.GetCmpsSize());
    proto->set_cmps_tab(desc.ext_meta_.GetCmpsTab());
    proto->set_cmps_tab_offset(desc.ext_meta_.GetCmpsTabOffset());

    // serialize attributes
    if (!ModelSerializeImp::SerializeAllAttrsFromAnyMap(desc.attrs_.GetAllAttrs(), proto->mutable_attr())) {
      GELOGE(GRAPH_FAILED, "GeTensorDesc attr serialize failed.");
      return;
    }

    // serialize member object
    (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_FORMAT].set_s(
        TypeUtils::FormatToSerialString(desc.GetOriginFormat()));
    if (desc.GetOriginDataType() != DT_UNDEFINED) {
      (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_DATA_TYPE].set_s(
          TypeUtils::DataTypeToSerialString(desc.GetOriginDataType()));
    }

    const bool is_origin_shape_init = desc.ext_meta_.IsOriginShapeInited();
    (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED].set_b(is_origin_shape_init);
    if (is_origin_shape_init) {
      auto const origin_shape_proto_list = (*proto->mutable_attr())[TENSOR_UTILS_ORIGIN_SHAPE].mutable_list();
      origin_shape_proto_list->clear_i();
      for (auto const dim : desc.OriginShapeReference().GetDims()) {
        origin_shape_proto_list->add_i(dim);
      }
      origin_shape_proto_list->set_val_type(proto::AttrDef::ListValue::VT_LIST_INT);
    }

    auto const iter = kDataTypeMap.find(desc.GetDataType());
    if (iter != kDataTypeMap.end()) {
      proto->set_dtype(iter->second);
    } else { // maybe custom data type
      proto->set_dtype(kDataTypeMap.at(DT_UNDEFINED));
    }
    proto->set_layout(TypeUtils::FormatToSerialString(desc.GetFormat()));
    GeTensorSerializeUtils::GeShapeAsProto(desc.ShapeReference(), proto->mutable_shape());
  }
}
void GeTensorSerializeUtils::GeTensorDescAsProto(const GeTensorDesc &desc, proto::TensorDescriptor *proto) {
  GeTensorSerializeUtils::GeTensorDescAsProto(*desc.impl_, proto);
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensorImpl &tensor, proto::TensorDef *proto) {
  if (tensor.tensor_def_.protoOwner_ != nullptr) {
    if (tensor.tensor_def_.protoMsg_ != nullptr) {
      *proto = *tensor.tensor_def_.protoMsg_;
      GeTensorDescAsProto(tensor.desc_, proto->mutable_desc());
    }
  } else {
    if ((tensor.tensor_data_.impl_ != nullptr) && (tensor.tensor_data_.impl_->tensor_descriptor_ != nullptr)) {
      GeTensorDescAsProto(*tensor.tensor_data_.impl_->tensor_descriptor_, proto->mutable_desc());
    }
    proto->set_data(tensor.tensor_data_.data(), tensor.tensor_data_.size());
  }
}
void GeTensorSerializeUtils::GeTensorAsProto(const GeTensor &tensor, proto::TensorDef *proto) {
  GeTensorSerializeUtils::GeTensorAsProto(*tensor.impl_, proto);
}

void GeTensorSerializeUtils::AssembleGeShapeFromProto(const proto::ShapeDef *proto, GeShape &shape) {
  if (proto != nullptr) {
    shape = GeShape(nullptr, const_cast<proto::ShapeDef *>(proto));
  }
}
void GeTensorSerializeUtils::AssembleGeTensorDescFromProto(
    const proto::TensorDescriptor *proto, GeTensorDesc &desc) {
  if (proto != nullptr) {
    desc = GeTensorDesc(nullptr, const_cast<proto::TensorDescriptor *>(proto));
  }
}
void GeTensorSerializeUtils::AssembleGeTensorFromProto(const proto::TensorDef *proto, GeTensor &tensor) {
  if (proto != nullptr) {
    tensor = GeTensor(nullptr, const_cast<proto::TensorDef *>(proto));
  }
}

void GeTensorSerializeUtils::NormalizeGeTensorDescProto(proto::TensorDescriptor *proto) {
  if (proto == nullptr) {
    return;
  }
  auto &attr_map = *(proto->mutable_attr());
  auto iter = attr_map.find(TENSOR_UTILS_SIZE);
  if (iter != attr_map.end()) {
    proto->set_size(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_WEIGHT_SIZE);
  if (attr_map.end() != iter) {
    proto->set_weight_size(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_REUSE_INPUT);
  if (attr_map.end() != iter) {
    proto->set_reuse_input(iter->second.b());
  }
  iter = attr_map.find(TENSOR_UTILS_OUTPUT_TENSOR);
  if (attr_map.end() != iter) {
    proto->set_output_tensor(iter->second.b());
  }
  iter = attr_map.find(TENSOR_UTILS_DEVICE_TYPE);
  if (attr_map.end() != iter) {
    proto->set_device_type(iter->second.s());
  }
  iter = attr_map.find(TENSOR_UTILS_INPUT_TENSOR);
  if (attr_map.end() != iter) {
    proto->set_input_tensor(iter->second.b());
  }
  iter = attr_map.find(TENSOR_UTILS_REAL_DIM_CNT);
  if (attr_map.end() != iter) {
    proto->set_real_dim_cnt(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_REUSE_INPUT_INDEX);
  if (attr_map.end() != iter) {
    proto->set_reuse_input_index(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_DATA_OFFSET);
  if (attr_map.end() != iter) {
    proto->set_data_offset(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_CMPS_SIZE);
  if (attr_map.end() != iter) {
    proto->set_cmps_size(iter->second.i());
  }
  iter = attr_map.find(TENSOR_UTILS_CMPS_TAB);
  if (attr_map.end() != iter) {
    proto->set_cmps_tab(iter->second.s());
  }
  iter = attr_map.find(TENSOR_UTILS_CMPS_TAB_OFFSET);
  if (attr_map.end() != iter) {
    proto->set_cmps_tab_offset(iter->second.i());
  }
}

void GeTensorSerializeUtils::GetShapeFromDescProto(const proto::TensorDescriptor *proto, GeShape &shape) {
  shape.SetDimNum(static_cast<size_t>(proto->shape().dim_size()));
  size_t i = 0U;
  for (auto const dim : proto->shape().dim()) {
    (void)shape.SetDim(i++, dim);
  }
}

void GeTensorSerializeUtils::GetOriginShapeFromDescProto(const proto::TensorDescriptor *proto, GeShape &shape) {
  auto &attrs = proto->attr();
  auto const iter = attrs.find(TENSOR_UTILS_ORIGIN_SHAPE);
  if (iter != attrs.end()) {
    shape.SetDimNum(static_cast<size_t>(iter->second.list().i_size()));
    size_t i = 0U;
    for (auto const dim : iter->second.list().i()) {
      (void)shape.SetDim(i++, dim);
    }
  }
}

void GeTensorSerializeUtils::GetDtypeFromDescProto(const proto::TensorDescriptor *proto, DataType &dtype) {
  dtype = DT_UNDEFINED;
  auto &attrs = proto->attr();
  auto const iter = attrs.find(kKeyDataTypeSelfDefined);
  if (iter == attrs.end()) {
    auto const proto_dtype = proto->dtype();
    auto const founded = std::find_if(
        kDataTypeMap.begin(), kDataTypeMap.end(),
        [proto_dtype](const std::pair<DataType, ge::proto::DataType> &item) {
          return item.second == proto_dtype;
        });
    if (founded != kDataTypeMap.end()) {
      dtype = founded->first;
      return;
    }
  } else {  // Custom defined data type set
    const int64_t data_type_proto = iter->second.i();
    auto const founded = std::find_if(kDataTypeSelfDefinedMap.begin(), kDataTypeSelfDefinedMap.end(),
        [data_type_proto](const std::pair<DataType, int> &item) {
        return item.second == data_type_proto;
        });
    if (founded != kDataTypeSelfDefinedMap.end()) {
      dtype = founded->first;
      return;
    }
  }
}

void GeTensorSerializeUtils::GetOriginDtypeFromDescProto(const proto::TensorDescriptor *proto, DataType &dtype) {
  auto &attrs = proto->attr();
  auto const iter = attrs.find(TENSOR_UTILS_ORIGIN_DATA_TYPE);
  if (iter != attrs.end()) {
    dtype = TypeUtils::SerialStringToDataType(iter->second.s());
  }
}

void GeTensorSerializeUtils::GetFormatFromDescProto(const proto::TensorDescriptor *proto, Format &format) {
  format = TypeUtils::SerialStringToFormat(proto->layout());
}

void GeTensorSerializeUtils::GetOriginFormatFromDescProto(const proto::TensorDescriptor *proto, Format &format) {
  auto &attrs = proto->attr();
  auto const iter = attrs.find(TENSOR_UTILS_ORIGIN_FORMAT);
  if (iter != attrs.end()) {
    format = TypeUtils::SerialStringToFormat(iter->second.s());
  }
}

const static size_t kDefaultDimsNum = 8U;

class GeShapeImpl {
  using DimsType = SmallVector<int64_t, kDefaultDimsNum>;
 public:
  GeShapeImpl() = default;
  ~GeShapeImpl() = default;
  explicit GeShapeImpl(const std::vector<int64_t> &dims);
  GeShapeImpl(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg);

  void SetDimNum(const size_t dim_num);
  void AppendDim(const int64_t dim_size);
  bool IsUnknownDimNum() const;
  void SetIsUnknownDimNum();
  size_t GetDimNum() const;
  int64_t GetDim(const size_t idx) const;
  graphStatus SetDim(const size_t idx, const int64_t value);
  std::vector<int64_t> ShapeImplGetDims() const;
  std::string ShapeImplToString() const;
  int64_t GetShapeSize() const;
  bool IsUnknownShape() const;
  bool IsScalar() const;

  bool operator==(const GeShapeImpl &other) const;

private:
  DimsType dims_;
  friend class GeTensorDesc;
};

// Default
GeShapeImpl::GeShapeImpl(const std::vector<int64_t> &dims) {
  dims_.reserve(dims.size());
  for (auto dim : dims) {
    dims_.emplace_back(dim);
  }
}

void GeShapeImpl::SetDimNum(const size_t dim_num) {
  dims_.resize(dim_num, UNKNOWN_DIM);
}

void GeShapeImpl::AppendDim(const int64_t dim_size) {
  dims_.push_back(dim_size);
}

bool GeShapeImpl::IsUnknownDimNum() const {
  return (dims_.size() == 1UL) && (dims_[0UL] == UNKNOWN_DIM_NUM);
}

void GeShapeImpl::SetIsUnknownDimNum() {
  dims_.resize(1UL, UNKNOWN_DIM_NUM);
  dims_[0UL] = UNKNOWN_DIM_NUM;
}

size_t GeShapeImpl::GetDimNum() const {
  if (IsUnknownDimNum()) {
    return 0UL;
  }
  return dims_.size();
}

int64_t GeShapeImpl::GetDim(const size_t idx) const {
  if (idx < dims_.size()) {
    return dims_[idx];
  } else {
    return 0;
  }
}

graphStatus GeShapeImpl::SetDim(const size_t idx, const int64_t value) {
  if (idx < dims_.size()) {
    dims_[idx] = value;
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

std::vector<int64_t> GeShapeImpl::ShapeImplGetDims() const {
  std::vector<int64_t> dims;
  dims.reserve(dims_.size());
  for (auto dim : dims_) {
    dims.emplace_back(dim);
  }
  return dims;
}

std::string GeShapeImpl::ShapeImplToString() const {
  if (dims_.empty()) {
    return "";
  }

  std::stringstream ss;
  ss << dims_[0UL];
  for (size_t i = 1UL; i < dims_.size(); i++) {
    ss << "," << dims_[i];
  }
  return ss.str();
}

int64_t GeShapeImpl::GetShapeSize() const {
  if (dims_.empty()) {
    return 0;
  }
  int64_t shape_size = 1;
  for (auto const dim : dims_) {
    if ((dim == UNKNOWN_DIM) || (dim == UNKNOWN_DIM_NUM) || (dim < 0)) {
      return -1;
    } else if (dim == 0) {
      return 0;
    } else {
      if (shape_size > (INT64_MAX / dim)) {
        return -1;
      }
      shape_size *= dim;
    }
  }
  return shape_size;
}

bool GeShapeImpl::IsUnknownShape() const {
  return std::any_of(dims_.begin(), dims_.end(), [&](const int64_t &dim) {
      return (dim == UNKNOWN_DIM) || (dim == UNKNOWN_DIM_NUM) || (dim < 0);
      });
}

bool GeShapeImpl::IsScalar() const {
  return dims_.empty();
}

GeShapeImpl::GeShapeImpl(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg) {
  if (proto_msg != nullptr) {
    for (auto &dim : *proto_msg->mutable_dim()) {
      dims_.emplace_back(dim);
    }
  }
}

bool GeShapeImpl::operator==(const GeShapeImpl &other) const {
  return this->ShapeImplGetDims() == other.ShapeImplGetDims();
}

GeShape::GeShape() : impl_(MakeShared<GeShapeImpl>()) {}
GeShape::GeShape(std::vector<int64_t> s)
    : impl_(MakeShared<GeShapeImpl>(std::move(s))) {}
GeShape::GeShape(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg)
    : impl_(MakeShared<GeShapeImpl>(proto_owner, proto_msg)) {}

GeShape::GeShape(const GeShape &other)
    : impl_(MakeShared<GeShapeImpl>(*(other.impl_))) {}

GeShape::GeShape(GeShape &&other)
    : impl_(MakeShared<GeShapeImpl>(std::move(*(other.impl_)))) {}

GeShape::~GeShape() = default;

size_t GeShape::GetDimNum() const {
  return impl_->GetDimNum();
}

void GeShape::SetDimNum(const size_t dim_num) {
  impl_->SetDimNum(dim_num);
}

void GeShape::AppendDim(const int64_t dim_size) {
  impl_->AppendDim(dim_size);
}

bool GeShape::IsUnknownDimNum() const {
  return impl_->IsUnknownDimNum();
}

void GeShape::SetIsUnknownDimNum() {
  impl_->SetIsUnknownDimNum();
}

int64_t GeShape::GetDim(const size_t idx) const {
  return impl_->GetDim(idx);
}

graphStatus GeShape::SetDim(const size_t idx, const int64_t value) {
  return impl_->SetDim(idx, value);
}

std::vector<int64_t> GeShape::GetDims() const {
  return impl_->ShapeImplGetDims();
}

std::string GeShape::ToString() const {
  return impl_->ShapeImplToString();
}

int64_t GeShape::GetShapeSize() const {
  return impl_->GetShapeSize();
}

bool GeShape::IsUnknownShape() const {
  return impl_->IsUnknownShape();
}

bool GeShape::IsScalar() const {
  return impl_->IsScalar();
}

GeShape &GeShape::operator=(const GeShape &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

GeShape &GeShape::operator=(GeShape &&other) {
  if (&other != this) {
    impl_ = other.impl_;
  }
  return *this;
}

bool GeShape::operator==(const GeShape &other) const {
  return *impl_ == *(other.impl_);
}

GeTensorDescImpl::GeTensorDescImpl(const GeShape &shape, const Format format, const DataType dt) : GeTensorDescImpl() {
  SetFormat(format);
  SetDataType(dt);
  shape_ = shape;
}

GeTensorDescImpl::GeTensorDescImpl(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
    : GeTensorDescImpl() {
  if (proto_msg == nullptr) {
    GELOGE(INTERNAL_ERROR, "Try assemble ge tensor desc from nullptr proto");
    return;
  }
  // normalize the input TensorDescriptor，A metadata information maybe stored in different fields of TensorDescriptor,
  // This function needs to prioritize and determine the final metadata information used.
  // After standardization, the direct member field on TensorDescriptor is always valid
  GeTensorSerializeUtils::NormalizeGeTensorDescProto(proto_msg);

  // store high frequency attributes to member field
  GeTensorSerializeUtils::GetOriginFormatFromDescProto(proto_msg, origin_format_);
  GeTensorSerializeUtils::GetOriginDtypeFromDescProto(proto_msg, origin_dtype_);
  GeTensorSerializeUtils::GetOriginShapeFromDescProto(proto_msg, origin_shape_);

  GeTensorSerializeUtils::GetFormatFromDescProto(proto_msg, format_);
  GeTensorSerializeUtils::GetDtypeFromDescProto(proto_msg, dtype_);
  GeTensorSerializeUtils::GetShapeFromDescProto(proto_msg, shape_);

  // get extension tensor desc metadata
  ext_meta_.SetSize(proto_msg->size());
  ext_meta_.SetWeightSize(proto_msg->weight_size());
  ext_meta_.SetReuseInput(proto_msg->reuse_input());
  ext_meta_.SetOutputTensor(proto_msg->output_tensor());
  if (kStrToDeviceMap.find(proto_msg->device_type()) != kStrToDeviceMap.end()) {
    ext_meta_.SetDeviceType(kStrToDeviceMap.at(proto_msg->device_type()));
  }
  ext_meta_.SetInputTensor(proto_msg->input_tensor());
  if (IntegerChecker<uint32_t>::Compat(proto_msg->real_dim_cnt())) {
    ext_meta_.SetRealDimCnt(static_cast<uint32_t>(proto_msg->real_dim_cnt()));
  }
  if (IntegerChecker<uint32_t>::Compat(proto_msg->reuse_input_index())) {
    ext_meta_.SetReuseInputIndex(static_cast<uint32_t>(proto_msg->reuse_input_index()));
  }
  ext_meta_.SetDataOffset(proto_msg->data_offset());
  ext_meta_.SetCmpsSize(proto_msg->cmps_size());
  ext_meta_.SetCmpsTab(proto_msg->cmps_tab());
  ext_meta_.SetCmpsTabOffset(proto_msg->cmps_tab_offset());

  auto &attr_map = *(proto_msg->mutable_attr());
  const auto iter = attr_map.find(TENSOR_UTILS_ORIGIN_SHAPE_INITIALIZED);
  if (iter != attr_map.end()) {
    ext_meta_.SetOriginShapeInited(iter->second.b());
  }

  // note that we deserialize attributes in implement of GeTensor constructor
}

void GeTensorDescImpl::SetDataType(const DataType dtype) {
  dtype_ = dtype;
}

void GeTensorDescImpl::SetOriginDataType(const DataType dtype) {
  origin_dtype_ = dtype;
}

DataType GeTensorDescImpl::GetOriginDataType() const {
  return origin_dtype_;
}

void GeTensorDescImpl::SetFormat(const Format format) {
  format_ = format;
}

void GeTensorDescImpl::SetOriginFormat(const Format format) {
  origin_format_ = format;
}

Format GeTensorDescImpl::GetOriginFormat() const {
  return origin_format_;
}

GeShape &GeTensorDescImpl::ShapeReference() const {
  return shape_;
}

GeShape &GeTensorDescImpl::OriginShapeReference() const {
  return origin_shape_;
}

bool GeTensorDescImpl::GeTensorDescAttrsAreEqual(const GeTensorDescImpl &other) const {
  // The definition of attribute equality remains unchanged
  return ((shape_ == other.shape_) &&
          (dtype_ == other.dtype_) &&
          (format_ == other.format_) &&
          (ext_meta_ == other.ext_meta_));
}

bool GeTensorDescImpl::operator==(const GeTensorDescImpl &other) const {
  // The definition of attribute equality remains unchanged
  return (origin_shape_ == other.origin_shape_) && (origin_format_ == other.origin_format_) &&
         (origin_dtype_ == other.origin_dtype_) && (GeTensorDescAttrsAreEqual(other));
}

ProtoAttrMap &GeTensorDescImpl::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &GeTensorDescImpl::GetAttrMap() const {
  return attrs_;
}

void GeTensorDescImpl::SetShape(const GeShape &shape) const {
  ShapeReference() = std::move(shape);
}

Format GeTensorDescImpl::GetFormat() const {
  return format_;
}

void GeTensorDescImpl::SetName(const std::string &name) {
  ext_meta_.SetName(name);
}

const std::string GeTensorDescImpl::GetName() const {
  return ext_meta_.GetName();
}

DataType GeTensorDescImpl::GetDataType() const {
  return dtype_;
}

std::string GeTensorDescImpl::ExtMeta::GetDeviceTypeStr() {
  auto const iter = kDeviceToStrMap.find(device_type);
  if (iter != kDeviceToStrMap.end()) {
    return iter->second;
  }
  const static std::string kDefaultTypeString{"NPU"};
  return kDefaultTypeString;
}

GeTensorDesc::GeTensorDesc() : AttrHolder(),
    impl_(ComGraphMakeShared<GeTensorDescImpl>()) {}

// Default
GeTensorDesc::GeTensorDesc(const GeShape &shape, const Format format, const DataType dt) : AttrHolder(),
    impl_(ComGraphMakeShared<GeTensorDescImpl>(shape, format, dt)) {}

// Default
GeTensorDesc::GeTensorDesc(const GeTensorDesc &desc) : AttrHolder(desc),
    impl_(ComGraphMakeShared<GeTensorDescImpl>(*(desc.impl_))) {}

// Default
GeTensorDesc::GeTensorDesc(GeTensorDesc &&desc) : AttrHolder(desc), impl_(desc.impl_) {}

GeTensorDesc::~GeTensorDesc() = default;

GeTensorDesc::GeTensorDesc(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg)
    : AttrHolder(), impl_(ComGraphMakeShared<GeTensorDescImpl>(proto_owner, proto_msg)) {
  if (proto_msg != nullptr) {
    if (!ModelSerializeImp::DeserializeAllAttrsToAttrHolder(proto_msg->attr(), this)) {
      GELOGW("GeTensorDesc attr deserialize failed.");
    }
  }
}

bool GeTensorDesc::GeTensorDescAttrsAreEqual(const GeTensorDesc &r_ge_tensor_desc) const {
  return impl_->GeTensorDescAttrsAreEqual(*(r_ge_tensor_desc.impl_));
}

bool GeTensorDesc::operator==(const GeTensorDesc &r_ge_tensor_desc) const {
  return *impl_ == *r_ge_tensor_desc.impl_;
}

GeShape &GeTensorDesc::ShapeReference() const {
  return impl_->ShapeReference();
}

ProtoAttrMap &GeTensorDesc::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMap &GeTensorDesc::GetAttrMap() const {
  return impl_->GetAttrMap();
}

void GeTensorDesc::Update(const GeShape &shape, const Format format, const DataType dt) {
  ShapeReference() = shape;
  SetFormat(format);
  SetDataType(dt);
}
const GeShape &GeTensorDesc::GetShape() const { return ShapeReference(); }

GeShape &GeTensorDesc::MutableShape() { return ShapeReference(); }

void GeTensorDesc::SetShape(const GeShape &shape) { ShapeReference() = shape; }

void GeTensorDesc::SetShape(GeShape &&shape) { ShapeReference() = std::move(shape); }

// set shape with -2, it stand for unknown shape
void GeTensorDesc::SetUnknownDimNumShape() { SetShape(GeShape({UNKNOWN_DIM_NUM})); }

// for unknown shape
graphStatus GeTensorDesc::SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<std::vector<int64_t>> value_range;
  for (const auto &ele : range) {
    value_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto const ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_VALUE_RANGE, value_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetValueRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<std::vector<int64_t>> value_range;
  (void) AttrUtils::GetListListInt(this, TENSOR_UTILS_VALUE_RANGE, value_range);

  for (const auto &ele : value_range) {
    // here must be only two elemenet because pair
    if (ele.size() != PAIR_ELEMENT_SIZE) {
      REPORT_INNER_ERROR("E19999", "value_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] value_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    range.emplace_back(std::make_pair(ele[PAIR_ELEMENT_KEY], ele[PAIR_ELEMENT_VALUE]));
  }

  return GRAPH_SUCCESS;
}

graphStatus GeTensorDesc::SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<std::vector<int64_t>> shape_range;
  for (const auto &ele : range) {
    shape_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto const ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::SetOriginShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range) {
  std::vector<std::vector<int64_t>> origin_shape_range;
  for (const auto &ele : range) {
    origin_shape_range.emplace_back(std::vector<int64_t>({ele.first, ele.second}));
  }
  auto const ret = AttrUtils::SetListListInt(this, TENSOR_UTILS_ORIGIN_SHAPE_RANGE, origin_shape_range);
  return ret ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus GeTensorDesc::GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<std::vector<int64_t>> shape_range;
  (void)AttrUtils::GetListListInt(this, TENSOR_UTILS_SHAPE_RANGE, shape_range);

  for (const auto &ele : shape_range) {
    // here must be only two elemenet because pair
    if (ele.size() != PAIR_ELEMENT_SIZE) {
      REPORT_INNER_ERROR("E19999", "shape_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] shape_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    std::pair<int64_t, int64_t> pair({ele[PAIR_ELEMENT_KEY], ele[PAIR_ELEMENT_VALUE]});
    range.emplace_back(pair);
  }

  return GRAPH_SUCCESS;
}

graphStatus GeTensorDesc::GetOriginShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const {
  std::vector<std::vector<int64_t>> origin_shape_range;
  (void)AttrUtils::GetListListInt(this, TENSOR_UTILS_ORIGIN_SHAPE_RANGE, origin_shape_range);

  for (const auto &ele : origin_shape_range) {
    // here must be only two elemenet because pair
    if (ele.size() != PAIR_ELEMENT_SIZE) {
      REPORT_INNER_ERROR("E19999", "origin_shape_range must contain only 2 value but really is %zu", ele.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] origin_shape_range must contain only 2 value but really is %zu", ele.size());
      return GRAPH_FAILED;
    }
    std::pair<int64_t, int64_t> pair({ele[PAIR_ELEMENT_KEY], ele[PAIR_ELEMENT_VALUE]});
    range.emplace_back(pair);
  }

  return GRAPH_SUCCESS;
}

const GeShape &GeTensorDesc::GetOriginShape() const {
  return impl_->OriginShapeReference();
}

void GeTensorDesc::SetOriginShape(const GeShape &origin_shape) {
  impl_->OriginShapeReference() = origin_shape;
  impl_->SetOriginShapeInited(true);
}

bool GeTensorDesc::IsOriginShapeInitialized() const {
  return impl_->IsOriginShapeInited();
}

Format GeTensorDesc::GetFormat() const {
  return impl_->GetFormat();
}

void GeTensorDesc::SetFormat(const Format format) {
  return impl_->SetFormat(format);
}

void GeTensorDesc::SetName(const std::string &name) {
  return impl_->SetName(name);
}

const std::string GeTensorDesc::GetName() const {
  return impl_->GetName();
}

Format GeTensorDesc::GetOriginFormat() const {
  return impl_->GetOriginFormat();
}

void GeTensorDesc::SetOriginFormat(const Format origin_format) {
  impl_->SetOriginFormat(origin_format);
}

void GeTensorDesc::SetDataType(const DataType data_type) {
  return impl_->SetDataType(data_type);
}

DataType GeTensorDesc::GetDataType() const {
  return impl_->GetDataType();
}

void GeTensorDesc::SetOriginDataType(const DataType origin_data_type) {
  impl_->SetOriginDataType(origin_data_type);
}

DataType GeTensorDesc::GetOriginDataType() const {
  return impl_->GetOriginDataType();
}

std::vector<uint32_t> GeTensorDesc::GetRefPortIndex() const {
  std::vector<uint32_t> ref_port_index;
  (void)AttrUtils::GetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, ref_port_index);
  return ref_port_index;
}

void GeTensorDesc::SetRefPortByIndex(const std::vector<uint32_t> &index) {
  (void)AttrUtils::SetListInt(this, TENSOR_UTILS_REF_PORT_INDEX, index);
}

Placement GeTensorDesc::GetPlacement() const {
  int64_t placement = 0;
  (void)AttrUtils::GetInt(this, TENSOR_UTILS_PLACEMENT, placement);
  return static_cast<Placement>(placement);
}

void GeTensorDesc::SetPlacement(const Placement placement) {
  (void)AttrUtils::SetInt(this, TENSOR_UTILS_PLACEMENT, static_cast<int64_t>(placement));
}

graphStatus GeTensorDesc::IsValid() const {
  auto const data_type = this->GetDataType();
  auto const format  = this->GetFormat();
  if ((data_type == DT_UNDEFINED) && (format == FORMAT_RESERVED)) {
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

GeTensorDesc GeTensorDesc::Clone() const { return *this; }

GeTensorDesc &GeTensorDesc::operator=(const GeTensorDesc &desc) {
  if (&desc != this) {
    AttrHolder::CopyFrom(desc);
    *impl_ = *(desc.impl_);
  }
  return *this;
}

GeTensorDesc &GeTensorDesc::operator=(GeTensorDesc &&desc) {
  if (&desc != this) {
    AttrHolder::CopyFrom(desc);
    impl_ = desc.impl_;
  }
  return *this;
}

uint32_t TensorDataImpl::invalid_data_ = 0x3A2D2900U;

TensorDataImpl::TensorDataImpl(const TensorDataImpl &other) {
  // Share data
  tensor_descriptor_ = other.tensor_descriptor_;
  aligned_ptr_ = other.aligned_ptr_;
  length_ = other.length_;
}

TensorDataImpl &TensorDataImpl::operator=(const TensorDataImpl &other) {
  if (&other != this) {
    // Share data
    tensor_descriptor_ = other.tensor_descriptor_;
    aligned_ptr_ = other.aligned_ptr_;
    length_ = other.length_;
  }
  return *this;
}

graphStatus TensorDataImpl::SetData(const uint8_t * const data, const size_t size) {
  if (size == 0UL) {
    GELOGI("size is 0");
    clear();
    return GRAPH_SUCCESS;
  }
  if (data == nullptr) {
    GELOGI("data addr is empty");
    return GRAPH_SUCCESS;
  }

  if (MallocAlignedPtr(size) == nullptr) {
    GELOGE(MEMALLOC_FAILED, "[Malloc][Memory] failed, size=%zu", size);
    return GRAPH_FAILED;
  }

  size_t remain_size = size;
  auto dst_addr = reinterpret_cast<uintptr_t>(aligned_ptr_->MutableGet());
  auto src_addr = reinterpret_cast<uintptr_t>(data);
  while (remain_size > SECUREC_MEM_MAX_LEN) {
    if (memcpy_s(reinterpret_cast<void *>(dst_addr), SECUREC_MEM_MAX_LEN,
                 reinterpret_cast<const void *>(src_addr), SECUREC_MEM_MAX_LEN) != EOK) {
      REPORT_CALL_ERROR("E19999", "memcpy failed, size = %lu", SECUREC_MEM_MAX_LEN);
      GELOGE(INTERNAL_ERROR, "[Memcpy][Data] failed, size = %lu", SECUREC_MEM_MAX_LEN);
      return GRAPH_FAILED;
    }
    remain_size -= SECUREC_MEM_MAX_LEN;
    dst_addr += SECUREC_MEM_MAX_LEN;
    src_addr += SECUREC_MEM_MAX_LEN;
  }
  if (memcpy_s(reinterpret_cast<void *>(dst_addr), remain_size,
               reinterpret_cast<const void *>(src_addr), remain_size) != EOK) {
    REPORT_CALL_ERROR("E19999", "memcpy failed, size=%zu", remain_size);
    GELOGE(INTERNAL_ERROR, "[Memcpy][Data] failed, size=%zu", remain_size);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void TensorDataImpl::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size) {
  aligned_ptr_ = std::move(aligned_ptr);
  length_ = size;
}

graphStatus TensorDataImpl::SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc) {
  if (size == 0UL) {
    GELOGW("[Set][Data] Input size is 0");
    clear();
    return GRAPH_SUCCESS;
  }
  if (data == nullptr) {
    REPORT_CALL_ERROR("E19999", "data is nullptr");
    GELOGE(GRAPH_FAILED, "[Check][Param] data is nullptr");
    return GRAPH_FAILED;
  }
  length_ = size;
  aligned_ptr_ = AlignedPtr::BuildFromData(data, delete_fuc);
  return GRAPH_SUCCESS;
}

const uint8_t *TensorDataImpl::MallocAlignedPtr(const size_t size) {
  if (size == 0UL) {
    GELOGW("[Check][Param] Input data size is 0");
    clear();
    return reinterpret_cast<const uint8_t *>(&invalid_data_);
  }
  if (length_ != size) {
    aligned_ptr_.reset();
  }
  length_ = size;
  if (aligned_ptr_ == nullptr) {
    aligned_ptr_ = MakeShared<AlignedPtr>(length_);
    if (aligned_ptr_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "create AlignedPtr failed.");
      GELOGE(INTERNAL_ERROR, "[Create][AlignedPtr] failed.");
      return nullptr;
    }
  }

  return aligned_ptr_->Get();
}

size_t TensorDataImpl::GetSize() const { return length_; }

const uint8_t *TensorDataImpl::GetData() const {
  if (length_ == 0UL) {
    return reinterpret_cast<const uint8_t *>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->Get();
}

uint8_t *TensorDataImpl::GetData() {
  if (length_ == 0UL) {
    return reinterpret_cast<uint8_t *>(&invalid_data_);
  }
  if (aligned_ptr_ == nullptr) {
    return nullptr;
  }
  return aligned_ptr_->MutableGet();
}

void TensorDataImpl::clear() {
  aligned_ptr_.reset();
  length_ = 0UL;
}

uint8_t TensorDataImpl::operator[](const size_t index) const {
  if ((aligned_ptr_ != nullptr) && (index < length_)) {
    return *(aligned_ptr_->MutableGet() + index);
  }
  return static_cast<uint8_t>(0xffU);
}

TensorData::TensorData()
    : impl_(MakeShared<TensorDataImpl>()) {}

TensorData::TensorData(const TensorData &other)
    : impl_(MakeShared<TensorDataImpl>(*(other.impl_))) {}

TensorData::~TensorData() = default;

TensorData &TensorData::operator=(const TensorData &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

graphStatus TensorData::SetData(std::vector<uint8_t> &&data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const std::vector<uint8_t> &data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const Buffer &data) { return SetData(data.data(), data.size()); }
graphStatus TensorData::SetData(const TensorData &data) { return SetData(data.data(), data.size()); }

graphStatus TensorData::SetData(const uint8_t * const data, const size_t size) {
  return impl_->SetData(data, size);
}

graphStatus TensorData::SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return impl_->SetData(data, size, delete_fuc);
}

void TensorData::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size) {
  impl_->SetData(aligned_ptr, size);
}

const uint8_t *TensorData::MallocAlignedPtr(const size_t size) {
  return impl_->MallocAlignedPtr(size);
}

size_t TensorData::GetSize() const {
  return impl_->GetSize();
}

const uint8_t *TensorData::GetData() const {
  return impl_->GetData();
}

uint8_t *TensorData::GetData() {
  return impl_->GetData();
}

const std::uint8_t *TensorData::data() const { return GetData(); }
std::uint8_t *TensorData::data() { return GetData(); }
std::size_t TensorData::size() const { return GetSize(); }
void TensorData::clear() {
  impl_->clear();
}

uint8_t TensorData::operator[](const size_t index) const {
  return (*impl_)[index];
}

const std::shared_ptr<AlignedPtr> &TensorData::GetAlignedPtr() {
  return impl_->GetAlignedPtr();
}

GeTensorImpl::GeTensorImpl() : tensor_def_(nullptr, nullptr), desc_(), tensor_data_()  {
  if (desc_.impl_ != nullptr) {
    if (tensor_data_.impl_ != nullptr) {
      tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
    }
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc) : GeTensorImpl() {
  DescReference() = tensor_desc;
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const uint8_t * const data, const size_t size)
    : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.SetData(data, size) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data) : GeTensorImpl() {
  DescReference() = std::move(tensor_desc);
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const Buffer &data) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (data.size() == 0UL) {
    GELOGI("GetSize res is 0.");
  }
  if (data.data() == nullptr) {
    GELOGI("data addr is null.");
  }
  if (tensor_data_.SetData(data) != GRAPH_SUCCESS) {
    GELOGW("[Set][Data] Set data failed");
  }
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size)
    : GeTensorImpl() {
  DescReference() = tensor_desc;
  tensor_data_.SetData(std::move(aligned_ptr), size);
}

GeTensorImpl::GeTensorImpl(const GeTensorDesc &tensor_desc, const size_t size) : GeTensorImpl() {
  DescReference() = tensor_desc;
  if (tensor_data_.MallocAlignedPtr(size) == nullptr) {
    GELOGW("[Malloc][Memory] Malloc memory failed, size=%zu", size);
  }
}

GeTensorImpl::GeTensorImpl(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg)
    : tensor_def_(proto_owner, proto_msg) {
  // 这里后续改为反序列化接口调用，从proto恢复GeTensorDesc
  desc_ = GeTensorDesc(proto_owner, (proto_msg == nullptr) ? nullptr : proto_msg->mutable_desc());
  tensor_data_ = TensorData();
  if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
    // 之前没有把TensorData上的proto变为GeTensorDesc，因为TensorData创建后不会修改，多个TensorData通过GeIrProto共享
    // 但是！原本的语义是TensorData上的proto::TensorDescriptor与Tensor上的GeTensorDesc是共享的，当GeTensorDesc改造完
    // 这种共享的能力就消失了，这会导致在GeTensor创建后，对GeTensorDesc的修改无法反应到TensorData上，看起来只能将TensorData
    // 上的proto::TensorDescriptor修改为GeTensorDescImpl，并且需要与GeTensor的GeTensorDesc共享
    tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
  }

  if (proto_msg != nullptr) {
    if (proto_owner != nullptr) {
      BuildAlignerPtrWithProtoData();
    } else {
      (void)tensor_data_.SetData(reinterpret_cast<const uint8_t *>(proto_msg->data().data()), proto_msg->data().size());
    }
  }
}

GeTensorDesc &GeTensorImpl::DescReference() const {
  return desc_;
}

void GeTensorImpl::BuildAlignerPtrWithProtoData() {
  auto const proto_msg = tensor_def_.GetProtoMsg();
  if ((proto_msg == nullptr) || (reinterpret_cast<const uint8_t *>(proto_msg->data().data()) == tensor_data_.data())) {
    return;
  }
  if (tensor_data_.impl_ == nullptr) {
    return;
  }

  tensor_data_.impl_->length_ = proto_msg->data().size();
  tensor_data_.impl_->aligned_ptr_.reset();
  tensor_data_.impl_->aligned_ptr_ =
      AlignedPtr::BuildFromAllocFunc([&proto_msg](std::unique_ptr<uint8_t[], AlignedPtr::Deleter> &ptr) {
                                       ptr.reset(const_cast<uint8_t *>(
                                           reinterpret_cast<const uint8_t *>(proto_msg->data().data())));
                                     },
                                     [](uint8_t *ptr) {
                                       ptr = nullptr;
                                     });
}

graphStatus GeTensorImpl::SetData(std::vector<uint8_t> &&data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto const proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const std::vector<uint8_t> &data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto const proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const uint8_t * const data, const size_t size) {
  if (size > 0UL) {
    GE_CHECK_NOTNULL(data);
  }
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto const proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    proto_msg->set_data(data, size);
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data, size);
}

graphStatus GeTensorImpl::SetData(const Buffer &data) {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto const proto_msg = tensor_def_.GetProtoMsg();
    GE_CHECK_NOTNULL(proto_msg);
    if (data.size() == 0UL) {
      GELOGI("GetSize res is 0.");
    }
    if (data.data() == nullptr) {
      GELOGI("data addr is null.");
    }
    proto_msg->set_data(data.data(), data.size());
    BuildAlignerPtrWithProtoData();
    return GRAPH_SUCCESS;
  }
  return tensor_data_.SetData(data);
}

graphStatus GeTensorImpl::SetData(const TensorData &data) {
  return SetData(data.data(), data.size());
}

graphStatus GeTensorImpl::SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return tensor_data_.SetData(data, size, delete_fuc);
}

void GeTensorImpl::ClearData() {
  if (tensor_def_.GetProtoOwner() != nullptr) {
    auto const proto_msg = tensor_def_.GetProtoMsg();
    if (proto_msg != nullptr) {
      proto_msg->clear_data();
    }
  }
  tensor_data_.clear();
}

void GeTensorImpl::Clone(GeTensorImpl &tensor) const {
  if (tensor.desc_.impl_ != nullptr && desc_.impl_ != nullptr) {
    *(tensor.desc_.impl_) = *(desc_.impl_);
  }
  if (tensor.tensor_data_.impl_ != nullptr && tensor.desc_.impl_ != nullptr) {
    tensor.tensor_data_.impl_->tensor_descriptor_ = tensor.desc_.impl_;
  }
  (void)tensor.SetData(GetData());
}

std::shared_ptr<AlignedPtr> GeTensorImpl::GetAlignedPtr() const {
  if (tensor_data_.impl_ != nullptr) {
    return tensor_data_.impl_->GetAlignedPtr();
  }
  return nullptr;
}

GeTensorImpl::GeTensorImpl(const GeTensorImpl &other) : GeTensorImpl() {
  *this = other;
}

GeTensorImpl &GeTensorImpl::operator=(const GeTensorImpl &other) {
  if (&other != this) {
    if (other.tensor_def_.GetProtoOwner() != nullptr) {
      // Old scene, share tensor_def, tensor_desc, tensor_data with `other`
      tensor_def_ = other.tensor_def_;
      // 这里修改了
      desc_ = other.desc_;
      if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
        tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
      }
      BuildAlignerPtrWithProtoData();
    } else {
      // share tensor_data, do not share tensor_desc, tensor_def is null
      desc_ = other.desc_;
      tensor_data_ = other.tensor_data_;
      if (tensor_data_.impl_ != nullptr && desc_.impl_ != nullptr) {
        tensor_data_.impl_->tensor_descriptor_ = desc_.impl_;
      }
    }
  }
  return *this;
}

GeTensor::GeTensor() : impl_(MakeShared<GeTensorImpl>()) {}

GeTensor::GeTensor(GeTensor &&other) noexcept : impl_(std::move(other.impl_)) {}

GeTensor::GeTensor(GeTensorImplPtr impl) : impl_(std::move(impl)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc, data)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const uint8_t * const data, const size_t size)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc, data, size)) {}

GeTensor::GeTensor(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data)
    : impl_(MakeShared<GeTensorImpl>(std::move(tensor_desc), std::move(data))) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const Buffer &data)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc, data)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc, aligned_ptr, size)) {}

GeTensor::GeTensor(const GeTensorDesc &tensor_desc, const size_t size)
    : impl_(MakeShared<GeTensorImpl>(tensor_desc, size)) {}

GeTensor::GeTensor(const ProtoMsgOwner &proto_owner, proto::TensorDef *protoMsg)
    : impl_(MakeShared<GeTensorImpl>(proto_owner, protoMsg)) {}

GeTensor::~GeTensor() = default;

void GeTensor::BuildAlignerPtrWithProtoData() {
  impl_->BuildAlignerPtrWithProtoData();
}

const GeTensorDesc &GeTensor::GetTensorDesc() const { return DescReference(); }

GeTensorDesc &GeTensor::MutableTensorDesc() { return DescReference(); }

GeTensorDesc &GeTensor::DescReference() const {
  return impl_->DescReference();
}

void GeTensor::SetTensorDesc(const GeTensorDesc &tensor_desc) { DescReference() = tensor_desc; }

graphStatus GeTensor::SetData(std::vector<uint8_t> &&data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const std::vector<uint8_t> &data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const uint8_t * const data, const size_t size) {
  return impl_->SetData(data, size);
}

graphStatus GeTensor::SetData(const Buffer &data) {
  return impl_->SetData(data);
}

graphStatus GeTensor::SetData(const TensorData &data) {
  return SetData(data.data(), data.size());
}

graphStatus GeTensor::SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc) {
  return impl_->SetData(data, size, delete_fuc);
}

void GeTensor::ClearData() {
  impl_->ClearData();
}

GeTensor GeTensor::Clone() const {
  const GeTensor tensor;
  impl_->Clone(*(tensor.impl_));
  return tensor;
}

GeTensor::GeTensor(const GeTensor &other)
    : impl_(MakeShared<GeTensorImpl>(*(other.impl_))) {}

GeTensor &GeTensor::operator=(const GeTensor &other) {
  if (&other != this) {
    *impl_ = *(other.impl_);
  }
  return *this;
}

GeTensor &GeTensor::operator=(GeTensor &&other) {
  if (&other != this) {
    impl_ = other.impl_;
  }
  return *this;
}

std::shared_ptr<AlignedPtr> GeTensor::GetAlignedPtr() {
  return impl_->GetAlignedPtr();
}

const TensorData &GeTensor::GetData() const {
  return impl_->GetData();
}
TensorData &GeTensor::MutableData() {
  return impl_->MutableData();
}
// zero copy SetData
void GeTensor::SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size) {
  impl_->SetData(std::move(aligned_ptr), size);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetSize(const GeTensorDesc &tensor_desc,
                                                                                int64_t &size) {
  if (tensor_desc.impl_ != nullptr) {
    size = tensor_desc.impl_->ext_meta_.GetSize();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetSize(
    GeTensorDesc &tensor_desc, const int64_t size) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetSize(size);
  }
}

uint32_t TensorUtils::GetWeightSize(const GeTensorDesc &tensor_desc) {
  if (tensor_desc.impl_ != nullptr && IntegerChecker<uint32_t>::Compat(tensor_desc.impl_->ext_meta_.GetWeightSize())) {
    return static_cast<uint32_t>(tensor_desc.impl_->ext_meta_.GetWeightSize());
  }
  return 0U;
}

uint32_t TensorUtils::GetWeightSize(const GeTensor &tensor) { return GetWeightSize(tensor.GetTensorDesc()); }

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t TensorUtils::GetWeightSize(const ConstGeTensorPtr &tensor_ptr) {
  if (tensor_ptr == nullptr) {
    return 0U;
  }
  return GetWeightSize(*tensor_ptr);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint8_t *TensorUtils::GetWeightAddr(const ConstGeTensorPtr &tensor_ptr,
                                                                                   uint8_t * const base) {
  if (tensor_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "param tensor_ptr is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] tensor_ptr is null.");
    return nullptr;
  }
  return GetWeightAddr(*tensor_ptr, base);
}

uint8_t *TensorUtils::GetWeightAddr(const GeTensor &tensor, uint8_t * const base) {
  if (base == nullptr) {
    REPORT_INNER_ERROR("E19999", "param base is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] base is null.");
    return nullptr;
  }
  int64_t weight_data_offset = 0;
  if (GetDataOffset(tensor.GetTensorDesc(), weight_data_offset) != GRAPH_SUCCESS) {
    return nullptr;
  }

  if (weight_data_offset == 0) {
    // The weight of offset 0 is still in const op, still get from ATTR_NAME_WEIGHTS.
    return const_cast<uint8_t *>(tensor.GetData().data());
  }

  return base + weight_data_offset;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetWeightSize(GeTensorDesc &tensor_desc,
                                                                               const uint32_t size) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetWeightSize(static_cast<int64_t>(size));
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetReuseInput(const GeTensorDesc &tensor_desc,
                                                                                      bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    flag = tensor_desc.impl_->ext_meta_.GetReuseInput();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInput(
    GeTensorDesc &tensor_desc, const bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetReuseInput(flag);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetOutputTensor(const GeTensorDesc &tensor_desc,
                                                                                        bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    flag = tensor_desc.impl_->ext_meta_.GetOutputTensor();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetOutputTensor(
    GeTensorDesc &tensor_desc, const bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetOutputTensor(flag);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDeviceType(const GeTensorDesc &tensor_desc,
                                                                                      DeviceType &type) {
  if (tensor_desc.impl_ != nullptr) {
    type = tensor_desc.impl_->ext_meta_.GetDeviceType();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDeviceType(GeTensorDesc &tensor_desc,
                                                                               const DeviceType type) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetDeviceType(type);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetInputTensor(const GeTensorDesc &tensor_desc,
                                                                                       bool &flag) {
  if (tensor_desc.impl_ != nullptr) {
    flag = tensor_desc.impl_->ext_meta_.GetInputTensor();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetInputTensor(
    GeTensorDesc &tensor_desc, const bool flag) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetInputTensor(flag);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRealDimCnt(const GeTensorDesc &tensor_desc,
                                                                                      uint32_t &cnt) {
  if (tensor_desc.impl_ != nullptr) {
    cnt = tensor_desc.impl_->ext_meta_.GetRealDimCnt();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRealDimCnt(GeTensorDesc &tensor_desc,
                                                                               const uint32_t cnt) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetRealDimCnt(cnt);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
TensorUtils::GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx) {
  if (tensor_desc.impl_ != nullptr) {
    idx = tensor_desc.impl_->ext_meta_.GetReuseInputIndex();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetReuseInputIndex(GeTensorDesc &tensor_desc,
                                                                                    const uint32_t idx) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetReuseInputIndex(idx);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetDataOffset(const GeTensorDesc &tensor_desc,
                                                                                      int64_t &offset) {
  if (tensor_desc.impl_ != nullptr) {
    offset = tensor_desc.impl_->ext_meta_.GetDataOffset();
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetDataOffset(GeTensorDesc &tensor_desc,
                                                                               const int64_t offset) {
  if (tensor_desc.impl_ != nullptr) {
    tensor_desc.impl_->ext_meta_.SetDataOffset(offset);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus TensorUtils::GetRC(const GeTensorDesc &tensor_desc,
                                                                              uint32_t &rc) {
  return AttrUtils::GetInt(&tensor_desc, TENSOR_UTILS_RC, rc) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void TensorUtils::SetRC(GeTensorDesc &tensor_desc, const uint32_t rc) {
  (void)AttrUtils::SetInt(&tensor_desc, TENSOR_UTILS_RC, static_cast<const int64_t>(rc));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY  bool TensorUtils::IsOriginShapeInited(const GeTensorDesc &tensor_desc) {
  return tensor_desc.impl_->IsOriginShapeInited();
}

GeTensor TensorUtils::CreateShareTensor(const GeTensor &other) {
  GeTensor tensor;
  ShareTensor(other, tensor);
  return tensor;
}

GeTensor TensorUtils::CreateShareTensor(const GeTensorDesc &tensor_desc,
                                        std::shared_ptr<AlignedPtr> aligned_ptr,
                                        const size_t size) {
  const GeTensor tensor(tensor_desc);
  if (tensor.impl_ != nullptr) {
    ShareAlignedPtr(std::move(aligned_ptr), size, tensor.impl_->tensor_data_);
  }
  return tensor;
}

void TensorUtils::ShareTensor(const GeTensor &from, GeTensor &to) {
  if (&from == &to) {
    return;
  }
  if (from.impl_ != nullptr && to.impl_ != nullptr) {
    if (from.impl_->tensor_def_.GetProtoOwner() != nullptr) {
      // 这种场景下看原来的逻辑，已经没有什么是不是共享的了，所以直接改成了impl共享，幸好impl是shared ptr
      // 但是之前似乎有个啥逻辑。是假定可以把shared ptr当成unique用的，得风暴下，记不得了
      to.impl_ = from.impl_;
    } else {
      // share tensor_data, do not share tensor_desc, tensor_def is null
      to.impl_->desc_ = from.impl_->desc_;
      to.impl_->tensor_data_ = from.impl_->tensor_data_;
      to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
    }
  }
}
void TensorUtils::ShareTensorData(const TensorData &from, TensorData &to) {
  if (&from == &to) {
    return;
  }
  // Share data
  if (from.impl_ != nullptr && to.impl_ != nullptr) {
    to.impl_->tensor_descriptor_ = from.impl_->tensor_descriptor_;
    to.impl_->aligned_ptr_ = from.impl_->aligned_ptr_;
    to.impl_->length_ = from.impl_->length_;
  }
}
TensorData TensorUtils::CreateShareTensorData(const TensorData &other) {
  TensorData td;
  ShareTensorData(other, td);
  return td;
}
void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, TensorData &to) {
  if (to.impl_ != nullptr) {
    to.impl_->aligned_ptr_ = std::move(ptr);
    to.impl_->length_ = size;
  }
}
void TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, GeTensor &to) {
  if (to.impl_ != nullptr) {
    ShareAlignedPtr(std::move(ptr), size, to.impl_->tensor_data_);
  }
}
// UT
void TensorUtils::CopyTensor(const GeTensor &from, GeTensor &to) {
  if (&from == &to) {
    return;
  }
  if (from.impl_ == nullptr || to.impl_ == nullptr) {
    return;
  }
  if (from.impl_->tensor_def_.GetProtoOwner() != nullptr) {
    to.impl_->tensor_def_.CopyValueFrom(from.impl_->tensor_def_);
    to.impl_->desc_.impl_ = GeTensorDesc(to.impl_->tensor_def_.GetProtoOwner(),
                                         to.impl_->tensor_def_.GetProtoMsg()->mutable_desc()).impl_;
    to.impl_->desc_.impl_->attrs_ = from.impl_->desc_.impl_->attrs_;
    to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
    to.BuildAlignerPtrWithProtoData();
  } else {
    // tensor_def is null, copy tensor_data, tensor_desc
    to.impl_->desc_ = from.impl_->desc_;
    (void)to.impl_->tensor_data_.SetData(from.impl_->tensor_data_);
    to.impl_->tensor_data_.impl_->tensor_descriptor_ = to.impl_->desc_.impl_;
  }
}
}  // namespace ge
