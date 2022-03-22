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

#ifndef INC_GRAPH_GE_TENSOR_H_
#define INC_GRAPH_GE_TENSOR_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include "detail/attributes_holder.h"
#include "graph/buffer.h"
#include "graph/aligned_ptr.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "any_value.h"

namespace ge {
class GeShapeImpl;
using GeShapeImplPtr = std::shared_ptr<GeShapeImpl>;

class TensorDataImpl;
using TensorDataImplPtr = std::shared_ptr<TensorDataImpl>;

class GeTensorDescImpl;
using GeTensorDescImplPtr = std::shared_ptr<GeTensorDescImpl>;

class GeTensorImpl;
using GeTensorImplPtr = std::shared_ptr<GeTensorImpl>;

class GeTensorSerializeUtils;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeShape {
 public:
  GeShape();
  ~GeShape();
  explicit GeShape(std::vector<int64_t> s);

  size_t GetDimNum() const;
  void SetDimNum(const size_t dim_num);
  void AppendDim(const int64_t dim_size);
  bool IsUnknownDimNum() const;
  void SetIsUnknownDimNum();
  // If the idx is invalid, return 0
  int64_t GetDim(const size_t idx) const;
  graphStatus SetDim(const size_t idx, const int64_t value);
  std::vector<int64_t> GetDims() const;

  int64_t GetShapeSize() const;
  std::string ToString() const;

  ///
  /// @brief Check is unknown shape
  /// @return bool
  ///
  bool IsUnknownShape() const;

  ///
  /// @brief Check is a scalar
  /// @return bool
  ///
  bool IsScalar() const;

  GeShape(const GeShape &other);
  GeShape(GeShape &&other);
  GeShape &operator=(const GeShape &other);
  GeShape &operator=(GeShape &&other);
  bool operator==(const GeShape &other) const;

 private:
  GeShapeImplPtr impl_;
  friend class GeTensorDesc;
  friend class GeTensorDescImpl;
  friend class GeTensorSerializeUtils;
  friend class ModelSerialize;
  // Create from proto obj
  GeShape(const ProtoMsgOwner &proto_owner, proto::ShapeDef *proto_msg);
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc : public AttrHolder {
  friend class TensorUtils;
  friend class ModelSerialize;

 public:
  GeTensorDesc();
  explicit GeTensorDesc(const GeShape &shape, const Format format = FORMAT_ND, const DataType dt = DT_FLOAT);
  GeTensorDesc(const GeTensorDesc &desc);
  GeTensorDesc(GeTensorDesc &&desc);

  ~GeTensorDesc() override;
  bool operator==(const GeTensorDesc &r_ge_tensor_desc) const;

  void Update(const GeShape &shape, const Format format = FORMAT_ND, const DataType dt = DT_FLOAT);

  const GeShape &GetShape() const;
  GeShape &MutableShape();
  void SetShape(const GeShape &shape);
  void SetShape(GeShape &&shape);

  // set shape with -2, it stand for unknown shape
  void SetUnknownDimNumShape();
  // for unknown shape
  graphStatus SetValueRange(const std::vector<std::pair<int64_t, int64_t>> &range);
  graphStatus GetValueRange(std::vector<std::pair<int64_t, int64_t>> &range) const;
  graphStatus SetShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range);
  graphStatus SetOriginShapeRange(const std::vector<std::pair<int64_t, int64_t>> &range);
  graphStatus GetShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const;
  graphStatus GetOriginShapeRange(std::vector<std::pair<int64_t, int64_t>> &range) const;

  const GeShape &GetOriginShape() const;
  // 该方法暂时不实现，因为一旦开放后，当前代码里判断OriginShape是否设置的逻辑就失效了
  GeShape &MutableOriginShape();

  void SetOriginShape(const GeShape &origin_shape);
  bool IsOriginShapeInitialized() const;

  Format GetFormat() const;
  void SetFormat(const Format format);

  Format GetOriginFormat() const;
  void SetOriginFormat(const Format origin_format);

  void SetName(const std::string &name);
  const std::string GetName() const;

  DataType GetDataType() const;
  void SetDataType(const DataType data_type);

  DataType GetOriginDataType() const;
  void SetOriginDataType(const DataType origin_data_type);

  std::vector<uint32_t> GetRefPortIndex() const;
  void SetRefPortByIndex(const std::vector<uint32_t> &index);

  Placement GetPlacement() const;
  void SetPlacement(Placement placement);

  GeTensorDesc Clone() const;
  GeTensorDesc &operator=(const GeTensorDesc &desc);
  GeTensorDesc &operator=(GeTensorDesc &&desc);

  graphStatus IsValid() const;

  using AttrHolder::DelAttr;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  bool GeTensorDescAttrsAreEqual(const GeTensorDesc &r_ge_tensor_desc) const;

  // Create from proto obj
  GeTensorDesc(const ProtoMsgOwner &proto_owner, proto::TensorDescriptor *proto_msg);
  friend class GeTensor;
  friend class GeTensorImpl;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  friend class OnnxUtils;

  GeTensorDescImplPtr impl_;

  void RefTo(const GeTensorDesc &tensor_desc);
  GeShape &ShapeReference() const;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorData {
 public:
  TensorData();
  ~TensorData();

  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const Buffer &data);
  graphStatus SetData(const TensorData &data);
  graphStatus SetData(const uint8_t * const data, const size_t size);
  graphStatus SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);  /*lint !e148*/

  const uint8_t *MallocAlignedPtr(const size_t size);

  const std::uint8_t *data() const;
  std::uint8_t *data();
  std::size_t size() const;
  void clear();
  uint8_t operator[](const size_t index) const;

  std::size_t GetSize() const;
  const std::uint8_t *GetData() const;
  std::uint8_t *GetData();

  const std::shared_ptr<AlignedPtr> &GetAlignedPtr();

  // share data, share tensor_descriptor/aligned_ptr
  // replace using TensorUtils::ShareTensorData(const TensorData &from, TensorData &to)
  TensorData &operator=(const TensorData &other);
  // share data share tensor_descriptor/aligned_ptr
  // replace using TensorUtils::CreateShareTensorData(const TensorData &other)
  TensorData(const TensorData &other);
  // zero copy SetData
  // replace using TensorUtils::ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, size_t size, TensorData &to)
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size);
 private:
  friend class GeTensor;
  friend class GeTensorImpl;
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  friend class TensorUtils;
  TensorDataImplPtr impl_;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensor {
 public:
  GeTensor();
  GeTensor(GeTensor &&other) noexcept;
  explicit GeTensor(const GeTensorDesc &tensor_desc);
  explicit GeTensor(const GeTensorDesc &tensor_desc, const std::vector<uint8_t> &data);
  explicit GeTensor(const GeTensorDesc &tensor_desc, const Buffer &data);
  explicit GeTensor(const GeTensorDesc &tensor_desc, const uint8_t * const data, const size_t size);
  explicit GeTensor(GeTensorDesc &&tensor_desc, std::vector<uint8_t> &&data);
  explicit GeTensor(const GeTensorDesc &tensor_desc, const size_t size);
  ~GeTensor();

  const GeTensorDesc &GetTensorDesc() const;
  GeTensorDesc &MutableTensorDesc();
  void SetTensorDesc(const GeTensorDesc &tensor_desc);

  std::shared_ptr<AlignedPtr> GetAlignedPtr();

  const TensorData &GetData() const;
  TensorData &MutableData();

  graphStatus SetData(std::vector<uint8_t> &&data);
  graphStatus SetData(const std::vector<uint8_t> &data);
  graphStatus SetData(const Buffer &data);
  graphStatus SetData(const uint8_t * const data, const size_t size);
  graphStatus SetData(const TensorData &data);
  graphStatus SetData(uint8_t * const data, const size_t size, const AlignedPtr::Deleter &delete_fuc);

  void ClearData();
  GeTensor Clone() const;

  // zero copy SetData
  // replace using TensorUtils::ShareAlignedPtr
  void SetData(std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size);
  // zero copy construction, share aligned_ptr, do not share tensor_desc
  // replace using TensorUtils::CreateShareTensor
  GeTensor(const GeTensorDesc &td, std::shared_ptr<AlignedPtr> aligned_ptr, const size_t size);
  // Share tensor_data, tensor_desc
  // replace using TensorUtils::CreateShareTensor
  GeTensor(const GeTensor &other);
  // Share tensor_data, tensor_desc
  // replace using TensorUtils::ShareTensor
  GeTensor &operator=(const GeTensor &other);
  GeTensor &operator=(GeTensor &&other);

 private:
  friend class GeAttrValueImp;
  friend class ModelSerializeImp;
  friend class GeTensorSerializeUtils;
  friend class OnnxUtils;
  friend class TensorData;
  friend class TensorUtils;
  friend class TensorAdapter;
  // Create from proto obj
  GeTensor(const ProtoMsgOwner &proto_owner, proto::TensorDef *proto_msg);
  explicit GeTensor(GeTensorImplPtr impl);
  void BuildAlignerPtrWithProtoData();
  GeTensorImplPtr impl_;
  GeTensorDesc &DescReference() const;
};

class GeTensorSerializeUtils {
 public:
  static void GeShapeAsProto(const GeShape &shape, proto::ShapeDef *proto);
  static void GeTensorDescAsProto(const GeTensorDescImpl &desc, proto::TensorDescriptor *proto);
  static void GeTensorDescAsProto(const GeTensorDesc &desc, proto::TensorDescriptor *proto);
  static void GeTensorAsProto(const GeTensorImpl &tensor, proto::TensorDef *proto);
  static void GeTensorAsProto(const GeTensor &tensor, proto::TensorDef *proto);

  static void AssembleGeShapeFromProto(const proto::ShapeDef *proto, GeShape &shape);
  static void AssembleGeTensorDescFromProto(const proto::TensorDescriptor *proto, GeTensorDesc &desc);
  static void AssembleGeTensorFromProto(const proto::TensorDef *proto, GeTensor &tensor);

  // normalize the input TensorDescriptor，A metadata information maybe stored in different fields of TensorDescriptor,
  // This function needs to prioritize and determine the final metadata information used.
  // After standardization, the direct member field on TensorDescriptor is always valid
  static void NormalizeGeTensorDescProto(proto::TensorDescriptor *proto);
  static void GetShapeFromDescProto(const proto::TensorDescriptor *proto, GeShape &shape);
  static void GetOriginShapeFromDescProto(const proto::TensorDescriptor *proto, GeShape &shape);
  static void GetDtypeFromDescProto(const proto::TensorDescriptor *proto, DataType &dtype);
  static void GetOriginDtypeFromDescProto(const proto::TensorDescriptor *proto, DataType &dtype);
  static void GetFormatFromDescProto(const proto::TensorDescriptor *proto, Format &format);
  static void GetOriginFormatFromDescProto(const proto::TensorDescriptor *proto, Format &format);
};

}  // namespace ge
#endif  // INC_GRAPH_GE_TENSOR_H_
