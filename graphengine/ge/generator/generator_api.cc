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
#include "framework/generator/generator_api.h"
#include "common/ge/ge_util.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/generator/ge_generator.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"

#define CHECK_PARAM_NOT_NULL(param)                                              \
  do {                                                                           \
    if (param == nullptr) {                                                      \
      REPORT_INNER_ERROR("E19999", "param:%s is null", #param);                  \
      GELOGE(ge::PARAM_INVALID, "[Check][Param] %s is null.", #param);           \
      return ge::PARAM_INVALID;                                                  \
    }                                                                            \
  } while (0)

#define CHECK_PARAM_OBJECT(object, param)                                        \
  ({                                                                             \
    object *obj_value = reinterpret_cast<object *>(param);                       \
    if (obj_value == nullptr) {                                                  \
      REPORT_INNER_ERROR("E19999", "param:%s is null.", #param);                 \
      GELOGE(ge::PARAM_INVALID, "[Check][Param] %s is null.", #param);           \
      return ge::PARAM_INVALID;                                                  \
    }                                                                            \
    obj_value;                                                                   \
  })

class OpAttr {
 public:
  OpAttr() = default;
  ~OpAttr() = default;

  const std::map<string, ge::GeAttrValue> &Attrs() const { return attrs_; }

  template <typename T>
  Status_t SetAttr(const char *name, T value) {
    CHECK_PARAM_NOT_NULL(name);
    auto attr_value = ge::GeAttrValue::CreateFrom<T>(value);
    attrs_[std::string(name)] = attr_value;
    return ge::SUCCESS;
  }

  template <typename T>
  Status_t SetAttr(const char *name, const T *value, int num) {
    CHECK_PARAM_NOT_NULL(name);
    CHECK_PARAM_NOT_NULL(value);

    std::vector<T> values;
    for (int i = 0; i < num; ++i) {
      values.push_back(value[i]);
    }

    auto attr_value = ge::GeAttrValue::CreateFrom<std::vector<T>>(values);
    attrs_[std::string(name)] = attr_value;
    return ge::SUCCESS;
  }

  Status_t SetAttr(const char *name, const char *value) {
    CHECK_PARAM_NOT_NULL(name);
    CHECK_PARAM_NOT_NULL(value);
    auto attr_value = ge::GeAttrValue::CreateFrom<string>(string(value));
    attrs_[std::string(name)] = attr_value;
    return ge::SUCCESS;
  }

  Status_t SetAttr(const char *name, const char **value, int num) {
    CHECK_PARAM_NOT_NULL(name);
    CHECK_PARAM_NOT_NULL(value);

    std::vector<string> values;
    for (int i = 0; i < num; ++i) {
      values.push_back(string(value[i]));
    }

    auto attr_value = ge::GeAttrValue::CreateFrom<std::vector<string>>(values);
    attrs_[std::string(name)] = attr_value;
    return ge::SUCCESS;
  }

 private:
  std::map<string, ge::GeAttrValue> attrs_;
};

/**
 * @ingroup ge
 * @brief Generate offline model for the op.
 * @param [in] op_type: type name of the op.
 * @param [in] in_tensor: input description array (created by OpTensorCreate).
 * @param [in] in_num: number of in_tensor.
 * @param [in] out_tensor: output description array (created by OpTensorCreate).
 * @param [in] out_num: number of out_tensor.
 * @param [in] attr: the attributes of the op (created by OpAttrCreate).
 * @param [in] om_file: file name for the om to save.
 * @return 0 for success / others for fail
 */
Status_t OpTaskGernerator(const char *op_type, const OpTensor_t *in_tensor, int in_num, const OpTensor_t *out_tensor,
                          int out_num, const OpAttr_t attr, const char *om_file) {
  CHECK_PARAM_NOT_NULL(op_type);
  CHECK_PARAM_NOT_NULL(om_file);
  const std::string om_file_name(om_file);

  std::string op_name = std::string(op_type) + "_" + std::to_string(ge::GetCurrentTimestamp());
  ge::OpDescPtr op_desc = ge::MakeShared<ge::OpDesc>(op_name, op_type);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "MakeShared ge::OpDesc failed, as return nullptr");
    GELOGE(ge::FAILED, "[Call][MakeShared] create ge::OpDesc failed.");
    return ge::FAILED;
  }
  std::vector<ge::GeTensor> inputs;
  for (int i = 0; i < in_num && in_tensor != nullptr; ++i) {
    const ge::TensorDesc *in_desc = CHECK_PARAM_OBJECT(ge::TensorDesc, in_tensor[i]);
    ge::GeTensorDesc tensor_desc(ge::GeShape(in_desc->GetShape().GetDims()), in_desc->GetFormat(),
                                 in_desc->GetDataType());

    tensor_desc.SetOriginFormat(in_desc->GetFormat());
    ge::TensorUtils::SetRealDimCnt(tensor_desc, static_cast<uint32_t>(in_desc->GetShape().GetDims().size()));
    ge::TensorUtils::SetInputTensor(tensor_desc, true);
    ge::TensorUtils::SetOutputTensor(tensor_desc, false);

    if (op_desc->AddInputDesc(tensor_desc) != ge::GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "add inputdesc failed, op:%s", op_desc->GetName().c_str());
      GELOGE(ge::FAILED, "[Add][InputDesc] fail, op:%s.", op_desc->GetName().c_str());
      return ge::FAILED;
    }
    inputs.emplace_back(tensor_desc);
  }

  std::vector<ge::GeTensor> outputs;
  for (int i = 0; i < out_num && out_tensor != nullptr; ++i) {
    const ge::TensorDesc *out_desc = CHECK_PARAM_OBJECT(ge::TensorDesc, out_tensor[i]);
    ge::GeTensorDesc tensor_desc(ge::GeShape(out_desc->GetShape().GetDims()), out_desc->GetFormat(),
                                 out_desc->GetDataType());

    tensor_desc.SetOriginFormat(out_desc->GetFormat());
    ge::TensorUtils::SetRealDimCnt(tensor_desc, static_cast<uint32_t>(out_desc->GetShape().GetDims().size()));
    ge::TensorUtils::SetInputTensor(tensor_desc, false);
    ge::TensorUtils::SetOutputTensor(tensor_desc, true);

    (void)op_desc->AddOutputDesc(tensor_desc);
    outputs.emplace_back(tensor_desc);
  }

  if (attr != nullptr) {
    OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);
    for (const auto &it : op_attr->Attrs()) {
      GE_IF_BOOL_EXEC(op_desc->SetAttr(it.first, it.second) != ge::SUCCESS, GELOGE(ge::FAILED, "SetAttr failed.");
                      REPORT_CALL_ERROR("E19999", "set attr:%s failed, op:%s",
                                        it.first.c_str(), op_desc->GetName().c_str());
                      return ge::FAILED);
    }
  }

  ge::GeGenerator generator;
  return generator.BuildSingleOpModel(op_desc, inputs, outputs, om_file_name);
}

/**
 * @ingroup ge
 * @brief Create Tensor Description.
 * @param [in] format: tensor format of the data.
 * @param [in] datatype: tensor type of the data.
 * @param [in] shape: tensor shape array.
 * @param [in] num: number of shape.
 * @return OpTensor_t for success / nullptr for fail
 */
OpTensor_t OpTensorCreate(int format, int datatype, const int64_t *shape, int num) {
  std::vector<int64_t> dims;
  if (shape != nullptr) {
    for (int i = 0; i < num; ++i) {
      dims.push_back(shape[i]);
    }
  }

  ge::Format fmt = static_cast<ge::Format>(format);
  ge::DataType dt = static_cast<ge::DataType>(datatype);

  return new (std::nothrow) ge::TensorDesc(ge::Shape(dims), fmt, dt);
}

/**
 * @ingroup ge
 * @brief Destroy Tensor Description.
 * @param [in] OpTensor_t tensor: created by OpTensorCreate.
 * @param [out] none
 * @return 0 for success / others for fail.
 */
Status_t OpTensorDestroy(OpTensor_t tensor) {
  ge::TensorDesc *op_tensor = CHECK_PARAM_OBJECT(ge::TensorDesc, tensor);
  delete op_tensor;
  op_tensor = nullptr;

  return ge::SUCCESS;
}

/**
 * @ingroup ge
 * @brief Create an attribute holder.
 * @param [in] none
 * @param [out] none
 * @return OpAttr_t for success / nullptr for fail.
 */
OpAttr_t OpAttrCreate() { return new (std::nothrow) OpAttr; }

/**
 * @ingroup ge
 * @brief Destroy Attribute holder.
 * @param [in] OpAttr_t attr: created by OpAttrCreate.
 * @param [out] none
 * @return 0 for success / others for fail.
 */
Status_t OpAttrDestroy(OpAttr_t attr) {
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);
  delete op_attr;

  return ge::SUCCESS;
}

/**
 * @ingroup ge
 * @brief Set a boolean attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrBool(OpAttr_t attr, const char *name, bool value) {
  CHECK_PARAM_NOT_NULL(name);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value);
}

/**
 * @ingroup ge
 * @brief Set an integer attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrInt(OpAttr_t attr, const char *name, int64_t value) {
  CHECK_PARAM_NOT_NULL(name);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value);
}

/**
 * @ingroup ge
 * @brief Set a float attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrFloat(OpAttr_t attr, const char *name, float value) {
  CHECK_PARAM_NOT_NULL(name);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value);
}

/**
 * @ingroup ge
 * @brief Set a string attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value (can`t be nullptr, end with '\0').
 * @return 0 for success / others for fail.
 */
Status_t SetAttrString(OpAttr_t attr, const char *name, const char *value) {
  CHECK_PARAM_NOT_NULL(name);
  CHECK_PARAM_NOT_NULL(value);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, string(value));
}

/**
 * @ingroup ge
 * @brief Set a boolean array attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value array.
 * @param [in] num: number of value array.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrBoolList(OpAttr_t attr, const char *name, const bool *value, int num) {
  CHECK_PARAM_NOT_NULL(name);
  CHECK_PARAM_NOT_NULL(value);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value, num);
}

/**
 * @ingroup ge
 * @brief Set an integer array attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value array.
 * @param [in] num: number of value array.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrIntList(OpAttr_t attr, const char *name, const int64_t *value, int num) {
  CHECK_PARAM_NOT_NULL(name);
  CHECK_PARAM_NOT_NULL(value);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value, num);
}

/**
 * @ingroup ge
 * @brief Set a float array attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value array.
 * @param [in] num: number of value array.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrFloatList(OpAttr_t attr, const char *name, const float *value, int num) {
  CHECK_PARAM_NOT_NULL(name);
  CHECK_PARAM_NOT_NULL(value);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value, num);
}

/**
 * @ingroup ge
 * @brief Set a string array attribute to the attribute holder.
 * @param [in] attr: attribute holder (created by OpAttrCreate).
 * @param [in] name: attribute name (can`t be nullptr, end with '\0').
 * @param [in] value: attribute value array (each value can`t be nullptr, end with '\0').
 * @param [in] num: number of value array.
 * @return 0 for success / others for fail.
 */
Status_t SetAttrStringList(OpAttr_t attr, const char *name, const char **value, int num) {
  CHECK_PARAM_NOT_NULL(name);
  CHECK_PARAM_NOT_NULL(value);
  OpAttr *op_attr = CHECK_PARAM_OBJECT(OpAttr, attr);

  return op_attr->SetAttr(name, value, num);
}
