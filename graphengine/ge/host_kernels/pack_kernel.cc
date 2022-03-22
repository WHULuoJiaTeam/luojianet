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
#include "host_kernels/pack_kernel.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"
#include "framework/common/types.h"

namespace {
const int64_t kShapeItemNumMAX = 2000000000;
}  // namespace
namespace ge {
Status PackKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                           std::vector<ge::GeTensorPtr> &v_output) {
  GELOGI("Pack kernel in.");
  Status validate_ret = ValidateKernelParams(op_desc_ptr, input);
  if (validate_ret != SUCCESS) {
    GELOGW("Pack kernel input is invalid , can not continue compute.");
    return NOT_CHANGED;
  }

  GeShape final_shape;
  ExpandDims(axis_, input, final_shape);

  // generate output
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGW("Fail to malloc output.");
    return OUT_OF_MEMORY;
  }
  Status ret = CopyOutputData(final_shape, input, output_ptr);
  if (ret != SUCCESS) {
    GELOGW("Pack inputs failed. Ignore pack kernel.");
    return NOT_CHANGED;
  }
  v_output.push_back(output_ptr);
  return SUCCESS;
}

Status PackKernel::ValidateKernelParams(const ge::OpDescPtr &op_desc_ptr,
                                        const std::vector<ge::ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGW("input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if (!(AttrUtils::GetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, n_))) {
    n_ = 0;
    GELOGD("Attr %s is not set, default value %ld is used.", PACK_ATTR_NAME_NUM.c_str(), n_);
  }
  if (!(AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_AXIS, axis_))) {
    GELOGW("Attr %s is not exist.", ATTR_NAME_AXIS.c_str());
    return PARAM_INVALID;
  }
  if (input.empty()) {
    GELOGW("The number of input for Pack should be %ld, in fact it is %zu ", n_, input.size());
    return NOT_CHANGED;
  }
  if (input.size() != static_cast<size_t>(n_)) {
    GELOGW("The number of input for Pack should be %d, in fact it is %ld ", static_cast<int>(n_),
           input.size());
    return PARAM_INVALID;
  }
  data_type_ = op_desc_ptr->GetInputDesc(0).GetDataType();
  GeShape shape = op_desc_ptr->GetInputDesc(0).GetShape();
  if (axis_ < 0 || axis_ > (static_cast<int64_t>(shape.GetDimNum()) + 1)) {
    GELOGW("Axis is %ld ,which is out of range [0,R+1].", axis_);
    return NOT_CHANGED;
  }

  Status validate_ret = ValidateInputs(op_desc_ptr, input);
  if (validate_ret != SUCCESS) {
    GELOGW("Validate inputs failed.Ignore pack kernel.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status PackKernel::ValidateInputs(const ge::OpDescPtr &op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input) {
  GeShape shape;
  for (int64_t i = 0; i < n_; i++) {
    if (input[i] == nullptr) {
      GELOGW("Input %ld of pack kernel %s is null.", i, op_desc_ptr->GetName().c_str());
      return PARAM_INVALID;
    }

    if (i == 0) {
      // get first input shape
      shape = input[0]->GetTensorDesc().GetShape();
    }

    GeTensorDesc tensor_desc = input[i]->GetTensorDesc();
    // check datatype of inputs is same or not
    if (tensor_desc.GetDataType() != data_type_) {
      GELOGW("Data type of inputs %ld for pack not matched, data type should be %s, but actual datatype is %s", i,
             TypeUtils::DataTypeToSerialString(data_type_).c_str(),
             TypeUtils::DataTypeToSerialString(tensor_desc.GetDataType()).c_str());
      return NOT_CHANGED;
    }
    // check shape of inputs is same or not
    auto dst_shape = tensor_desc.GetShape();
    int64_t num = 1;
    for (auto dim : dst_shape.GetDims()) {
      if (dim < 0) {
        GELOGW("Invalid dim %ld in the shape %s", dim, formats::ShapeToString(shape).c_str());
        return NOT_CHANGED;
      }
      num *= dim;
      if (num > kShapeItemNumMAX) {
        GELOGW("Shape overflow, the total count should be less than %ld!", kShapeItemNumMAX);
        return NOT_CHANGED;
      }
    }
    if (!formats::IsShapeEqual(shape, dst_shape)) {
      GELOGW("Shape of input %ld is not equal wiht input 0.", i);
      return NOT_CHANGED;
    }

    // check tensor data size is zero ot not
    if (input[i]->GetData().size() == 0 && num != 0) {
      GELOGW("Inputs %ld do not have value.", i);
      return NOT_CHANGED;
    }
  }
  return SUCCESS;
}

void PackKernel::ExpandDims(const int64_t axis, const std::vector<ge::ConstGeTensorPtr> &input, GeShape &final_shape) {
  // expand dims
  vector<int64_t> current_dims = input[0]->GetTensorDesc().GetShape().GetDims();
  vector<int64_t> final_dims;
  final_dims.assign(current_dims.begin(), current_dims.end());

  // expand dim of N
  // assume there are N inputs, and shape is [A,B,C],
  // if axis = 0, after pack, the output shape should be [N,A,B,C].
  // if axis = 1, after pack, the output shape should be [A,N,B,C].
  // ...etc
  // if axis = 3, after pack, the output shape should be [A,B,C,N]
  if (axis >= static_cast<int64_t>(final_dims.size())) {
    final_dims.emplace_back(n_);
  } else {
    final_dims.insert(final_dims.begin() + axis, n_);
  }
  final_shape = GeShape(final_dims);
}

Status PackKernel::CopyOutputData(const GeShape &final_shape,
                                  const std::vector<ge::ConstGeTensorPtr> &input,
                                  ge::GeTensorPtr &output_ptr) {
  output_ptr->MutableTensorDesc().SetShape(final_shape);
  output_ptr->MutableTensorDesc().SetDataType(DataType(data_type_));
  if (final_shape.GetShapeSize() == 0 && final_shape.GetDims().size() != 0) {
    // means has zero in shape list, output tnesor data is [].
    return SUCCESS;
  }

  int64_t times = 1;
  int64_t unit = 1;
  // calculate data unit
  for (int64_t i = (axis_ + 1); i < static_cast<int64_t>(final_shape.GetDimNum()); i++) {
    unit *= final_shape.GetDim(static_cast<size_t>(i));
  }
  // calculate get times
  for (int64_t i = 0; i < axis_; i++) {
    times *= final_shape.GetDim(static_cast<size_t>(i));
  }
  GELOGD("Copy output data times is %ld, unit is %ld.", times, unit);

  uint32_t data_size = GetSizeByDataType(data_type_);
  // assume output shape is [A,N,B,C], time=A,unit=B*C
  // when copy data from input, we follow time*N*unit
  auto output_size = final_shape.GetShapeSize();
  std::shared_ptr<uint8_t> buf(new (std::nothrow) uint8_t[output_size * data_size], std::default_delete<uint8_t[]>());
  if (buf == nullptr) {
    GELOGW("malloc buf is null.Ignore pack kernel.");
    return NOT_CHANGED;
  }

  size_t dst_offset = 0;
  size_t src_offset = 0;
  // data copy follow times*N*offset, which offset = time*unit
  for (int64_t i = 0; i < times; i++) {
    for (int64_t j = 0; j < n_; j++) {
      // input range already check before. Range is [0,n_).
      const uint8_t *in_data = input[j]->GetData().data();
      auto ret = memcpy_s(buf.get() + dst_offset, output_size * data_size - dst_offset, in_data + src_offset,
                          data_size * unit);
      if (ret != EOK) {
        GELOGW("Memory copy failed.");
        return NOT_CHANGED;
      }
      dst_offset += data_size * unit;
    }
    src_offset += unit * data_size;
  }

  if (output_ptr->SetData(buf.get(), static_cast<size_t>(output_size * data_size)) != GRAPH_SUCCESS) {
    GELOGW("CopyOutputData: SetData failed");
  }
  return SUCCESS;
}

REGISTER_KERNEL(PACK, PackKernel);
}  // namespace ge
