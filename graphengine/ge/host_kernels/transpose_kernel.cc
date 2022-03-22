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

#include "host_kernels/transpose_kernel.h"
#include <memory>
#include <vector>
#include "framework/common/debug/log.h"
#include "common/formats/format_transfers/format_transfer_transpose.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kTransposeInputX = 0;
const size_t kTransposeInputPerm = 1;
const size_t kTransposeInputSize = 2;
const size_t kTransposeOutputY = 0;
const size_t kTransposeOutputSize = 1;
}  // namespace

Status TransposeKernel::ValidateInput(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGW("Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetInputsSize() != kTransposeInputSize || op_desc_ptr->GetOutputsSize() != kTransposeOutputSize) {
    GELOGW("The input_size(%zu) and output_size(%zu) of op are invalid, op name: %s.", op_desc_ptr->GetInputsSize(),
           op_desc_ptr->GetOutputsSize(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  if (input.size() != kTransposeInputSize) {
    GELOGW("The size of input tensor vector is invalid, input size is %zu, op name: %s.", input.size(),
           op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  ConstGeTensorPtr tensor_x_ptr = input[kTransposeInputX];
  ConstGeTensorPtr tensor_perm_ptr = input[kTransposeInputPerm];
  if (tensor_x_ptr == nullptr || tensor_perm_ptr == nullptr) {
    GELOGW("Input tensor of op is nullptr, node name: %s.", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status TransposeKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                std::vector<GeTensorPtr> &v_output) {
  GELOGD("TransposeKernel in.");
  Status status = ValidateInput(op_desc_ptr, input);
  if (status != SUCCESS) {
    GELOGW("TransposeKernel input is invalid, failed to fold node.");
    return NOT_CHANGED;
  }

  ConstGeTensorPtr const_weight_ptr = input[kTransposeInputX];
  GeTensorDesc op_desc = op_desc_ptr->GetOutputDesc(kTransposeOutputY);
  GeTensorDesc op_desc_in = op_desc_ptr->GetInputDesc(kTransposeInputX);
  auto src_format = op_desc_in.GetFormat();
  auto src_shape = op_desc_in.GetShape().GetDims();
  auto src_data_type = op_desc_in.GetDataType();
  auto data_shape = op_desc.GetShape().GetDims();
  auto data_format = op_desc.GetFormat();
  auto data_type = op_desc.GetDataType();
  GELOGD(
      "current node %s, format %s, input shape %s, data type %s,  weight format %s, shape %s, data type %s. "
      "output format %s, shape %s, data type %s",
      op_desc_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(src_format).c_str(),
      formats::ShapeToString(src_shape).c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
      TypeUtils::FormatToSerialString(const_weight_ptr->GetTensorDesc().GetFormat()).c_str(),
      formats::ShapeToString(const_weight_ptr->GetTensorDesc().GetShape()).c_str(),
      TypeUtils::DataTypeToSerialString(const_weight_ptr->GetTensorDesc().GetDataType()).c_str(),
      TypeUtils::FormatToSerialString(data_format).c_str(), formats::ShapeToString(data_shape).c_str(),
      TypeUtils::DataTypeToSerialString(data_type).c_str());

  ConstGeTensorPtr tensor_perm_ptr = input[kTransposeInputPerm];
  DataType data_dtype = tensor_perm_ptr->GetTensorDesc().GetDataType();
  auto input_perm_shape = tensor_perm_ptr->GetTensorDesc().GetShape();
  auto output_size = input_perm_shape.GetShapeSize();
  uint32_t data_size = GetSizeByDataType(data_dtype);
  if (static_cast<size_t>(output_size * data_size) != tensor_perm_ptr->GetData().size()) {
    GELOGW("TransposeKernel input perm shape size and data size do not match.");
    return NOT_CHANGED;
  }

  vector<int64_t> perm_list;
  auto input_perm = tensor_perm_ptr->GetData().data();
  if (data_dtype == DT_INT32) {
    int32_t *input_perm_data = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(input_perm));
    for (int64_t i = 0; i < output_size; i++) {
      perm_list.push_back(static_cast<int64_t>(input_perm_data[i]));
    }
  } else if (data_dtype == DT_INT64) {
    int64_t *input_perm_data = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(input_perm));
    for (int64_t i = 0; i < output_size; i++) {
      perm_list.push_back(input_perm_data[i]);
    }
  } else {
    GELOGW("TransposeKernel input perm data type is invalid, data type is %s.",
           TypeUtils::DataTypeToSerialString(data_dtype).c_str());
    return NOT_CHANGED;
  }

  GELOGD("Transpose from %s to %s, shape %s to  %s, perm_list %s, data type %s",
         TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
         formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
         formats::ShapeToString(perm_list).c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
  if ((data_shape.empty()) || (src_data_type != data_type)) {
    GELOGW("Transpose is not supported. Invalid shape (src: %s, dst: %s) or inconsistent datatype (src: %s, dst: %s)",
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
           TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }
  if (!KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr)) {
    GELOGW("CheckSize failed, input size is not equal to weight size");
    return NOT_CHANGED;
  }
  const uint8_t *src_data = const_weight_ptr->GetData().data();
  formats::TransResult trans_result;
  auto ret = formats::TransposeWithShapeCheck(src_data, src_shape, data_shape, src_data_type, perm_list, trans_result);
  if (ret != SUCCESS) {
    GELOGW("Failed to Transpose from %s to %s, shape %s to  %s, perm_list %s, data type %s",
           TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
           formats::ShapeToString(perm_list).c_str(), TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kTransposeOutputY));
  GE_CHECK_NOTNULL(output_ptr);
  if (output_ptr->SetData(trans_result.data.get(), trans_result.length) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.push_back(output_ptr);

  GELOGI("TransposeKernel success.");
  return SUCCESS;
}

REGISTER_KERNEL(TRANSPOSE, TransposeKernel);
}  // namespace ge
