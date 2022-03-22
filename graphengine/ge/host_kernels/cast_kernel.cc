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

#include "host_kernels/cast_kernel.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/fp16_t.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/bcast.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kCastInputSize = 1;
}
Status CastKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                           std::vector<GeTensorPtr> &v_output) {
  GELOGD("CastKernel begin.");
  if (input.size() != kCastInputSize) {
    GELOGE(PARAM_INVALID, "The number of input for cast must be %zu.", kCastInputSize);
    return PARAM_INVALID;
  }
  ConstGeTensorPtr const_weight_ptr = input[0];
  if (const_weight_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input const_weight_ptr is nullptr.");
    return PARAM_INVALID;
  }

  const uint8_t *src_data = const_weight_ptr->GetData().data();
  // src_data == nullptr is supported
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  GeTensorDesc op_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorDesc op_desc_in = op_desc_ptr->GetInputDesc(0);
  auto src_data_type = op_desc_in.GetDataType();
  auto src_shape   = op_desc_in.GetShape();
  auto src_format  = op_desc_in.GetFormat();
  auto data_type   = op_desc.GetDataType();
  auto data_shape  = op_desc.GetShape();
  auto data_format = op_desc.GetFormat();
  GELOGD("Current node %s, format %s, input shape %s, data type %s,  weight format %s, shape %s, data type %s. "
         "output format %s, shape %s, data type %s", op_desc_ptr->GetName().c_str(),
         TypeUtils::FormatToSerialString(src_format).c_str(),
         formats::ShapeToString(src_shape).c_str(),
         TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
         TypeUtils::FormatToSerialString(const_weight_ptr->GetTensorDesc().GetFormat()).c_str(),
         formats::ShapeToString(const_weight_ptr->GetTensorDesc().GetShape()).c_str(),
         TypeUtils::DataTypeToSerialString(const_weight_ptr->GetTensorDesc().GetDataType()).c_str(),
         TypeUtils::FormatToSerialString(data_format).c_str(),
         formats::ShapeToString(data_shape).c_str(),
         TypeUtils::DataTypeToSerialString(data_type).c_str());

  // const_weight_ptr->GetData().GetSize() == 0 is supported
  auto src_data_size = src_shape.GetShapeSize();
  if (src_data_size == 0 &&
      static_cast<int>(const_weight_ptr->GetData().GetSize()) == GetSizeByDataType(src_data_type)) {
    src_data_size = 1;
    GELOGD("Weight of the current const node is scalar");
  }
  const formats::CastArgs cast_args{src_data, static_cast<size_t>(src_data_size), src_data_type, data_type};
  formats::TransResult trans_result;
  GELOGD("Trans data type from %s to %s, shape %s, data size %ld",
         TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
         TypeUtils::DataTypeToSerialString(data_type).c_str(),
         formats::ShapeToString(src_shape).c_str(), src_data_size);

  if ((src_format != data_format) || (src_shape.GetDims() != data_shape.GetDims()) ||
      (!formats::IsTransDataTypeSupport(cast_args))) {
    GELOGW("Transfer from data type %s to %s, format %s to %s, shape %s to %s is not supported",
           TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str(),
           TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str());
    return NOT_CHANGED;
  }
  if (!KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr)) {
    GELOGE(FAILED, "CheckSize failed, input size is not equal to weight size");
    return NOT_CHANGED;
  }
  if (formats::TransDataType(cast_args, trans_result) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to trans data type from %s to %s, shape %s, data size %ld.",
           TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str(),
           formats::ShapeToString(src_shape).c_str(), src_data_size);
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    return FAILED;
  }
  if (output_ptr->SetData(trans_result.data.get(), trans_result.length) != SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.push_back(output_ptr);
  return SUCCESS;
}

REGISTER_KERNEL(CAST, CastKernel);
}  // namespace ge
