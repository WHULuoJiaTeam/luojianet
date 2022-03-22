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

#include "host_kernels/permute_kernel.h"

#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"
#include "common/formats/formats.h"
#include "common/formats/format_transfers/format_transfer_transpose.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "host_kernels/kernel_utils.h"
#include "framework/common/ge_inner_error_codes.h"


namespace ge {
namespace {
const char *const kAttrOrder = "order";
const char *const kAttrPerm = "perm";
const size_t kTbePermuteInputSize = 2;
}  // namespace

Status PermuteKernel::ValidateInput(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Input tensor vector is empty");
    return PARAM_INVALID;
  }
  ConstGeTensorPtr const_weight_ptr = input[0];
  if (const_weight_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input const_weight_ptr is nullptr.");
    return PARAM_INVALID;
  }
  const uint8_t *src_data = const_weight_ptr->GetData().data();
  if ((op_desc_ptr == nullptr) || (src_data == nullptr)) {
    GELOGW("Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetInputsSize() >= kTbePermuteInputSize) {
    GELOGW("trans_op has more than 1 input_size.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status PermuteKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                              std::vector<GeTensorPtr> &v_output) {
  GELOGD("PermuteKernel begin.");
  Status status = ValidateInput(op_desc_ptr, input);
  if (status != SUCCESS) {
    return status;
  }

  ConstGeTensorPtr const_weight_ptr = input[0];
  GeTensorDesc op_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorDesc op_desc_in = op_desc_ptr->GetInputDesc(0);
  auto src_format = op_desc_in.GetFormat();
  auto src_shape  = op_desc_in.GetShape().GetDims();
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

  vector<int64_t> perm_list;
  if (!AttrUtils::GetListInt(op_desc_ptr, kAttrOrder, perm_list) &&
      !AttrUtils::GetListInt(op_desc_ptr, kAttrPerm, perm_list)) {
    GELOGW("Get perm_list failed, Transpose from shape %s to %s is not supported, ",
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str());
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

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  GE_CHECK_NOTNULL(output_ptr);
  GE_CHK_STATUS_RET(output_ptr->SetData(trans_result.data.get(), trans_result.length));
  v_output.push_back(output_ptr);
  return SUCCESS;
}

REGISTER_KERNEL(PERMUTE, PermuteKernel);
REGISTER_KERNEL(TRANSPOSED, PermuteKernel);
}  // namespace ge
