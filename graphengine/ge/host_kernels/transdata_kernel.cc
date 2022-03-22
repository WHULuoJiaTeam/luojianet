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

#include "host_kernels/transdata_kernel.h"

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
const size_t kTransdataInputSize = 1;
}

Status TransdataKernel::ValidateInput(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Input tensor vector is empty");
    return PARAM_INVALID;
  }
  ConstGeTensorPtr const_weight_ptr = input[0];
  if (const_weight_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input const_weight_ptr is nullptr.");
    return PARAM_INVALID;
  }

  // src_data == nullptr is supported
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetInputsSize() != kTransdataInputSize) {
    GELOGW("trans_op has more than 1 input_size.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status TransdataKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                std::vector<GeTensorPtr> &v_output) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  GELOGD("TransdataKernel begin.");
  Status status = ValidateInput(op_desc_ptr, input);
  if (status != SUCCESS) {
    return status;
  }

  ConstGeTensorPtr const_weight_ptr = input[0];
  const auto &op_desc = op_desc_ptr->MutableOutputDesc(0);
  const auto &op_desc_in = op_desc_ptr->MutableInputDesc(0);
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(op_desc_in);
  const auto &src_format = op_desc_in->GetFormat();
  const auto &src_shape = op_desc_in->GetShape().GetDims();
  const auto &src_data_type = op_desc_in->GetDataType();
  const auto &data_shape = op_desc->GetShape().GetDims();
  const auto &data_format = op_desc->GetFormat();
  const auto &data_type = op_desc->GetDataType();
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

  const uint8_t *src_data = const_weight_ptr->GetData().data();
  const formats::TransArgs trans_args{src_data, src_format, data_format, src_shape, data_shape, src_data_type};
  formats::TransResult trans_result;
  GELOGD("Trans formats from %s to %s, shape %s to %s, data type %s",
         TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
         formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
         TypeUtils::DataTypeToSerialString(src_data_type).c_str());

  if (src_data_type != data_type || data_shape.empty() || !formats::IsTransFormatSupport(trans_args)) {
    GELOGW("Transfer from format %s to %s, shape %s to %s, data type %s to %s is not supported",
           TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
           TypeUtils::DataTypeToSerialString(src_data_type).c_str(),
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }
  if (!KernelUtils::CheckSizeForTransOp(const_weight_ptr, op_desc_ptr)) {
    GELOGI("CheckSize failed, input size is not equal to weight size");
    return NOT_CHANGED;
  }
  if (formats::TransFormat(trans_args, trans_result) != SUCCESS) {
    GELOGW("Failed to trans formats from %s to %s, shape %s to  %s, data type %s",
           TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(data_format).c_str(),
           formats::ShapeToString(src_shape).c_str(), formats::ShapeToString(data_shape).c_str(),
           TypeUtils::DataTypeToSerialString(src_data_type).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Make shared failed");
    return ge::PARAM_INVALID;
  }
  if (output_ptr->SetData(trans_result.data.get(), trans_result.length) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.push_back(output_ptr);
  return SUCCESS;
}

REGISTER_KERNEL(TRANSDATA, TransdataKernel);
}  // namespace ge
