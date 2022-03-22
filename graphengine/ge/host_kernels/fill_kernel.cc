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

#include "host_kernels/fill_kernel.h"

#include <memory>
#include <vector>

#include "common/fp16_t.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"
#include "framework/common/types.h"

namespace {
const int kFillInputSize = 2;
const int kFillDimsInputIndex = 0;
const int kFillDataInputIndex = 1;
}  // namespace

namespace ge {
Status FillKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                           std::vector<ge::GeTensorPtr> &v_output) {
  if (input.size() != kFillInputSize) {
    GELOGW("fill input size must be %d", kFillInputSize);
    return NOT_CHANGED;
  }
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  GELOGD("FillKernel in, name: %s.", op_desc_ptr->GetName().c_str());

  GE_CHECK_NOTNULL(input.at(kFillDimsInputIndex));
  GE_CHECK_NOTNULL(input.at(kFillDataInputIndex));

  ConstGeTensorPtr dims = input.at(kFillDimsInputIndex);
  ConstGeTensorPtr value = input.at(kFillDataInputIndex);
  // Check if the value is a scalar
  if (value->GetTensorDesc().GetShape().GetDimNum() != 0) {
    GELOGW("value must be a scalar.");
    return NOT_CHANGED;
  }

  auto output_desc = op_desc_ptr->GetOutputDescPtr(0);
  GE_CHECK_NOTNULL(output_desc);
  if (output_desc->GetShape().IsUnknownShape()) {
    GELOGD("Output is unknown shape, [%s] skip FillKernel.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr;
  output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "make_shared ge::GeTensor failed");
    return MEMALLOC_FAILED;
  }

  int64_t fill_size = 1;
  std::vector<int64_t> vec_dim;
  DataType dim_type = dims->GetTensorDesc().GetDataType();

  // Calculate user input dim
  Status ret = PARAM_INVALID;
  if (dim_type == DT_INT32) {
    ret = KernelUtils::CalcDims<int32_t>(dims, vec_dim, fill_size);
  } else if (dim_type == DT_INT64) {
    ret = KernelUtils::CalcDims<int64_t>(dims, vec_dim, fill_size);
  } else {
    GELOGE(PARAM_INVALID, "dim type must be DT_INT32 or DT_INT64.");
    return PARAM_INVALID;
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "CalcDims failed, dim_type: %s", TypeUtils::DataTypeToSerialString(dim_type).c_str());
    return ret;
  }

  // Generating a sequence of numbers
  DataType data_type = value->GetTensorDesc().GetDataType();
  ret = PARAM_INVALID;
  switch (data_type) {
#define CASE(dtype, type)                                                                                        \
  case dtype:                                                                                                    \
    ret = KernelUtils::GenData(fill_size, *reinterpret_cast<const type *>(value->GetData().data()), output_ptr); \
    break;
    CASE(DT_FLOAT, float)
    CASE(DT_FLOAT16, fp16_t)
    CASE(DT_INT8, int8_t)
    CASE(DT_INT16, int16_t)
    CASE(DT_UINT16, uint16_t)
    CASE(DT_UINT8, uint8_t)
    CASE(DT_INT32, int32_t)
    CASE(DT_INT64, int64_t)
    CASE(DT_UINT32, uint32_t)
    CASE(DT_UINT64, uint64_t)
    CASE(DT_BOOL, bool)
    CASE(DT_DOUBLE, double)
#undef CASE
    default:
      GELOGW("invalid data type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }
  if (ret != SUCCESS) {
    GELOGE(ret, "GenData failed, data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return ret;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(vec_dim));
  output_ptr->MutableTensorDesc().SetDataType(DataType(data_type));
  v_output.push_back(output_ptr);

  return SUCCESS;
}
REGISTER_KERNEL(FILL, FillKernel);
}  // namespace ge
