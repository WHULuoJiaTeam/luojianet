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

#include "host_kernels/sub_kernel.h"

#include <cfloat>
#include <cmath>
#include <memory>

#include "framework/common/debug/log.h"
#include "common/math/math_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kSubFirstInput = 0;
const size_t kSubSecondInput = 1;
const size_t kSubFirstOutput = 0;
const size_t kSubOutputSize = 1;
const size_t kSubInputSize = 2;

template <typename T>
Status OverflowCheck(T const &x, T const &y, DataType &data_type) {
  switch (data_type) {
    case DT_INT8:
      FMK_INT8_SUBCHECK(x, y)
      break;
    case DT_INT16:
      FMK_INT16_SUBCHECK(x, y)
      break;
    case DT_INT32:
      FMK_INT32_SUBCHECK(x, y)
      break;
    case DT_INT64:
      FMK_INT64_SUBCHECK(x, y)
      break;
    case DT_UINT8:
      FMK_UINT8_SUBCHECK(x, y)
      break;
    case DT_UINT16:
      FMK_UINT16_SUBCHECK(x, y)
      break;
    case DT_UINT32:
      FMK_UINT32_SUBCHECK(x, y)
      break;
    case DT_UINT64:
      FMK_UINT64_SUBCHECK(x, y)
      break;
    case DT_FLOAT16:
      FMK_FP16_SUBCHECK(x, y)
      break;
    case DT_FLOAT:
      FMK_FLOAT_SUBCHECK(x, y)
      break;
    case DT_DOUBLE:
      FMK_DOUBLE_SUBCHECK(x, y)
      break;
    default:
      break;
  }

  return SUCCESS;
}

#define DEFINE_FUNC_WITH_STATUS_BY_TYPE(TYPE)                                         \
  std::function<TYPE(TYPE const &, TYPE const &, DataType &, Status &)> func_##TYPE = \
    [](TYPE const &x, TYPE const &y, DataType &type, Status &ret) -> TYPE {           \
    ret = OverflowCheck<TYPE>(x, y, type);                                            \
    if (ret != SUCCESS) {                                                             \
      GELOGE(PARAM_INVALID, "Result of sub is overflow.");                            \
      return static_cast<TYPE>(0);                                                    \
    }                                                                                 \
    return static_cast<TYPE>(x) - static_cast<TYPE>(y);                               \
  };

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                              \
  case DTYPE:                                                            \
    ret = bcast.BCastComputeCheck(input, y_data_##TYPE##_, func_##TYPE); \
    break;

#define SET_OUTPUT(DTYPE, TYPE)                                                                                        \
  case DTYPE:                                                                                                          \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE##_.data()), y_data_##TYPE##_.size() * length); \
    break;

DEFINE_FUNC_WITH_STATUS_BY_TYPE(int8_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int16_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int32_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(int64_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint8_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint16_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint32_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(uint64_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(fp16_t)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(float)
DEFINE_FUNC_WITH_STATUS_BY_TYPE(double)
}  // namespace

Status SubKernel::Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                          vector<ge::GeTensorPtr> &v_output) {
  GE_CHECK_NOTNULL(op_desc_ptr);
  // check how many inputs
  if ((input.size() != kSubInputSize) || (op_desc_ptr->GetOutputsSize() != kSubOutputSize)) {
    GELOGW("The number of input for sub must be %zu.", kSubInputSize);
    return NOT_CHANGED;
  }

  GE_CHECK_NOTNULL(input[kSubFirstInput]);
  GE_CHECK_NOTNULL(input[kSubSecondInput]);
  ConstGeTensorPtr weight0 = input[kSubFirstInput];
  ConstGeTensorPtr weight1 = input[kSubSecondInput];

  Status ret;
  DataType data_type = input[kSubFirstInput]->GetTensorDesc().GetDataType();
  BCast bcast;
  switch (data_type) {
    SET_BCAST_COMPUTE_CASE(DT_INT8, int8_t)
    SET_BCAST_COMPUTE_CASE(DT_INT16, int16_t)
    SET_BCAST_COMPUTE_CASE(DT_INT32, int32_t)
    SET_BCAST_COMPUTE_CASE(DT_INT64, int64_t)
    SET_BCAST_COMPUTE_CASE(DT_UINT8, uint8_t)
    SET_BCAST_COMPUTE_CASE(DT_UINT16, uint16_t)
    SET_BCAST_COMPUTE_CASE(DT_UINT32, uint32_t)
    SET_BCAST_COMPUTE_CASE(DT_UINT64, uint64_t)
    SET_BCAST_COMPUTE_CASE(DT_FLOAT16, fp16_t)
    SET_BCAST_COMPUTE_CASE(DT_FLOAT, float)
    SET_BCAST_COMPUTE_CASE(DT_DOUBLE, double)
    default:
      GELOGI("Sub kernel data type %s not support.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      ret = NOT_CHANGED;
      break;
  }

  if (ret != SUCCESS) {
    GELOGW("BCastCompute fail, data_type:%s, ret:%s", TypeUtils::DataTypeToSerialString(data_type).c_str(),
           GET_ERRORNO_STR(ret).c_str());
    return NOT_CHANGED;
  }

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kSubFirstOutput);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("make_shared ge::GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  // only return GRAPH_SUCCESS here
  switch (data_type) {
    SET_OUTPUT(DT_INT8, int8_t)
    SET_OUTPUT(DT_INT16, int16_t)
    SET_OUTPUT(DT_INT32, int32_t)
    SET_OUTPUT(DT_INT64, int64_t)
    SET_OUTPUT(DT_UINT8, uint8_t)
    SET_OUTPUT(DT_UINT16, uint16_t)
    SET_OUTPUT(DT_UINT32, uint32_t)
    SET_OUTPUT(DT_UINT64, uint64_t)
    SET_OUTPUT(DT_FLOAT16, fp16_t)
    SET_OUTPUT(DT_FLOAT, float)
    SET_OUTPUT(DT_DOUBLE, double)
    default:
      break;
  }
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  v_output.push_back(output_ptr);

  return SUCCESS;
}

REGISTER_KERNEL(SUB, SubKernel);
}  // namespace ge
