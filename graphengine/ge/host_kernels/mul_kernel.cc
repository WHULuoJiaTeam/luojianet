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

#include "host_kernels/mul_kernel.h"

#include <memory>
#include <set>

#include "framework/common/debug/log.h"
#include "common/math/math_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const std::set<DataType> kMulSupportedType = {DT_INT8,   DT_INT16,  DT_INT32,   DT_INT64, DT_UINT8, DT_UINT16,
                                              DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT, DT_DOUBLE};
template <typename T>
Status OverflowCheck(T const &x, T const &y, DataType &type) {
  switch (type) {
    case DT_INT8:
      FMK_INT8_MULCHECK(x, y)
      break;
    case DT_INT16:
      FMK_INT16_MULCHECK(x, y)
      break;
    case DT_INT32:
      FMK_INT32_MULCHECK(x, y)
      break;
    case DT_INT64:
      FMK_INT64_MULCHECK(x, y)
      break;
    case DT_UINT8:
      FMK_UINT8_MULCHECK(x, y)
      break;
    case DT_UINT16:
      FMK_UINT16_MULCHECK(x, y)
      break;
    case DT_UINT32:
      FMK_UINT32_MULCHECK(x, y)
      break;
    case DT_UINT64:
      FMK_UINT64_MULCHECK(x, y)
      break;
    case DT_FLOAT16:
      FMK_FP16_MULCHECK(x, y)
      break;
    case DT_FLOAT:
      FMK_FLOAT_MULCHECK(x, y)
      break;
    case DT_DOUBLE:
      FMK_DOUBLE_MULCHECK(x, y)
      break;
    default:
      break;
  }

  return SUCCESS;
}

#define DEFINE_FUNC_WITH_STATUS_BY_TYPE(TYPE)                                         \
  std::function<TYPE(TYPE const &, TYPE const &, DataType &, Status &)> func_##TYPE = \
    [](TYPE const &a, TYPE const &b, DataType &type, Status &ret) -> TYPE {           \
    ret = OverflowCheck(a, b, type);                                                  \
    if (ret != SUCCESS) {                                                             \
      GELOGE(PARAM_INVALID, "Result of mul is overflow.");                            \
      return static_cast<TYPE>(0);                                                    \
    }                                                                                 \
    return static_cast<TYPE>(a) * static_cast<TYPE>(b);                               \
  };

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                              \
  case DTYPE:                                                            \
    ret = bcast.BCastComputeCheck(input, y_data_##TYPE##_, func_##TYPE); \
    break;

#define SET_OUTPUT(DTYPE, TYPE)                                                                                        \
  case DTYPE:                                                                                                          \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE##_.data()), y_data_##TYPE##_.size() * length); \
    break;
// [no need to check result]
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

Status MulKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                          std::vector<GeTensorPtr> &v_output) {
  GELOGD("MulKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = MulCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  DataType data_type = input[0]->GetTensorDesc().GetDataType();
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
      ret = NOT_CHANGED;
      break;
  }

  if (ret != SUCCESS) {
    GELOGW("BCastCompute fail, data_type: %s, ret: %s", TypeUtils::DataTypeToSerialString(data_type).c_str(),
           GET_ERRORNO_STR(ret).c_str());
    return NOT_CHANGED;
  }

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGW("Can't GetDataTypeLength of data_type: %s", TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
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
  GELOGD("MulKernel success");

  return SUCCESS;
}

Status MulKernel::MulCheck(const std::vector<ConstGeTensorPtr> &input) {
  // check input number
  if (input.size() != static_cast<size_t>(MUL_INPUT_NUM)) {
    GELOGI("The number of input for Mul must be %u.", MUL_INPUT_NUM);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr input_x1 = input.at(0);
  ConstGeTensorPtr input_x2 = input.at(1);
  GE_CHECK_NOTNULL(input_x1);
  GE_CHECK_NOTNULL(input_x2);
  // check whether there is data in Tensor
  if (input_x1->GetData().size() == 0 || input_x2->GetData().size() == 0) {
    GELOGI("Check data size fail. x1: %zu, x2: %zu", input_x1->GetData().size(), input_x2->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  DataType type = input_x1->GetTensorDesc().GetDataType();
  if (type != input_x2->GetTensorDesc().GetDataType()) {
    GELOGI("Data type of inputs for Mul not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (kMulSupportedType.find(type) == kMulSupportedType.end()) {
    GELOGI("Mul does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_KERNEL(MUL, MulKernel);
}  // namespace ge
