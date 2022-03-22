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

#include "host_kernels/floormod_kernel.h"

#include <memory>
#include <set>

#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kFloorModInputX = 0;
const size_t kFloorModInputY = 1;
const size_t kFloorModFirstOutput = 0;
const size_t kFloorModInputNum = 2;
const std::set<DataType> kFloorModSupportedType = {DT_INT32};

// func FloorDiv is compute floor(x/y); quotient is integer
template <typename T>
T FloorDiv(T const &x, T const &y) {
  if ((x < static_cast<T>(0)) != (y < static_cast<T>(0))) {
    T abs_x = std::abs(x);
    T abs_y = std::abs(y);
    return static_cast<int32_t>((1 - abs_x) / abs_y - 1);
  } else {
    return x / y;
  }
}

template <typename T>
Status CheckYIsZero(T const &y, DataType &type) {
  switch (type) {
    case DT_INT32:
      if (y == static_cast<T>(0)) {
        GELOGE(INTERNAL_ERROR, "CheckYIsZero failed, y is zero.");
        return INTERNAL_ERROR;
      }
      break;
    default:
      return INTERNAL_ERROR;
  }
  return SUCCESS;
}

// mod(x,y) equals to x - y * floor(x/y)
#define DEFINE_FUNC_BY_TYPE(TYPE)                                                     \
  std::function<TYPE(TYPE const &, TYPE const &, DataType &, Status &)> func_##TYPE = \
    [](TYPE const &a, TYPE const &b, DataType &type, Status &ret) -> TYPE {           \
    ret = CheckYIsZero(b, type);                                                      \
    if (ret != SUCCESS) {                                                             \
      return static_cast<TYPE>(0);                                                    \
    }                                                                                 \
    return (a - b * FloorDiv(a, b));                                                  \
  };

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                           \
  case DTYPE:                                                         \
    ret = bcast.BCastComputeCheck(input, y_data_##TYPE, func_##TYPE); \
    break;

#define SET_OUTPUT(DTYPE, TYPE)                                                                                  \
  case DTYPE:                                                                                                    \
    (void)output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE.data()), y_data_##TYPE.size() * length); \
    break;

DEFINE_FUNC_BY_TYPE(int32_t)
}  // namespace

Status FloorModKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                               std::vector<GeTensorPtr> &v_output) {
  GELOGD("FloorModKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = FloorModCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int32_t> y_data_int32_t;
  DataType data_type = input[kFloorModInputX]->GetTensorDesc().GetDataType();
  BCast bcast;
  switch (data_type) {
    SET_BCAST_COMPUTE_CASE(DT_INT32, int32_t)
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

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kFloorModFirstOutput));
  if (output_ptr == nullptr) {
    GELOGW("make_shared ge::GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast.GetOutputShape()));
  // only return GRAPH_SUCCESS here
  switch (data_type) {
    SET_OUTPUT(DT_INT32, int32_t)
    default:
      break;
  }
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  v_output.push_back(output_ptr);
  GELOGD("FloorModKernel success");

  return SUCCESS;
}

Status FloorModKernel::FloorModCheck(const std::vector<ConstGeTensorPtr> &input) {
  // check input number
  if (input.size() != kFloorModInputNum) {
    GELOGI("The number of input for FloorMod must be %zu.", kFloorModInputNum);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr input_x1 = input.at(kFloorModInputX);
  ConstGeTensorPtr input_x2 = input.at(kFloorModInputY);
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
    GELOGI("Data type of inputs for FloorMod not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (kFloorModSupportedType.find(type) == kFloorModSupportedType.end()) {
    GELOGI("FloorMod does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_KERNEL(FLOORMOD, FloorModKernel);
}  // namespace ge
