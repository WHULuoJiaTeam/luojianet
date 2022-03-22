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

#include "host_kernels/maximum_kernel.h"

#include <memory>
#include <set>

#include "framework/common/debug/log.h"
#include "common/fp16_t.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kMaximumInputNum = 2;
const size_t kMaximumFirstInput = 0;
const size_t kMaximumSecondInput = 1;
const size_t kMaximumFirstOutput = 0;
const std::set<DataType> kMaximumSupportedType = {DT_FLOAT, DT_FLOAT16, DT_INT8,   DT_INT16,  DT_UINT16, DT_UINT8,
                                                  DT_INT32, DT_INT64,   DT_UINT32, DT_UINT64, DT_DOUBLE};

#define DEFINE_FUNC_BY_TYPE(TYPE)                                                                          \
  std::function<TYPE(TYPE const &, TYPE const &)> func_##TYPE = [](TYPE const &a, TYPE const &b) -> TYPE { \
    return (a > b ? a : b);                                                                                \
  };

#define SET_BCAST_COMPUTE_CASE(DTYPE, TYPE)                      \
  case DTYPE:                                                    \
    ret = bcast.BCastCompute(input, y_data_##TYPE, func_##TYPE); \
    break;

#define SET_OUTPUT(DTYPE, TYPE)                                                                                  \
  case DTYPE:                                                                                                    \
    if (output_ptr->SetData(reinterpret_cast<uint8_t *>(y_data_##TYPE.data()), y_data_##TYPE.size() * length) != \
        GRAPH_SUCCESS) {                                                                                         \
      GELOGW("GenData: SetData failed");                                                                         \
    }                                                                                                            \
    break;

DEFINE_FUNC_BY_TYPE(int8_t)
DEFINE_FUNC_BY_TYPE(int16_t)
DEFINE_FUNC_BY_TYPE(int32_t)
DEFINE_FUNC_BY_TYPE(int64_t)
DEFINE_FUNC_BY_TYPE(uint8_t)
DEFINE_FUNC_BY_TYPE(uint16_t)
DEFINE_FUNC_BY_TYPE(uint32_t)
DEFINE_FUNC_BY_TYPE(uint64_t)
DEFINE_FUNC_BY_TYPE(fp16_t)
DEFINE_FUNC_BY_TYPE(float)
DEFINE_FUNC_BY_TYPE(double)
}  // namespace

Status MaximumKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                              std::vector<GeTensorPtr> &v_output) {
  GELOGD("MaximumKernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  Status ret = MaximumCheck(input);
  if (ret != SUCCESS) {
    return ret;
  }

  std::vector<int8_t> y_data_int8_t;
  std::vector<int16_t> y_data_int16_t;
  std::vector<int32_t> y_data_int32_t;
  std::vector<int64_t> y_data_int64_t;
  std::vector<uint8_t> y_data_uint8_t;
  std::vector<uint16_t> y_data_uint16_t;
  std::vector<uint32_t> y_data_uint32_t;
  std::vector<uint64_t> y_data_uint64_t;
  std::vector<fp16_t> y_data_fp16_t;
  std::vector<float> y_data_float;
  std::vector<double> y_data_double;

  if (input.empty()) {
    GELOGE(FAILED, "input is empty.");
    return FAILED;
  }
  DataType data_type = input[kMaximumFirstInput]->GetTensorDesc().GetDataType();
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

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kMaximumFirstOutput));
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
  GELOGD("MaximumKernel success");

  return SUCCESS;
}

Status MaximumKernel::MaximumCheck(const std::vector<ConstGeTensorPtr> &input) {
  // check input number
  if (input.size() != kMaximumInputNum) {
    GELOGI("The number of input for Maximum must be %zu.", kMaximumInputNum);
    return NOT_CHANGED;
  }
  ConstGeTensorPtr input_x1 = input.at(kMaximumFirstInput);
  ConstGeTensorPtr input_x2 = input.at(kMaximumSecondInput);
  GE_CHECK_NOTNULL(input_x1);
  GE_CHECK_NOTNULL(input_x2);

  // check whether there is data in Tensor
  if ((input_x1->GetData().size() == 0) || (input_x2->GetData().size() == 0)) {
    GELOGI("Check data size fail. x1: %zu, x2: %zu", input_x1->GetData().size(), input_x2->GetData().size());
    return NOT_CHANGED;
  }

  // check whether the data types are the same
  DataType type = input_x1->GetTensorDesc().GetDataType();
  if (type != input_x2->GetTensorDesc().GetDataType()) {
    GELOGI("Data type of inputs for Maximum not matched.");
    return NOT_CHANGED;
  }

  // check if input data type is supported
  if (kMaximumSupportedType.find(type) == kMaximumSupportedType.end()) {
    GELOGI("Maximum does not support this Data type: %s", TypeUtils::DataTypeToSerialString(type).c_str());
    return NOT_CHANGED;
  }

  return SUCCESS;
}

REGISTER_KERNEL(MAXIMUM, MaximumKernel);
}  // namespace ge
