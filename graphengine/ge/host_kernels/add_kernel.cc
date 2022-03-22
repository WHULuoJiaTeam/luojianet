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

#include "host_kernels/add_kernel.h"

#include <cfloat>

#include "common/math/math_util.h"
#include "common/bcast.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kAddFirstInput = 0;
const size_t kAddSecondInput = 1;
const size_t kAddFirstOutput = 0;
const size_t kAddInputSize = 2;
const size_t kAddOutputSize = 1;

#define SET_BCAST_ADD_CASE(DTYPE, TYPE)                 \
  case (DTYPE):                                         \
    ret = BCastAdd<TYPE>(op_desc_ptr, input, v_output); \
    break;
}  // namespace

template <typename T>
Status AddKernel::OverflowCheck(const T &x, const T &y, DataType data_type) {
  switch (data_type) {
    case DT_INT8:
      FMK_INT8_ADDCHECK(x, y)
      break;
    case DT_INT16:
      FMK_INT16_ADDCHECK(x, y)
      break;
    case DT_INT32:
      FMK_INT32_ADDCHECK(x, y)
      break;
    case DT_INT64:
      FMK_INT64_ADDCHECK(x, y)
      break;
    case DT_UINT8:
      FMK_UINT8_ADDCHECK(x, y)
      break;
    case DT_UINT16:
      FMK_UINT16_ADDCHECK(x, y)
      break;
    case DT_UINT32:
      FMK_UINT32_ADDCHECK(x, y)
      break;
    case DT_UINT64:
      FMK_UINT64_ADDCHECK(x, y)
      break;
    case DT_FLOAT16:
      FMK_FP16_ADDCHECK(x, y)
      break;
    case DT_FLOAT:
      FMK_FLOAT_ADDCHECK(x, y)
      break;
    case DT_DOUBLE:
      FMK_DOUBLE_ADDCHECK(x, y)
      break;
    default:
      break;
  }

  return SUCCESS;
}

template <typename InT>
Status AddKernel::BCastAdd(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                           std::vector<GeTensorPtr> &v_output) {
  // only broadcast shape
  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(BCast::TransShapeToDimVec(input[kAddFirstInput]->GetTensorDesc()),
                                       BCast::TransShapeToDimVec(input[kAddSecondInput]->GetTensorDesc()));
  if (ret != SUCCESS) {
    GELOGE(ret, "Greater broadcasting failed.");
    return ret;
  }

  std::vector<int64_t> x_indexes;
  std::vector<int64_t> y_indexes;
  bcast.BCastIndexes(x_indexes, y_indexes);

  auto x1_data = reinterpret_cast<const InT *>(input[kAddFirstInput]->GetData().data());
  auto x2_data = reinterpret_cast<const InT *>(input[kAddSecondInput]->GetData().data());

  size_t data_num = x_indexes.size();
  std::unique_ptr<InT[]> buf(new (std::nothrow) InT[data_num]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "New sizeof(T) * data_num(%zu) memory failed", static_cast<size_t>(sizeof(InT) * data_num));
    return MEMALLOC_FAILED;
  }

  DataType data_type = input[kAddFirstInput]->GetTensorDesc().GetDataType();
  for (size_t i = 0; i < data_num; i++) {
    auto x_index = *(x1_data + x_indexes[i]);
    auto y_index = *(x2_data + y_indexes[i]);
    if (OverflowCheck<InT>(x_index, y_index, data_type) != SUCCESS) {
      GELOGE(PARAM_INVALID, "Result of add is overflow.");
      return PARAM_INVALID;
    }
    *(buf.get() + i) = x_index + y_index;
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kAddFirstOutput));
  if (output_ptr == nullptr) {
    GELOGE(MEMALLOC_FAILED, "Make shared failed");
    return MEMALLOC_FAILED;
  }
  output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_num * sizeof(InT));
  output_ptr->MutableTensorDesc().SetDataType(data_type);
  vector<int64_t> bcast_dims = bcast.GetOutputShape();
  output_ptr->MutableTensorDesc().SetShape(GeShape(bcast_dims));
  v_output.push_back(output_ptr);

  return SUCCESS;
}

Status AddKernel::AddCheck(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGW("Op_desc_ptr must not be null.");
    return PARAM_INVALID;
  }
  // check how many inputs
  if ((input.size() != kAddInputSize) || (op_desc_ptr->GetOutputsSize() != kAddOutputSize)) {
    GELOGW("The number of input for add must be %zu, output number must be %zu.", kAddInputSize,
           kAddOutputSize);
    return PARAM_INVALID;
  }
  // input vector elements must not be null
  if ((input[kAddFirstInput] == nullptr) || (input[kAddSecondInput] == nullptr)) {
    GELOGW("Input vector elements must not be null.");
    return PARAM_INVALID;
  }
  // Inputs must have the same datatype.
  DataType data_type_0 = input[kAddFirstInput]->GetTensorDesc().GetDataType();
  DataType data_type_1 = input[kAddSecondInput]->GetTensorDesc().GetDataType();
  if (data_type_0 != data_type_1) {
    GELOGW("Data type of inputs for add not matched, data_type_0:%s, data_type_1:%s",
           TypeUtils::DataTypeToSerialString(data_type_0).c_str(),
           TypeUtils::DataTypeToSerialString(data_type_1).c_str());
    return PARAM_INVALID;
  }
  // Checking whether the weightdef contains data
  if ((input[kAddFirstInput]->GetData().size() == 0) || (input[kAddSecondInput]->GetData().size() == 0)) {
    GELOGW("Data size of input0 is %zu, input1 is %zu.", input[kAddFirstInput]->GetData().size(),
           input[kAddSecondInput]->GetData().size());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

Status AddKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                          std::vector<GeTensorPtr> &v_output) {
  if (AddCheck(op_desc_ptr, input) != SUCCESS) {
    return NOT_CHANGED;
  }

  Status ret = NOT_CHANGED;
  DataType data_type = input[kAddFirstInput]->GetTensorDesc().GetDataType();
  switch (data_type) {
    SET_BCAST_ADD_CASE(DT_INT8, int8_t)
    SET_BCAST_ADD_CASE(DT_INT16, int16_t)
    SET_BCAST_ADD_CASE(DT_INT32, int32_t)
    SET_BCAST_ADD_CASE(DT_INT64, int64_t)
    SET_BCAST_ADD_CASE(DT_UINT8, uint8_t)
    SET_BCAST_ADD_CASE(DT_UINT16, uint16_t)
    SET_BCAST_ADD_CASE(DT_UINT32, uint32_t)
    SET_BCAST_ADD_CASE(DT_UINT64, uint64_t)
    SET_BCAST_ADD_CASE(DT_FLOAT16, fp16_t)
    SET_BCAST_ADD_CASE(DT_FLOAT, float)
    SET_BCAST_ADD_CASE(DT_DOUBLE, double)
    default:
      GELOGI("Add kernel data type %s not support.", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }

  if (ret != SUCCESS) {
    GELOGW("Greater broadcasting failed.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

REGISTER_KERNEL(ADD, AddKernel);
}  // namespace ge
