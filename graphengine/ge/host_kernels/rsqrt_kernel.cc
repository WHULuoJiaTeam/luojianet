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
#include "host_kernels/rsqrt_kernel.h"

#include <cfloat>

#include <memory>

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "inc/kernel_factory.h"
#include "common/math/math_util.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const size_t kRsqrtInputSize = 1;
const size_t kRsqrtInputIndex0 = 0;

template <typename T>
Status ZeroCheck(T x, const DataType &data_type) {
  switch (data_type) {
    case DT_FLOAT16:
      FMK_FP16_ZEROCHECK(static_cast<double>(x))
      break;
    case DT_FLOAT:
      FMK_FLOAT_ZEROCHECK(static_cast<float>(x))
      break;
    case DT_DOUBLE:
      FMK_DOUBLE_ZEROCHECK(static_cast<double>(x))
      break;
    default:
      break;
  }
  return SUCCESS;
}
#define SET_RSQRT_CASE(DTYPE, TYPE)                                 \
  case (DTYPE):                                                     \
    ret = RsqrtKernel::RsqrtCompute<TYPE>(input_ptr, output_ptr);   \
    break;
}  // namespace

template<typename T>
Status RsqrtKernel::RsqrtCompute(ConstGeTensorPtr &input_tensor_ptr, GeTensorPtr &output_tensor_ptr) {
  GE_CHECK_NOTNULL(input_tensor_ptr);
  GE_CHECK_NOTNULL(output_tensor_ptr);
  size_t data_size = input_tensor_ptr->GetData().size();
  size_t data_count = data_size / sizeof(T);
  auto data_type = input_tensor_ptr->GetTensorDesc().GetDataType();
  if (data_count > 0) {
    unique_ptr<T[]> buf(new(std::nothrow) T[data_count]());
    if (buf == nullptr) {
      GELOGW("New buf failed");
      return NOT_CHANGED;
    }
    auto ptr = const_cast<T * >(reinterpret_cast<const T *>(input_tensor_ptr->GetData().data()));
    for (size_t i = 0; i < data_count; i++) {
      if (ZeroCheck(*(ptr + i), data_type) != SUCCESS) {
        GELOGW("Rsqrt: The input data can not less than or equal to zero, rsqrt folding failed.");
        return NOT_CHANGED;
      }
      switch (data_type) {
        case DT_FLOAT16: {
          double val = static_cast<double>(*(reinterpret_cast<const fp16_t*>(input_tensor_ptr->GetData().data()) + i));
          double drSqrt = 1.0 / std::sqrt(val);
          buf[i] = drSqrt;
          break;
        }
        case DT_FLOAT:{
          float denominator = std::sqrt(*(reinterpret_cast<const float*>(input_tensor_ptr->GetData().data()) + i));
          buf[i] = static_cast<float >(1 / denominator);
          break;
        }
        case DT_DOUBLE: {
          double denominator = std::sqrt(*(reinterpret_cast<const double*>(input_tensor_ptr->GetData().data()) + i));
          buf[i] = static_cast<double>(1 / denominator);
          break;
        }
        default:
          GELOGW("Input data type must be FP16, FP32 and DOUBLE.");
          return NOT_CHANGED;
      }
    }
    GE_IF_BOOL_EXEC(output_tensor_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_size) != GRAPH_SUCCESS,
                    GELOGW("Set data failed");  return NOT_CHANGED);
    output_tensor_ptr->MutableTensorDesc().SetDataType(data_type);
    output_tensor_ptr->MutableTensorDesc().SetShape(input_tensor_ptr->GetTensorDesc().GetShape());
  }
  return SUCCESS;
}

Status RsqrtKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                            std::vector<GeTensorPtr> &v_output) {
  GELOGI("RsqrtKernel in.");
  GE_CHECK_NOTNULL(op_desc_ptr);

  // check input size
  if (input.size() != kRsqrtInputSize) {
    GELOGW("The number of input for rsqrt must be %zu.", kRsqrtInputSize);
    return NOT_CHANGED;
  }

  ConstGeTensorPtr input_ptr = input.at(kRsqrtInputIndex0);
  GE_CHECK_NOTNULL(input_ptr);

  // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("MakeShared GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }
  Status ret = NOT_CHANGED;
  auto dtype = input_ptr->GetTensorDesc().GetDataType();
  switch (dtype) {
    SET_RSQRT_CASE(DT_FLOAT16, fp16_t)
    SET_RSQRT_CASE(DT_FLOAT, float)
    SET_RSQRT_CASE(DT_DOUBLE, double)
    default:
      GELOGW("Input data type must be FP16, FP32 and DOUBLE.");
      return NOT_CHANGED;
  }
  if (ret != SUCCESS) {
    GELOGW("Rsqrt folding failed.");
    return NOT_CHANGED;
  }
  v_output.push_back(output_ptr);
  GELOGI("RsqrtKernel success.");
  return SUCCESS;
}

REGISTER_KERNEL(RSQRT, RsqrtKernel);
}  // namespace ge
