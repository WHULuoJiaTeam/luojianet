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

#include "host_kernels/unpack_kernel.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kUnpackInputNum = 1;
}  // namespace
template <typename T>
Status CalcUpack(const int32_t num, const DataType data_type, const T *value, std::vector<GeTensorPtr> &v_output) {
  GE_CHECK_NOTNULL(value);
  // not support num=0
  if (num > 0) {
    unique_ptr<T[]> buf(new (std::nothrow) T[num]());
    GE_CHECK_NOTNULL(buf);
    for (int32_t i = 0; i < num; ++i) {
      GeTensorPtr output_ptr = ge::MakeShared<ge::GeTensor>();
      GE_CHECK_NOTNULL(output_ptr);

      buf[i] = *value;
      ++value;
      GE_CHK_STATUS_RET(output_ptr->SetData(reinterpret_cast<uint8_t *>(&buf[i]), sizeof(T)),
                        "unpack set data failed!");
      output_ptr->MutableTensorDesc().SetDataType(data_type);
      v_output.push_back(output_ptr);
    }
  } else {
    GELOGW("num <= 0 is not support.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status UnpackKernel::Compute(const OpDescPtr attr, const std::vector<ge::ConstGeTensorPtr> &input,
                             std::vector<ge::GeTensorPtr> &v_output) {
  GE_CHECK_NOTNULL(attr);
  // check input num
  GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(input.size() == kUnpackInputNum,
                                        "The number of input for unpack must be %zu, real is %zu.", kUnpackInputNum,
                                        input.size());

  ConstGeTensorPtr dims = input[0];
  GE_CHECK_NOTNULL(dims);

  if (dims->GetTensorDesc().GetShape().GetDimNum() != 1) {
    GELOGW("input tensor not 1 dim");
    return NOT_CHANGED;
  }

  ge::DataType data_type;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetDataType(attr, ATTR_NAME_T, data_type), PARAM_INVALID, "get T attr failed.");
  // data_type must be FLOAT or INT32
  GE_CHK_BOOL_RET_STATUS((data_type == DT_FLOAT || data_type == DT_INT32), PARAM_INVALID, "T must be float or int32.");

  int64_t num = 0;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetInt(attr, UNPACK_ATTR_NAME_NUM, num), PARAM_INVALID, "get num attr failed.");
  size_t data_count = dims->GetData().size() / sizeof(float);
  // num must equal to input_data size
  GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(data_count == static_cast<size_t>(num),
                                        "input tensor size not equal num, data_count:%zu, num:%ld.", data_count, num);
  // calculate result
  if (data_type == DT_FLOAT) {
    GE_RETURN_IF_ERROR(CalcUpack(num, data_type, reinterpret_cast<const float *>(dims->GetData().data()), v_output));
  } else {
    GE_RETURN_IF_ERROR(CalcUpack(num, data_type, reinterpret_cast<const int32_t *>(dims->GetData().data()), v_output));
  }

  return SUCCESS;
}

REGISTER_KERNEL(UNPACK, UnpackKernel);
}  // namespace ge

