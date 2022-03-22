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

#ifndef INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_
#define INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_

#include "external/graph/ge_error_codes.h"
#include "external/graph/operator.h"

namespace ge {
using InferDataSliceFunc = std::function<graphStatus(Operator &)>;

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferDataSliceFuncRegister {
 public:
  InferDataSliceFuncRegister(const char *operator_type, const InferDataSliceFunc &infer_data_slice_func);
  ~InferDataSliceFuncRegister() = default;
};

// infer data slice func register
#define IMPLEMT_COMMON_INFER_DATA_SLICE(func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(Operator &op)

#define IMPLEMT_INFER_DATA_SLICE(op_name, func_name) \
  GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY static graphStatus func_name(op::op_name &op)

#define INFER_DATA_SLICE_FUNC(op_name, x) [&](Operator &v) { return x((op::op_name &)v); }

#define __INFER_DATA_SLICE_FUNC_REG_IMPL__(op_name, x, n) \
  static const InferDataSliceFuncRegister PASTE(ids_register, n)(#op_name, x)

#define INFER_DATA_SLICE_FUNC_REG(op_name, x) \
  __INFER_DATA_SLICE_FUNC_REG_IMPL__(op_name, INFER_DATA_SLICE_FUNC(op_name, x), __COUNTER__)
}  // namespace ge

#endif  // INC_REGISTER_INFER_DATA_SLICE_REGISTRY_H_
