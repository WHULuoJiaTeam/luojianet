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

#ifndef INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
#define INC_GRAPH_UTILS_TENSOR_ADAPTER_H_

#include <memory>
#include "graph/ge_tensor.h"
#include "graph/tensor.h"
#include "graph/ge_attr_value.h"

namespace ge {
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TensorAdapter {
 public:
  static GeTensorDesc TensorDesc2GeTensorDesc(const TensorDesc &tensor_desc);
  static TensorDesc GeTensorDesc2TensorDesc(const GeTensorDesc &ge_tensor_desc);
  static Tensor GeTensor2Tensor(const ConstGeTensorPtr &ge_tensor);

  static ConstGeTensorPtr AsGeTensorPtr(const Tensor &tensor);  // Share value
  static GeTensorPtr AsGeTensorPtr(Tensor &tensor);             // Share value
  static const GeTensor AsGeTensor(const Tensor &tensor);       // Share value
  static GeTensor AsGeTensor(Tensor &tensor);                   // Share value
  static const Tensor AsTensor(const GeTensor &ge_tensor);         // Share value
  static Tensor AsTensor(GeTensor &ge_tensor);                     // Share value
  static GeTensor AsGeTensorShared(const Tensor &tensor);
  static GeTensor NormalizeGeTensor(const GeTensor &tensor);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TENSOR_ADAPTER_H_
