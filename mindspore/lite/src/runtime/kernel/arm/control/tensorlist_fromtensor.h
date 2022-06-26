/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONTROL_TENSORLISTFROMTENSOR_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONTROL_TENSORLISTFROMTENSOR_H_

#include <vector>
#include "src/inner_kernel.h"
#include "src/tensorlist.h"
#include "schema/model_generated.h"
#include "nnacl/tensorlist_parameter.h"

namespace mindspore::kernel {
class TensorListFromTensorCPUKernel : public InnerKernel {
 public:
  TensorListFromTensorCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx),
        dtype_(static_cast<TypeId>(reinterpret_cast<TensorListParameter *>(parameter)->element_dtype_)) {}
  ~TensorListFromTensorCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int IsCompatibleShape();

 private:
  std::vector<int> output_shape_;
  lite::Tensor *output0_ = nullptr;
  lite::Tensor *input0_ = nullptr;
  lite::Tensor *input1_ = nullptr;
  TypeId dtype_ = kTypeUnknown;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_CONTROL_TENSORLISTFROMTENSOR_H_
