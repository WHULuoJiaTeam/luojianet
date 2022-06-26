/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BIASADD_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BIASADD_FP16_H_
#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp16/arithmetic_fp16.h"

namespace mindspore::kernel {
class BiasAddCPUFp16Kernel : public InnerKernel {
 public:
  BiasAddCPUFp16Kernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    bias_param_ = reinterpret_cast<ArithmeticParameter *>(parameter);
  }
  ~BiasAddCPUFp16Kernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int Eval() override;

 private:
  int GetBiasData();
  void PackWeight();
  bool IsRepack() { return is_repack_; }
  ArithmeticParameter *bias_param_ = nullptr;
  float16_t *bias_data_ = nullptr;
  lite::Tensor *bias_tensor_ = nullptr;
  TypeId bias_data_type_ = kNumberTypeFloat16;
  bool is_repack_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_BIASADD_FP16_H_
