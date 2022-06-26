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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ADDN_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ADDN_FP16_H_

#include <vector>
#include "src/inner_kernel.h"
#include "schema/model_generated.h"

namespace mindspore::kernel {
class AddNFp16CPUKernel : public InnerKernel {
 public:
  AddNFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {}
  ~AddNFp16CPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int AddNParallelRun(int thread_id, float lhs_scale, float rhs_scale);

 private:
  float16_t *in1_addr_;
  float16_t *in2_addr_;
  float16_t *out_addr_;
  int elements_num_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_ADDN_FP16_H_
