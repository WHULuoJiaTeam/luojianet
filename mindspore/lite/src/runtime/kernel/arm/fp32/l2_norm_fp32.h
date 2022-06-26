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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_L2_NORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_L2_NORM_H_

#include <vector>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "nnacl/l2_norm_parameter.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class L2NormCPUKernel : public InnerKernel {
 public:
  L2NormCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                  const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    l2_norm_param_ = reinterpret_cast<L2NormParameter *>(op_parameter_);
  }
  ~L2NormCPUKernel() { FreeTmpBuffer(); }

  int CalcSquareSum(int task_id) const;
  int DivSqrtSum(int task_id) const;
  int CalcL2NormTrailingAxis(int task_id) const;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 protected:
  L2NormParameter *l2_norm_param_ = nullptr;

 private:
  int MallocTmpBuffer();
  void FreeTmpBuffer();
  float sqrt_sum_ = 0;
  float *input_ptr_ = nullptr;
  float *output_ptr_ = nullptr;
  float *tmp_sum_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_L2_NORM_H_
