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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_

#include <vector>
#include "src/runtime/kernel/arm/fp32/fused_batchnorm_fp32.h"

namespace mindspore::kernel {
class FusedBatchnormFp16CPUKernel : public FusedBatchnormCPUKernel {
 public:
  FusedBatchnormFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : FusedBatchnormCPUKernel(parameter, inputs, outputs, ctx) {}
  virtual ~FusedBatchnormFp16CPUKernel() {}

  int DoExecute(int task_id) override;
  int Eval() override;
  int Batchnorm2Scale(const void *scale_data, const void *bias_data, const void *mean_data, const void *var_data,
                      float eps, int kernel_num) override;

 protected:
  void CalcMeanVar(float16_t *in, float16_t *scale, float16_t *offset, float16_t *save_mean, float16_t *save_variance);
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_FUSED_BATCHNORM_FP16_H_
