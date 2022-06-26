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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LAYER_NORM_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LAYER_NORM_FP32_H_
#include <vector>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "nnacl/fp32/layer_norm_fp32.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class LayerNormCPUKernel : public InnerKernel {
 public:
  LayerNormCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                     const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<LayerNormParameter *>(parameter);
  }
  ~LayerNormCPUKernel() override{};

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoLayerNorm(int thread_id) const;

 private:
  LayerNormParameter *param_ = nullptr;
  float *src_data_ = nullptr;
  float *dst_data_ = nullptr;
  float *gamma_data_ = nullptr;
  float *beta_data_ = nullptr;
  float *mean_data_ = nullptr;
  float *var_data_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_LAYER_NORM_FP32_H_
