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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_INSTANCE_NORM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_INSTANCE_NORM_H_
#include <vector>
#include "src/inner_kernel.h"
#include "include/context.h"
#include "nnacl/instance_norm_parameter.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class InstanceNormFp16CPUKernel : public InnerKernel {
 public:
  InstanceNormFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                            const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<InstanceNormParameter *>(parameter);
  }
  ~InstanceNormFp16CPUKernel() override { FreeTmpBuffer(); };

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoInstanceNorm(int task_id);
  void FreeTmpSrcData() {
    if (tmp_src_data_ != nullptr) {
      ms_context_->allocator->Free(tmp_src_data_);
      tmp_src_data_ = nullptr;
    }
  }

 private:
  void FreeTmpBuffer();
  InstanceNormParameter *param_ = nullptr;
  float16_t *src_data_ = nullptr;
  float16_t *dst_data_ = nullptr;
  float16_t *tmp_src_data_ = nullptr;
  float16_t *gamma_data_ = nullptr;
  float16_t *beta_data_ = nullptr;
  bool input_pack_to_nc8hw8_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_INSTANCE_NORM_H_
