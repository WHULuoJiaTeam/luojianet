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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_TOPK_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_TOPK_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/topk_fp32.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/topk_fp16.h"
#endif

namespace mindspore::kernel {
typedef void (*TopKFunc)(void *input_data, void *output_data, int32_t *output_index, TopkParameter *parameter);
class TopKCPUKernel : public InnerKernel {
 public:
  explicit TopKCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    topk_param_ = reinterpret_cast<TopkParameter *>(op_parameter_);
    switch (inputs.front()->data_type()) {
      case kNumberTypeFloat:
      case kNumberTypeFloat32:
        topk_func_ = Topk;
        break;
      case kNumberTypeInt:
      case kNumberTypeInt32:
        topk_func_ = TopkInt;
        break;
#ifdef ENABLE_FP16
      case kNumberTypeFloat16:
        topk_func_ = TopkFp16;
        break;
#endif
      default:
        MS_LOG(ERROR) << "Unsupported input data type: " << inputs.front()->data_type();
        topk_func_ = nullptr;
        break;
    }
  }
  ~TopKCPUKernel() override {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;

 private:
  TopkParameter *topk_param_;
  TopKFunc topk_func_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_TOPK_H_
