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
#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ROI_POOLING_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_ROI_POOLING_H_

#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/fp32/roi_pooling_fp32.h"

namespace luojianet_ms::kernel {
class ROIPoolingCPUKernel : public InnerKernel {
 public:
  ROIPoolingCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                      const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<ROIPoolingParameter *>(parameter);
  }
  ~ROIPoolingCPUKernel() override {
    if (max_c_ != nullptr) {
      free(max_c_);
      max_c_ = nullptr;
    }
  };

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int DoExecute(int task_id);

 private:
  float *in_ptr_ = nullptr;
  float *out_ptr_ = nullptr;
  float *roi_ptr_ = nullptr;
  float *max_c_ = nullptr;
  ROIPoolingParameter *param_ = nullptr;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_REVERSE_H_
