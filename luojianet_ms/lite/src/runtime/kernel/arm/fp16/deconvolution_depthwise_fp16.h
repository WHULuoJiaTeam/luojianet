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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_DEPTHWISE_FP16_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_DEPTHWISE_FP16_H_

#include <vector>
#include "src/inner_kernel.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/fp16/conv_depthwise_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif
void DeconvDwC8Fp16(float16_t *output_data, const float16_t *input_data, const float16_t *weight_data,
                    const float16_t *bias_data, const ConvParameter *conv_param, const SlidingWindowParam *sliding,
                    int task_id);
void ComputeStrides(int *shape, int *strides, int ndim);
#ifdef __cplusplus
}
#endif

namespace luojianet_ms::kernel {
class DeconvolutionDepthwiseFp16CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  DeconvolutionDepthwiseFp16CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, inputs.at(kWeightIndex)->data(),
                                 inputs.size() == kInputSize2 ? inputs.at(kBiasIndex)->data() : nullptr) {}
  ~DeconvolutionDepthwiseFp16CPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;

  int InitPackedInputOutput();
  int InitSlideParam();
  int Execute(int task_id);

 private:
  int MallocWeightBiasData() override;
  void PackWeight() override;
  void FreePackedInputOutput();
  SlidingWindowParam *sliding_ = nullptr;
  float16_t *packed_input_ = nullptr;
  float16_t *packed_output_ = nullptr;
  bool need_align_ = false;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP16_DECONVOLUTION_DEPTHWISE_FP16_H_
