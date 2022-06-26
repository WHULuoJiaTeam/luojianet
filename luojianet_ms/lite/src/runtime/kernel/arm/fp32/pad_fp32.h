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
#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_PAD_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_PAD_H_

#include <vector>
#include <string>
#include <cmath>
#include <utility>
#include "include/errorcode.h"
#include "nnacl/fp32/pad_fp32.h"
#include "nnacl/errorcode.h"
#include "nnacl/common_func.h"
#include "src/inner_kernel.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"

namespace luojianet_ms::kernel {
class PadCPUKernel : public InnerKernel {
 public:
  PadCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
               const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : InnerKernel(parameter, inputs, outputs, ctx) {
    pad_param_ = reinterpret_cast<PadParameter *>(parameter);
  }

  ~PadCPUKernel() {}

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  virtual int RunImpl(int task_id) const;
  virtual int RunMirrorPadImpl(int task_id) const;

 private:
  int CheckPaddings(const int *paddings, int length, const int *input_shape, int mode);
  void CalculateStrides();
  int ExtendShape(int *shape, int length, const int *ori_shape, int rank) const;
  int ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) const;
  void InitMirrorPadBlock();
  void RunMirrorPadImplFast(const MirrorPadBlock &block, const float *input_data, float *output_data) const;

 protected:
  int HandleMirrorPad();
  int CopyPaddingFromInput();
  PadParameter *pad_param_ = nullptr;
  int in_[DEFAULT_PAD_NDIMS] = {0};
  int out_[DEFAULT_PAD_NDIMS] = {0};
  std::vector<MirrorPadBlock> mirror_pad_block_;
};

int PadImpl(void *cdata, int task_id, float, float);
int MirrorPadImpl(void *cdata, int task_id, float, float);
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_PAD_H_
