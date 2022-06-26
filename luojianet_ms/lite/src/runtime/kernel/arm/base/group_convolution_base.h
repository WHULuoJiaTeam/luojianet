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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_GROUP_CONVOLUTION_BASE_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_GROUP_CONVOLUTION_BASE_H_

#include <utility>
#include <vector>
#include "src/inner_kernel.h"
#include "nnacl/op_base.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "src/runtime/kernel/arm/base/group_convolution_creator.h"

namespace luojianet_ms::kernel {
class GroupConvolutionBaseCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  GroupConvolutionBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                GroupConvCreator *group_conv_creator, const int group_num)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, nullptr, nullptr),
        group_conv_creator_(group_conv_creator),
        group_num_(group_num) {}  // opParameter(in channel, out channel) in this kernel has been split to groups, if
                                  // you want to get real params, multiply in channel / out channel with group num
  ~GroupConvolutionBaseCPUKernel() override;

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int PreProcess() override;
  virtual int SeparateInput(int group_id) = 0;
  virtual int PostConcat(int group_id) = 0;
  void FreeSubKernel();

 protected:
  int InitGroupParam();
  GroupConvCreator *group_conv_creator_ = nullptr;
  std::vector<kernel::InnerKernel *> group_convs_;
  const int group_num_;
  void *ori_in_data_ = nullptr;   // do not free
  void *ori_out_data_ = nullptr;  // do not free

  int in_plane_ = 0;
  int sub_in_channel_ = 0;
  int ori_in_channel_ = 0;
  int in_thread_num_ = 0;
  int out_plane_ = 0;
  int sub_out_channel_ = 0;
  int ori_out_channel_ = 0;
  int out_thread_num_ = 0;
};
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_KERNEL_ARM_GROUP_CONVOLUTION_BASE_H_
