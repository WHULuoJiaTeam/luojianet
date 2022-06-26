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

#include "plugin/device/cpu/kernel/mkldnn/batch_norm_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"

namespace luojianet_ms {
namespace kernel {
namespace {
constexpr size_t kBatchNormGradInputsNum = 6;
constexpr size_t kBatchNormGradOutputsNum = 3;
constexpr size_t kBatchNormGradInputShapeSize = 4;
constexpr size_t kBatchNormGradInputShapeSize2 = 2;
constexpr size_t kScaleShiftNum = 2;
}  // namespace

void BatchNormGradCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  size_t type_size = sizeof(float);
  std::vector<size_t> shape = AnfAlgo::GetInputDeviceShape(kernel_node, Y_BACKPROP);
  size_t tensor_size = shape[C] * kScaleShiftNum * type_size;
  input_size_list_.pop_back();
  // [2, c] to store scale and bias
  (void)workspace_size_list_.emplace_back(tensor_size);
  // [2, c] to store diff_scale and diff_bias
  (void)workspace_size_list_.emplace_back(tensor_size);
}

void BatchNormGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> x_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (x_shape.size() == NC_LEN) {
    (void)x_shape.insert(x_shape.end(), (SHAPE_4D - NC_LEN), 1);
  } else if (x_shape.size() != SHAPE_4D) {
    MS_LOG(EXCEPTION) << "Fused batchnorm support nc or nchw input!";
  }
  batch_size = x_shape[N];
  channel = x_shape[C];
  hw_size = x_shape[H] * x_shape[W];
  nhw_size = batch_size * hw_size;
  dnnl::memory::desc x_desc = GetDefaultMemDesc(x_shape);
  dnnl::memory::desc scale_bias_desc = GetDefaultMemDesc({kScaleShiftNum, channel});
  auto epsilon = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "epsilon");
  auto is_train = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "is_training");
  auto prop_kind = dnnl::prop_kind::forward_inference;
  auto normalization_flags = dnnl::normalization_flags::use_scale_shift | dnnl::normalization_flags::use_global_stats;
  if (is_train) {
    prop_kind = dnnl::prop_kind::forward_training;
    normalization_flags = dnnl::normalization_flags::use_scale_shift;
  }

  // fused Batch Normalization forward description
  auto desc = CreateDesc<dnnl::batch_normalization_forward::desc>(prop_kind, x_desc, epsilon, normalization_flags);
  auto forward_prim_desc = CreateDesc<dnnl::batch_normalization_forward::primitive_desc>(desc, engine_);

  // fused Batch Normalization backward description
  auto backward_desc = CreateDesc<dnnl::batch_normalization_backward::desc>(dnnl::prop_kind::backward, x_desc, x_desc,
                                                                            epsilon, normalization_flags);
  auto backward_prim_desc =
    CreateDesc<dnnl::batch_normalization_backward::primitive_desc>(backward_desc, engine_, forward_prim_desc);
  auto wksp_desc = GetWorkspaceDesc(forward_prim_desc);
  auto mean = GetMeanDesc(forward_prim_desc);
  auto variance = GetVarianceDesc(forward_prim_desc);
  primitive_ = CreatePrimitive<dnnl::batch_normalization_backward>(backward_prim_desc);
  AddArgument(DNNL_ARG_SRC, x_desc);
  AddArgument(DNNL_ARG_MEAN, mean);
  AddArgument(DNNL_ARG_VARIANCE, variance);
  AddArgument(DNNL_ARG_SCALE_SHIFT, scale_bias_desc);
  AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
  AddArgument(DNNL_ARG_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_DST, x_desc);
  AddArgument(DNNL_ARG_DIFF_SRC, x_desc);
  AddArgument(DNNL_ARG_DIFF_SCALE_SHIFT, scale_bias_desc);
}

bool BatchNormGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kBatchNormGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kBatchNormGradOutputsNum, kernel_name_);

  auto wksp_in = reinterpret_cast<float *>(workspace[SCALE_BIAS]->addr);
  auto scale_ret = memcpy_s(wksp_in, workspace[SCALE_BIAS]->size, inputs[SCALE]->addr, inputs[SCALE]->size);
  if (scale_ret != 0) {
    MS_LOG(EXCEPTION) << "Scale memcpy error!";
  }
  auto max_size = workspace[SCALE_BIAS]->size - inputs[SCALE]->size;
  auto bias_ret = memset_s(wksp_in + (inputs[SCALE]->size / sizeof(float)), max_size, 0, max_size);
  if (bias_ret != 0) {
    MS_LOG(EXCEPTION) << "Bias memset 0 error.";
  }

  SetArgumentHandle(DNNL_ARG_DIFF_DST, inputs[Y_BACKPROP]->addr);
  SetArgumentHandle(DNNL_ARG_SRC, inputs[X]->addr);
  SetArgumentHandle(DNNL_ARG_MEAN, inputs[SAVE_MEAN]->addr);
  SetArgumentHandle(DNNL_ARG_VARIANCE, inputs[SAVE_VARIANCE]->addr);
  SetArgumentHandle(DNNL_ARG_SCALE_SHIFT, workspace[SCALE_BIAS]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC, outputs[DX]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SCALE_SHIFT, workspace[DIFF_SCALE_BIAS]->addr);
  ExecutePrimitive();

  auto wksp_out = reinterpret_cast<float *>(workspace[DIFF_SCALE_BIAS]->addr);
  auto diff_scale_ret = memcpy_s(outputs[DSCALE]->addr, outputs[DSCALE]->size, wksp_out, inputs[SCALE]->size);
  if (diff_scale_ret != 0) {
    MS_LOG(EXCEPTION) << "Diff_scale memcpy to output[1] error.";
  }
  auto diff_bias_ret = memcpy_s(outputs[DBIAS]->addr, outputs[DBIAS]->size,
                                wksp_out + (outputs[DSCALE]->size / sizeof(float)), outputs[DBIAS]->size);
  if (diff_bias_ret != 0) {
    MS_LOG(EXCEPTION) << "Diff_bias memcpy to  to output[2] error.";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BatchNormGrad, BatchNormGradCpuKernelMod);
}  // namespace kernel
}  // namespace luojianet_ms
