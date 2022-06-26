/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/quant/fake_learned_scale_quant_perchannel_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fake_learned_scale_quant_perchannel_impl.cuh"
#include <thrust/extrema.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

namespace mindspore {
namespace kernel {
FakeLearnedScaleQuantPerChannelGpuKernelMod::FakeLearnedScaleQuantPerChannelGpuKernelMod()
    : input_size_(0),
      quant_num_(1),
      global_step_(0),
      quant_delay_(0),
      training_(false),
      neg_trunc_(false),
      num_channels_(0) {}

bool FakeLearnedScaleQuantPerChannelGpuKernelMod::Init(const CNodePtr &kernel_node) {
  auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  kernel_node_ = kernel_node;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num != 3) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 3, but got " << input_num;
  }

  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
  }

  quant_delay_ =
    static_cast<int>(GetValue<int64_t>(common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("quant_delay")));
  training_ = GetValue<bool>(common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("training"));
  neg_trunc_ = GetValue<bool>(common::AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("neg_trunc"));

  // init size
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  num_channels_ = SizeToInt(input_shape[0]);
  for (size_t i = 0; i < input_shape.size(); ++i) {
    quant_num_ *= SizeToInt(input_shape[i]);
  }
  input_size_ = sizeof(float) * quant_num_;
  InitSizeLists();
  return true;
}

void FakeLearnedScaleQuantPerChannelGpuKernelMod::InitSizeLists() {
  input_size_list_.push_back(input_size_);                    // x
  input_size_list_.push_back(sizeof(float) * num_channels_);  // alpha
  input_size_list_.push_back(sizeof(float));                  // quant_max
  output_size_list_.push_back(input_size_);                   // y
  workspace_size_list_.push_back(input_size_);                // input_div_alpha
  workspace_size_list_.push_back(input_size_);                // input_quant
}

bool FakeLearnedScaleQuantPerChannelGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &workspace,
                                                         const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  float *input = GetDeviceAddress<float>(inputs, 0);
  float *input_alpha = GetDeviceAddress<float>(inputs, 1);
  float *input_quant_max = GetDeviceAddress<float>(inputs, 2);
  float *output = GetDeviceAddress<float>(outputs, 0);
  float *input_div_alpha = GetDeviceAddress<float>(workspace, 0);
  float *input_quant = GetDeviceAddress<float>(workspace, 1);

  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(input_alpha);
  MS_EXCEPTION_IF_NULL(input_quant_max);
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(input_div_alpha);
  MS_EXCEPTION_IF_NULL(input_quant);

  if (training_) {
    // control flow for quant_delay
    if (global_step_ >= quant_delay_) {
      // real launch
      CalLSQNudgePerChannel(input, quant_num_, input_alpha, input_quant_max, input_div_alpha, input_quant, neg_trunc_,
                            num_channels_, reinterpret_cast<cudaStream_t>(stream_ptr));
      CalFakeLearnedScaleQuantPerChannel(output, quant_num_, input_alpha, input_quant, num_channels_,
                                         reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDA_RET_WITH_ERROR(kernel_node_,
                                cudaMemcpyAsync(output, input, input_size_, cudaMemcpyDeviceToDevice,
                                                reinterpret_cast<cudaStream_t>(stream_ptr)),
                                "Copy gpu memory failed");
    }
    global_step_++;
  } else {
    // real launch
    CalLSQNudgePerChannel(input, quant_num_, input_alpha, input_quant_max, input_div_alpha, input_quant, neg_trunc_,
                          num_channels_, reinterpret_cast<cudaStream_t>(stream_ptr));
    CalFakeLearnedScaleQuantPerChannel(output, quant_num_, input_alpha, input_quant, num_channels_,
                                       reinterpret_cast<cudaStream_t>(stream_ptr));
  }

  return true;
}

MS_REG_GPU_KERNEL(FakeLearnedScaleQuantPerChannel, FakeLearnedScaleQuantPerChannelGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore
