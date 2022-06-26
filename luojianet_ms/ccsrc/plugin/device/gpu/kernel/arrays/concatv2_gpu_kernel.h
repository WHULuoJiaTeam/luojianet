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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/concatv2_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class ConcatV2FwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  ConcatV2FwdGpuKernelMod()
      : axis_(0),
        input_num_(1),
        output_size_(0),
        all_size_before_axis_(1),
        all_size_axis_(1),
        kernel_name_("ConcatV2"),
        inputs_host_(nullptr),
        len_axis_(nullptr) {}
  ~ConcatV2FwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (input_num_ == 0) {
      return true;
    }

    T *output = GetDeviceAddress<T>(outputs, 0);
    T **inputs_device = GetDeviceAddress<T *>(workspace, 0);
    int *len_axis_device = GetDeviceAddress<int>(workspace, 1);
    int current_dim = 0;
    for (size_t i = 0; i < inputs.size(); i++) {
      T *input = GetPossiblyNullDeviceAddress<T>(inputs, i);
      if (input != nullptr) {
        inputs_host_[current_dim] = input;
        current_dim++;
      }
    }
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(inputs_device, inputs_host_.get(), sizeof(T *) * input_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "ConcatV2 opt cudaMemcpyAsync inputs failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemcpyAsync(len_axis_device, len_axis_.get(), sizeof(int) * input_num_,
                                               cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "ConcatV2 opt cudaMemcpyAsync length on axis failed");
    ConcatKernel(output_size_, input_num_, all_size_before_axis_, all_size_axis_, len_axis_device, inputs_device,
                 output, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    int dims = SizeToInt(input_shape.size());
    axis_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis_ < -dims || axis_ >= dims) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'axis' should be in the range [-" << dims << "," << dims
                        << "), but got " << axis_;
    }
    if (axis_ < 0) {
      axis_ += dims;
    }
    auto origin_data_format = AnfAlgo::GetOriginDataFormat(kernel_node);
    auto input_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    axis_ = AxisTransform(origin_data_format, input_format, axis_);

    input_num_ = SizeToInt(common::AnfAlgo::GetInputTensorNum(kernel_node));
    inputs_host_ = std::make_unique<T *[]>(input_num_);
    len_axis_ = std::make_unique<int[]>(input_num_);
    int current_dim = 0;
    for (int i = 0; i < input_num_; i++) {
      size_t input_size = 1;
      auto input_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, i);
      for (size_t j = 0; j < input_shape.size(); j++) {
        input_size *= input_shape[j];
      }

      if (input_size == 0) {
        input_num_--;
      } else {
        input_size_list_.push_back(input_size * sizeof(T));
        len_axis_[current_dim] = SizeToInt(input_shape[axis_]);
        current_dim++;
      }
    }
    workspace_size_list_.push_back(sizeof(T *) * input_num_);
    workspace_size_list_.push_back(sizeof(int) * input_num_);

    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    output_size_ = 1;
    for (int i = 0; i < SizeToInt(output_shape.size()); i++) {
      output_size_ *= output_shape[i];
      if (i > axis_) {
        all_size_before_axis_ *= output_shape[i];
        all_size_axis_ *= output_shape[i];
      }
      if (i == axis_) {
        all_size_before_axis_ *= output_shape[i];
      }
    }
    output_size_list_.push_back(output_size_ * sizeof(T));
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    ResetSizeLists();
    axis_ = 0;
    input_num_ = 1;
    output_size_ = 0;
    all_size_before_axis_ = 1;
    all_size_axis_ = 1;
    kernel_name_ = "ConcatV2";
    inputs_host_ = nullptr;
    len_axis_ = nullptr;
  }

 protected:
  void InitSizeLists() override {}

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    return true;
  }
  int axis_;
  int input_num_;
  size_t output_size_;
  int all_size_before_axis_;
  int all_size_axis_;
  std::string kernel_name_;
  std::unique_ptr<T *[]> inputs_host_;
  std::unique_ptr<int[]> len_axis_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_CONCATV2_GPU_KERNEL_H_
