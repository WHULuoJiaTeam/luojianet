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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename S>
class DynamicReshapeKernelMod : public NativeGpuKernelMod {
 public:
  DynamicReshapeKernelMod() : data_type_size_(0), shape_size_(0) { ResetResource(); }
  ~DynamicReshapeKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    auto data_addr = GetDeviceAddress<unsigned char>(inputs, 0);
    auto shape_addr = GetDeviceAddress<S>(inputs, 1);
    auto output_addr = GetDeviceAddress<unsigned char>(outputs, 0);

    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_, cudaMemcpyAsync(output_addr, data_addr, input_size_list_[0], cudaMemcpyDeviceToDevice, cuda_stream),
      "DynamicReshape cpy data failed");
    real_output_shape_ = std::vector<S>(input_size_list_[1] / sizeof(S), 0);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(&real_output_shape_[0], shape_addr, input_size_list_[1], cudaMemcpyDeviceToHost, cuda_stream),
      "DynamicReshape cpy real output shape value failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto output_shape = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    auto input_x_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto input_shape_shape = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    data_type_size_ = luojianet_ms::kernel::GetDtypeNbyte(TypeIdToString(data_type, true));
    shape_size_ = input_shape_shape.size();
    size_t input_x_size =
      std::accumulate(input_x_shape.begin(), input_x_shape.end(), data_type_size_, std::multiplies<size_t>());
    input_size_list_.push_back(input_x_size);
    size_t input_shape_size =
      std::accumulate(input_shape_shape.begin(), input_shape_shape.end(), sizeof(S), std::multiplies<size_t>());
    input_size_list_.push_back(input_shape_size);
    size_t output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), data_type_size_, std::multiplies<size_t>());
    output_size_list_.push_back(output_size);
    is_need_updateop_ = true;
    return true;
  }
  void ResetResource() noexcept override {
    real_output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }
  void UpdateOp() override {
    auto data_type = AnfAlgo::GetInputDeviceDataType(kernel_node_.lock(), 0);
    std::vector<size_t> output_shape;
    std::transform(real_output_shape_.begin(), real_output_shape_.end(), std::back_inserter(output_shape),
                   [](const S &value) { return static_cast<size_t>(value); });
    common::AnfAlgo::SetOutputInferTypeAndShape({data_type}, {output_shape}, kernel_node_.lock().get());
    MS_LOG(DEBUG) << "Run PostExecute for DynamicReshape, real output shape is " << output_shape;
  }

 protected:
  void InitSizeLists() override { return; }

 private:
  size_t data_type_size_;
  size_t shape_size_;
  std::vector<S> real_output_shape_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_OTHER_DYNAMIC_RESHAPE_GPU_KERNEL_H_
