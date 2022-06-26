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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ACTIVATION_GRAD_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ACTIVATION_GRAD_KERNEL_H_

#include <vector>
#include <map>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace luojianet_ms {
namespace kernel {
constexpr float ReLU6_UP_TURNING_POINT = 5.999999;
template <typename T>
class ActivationGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ActivationGradGpuKernelMod() { ResetResource(); }
  ~ActivationGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *dy = nullptr;
    T *y = nullptr;
    if (mode_ == CUDNN_ACTIVATION_ELU || mode_ == CUDNN_ACTIVATION_CLIPPED_RELU) {
      dy = GetDeviceAddress<T>(inputs, 0);
      y = GetDeviceAddress<T>(inputs, 1);
    } else {
      y = GetDeviceAddress<T>(inputs, 0);
      dy = GetDeviceAddress<T>(inputs, 1);
    }
    T *dx = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnActivationBackward(cudnn_handle_, activation_desc_, &alpha, data_descriptor_, y, data_descriptor_, dy,
                              data_descriptor_, y, &beta, data_descriptor_, dx),
      "cudnnActivationBackward failed");

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    auto iter = kernel_map.find(node_name);
    if (iter == kernel_map.end()) {
      MS_LOG(EXCEPTION) << "Only support these activations: ReLU6, Tanh, Elu, Sigmoid currently, but got " << node_name;
    }
    mode_ = iter->second;

    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << node_name << "', the number of inputs should be 2, but got " << input_num;
    }
    auto input_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(input_shape, node_name, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({input_shape});
    std::vector<size_t> shape;
    double coef = (mode_ == CUDNN_ACTIVATION_CLIPPED_RELU) ? ReLU6_UP_TURNING_POINT : 0.0;
    if (mode_ == CUDNN_ACTIVATION_ELU) coef = 1.0;
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetActivationDescriptor(activation_desc_, mode_, CUDNN_PROPAGATE_NAN, coef),
                                "SetActivationDescriptor failed");

    const int split_dim = 4;
    if (input_shape.size() <= split_dim) {
      ShapeNdTo4d(input_shape, &shape);
      if (AnfAlgo::GetInputFormat(kernel_node, 0) == kOpFormat_NHWC) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NHWC, cudnn_data_type_, SizeToInt(shape[0]),
                                     SizeToInt(shape[3]), SizeToInt(shape[1]), SizeToInt(shape[2])),
          "cudnnSetTensor4dDescriptor failed");
      } else {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnSetTensor4dDescriptor(data_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(shape[0]),
                                     SizeToInt(shape[1]), SizeToInt(shape[2]), SizeToInt(shape[3])),
          "cudnnSetTensor4dDescriptor failed");
      }
    } else {
      CudnnSetTensorNdDescriptor(input_shape, data_descriptor_, cudnn_data_type_, kernel_node_);
    }

    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyActivationDescriptor(activation_desc_),
                               "cudnnDestroyActivationDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(data_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    activation_desc_ = nullptr;
    mode_ = CUDNN_ACTIVATION_SIGMOID;
    data_descriptor_ = nullptr;
    is_null_input_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    input_size_ = 0;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&data_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateActivationDescriptor(&activation_desc_),
                                "cudnnCreateActivationDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(data_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(input_size_);
    input_size_list_.push_back(input_size_);
  }

 private:
  std::map<std::string, cudnnActivationMode_t> kernel_map = {{"ReLU6Grad", CUDNN_ACTIVATION_CLIPPED_RELU},
                                                             {"TanhGrad", CUDNN_ACTIVATION_TANH},
                                                             {"EluGrad", CUDNN_ACTIVATION_ELU},
                                                             {"SigmoidGrad", CUDNN_ACTIVATION_SIGMOID}};
  cudnnHandle_t cudnn_handle_;
  cudnnActivationDescriptor_t activation_desc_;
  cudnnActivationMode_t mode_;
  cudnnTensorDescriptor_t data_descriptor_;
  bool is_null_input_;

  cudnnDataType_t cudnn_data_type_;
  size_t input_size_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_ACTIVATION_GRAD_KERNEL_H_
