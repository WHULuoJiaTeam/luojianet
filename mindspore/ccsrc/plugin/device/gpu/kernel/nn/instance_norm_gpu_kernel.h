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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_

#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "include/common/utils/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/instance_norm_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputXDimSize = 4;
template <typename T>
class InstanceNormGpuKernelMod : public NativeGpuKernelMod {
 public:
  InstanceNormGpuKernelMod()
      : input_x_size_(0),
        input_z_size_(0),
        para_size_(0),
        output_size_(0),
        workspace_size_(0),
        mode_(CUDNN_BATCHNORM_SPATIAL),
        bn_ops_(CUDNN_BATCHNORM_OPS_BN),
        epsilon_(10e-5),
        exp_avg_factor_(0.1),
        is_null_input_(false),
        x_desc_(nullptr),
        y_desc_(nullptr),
        z_desc_(nullptr),
        scale_bias_mean_var_desc_(nullptr),
        handle_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT) {}
  ~InstanceNormGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    VARIABLE_NOT_USED(workspace);
    VARIABLE_NOT_USED(stream_ptr);
    if (is_null_input_) {
      return true;
    }
    auto x_addr = GetDeviceAddress<T>(inputs, 0);
    auto gamma_addr = GetDeviceAddress<float>(inputs, 1);
    auto beta_addr = GetDeviceAddress<float>(inputs, 2);
    auto runing_mean_addr = GetDeviceAddress<float>(inputs, 3);
    auto runnig_variance_addr = GetDeviceAddress<float>(inputs, 4);
    T *z = nullptr;

    auto y_addr = GetDeviceAddress<T>(outputs, 0);
    auto save_mean_addr = GetDeviceAddress<float>(outputs, 1);
    auto save_variance_addr = GetDeviceAddress<float>(outputs, 2);

    float *ws_gamma = GetDeviceAddress<float>(workspace, 0);
    float *ws_beta = GetDeviceAddress<float>(workspace, 1);
    float *ws_mean = GetDeviceAddress<float>(workspace, 2);
    float *ws_var = GetDeviceAddress<float>(workspace, 3);
    T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 4);

    size_t N = input_shape_[0];
    size_t C = input_shape_[1];
    CopyMemDevice2Device(N, C, gamma_addr, beta_addr, runing_mean_addr, runnig_variance_addr, ws_gamma, ws_beta,
                         ws_mean, ws_var, reinterpret_cast<cudaStream_t>(stream_ptr));

    const float alpha = 1;
    const float beta = 0;
    float *reserve_addr = nullptr;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnBatchNormalizationForwardTrainingEx(
        handle_, mode_, bn_ops_, &alpha, &beta, x_desc_, x_addr, z_desc_, z, y_desc_, y_addr, scale_bias_mean_var_desc_,
        ws_gamma, ws_beta, exp_avg_factor_, ws_mean, ws_var, epsilon_, save_mean_addr, save_variance_addr, nullptr,
        workspace_addr, workspace_size_, reserve_addr, 0),
      "Kernel launch failed");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    MS_EXCEPTION_IF_NULL(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    bn_ops_ = CUDNN_BATCHNORM_OPS_BN;

    InitResource();
    mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
    epsilon_ = GetAttr<float>(kernel_node, "epsilon");
    exp_avg_factor_ = GetAttr<float>(kernel_node, "momentum");

    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 5, but got " << input_num;
    }
    input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    if (input_shape_.size() != kInputXDimSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of input_x should be 4, but got "
                        << input_shape_.size();
    }
    is_null_input_ = CHECK_SHAPE_NULL(input_shape_, kernel_name, "input_x");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({input_shape_});
    SetTensorDescriptor();
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_), "Destroy x desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_desc_), "Destroy y desc failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_),
                               "Destroy para desc failed");
  }

 protected:
  void InitResource() override {
    handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_), "Create x desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_desc_), "Create y desc failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&scale_bias_mean_var_desc_),
                                "Create para desc failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(x_desc_, &input_x_size_),
                                  "Get input x size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(scale_bias_mean_var_desc_, &para_size_),
                                  "Get para size failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(y_desc_, &output_size_),
                                  "Get output size failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(handle_, mode_, bn_ops_, x_desc_, z_desc_, y_desc_,
                                                                 scale_bias_mean_var_desc_, nullptr, &workspace_size_),
        "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize failed");
    }

    input_size_list_.push_back(input_x_size_);    // input x
    input_size_list_.push_back(input_shape_[1]);  // gamma
    input_size_list_.push_back(input_shape_[1]);  // beta
    input_size_list_.push_back(input_shape_[1]);  // mean
    input_size_list_.push_back(input_shape_[1]);  // variance

    output_size_list_.push_back(output_size_);  // output
    output_size_list_.push_back(para_size_);    // save mean
    output_size_list_.push_back(para_size_);    // save variance

    workspace_size_list_.push_back(para_size_);  // ws gamma
    workspace_size_list_.push_back(para_size_);  // ws beta
    workspace_size_list_.push_back(para_size_);  // ws mean
    workspace_size_list_.push_back(para_size_);  // ws variance
    workspace_size_list_.push_back(workspace_size_);
  }

 private:
  void SetTensorDescriptor() {
    cudnnTensorFormat_t cudnn_format;
    int batch = 1;
    int channel = SizeToInt(input_shape_[0]) * SizeToInt(input_shape_[1]);
    int height = SizeToInt(input_shape_[2]);
    int width = SizeToInt(input_shape_[3]);
    cudnn_format = CUDNN_TENSOR_NCHW;

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensor4dDescriptor(x_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set x desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensor4dDescriptor(y_desc_, cudnn_format, cudnn_data_type_, batch, channel, height, width),
      "Set y desc failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(scale_bias_mean_var_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, channel, 1, 1),
      "Set para desc failed");
  }

  size_t input_x_size_;
  size_t input_z_size_;
  size_t para_size_;
  size_t output_size_;
  size_t workspace_size_;
  cudnnBatchNormMode_t mode_;
  cudnnBatchNormOps_t bn_ops_;
  double epsilon_;
  double exp_avg_factor_;
  bool is_null_input_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnTensorDescriptor_t z_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;

  cudnnHandle_t handle_;
  cudnnDataType_t cudnn_data_type_;
  std::vector<size_t> input_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_INSTANCE_NORM_GPU_KERNEL_H_
