/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PoolingFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  PoolingFwdGpuKernelMod()
      : cudnn_handle_(nullptr),
        input_descriptor_(nullptr),
        output_descriptor_(nullptr),
        pooling_descriptor_(nullptr),
        pooling_mode_(CUDNN_POOLING_MAX),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
        old_depth_(0),
        old_height_(0),
        old_width_(0),
        pad_depth_(0),
        pad_height_(0),
        pad_width_(0),
        pad_front_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        pad_value_(0),
        is_null_input_(false),
        kernel_name_("Pooling"),
        input_size_(0),
        output_size_(0),
        workspace_size_(0) {}
  ~PoolingFwdGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alpha, input_descriptor_,
                                                    input_addr, &beta, output_descriptor_, output_addr),
                                "cudnnPoolingForward failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (Anyone(format_attr, kOpFormat_NHWC, kOpFormat_NDHWC)) {
      data_format_ = format_attr;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({input_shape, output_shape});
    auto dim = input_shape.size();
    if (dim == kDim2DShapeSize) {
      SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    } else if (dim == kDim3DShapeSize) {
      SetNCDHW(input_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format_);
    }

    int dimA[kPoolingNbDims];
    int strideAin[kPoolingNbDims];
    int dimAout[kPoolingNbDims];
    int strideAout[kPoolingNbDims];
    SetDimA(input_shape, dimA, dim, data_format_);
    SetStrideA(input_shape, strideAin, dim, data_format_);
    SetDimA(output_shape, dimAout, dim, data_format_);
    SetStrideA(output_shape, strideAout, dim, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(input_descriptor_, cudnn_data_type_, dim, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(output_descriptor_, cudnn_data_type_, dim, dimAout, strideAout),
      "cudnnSetTensorNdDescriptor failed");
    SetPoolingMode(kernel_node);
    if (dim == kDim2DShapeSize) {
      SetPad(kernel_node);
    } else if (dim == kDim3DShapeSize) {
      SetPad3D(kernel_node);
    }
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyPoolingDescriptor(pooling_descriptor_),
                               "cudnnDestroyPoolingDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreatePoolingDescriptor(&pooling_descriptor_),
                                "cudnnCreatePoolingDescriptor failed");
  }
  void InitSizeLists() {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(input_descriptor_, reinterpret_cast<size_t *>(&input_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(output_descriptor_, reinterpret_cast<size_t *>(&output_size_)),
        "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
  }

  void SetPoolingMode(const CNodePtr &kernel_node) {
    mode_ = common::AnfAlgo::GetCNodeName(kernel_node);
    if (mode_ == "AvgPool" || mode_ == "AvgPool3D") {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }
  void SetPad(const CNodePtr &kernel_node) {
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
    std::vector<int> window;
    std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 4, but got "
                        << window.size();
    }
    int window_height = window[2];
    int window_width = window[3];
    std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[2] = {window_height, window_width};
    int paddingA[2] = {0, 0};
    if (stride_.size() < 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 4, but got "
                        << stride_.size();
    }
    int strideA[2] = {stride_[2], stride_[3]};
    int stride_h = stride_[2];
    int stride_w = stride_[3];
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      pad_top_ = pad_height_ / 2;
      pad_left_ = pad_width_ / 2;
      paddingA[0] = pad_top_;
      paddingA[1] = pad_left_;
    } else {
      pad_height_ = 0;
      pad_width_ = 0;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            2, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
  }

  void SetPad3D(const CNodePtr &kernel_node) {
    auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
    MS_EXCEPTION_IF_NULL(prim);
    pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
    std::vector<int> window;
    std::vector<int64_t> window_me = GetValue<std::vector<int64_t>>(prim->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (window.size() < 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 5, but got "
                        << window.size();
    }
    int window_depth = window[2];
    int window_height = window[3];
    int window_width = window[4];
    std::vector<int64_t> stride_me = GetValue<std::vector<int64_t>>(prim->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[3] = {window_depth, window_height, window_width};
    int paddingA[3] = {0, 0, 0};
    if (stride_.size() < 5) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 5, but got "
                        << stride_.size();
    }
    int strideA[3] = {stride_[2], stride_[3], stride_[4]};
    int stride_d = stride_[2];
    int stride_h = stride_[3];
    int stride_w = stride_[4];
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      pad_depth_ = GetPad(old_depth_, window_depth, stride_d);
      pad_height_ = GetPad(old_height_, window_height, stride_h);
      pad_width_ = GetPad(old_width_, window_width, stride_w);
      pad_front_ = pad_depth_ / 2;
      pad_top_ = pad_height_ / 2;
      pad_left_ = pad_width_ / 2;
      paddingA[0] = pad_front_;
      paddingA[1] = pad_top_;
      paddingA[2] = pad_left_;
    } else {
      pad_depth_ = 0;
      pad_height_ = 0;
      pad_width_ = 0;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            3, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;
  std::string mode_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;

  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_depth_;
  int old_height_;
  int old_width_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_front_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  float pad_value_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
