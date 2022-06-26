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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_FILTER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_FILTER_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cast_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kInputDimSize = 5;
constexpr size_t kInDimIdxForN = 0;
constexpr size_t kInDimIdxForC = 1;
constexpr size_t kInDimIdxForD = 2;
constexpr size_t kInDimIdxForH = 3;
constexpr size_t kInDimIdxForW = 4;

constexpr size_t k3DPadSize = 6;
constexpr size_t kHead3DPadIdx = 0;
constexpr size_t kTail3DPadIdx = 1;
constexpr size_t kTop3DPadIdx = 2;
constexpr size_t kBottom3DPadIdx = 3;
constexpr size_t kLeft3DPadIdx = 4;
constexpr size_t kRight3DPadIdx = 5;

constexpr size_t kPadDepthIdx = 0;
constexpr size_t kPadHeightIdx = 1;
constexpr size_t kPadWidthIdx = 2;

constexpr size_t k3DStrideSize = 5;
constexpr size_t kDepth3DStrideIdx = 2;
constexpr size_t kHeight3DStrideIdx = 3;
constexpr size_t kWidth3DStrideIdx = 4;

constexpr size_t k3DDilationSize = 5;
constexpr size_t kDepth3DDilationIdx = 2;
constexpr size_t kHeight3DDilationIdx = 3;
constexpr size_t kWidth3DDilationIdx = 4;

template <typename T>
class Conv3dGradFilterGpuKernelMod : public NativeGpuKernelMod {
 public:
  Conv3dGradFilterGpuKernelMod() { ResetResource(); }
  ~Conv3dGradFilterGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x = GetDeviceAddress<T>(inputs, 0);
    T *dy = GetDeviceAddress<T>(inputs, 1);

    T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    T *dw = nullptr;
    float *dw_float32 = nullptr;
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      dw = GetDeviceAddress<T>(workspace, 1);
      dw_float32 = GetDeviceAddress<float>(outputs, 0);
    } else {
      dw = GetDeviceAddress<T>(outputs, 0);
    }

    const float alpha = 1;
    const float beta = 0;
    if (use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);
      CalPad3d(padded_size_ / sizeof(T), x, n_, c_, old_depth_, old_height_, old_width_, old_depth_ + pad_depth_,
               old_height_ + pad_height_, old_width_ + pad_width_, pad_head_, pad_top_, pad_left_, pad_value_, padded,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, padded_descriptor_, padded, dy_desc_, dy, conv_desc_,
                                       algo_, work_space, workspace_size_, &beta, dw_desc_, dw),
        "ConvolutionBackwardFilter failed");
      return true;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, x_desc_, x, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta, dw_desc_, dw),
      "ConvolutionBackwardFilter failed");

    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      Cast(num_output_elements_, dw, dw_float32, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  void CheckSize(const size_t value, const size_t expect_value, const string arg_name) {
    if (value != expect_value) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of " << arg_name << " should be " << expect_value
                        << ", but got " << value;
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "x") || CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({in_shape});
    data_format_ = kOpFormat_NCDHW;

    std::vector<size_t> filter_shape;
    GetFilterShape(kernel_node, &filter_shape);
    num_output_elements_ = 1;
    for (auto x : filter_shape) {
      num_output_elements_ *= x;
    }

    compute_format_ = CUDNN_TENSOR_NCHW;
    (void)CheckSize(in_shape.size(), kInputDimSize, "input shape");
    n_ = SizeToInt(in_shape[kInDimIdxForN]);
    c_ = SizeToInt(in_shape[kInDimIdxForC]);
    old_depth_ = SizeToInt(in_shape[kInDimIdxForD]);
    old_height_ = SizeToInt(in_shape[kInDimIdxForH]);
    old_width_ = SizeToInt(in_shape[kInDimIdxForW]);
    SetNDDesc(dy_shape, filter_shape, in_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    SetPad(kernel_node, pad_list);
    SetStrideAndDilation(kernel_node);
    auto x_desc_real = GetXDescReal(pad_list);
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(x_desc_real);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    dw_desc_ = nullptr;
    conv_desc_ = nullptr;
    dy_desc_ = nullptr;
    x_desc_ = nullptr;
    padded_descriptor_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    compute_format_ = CUDNN_TENSOR_NCHW;
    old_depth_ = 0;
    old_height_ = 0;
    old_width_ = 0;
    pad_depth_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_head_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    n_ = 0;
    c_ = 0;
    group_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Conv3dGradFilter";
    input_size_ = 0;
    dy_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
    num_output_elements_ = 1;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(dw_desc_),
                               "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(x_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&x_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&dw_desc_),
                                "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(dy_desc_, reinterpret_cast<size_t *>(&dy_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(x_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetFilterSizeInBytes(dw_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
    }
    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(input_size_);

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(padded_descriptor_, reinterpret_cast<size_t *>(&padded_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, padded_descriptor_, dy_desc_, conv_desc_,
                                                       dw_desc_, algo_, reinterpret_cast<size_t *>(&workspace_size_)),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, x_desc_, dy_desc_, conv_desc_, dw_desc_, algo_,
                                                         reinterpret_cast<size_t *>(&workspace_size_)),
          "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);

    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      workspace_size_list_.push_back(output_size_);
      output_size_list_.push_back(num_output_elements_ * sizeof(float));
    } else {
      output_size_list_.push_back(output_size_);
    }
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  void SelectAlgorithm(cudnnTensorDescriptor_t x_desc_real) {
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                    requested_algo_count, &returned_algo_count, &perf_results),
      "GetConvolutionBackwardFilterAlgorithm failed");
    algo_ = perf_results.algo;
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    }
  }

  void GetFilterShape(const CNodePtr &kernel_node, std::vector<size_t> *filter_shape) {
    auto shp_tuple_x = GetAttrAndConvertValueTuple(kernel_node, "filter_size");
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*filter_shape),
                         [](const ValuePtr &e) -> size_t {
                           auto cast_value = e->cast<Int64ImmPtr>();
                           MS_EXCEPTION_IF_NULL(cast_value);
                           return static_cast<int>(cast_value->value());
                         });
  }

  void SetNDDesc(const std::vector<size_t> &dy_shape, const std::vector<size_t> &filter_shape,
                 const std::vector<size_t> &in_shape) {
    const int kDims = 5;
    int dimA[kDims];
    int strideAin[kDims];
    int dimAdy[kDims];
    int strideAdy[kDims];
    int filterDimA[kDims];
    SetDimA(in_shape, dimA, kDims, data_format_);
    SetStrideA(in_shape, strideAin, kDims, data_format_);
    SetDimA(dy_shape, dimAdy, kDims, data_format_);
    SetStrideA(dy_shape, strideAdy, kDims, data_format_);
    SetDimA(filter_shape, filterDimA, kDims, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, kDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(dw_desc_, cudnn_data_type_, compute_format_, kDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(x_desc_, cudnn_data_type_, kDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k3DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' should be 5, but got "
                        << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis should be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != k3DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' should be 5, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis should be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }

  void SetPad(const CNodePtr &kernel_node, const std::vector<int> &pad_list) {
    (void)CheckSize(pad_list.size(), k3DPadSize, "pad");
    pad_depth_ = pad_list[kHead3DPadIdx];
    pad_height_ = pad_list[kTop3DPadIdx];
    pad_width_ = pad_list[kLeft3DPadIdx];
    use_pad_ = !((pad_depth_ == pad_list[kTail3DPadIdx]) && (pad_height_ == pad_list[kBottom3DPadIdx]) &&
                 (pad_width_ == pad_list[kRight3DPadIdx]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
  }

  cudnnTensorDescriptor_t GetXDescReal(const std::vector<int> &pad_list) {
    cudnnTensorDescriptor_t x_desc_real = nullptr;
    const int kNumDims = 5;
    const int kConvDims = 3;
    int padA[kConvDims];
    int strideA[kConvDims] = {stride_[kDepth3DStrideIdx], stride_[kHeight3DStrideIdx], stride_[kWidth3DStrideIdx]};
    int dilaA[kConvDims] = {dilation_[kDepth3DDilationIdx], dilation_[kHeight3DDilationIdx],
                            dilation_[kWidth3DDilationIdx]};
    if (use_pad_) {
      pad_depth_ = pad_list[kHead3DPadIdx] + pad_list[kTail3DPadIdx];
      pad_height_ = pad_list[kTop3DPadIdx] + pad_list[kBottom3DPadIdx];
      pad_width_ = pad_list[kLeft3DPadIdx] + pad_list[kRight3DPadIdx];
      pad_head_ = pad_list[kHead3DPadIdx];
      pad_top_ = pad_list[kTop3DPadIdx];
      pad_left_ = pad_list[kLeft3DPadIdx];
      int dimA[kNumDims];
      int strideApadded[kNumDims];
      if (data_format_ != kOpFormat_NCDHW) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'data_format' only support 'NCDHW' right now "
                          << ", but got " << data_format_;
      }
      auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_depth_ + pad_depth_),
                           IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_)};
      SetDimA(padded_shape, dimA, kNumDims, data_format_);
      SetStrideA(padded_shape, strideApadded, kNumDims, data_format_);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, kNumDims, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      padA[kPadDepthIdx] = 0;
      padA[kPadHeightIdx] = 0;
      padA[kPadWidthIdx] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      x_desc_real = padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[kPadDepthIdx] = pad_depth_;
      padA[kPadHeightIdx] = pad_height_;
      padA[kPadWidthIdx] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      x_desc_real = x_desc_;
    }

    return x_desc_real;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t dw_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdFilterAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCDHW;

  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_depth_;
  int old_height_;
  int old_width_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_head_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t dy_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  size_t num_output_elements_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_FILTER_GPU_KERNEL_H_
