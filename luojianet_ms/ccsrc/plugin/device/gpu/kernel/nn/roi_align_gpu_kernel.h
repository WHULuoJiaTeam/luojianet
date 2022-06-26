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

#ifndef LUOJIANET_MS_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H
#define LUOJIANET_MS_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roi_align_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T>
class ROIAlignFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  ROIAlignFwdGpuKernelMod()
      : pooled_height_(0),
        pooled_width_(0),
        spatial_scale_(),
        sample_num_(0),
        roi_end_mode_(0),
        roi_rows_(0),
        roi_cols_(0),
        batch_N_(0),
        channels_(0),
        height_(0),
        width_(0),
        is_null_input_(false),
        x_size_(0),
        rois_size_(0),
        output_size_(0) {}
  ~ROIAlignFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    const T *x = GetDeviceAddress<T>(inputs, 0);
    const T *rois = GetDeviceAddress<T>(inputs, 1);

    T *out_data = GetDeviceAddress<T>(outputs, 0);

    ROIAlign(x, rois, roi_rows_, roi_cols_, out_data, spatial_scale_, sample_num_, roi_end_mode_, channels_, height_,
             width_, pooled_height_, pooled_width_, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    // Get the number of input args
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs should be 2, but got " << input_num;
    }

    // Get the number of output args
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of outputs should be 1, but got " << output_num;
    }

    // Get the input shapes
    auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto rois_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ =
      CHECK_SHAPE_NULL(x_shape, kernel_name, "features") || CHECK_SHAPE_NULL(rois_shape, kernel_name, "rois");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    auto x_shape_size = x_shape.size();
    if (x_shape_size != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of features should be equal to 4, but got "
                        << x_shape_size;
    }

    // Get channels, height & width
    batch_N_ = x_shape[0];
    channels_ = x_shape[1];
    height_ = x_shape[2];
    width_ = x_shape[3];
    x_shape_ = {batch_N_, channels_, height_, width_};
    x_size_ = batch_N_ * channels_ * height_ * width_ * sizeof(T);

    if (rois_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the dimension of rois cannot be less than 2, but got "
                        << rois_shape.size();
    }
    // Get rois rows and cols
    roi_rows_ = rois_shape[0];
    roi_cols_ = rois_shape[1];
    rois_size_ = roi_rows_ * roi_cols_ * sizeof(T);
    rois_shape_ = {roi_rows_, roi_cols_};

    // Get primitive args
    pooled_height_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_height"));
    pooled_width_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "pooled_width"));
    spatial_scale_ = static_cast<T>(GetAttr<float>(kernel_node, "spatial_scale"));
    sample_num_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "sample_num"));
    roi_end_mode_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "roi_end_mode"));

    // Get output_shape
    output_shape_ = {roi_rows_, channels_, pooled_height_, pooled_width_};
    output_size_ = 1;
    for (size_t i = 0; i < 4; i++) {
      output_size_ *= output_shape_[i];
    }
    output_size_ *= sizeof(T);

    InitSizeLists();
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(x_size_);
    input_size_list_.push_back(rois_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  int pooled_height_;
  int pooled_width_;
  T spatial_scale_;
  int sample_num_;
  int roi_end_mode_;

  int roi_rows_;
  int roi_cols_;
  int batch_N_;
  int channels_;
  int height_;
  int width_;
  bool is_null_input_;

  std::vector<int> x_shape_;
  std::vector<int> rois_shape_;
  std::vector<int> output_shape_;

  size_t x_size_;
  size_t rois_size_;
  size_t output_size_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_KERNEL_GPU_ROI_ALIGN_GPU_KERNEL_H
