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

#include "plugin/device/cpu/kernel/mirror_pad_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace luojianet_ms {
namespace kernel {
namespace {
// preset size of paddings
constexpr int MAX_PADDINGS = 4;
constexpr int PADDING_SIZE = 2;

// define constants for kernel indexing use
constexpr int BATCH = 0;
constexpr int CHANNEL = 1 * PADDING_SIZE;
constexpr int HEIGHT = 2 * PADDING_SIZE;
constexpr int WIDTH = 3 * PADDING_SIZE;
constexpr int TOP = 0;
constexpr int BOTTOM = 1;
constexpr int LEFT = 0;
constexpr int RIGHT = 1;
constexpr size_t kMirrorPadGradInputsNum = 2;
constexpr size_t kMirrorPadGradOutputsNum = 1;
constexpr size_t kPadMaxSupportDim = 4;

template <typename T>
void extract_paddings(const T *paddings_arg, int64_t padd_dim, int64_t *extracted_paddings) {
  const int64_t paddings_offset = MAX_PADDINGS - padd_dim;
  for (int64_t i = 0; i < padd_dim; i++) {
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE] = int64_t(paddings_arg[i * PADDING_SIZE]);
    extracted_paddings[(paddings_offset + i) * PADDING_SIZE + 1] = int64_t(paddings_arg[i * PADDING_SIZE + 1]);
  }
}

bool range_check(int64_t x, int64_t y, int64_t padded_width, int64_t padded_height) {
  if (((x >= 0) && (x <= padded_width - 1)) && ((y >= 0) && (y <= padded_height - 1))) {
    return true;
  }
  return false;
}
}  // namespace

void MirrorPadGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::string mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "mode");
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  pad_dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 1);
  if (mode == "REFLECT") {
    mode_ = 0;
  } else if (mode == "SYMMETRIC") {
    mode_ = 1;
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'mode' should be 'REFLECT' or 'SYMMETRIC', but got "
                      << mode;
  }

  std::vector<size_t> input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  shape_size_ = input_shape.size();
  (void)input_shape.insert(input_shape.begin(), kPadMaxSupportDim - shape_size_, 1);
  shape_size_ = kPadMaxSupportDim;

  for (size_t i = 0; i < shape_size_; ++i) {
    tensor_size_ *= input_shape[i];
    input_shape_.push_back(SizeToLong(input_shape[i]));
  }

  std::vector<size_t> padding_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  num_paddings_ = SizeToLong(padding_shape[0]);

  std::vector<size_t> output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

  if (output_shape.size() == 4) {
  } else if (output_shape.size() == 3) {
    (void)output_shape.insert(output_shape.begin(), 1);  // batch padding
  } else if (output_shape.size() == 2) {
    (void)output_shape.insert(output_shape.begin(), 2, 1);  // channel padding
  }
  for (auto x : output_shape) {
    output_size_ *= x;
    output_shape_.push_back(SizeToLong(x));
  }

  for (size_t i = 0; i < 2; i++) {
    workspace_size_ *= output_shape[i];
    workspace_size_ *= input_shape[i + 2];
  }

  int64_t max_width = input_shape_[3];
  int64_t max_height = input_shape_[2];
  // basic error check for padding value
  if (mode_ == 1) {  // symmetric
    max_width = max_width + (2 * max_width);
    max_height = max_height + (2 * max_height);
  } else {  // reflect
    max_width = max_width + (2 * (max_width - 1));
    max_height = max_height + (2 * (max_height - 1));
  }

  if (output_shape_[(output_shape_.size() - 2)] > max_height ||
      output_shape_[(output_shape_.size() - 2) + 1] > max_width) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the 'paddings' should be not too high for input Tensor on 1 or more dimensions";
  }
}

bool MirrorPadGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                       const std::vector<kernel::AddressPtr> &workspace,
                                       const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMirrorPadGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMirrorPadGradOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16 && pad_dtype_ == kNumberTypeInt32) {
    LaunchKernel<float16, int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32 && pad_dtype_ == kNumberTypeInt32) {
    LaunchKernel<float, int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat64 && pad_dtype_ == kNumberTypeInt32) {
    LaunchKernel<double, int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt32 && pad_dtype_ == kNumberTypeInt32) {
    LaunchKernel<int, int32_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat16 && pad_dtype_ == kNumberTypeInt64) {
    LaunchKernel<float16, int64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat32 && pad_dtype_ == kNumberTypeInt64) {
    LaunchKernel<float, int64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeFloat64 && pad_dtype_ == kNumberTypeInt64) {
    LaunchKernel<double, int64_t>(inputs, workspace, outputs);
  } else if (dtype_ == kNumberTypeInt32 && pad_dtype_ == kNumberTypeInt64) {
    LaunchKernel<int, int64_t>(inputs, workspace, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be float16, float32, float64, or int32, and the dtype of "
                         "'paddings' should be int32 or int64, but got "
                      << TypeIdLabel(dtype_) << " and " << TypeIdLabel(pad_dtype_);
  }
  return true;
}

template <typename T>
void MirrorPadGradCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(workspace_size_ * sizeof(T));
}

void MirrorPadGradCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  if (dtype_ == kNumberTypeFloat16) {
    InitWorkspaceSize<float16>();
  } else if (dtype_ == kNumberTypeFloat32) {
    InitWorkspaceSize<float>();
  } else if (dtype_ == kNumberTypeFloat64) {
    InitWorkspaceSize<double>();
  } else if (dtype_ == kNumberTypeInt32) {
    InitWorkspaceSize<int>();
  }
}

template <typename T1, typename T2>
void MirrorPadGradCpuKernelMod::MirrorPadGrad_Width_Height(const size_t size, const T1 *interim_dy,
                                                           const int64_t dx_height, const int64_t dx_width,
                                                           const int64_t dy_height, const int64_t dy_width,
                                                           const int64_t padd_dim, const T2 *paddings_arg, int64_t mode,
                                                           T1 *dx) const {
  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;  // init all to 0
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create required anchor points for non-mirrored data inside new tensor
  int64_t ap1_x = paddings[WIDTH];
  int64_t ap2_x = paddings[WIDTH] + dx_width - 1;
  int64_t ap1_y = paddings[HEIGHT];
  int64_t ap2_y = paddings[HEIGHT] + dx_height - 1;
  if (dx_width == 0 || dx_height == 0) {
    MS_LOG(EXCEPTION)
      << "For  MirrorPadGrad_Width_Height, the input argument 'dx_height' and 'dx_width' should not be 0, but got "
      << "dy_height: " << dx_height << " dy_width: " << dx_width;
  }
  for (size_t pos = 0; pos < size; ++pos) {
    int64_t dx_block_num = (SizeToLong(pos) / dx_width) / dx_height;
    const int64_t grad_x = (SizeToLong(pos) % dx_width) + paddings[WIDTH];
    const int64_t grad_y = ((SizeToLong(pos) / dx_width) % dx_height) + paddings[HEIGHT];

    // copy position's own value into output
    dx[pos] = interim_dy[(dx_block_num * dy_height + grad_y) * dy_width + grad_x];

    int64_t x_dist_1 = ((ap1_x - grad_x) - mode);
    int64_t y_dist_1 = ((ap1_y - grad_y) - mode);
    int64_t x_dist_2 = ((ap2_x - grad_x) + mode);
    int64_t y_dist_2 = ((ap2_y - grad_y) + mode);
    int64_t axis_dist[] = {x_dist_1, x_dist_2, y_dist_1, y_dist_2};
    int64_t anch_point[] = {ap1_x, ap2_x, ap1_y, ap2_y};
    bool x_axis_check[] = {true, true, false, false};  // true - update X , false - update Y

    int64_t temp_x = 0;
    int64_t temp_y = 0;
    // mirroring in axis lines
    for (int x = 0; x < 4; x++) {
      if (axis_dist[x] != 0) {
        if (x_axis_check[x]) {
          temp_y = grad_y;
          temp_x = anch_point[x] + axis_dist[x];
        } else {
          temp_x = grad_x;
          temp_y = anch_point[x] + axis_dist[x];
        }
        if (range_check(temp_x, temp_y, dy_width, dy_height)) {
          dx[pos] = dx[pos] + interim_dy[(dx_block_num * dy_height + temp_y) * dy_width + temp_x];
        }
      }
    }
    // mirroring at corners
    for (int x = 0; x < 2; x++) {
      for (int y = 2; y < 4; y++) {
        if ((axis_dist[x] != 0) && (axis_dist[y] != 0)) {
          temp_x = anch_point[x] + axis_dist[x];
          temp_y = anch_point[y] + axis_dist[y];
          if (range_check(temp_x, temp_y, dy_width, dy_height)) {
            dx[pos] = dx[pos] + interim_dy[(dx_block_num * dy_height + temp_y) * dy_width + temp_x];
          }
        }
      }
    }
  }
}

template <typename T1, typename T2>
void MirrorPadGradCpuKernelMod::MirrorPadGradBatchChannel(const size_t size, T1 *dy, T1 *interim_dy,
                                                          const int64_t dx_batches, const int64_t dx_channels,
                                                          const int64_t dy_height, const int64_t dy_width,
                                                          const int64_t padd_dim, const T2 *paddings_arg,
                                                          int64_t mode) const {
  if (dy_height == 0 || dy_width == 0 || dx_channels == 0) {
    MS_LOG(EXCEPTION) << "For  MirrorPadGradBatchChannel, the input argument 'dy_height', 'dy_width' and 'dx_channels' "
                         "should not be 0, but got "
                      << "dy_height: " << dy_height << " dy_width: " << dy_width << " dx_channels: " << dx_channels;
  }
  int64_t paddings[MAX_PADDINGS * PADDING_SIZE];  // local and fixed size to keep in registers
  for (int i = 0; i < MAX_PADDINGS * PADDING_SIZE; i++) {
    paddings[i] = 0;  // init all to 0
  }
  extract_paddings(paddings_arg, padd_dim, paddings);
  // Create anchor points for non mirrored data inside new tensor
  int64_t ap1_channel = paddings[CHANNEL];
  int64_t ap2_channel = paddings[CHANNEL] + dx_channels - 1;
  int64_t ap1_batch = paddings[BATCH];
  int64_t ap2_batch = paddings[BATCH] + dx_batches - 1;
  int64_t dy_channels = dx_channels + paddings[CHANNEL] + paddings[CHANNEL + RIGHT];
  int64_t dy_batches = dx_batches + paddings[BATCH] + paddings[RIGHT];

  for (size_t pos = 0; pos < size; ++pos) {
    int64_t block_num = (SizeToLong(pos) / dy_width) / dy_height;
    // Select exact position inside the dy_interim array
    const int64_t interim_x = SizeToLong(pos) % dy_width;
    const int64_t interim_y = (SizeToLong(pos) / dy_width) % dy_height;
    const int64_t interim_channel = block_num % dx_channels;
    const int64_t interim_batch = block_num / dx_channels;
    interim_dy[pos] = T1(0);  // init
    // map cur interim channel and batch to equivalent in padded dy array
    const int64_t equiv_dy_channel = interim_channel + paddings[CHANNEL];
    const int64_t equiv_dy_batch = interim_batch + paddings[BATCH];
    int64_t target_batch = 0;
    int64_t target_channel = 0;
    int64_t equiv_block_num = ((equiv_dy_batch * dy_channels) + equiv_dy_channel);
    // generate values to sweep over all possible mirrored points
    int64_t batch_offsets[] = {2 * (ap1_batch - equiv_dy_batch) - mode, 0, 2 * (ap2_batch - equiv_dy_batch) + mode};
    int64_t channel_offsets[] = {2 * (ap1_channel - equiv_dy_channel) - mode, 0,
                                 2 * (ap2_channel - equiv_dy_channel) + mode};
    for (int64_t b_adjust : batch_offsets) {
      for (int64_t c_adjust : channel_offsets) {
        target_batch = equiv_dy_batch + b_adjust;
        target_channel = equiv_dy_channel + c_adjust;
        // bounds check - if within bounds, mirrored value exists - copy dy
        if ((target_batch < 0) || (target_batch > (dy_batches - 1)) || (target_channel < 0) ||
            (target_channel > (dy_channels - 1))) {
          continue;  // no mirrored value with these target values
        }
        equiv_block_num = ((target_batch * dy_channels) + target_channel);
        // Copy data and set value at input to 0 to avoid duplicates in reflect mode
        interim_dy[pos] = T1(interim_dy[pos] + dy[(equiv_block_num * dy_height + interim_y) * dy_width + interim_x]);
        dy[(equiv_block_num * dy_height + interim_y) * dy_width + interim_x] = T1(0);
      }
    }
  }
}

template <typename T1, typename T2>
void MirrorPadGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                             const std::vector<AddressPtr> &workspace,
                                             const std::vector<AddressPtr> &outputs) const {
  auto *inputs_addr = reinterpret_cast<T1 *>(inputs[0]->addr);
  auto *paddings = reinterpret_cast<T2 *>(inputs[1]->addr);
  auto *interim = reinterpret_cast<T1 *>(workspace[0]->addr);
  auto *outputs_addr = reinterpret_cast<T1 *>(outputs[0]->addr);

  MirrorPadGradBatchChannel(workspace_size_, inputs_addr, interim, output_shape_[0], output_shape_[1], input_shape_[2],
                            input_shape_[3], num_paddings_, paddings, mode_);

  MirrorPadGrad_Width_Height(output_size_, interim, output_shape_[2], output_shape_[3], input_shape_[2],
                             input_shape_[3], num_paddings_, paddings, mode_, outputs_addr);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MirrorPadGrad, MirrorPadGradCpuKernelMod);
}  // namespace kernel
}  // namespace luojianet_ms
