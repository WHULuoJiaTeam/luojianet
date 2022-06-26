/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/kernel/grid_sampler_3d_grad_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace {
const size_t kDataSizeThreshold = 64 * 1024;
const size_t kZero = 0;
const size_t kOne = 1;
const size_t kTwo = 2;
const size_t kThree = 3;
const size_t kFour = 4;
const int64_t kNumber1 = 1;
}  // namespace

namespace mindspore {
namespace kernel {
void GridSampler3DGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  grad_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kZero);
  x_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kOne);
  grid_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, kTwo);
  dx_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kZero);
  dgrid_shape_ = AnfAlgo::GetOutputDeviceShape(kernel_node, kOne);
  dx_size_ = dx_shape_[kZero] * dx_shape_[kOne] * dx_shape_[kTwo] * dx_shape_[kThree] * dx_shape_[kFour];
  grid_size_ = grid_shape_[kZero] * grid_shape_[kOne] * grid_shape_[kTwo] * grid_shape_[kThree];
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, kZero);
  size_t stride_tmp = kOne;
  auto stride_compute = [&](std::vector<size_t> &stride, std::vector<size_t> shape) {
    for (int i = kFour; i > -static_cast<int>(kOne); i--) {
      stride.insert(stride.begin(), stride_tmp);
      stride_tmp *= shape[i];
    }
    stride_tmp = kOne;
  };
  stride_compute(grad_stride_, grad_shape_);
  stride_compute(x_stride_, x_shape_);
  stride_compute(grid_stride_, grid_shape_);
  stride_compute(dx_stride_, dx_shape_);
  stride_compute(dgrid_stride_, dgrid_shape_);
  interpolation_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "interpolation_mode");
  padding_mode = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, "padding_mode");
  align_corners = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "align_corners");
}

bool GridSampler3DGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  if (dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else {
    MS_EXCEPTION(TypeError) << "Input dtype only support float32 and float64!";
  }
  return true;
}

template <typename T>
void GridSampler3DGradCpuKernelMod::BilinearKernel(std::vector<T *> addr, std::vector<T> location, std::vector<T> mult,
                                                   std::vector<size_t> ptr) {
  T x = location[kZero], y = location[kOne], z = location[kTwo];
  int64_t x_tnw = static_cast<int64_t>(std::floor(x));
  int64_t y_tnw = static_cast<int64_t>(std::floor(y));
  int64_t z_tnw = static_cast<int64_t>(std::floor(z));
  int64_t x_tne = x_tnw + kNumber1, y_tne = y_tnw, z_tne = z_tnw;
  int64_t x_tsw = x_tnw, y_tsw = y_tnw + kNumber1, z_tsw = z_tnw;
  int64_t x_tse = x_tnw + kNumber1, y_tse = y_tnw + kNumber1, z_tse = z_tnw;
  int64_t x_bnw = x_tnw, y_bnw = y_tnw, z_bnw = z_tnw + kNumber1;
  int64_t x_bne = x_tnw + kNumber1, y_bne = y_tnw, z_bne = z_tnw + kNumber1;
  int64_t x_bsw = x_tnw, y_bsw = y_tnw + kNumber1, z_bsw = z_tnw + kNumber1;
  int64_t x_bse = x_tnw + kNumber1, y_bse = y_tnw + kNumber1, z_bse = z_tnw + kNumber1;
  T tnw = (x_bse - x) * (y_bse - y) * (z_bse - z), tne = (x - x_bsw) * (y_bsw - y) * (z_bsw - z);
  T tsw = (x_bne - x) * (y - y_bne) * (z_bne - z), tse = (x - x_bnw) * (y - y_bnw) * (z_bnw - z);
  T bnw = (x_tse - x) * (y_tse - y) * (z - z_tse), bne = (x - x_tsw) * (y_tsw - y) * (z - z_tsw);
  T bsw = (x_tne - x) * (y - y_tne) * (z - z_tne), bse = (x - x_tnw) * (y - y_tnw) * (z - z_tnw);
  T gx = static_cast<T>(kZero), gy = static_cast<T>(kZero), gz = static_cast<T>(kZero);
  for (size_t c = kZero; c < x_shape_[kOne];
       c++, ptr[kZero] += grad_stride_[kOne], ptr[kOne] += x_stride_[kOne], ptr[kTwo] += dx_stride_[kOne]) {
    T grad_out = addr[kZero][ptr[kZero]];
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_tnw, y_tnw, x_tnw, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], tnw * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_tne, y_tne, x_tne, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], tne * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_tsw, y_tsw, x_tsw, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], tsw * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_tse, y_tse, x_tse, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], tse * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_bnw, y_bnw, x_bnw, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], bnw * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_bne, y_bne, x_bne, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], bne * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_bsw, y_bsw, x_bsw, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], bsw * grad_out);
    safe_add_3d(&addr[kTwo][ptr[kTwo]], z_bse, y_bse, x_bse, dx_stride_[kTwo], dx_stride_[kThree], dx_stride_[kFour],
                x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour], bse * grad_out);
    if (within_bounds_3d(z_tnw, y_tnw, x_tnw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_tnw * x_stride_[kTwo] + y_tnw * x_stride_[kThree] + x_tnw * x_stride_[kFour];
      T tnw_val = addr[kOne][offset];
      gx -= tnw_val * (y_bse - y) * (z_bse - z) * grad_out;
      gy -= tnw_val * (x_bse - x) * (z_bse - z) * grad_out;
      gz -= tnw_val * (x_bse - x) * (y_bse - y) * grad_out;
    }
    if (within_bounds_3d(z_tne, y_tne, x_tne, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_tne * x_stride_[kTwo] + y_tne * x_stride_[kThree] + x_tne * x_stride_[kFour];
      T tne_val = addr[kOne][offset];
      gx += tne_val * (y_bsw - y) * (z_bsw - z) * grad_out;
      gy -= tne_val * (x - x_bsw) * (z_bsw - z) * grad_out;
      gz -= tne_val * (x - x_bsw) * (y_bsw - y) * grad_out;
    }
    if (within_bounds_3d(z_tsw, y_tsw, x_tsw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_tsw * x_stride_[kTwo] + y_tsw * x_stride_[kThree] + x_tsw * x_stride_[kFour];
      T tsw_val = addr[kOne][offset];
      gx -= tsw_val * (y - y_bne) * (z_bne - z) * grad_out;
      gy += tsw_val * (x_bne - x) * (z_bne - z) * grad_out;
      gz -= tsw_val * (x_bne - x) * (y - y_bne) * grad_out;
    }
    if (within_bounds_3d(z_tse, y_tse, x_tse, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_tse * x_stride_[kTwo] + y_tse * x_stride_[kThree] + x_tse * x_stride_[kFour];
      T tse_val = addr[kOne][offset];
      gx += tse_val * (y - y_bnw) * (z_bnw - z) * grad_out;
      gy += tse_val * (x - x_bnw) * (z_bnw - z) * grad_out;
      gz -= tse_val * (x - x_bnw) * (y - y_bnw) * grad_out;
    }
    if (within_bounds_3d(z_bnw, y_bnw, x_bnw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_bnw * x_stride_[kTwo] + y_bnw * x_stride_[kThree] + x_bnw * x_stride_[kFour];
      T bnw_val = addr[kOne][offset];
      gx -= bnw_val * (y_tse - y) * (z - z_tse) * grad_out;
      gy -= bnw_val * (x_tse - x) * (z - z_tse) * grad_out;
      gz += bnw_val * (x_tse - x) * (y_tse - y) * grad_out;
    }
    if (within_bounds_3d(z_bne, y_bne, x_bne, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_bne * x_stride_[kTwo] + y_bne * x_stride_[kThree] + x_bne * x_stride_[kFour];
      T bne_val = addr[kOne][offset];
      gx += bne_val * (y_tsw - y) * (z - z_tsw) * grad_out;
      gy -= bne_val * (x - x_tsw) * (z - z_tsw) * grad_out;
      gz += bne_val * (x - x_tsw) * (y_tsw - y) * grad_out;
    }
    if (within_bounds_3d(z_bsw, y_bsw, x_bsw, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_bsw * x_stride_[kTwo] + y_bsw * x_stride_[kThree] + x_bsw * x_stride_[kFour];
      T bsw_val = addr[kOne][offset];
      gx -= bsw_val * (y - y_tne) * (z - z_tne) * grad_out;
      gy += bsw_val * (x_tne - x) * (z - z_tne) * grad_out;
      gz += bsw_val * (x_tne - x) * (y - y_tne) * grad_out;
    }
    if (within_bounds_3d(z_bse, y_bse, x_bse, x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour])) {
      size_t offset = ptr[kOne] + z_bse * x_stride_[kTwo] + y_bse * x_stride_[kThree] + x_bse * x_stride_[kFour];
      T bse_val = addr[kOne][offset];
      gx += bse_val * (y - y_tnw) * (z - z_tnw) * grad_out;
      gy += bse_val * (x - x_tnw) * (z - z_tnw) * grad_out;
      gz += bse_val * (x - x_tnw) * (y - y_tnw) * grad_out;
    }
  }
  addr[kThree][ptr[kThree]] = mult[kZero] * gx;
  addr[kThree][ptr[kThree] + kOne] = mult[kOne] * gy;
  addr[kThree][ptr[kThree] + kTwo] = mult[kTwo] * gz;
}

template <typename T>
void GridSampler3DGradCpuKernelMod::ComputeTask(T *grad_addr, T *x_addr, T *grid_addr, T *dx_addr, T *dgrid_addr,
                                                const size_t &n) {
  size_t grid_ptr_N = n * grid_stride_[kZero];
  size_t dgrid_ptr_NDHW = n * dgrid_stride_[kZero];
  for (size_t d = kZero; d < grid_shape_[kOne]; d++) {
    for (size_t h = kZero; h < grid_shape_[kTwo]; h++) {
      for (size_t w = kZero; w < grid_shape_[kThree]; w++, dgrid_ptr_NDHW += dgrid_stride_[kThree]) {
        size_t grid_ptr_NDHW = grid_ptr_N + d * grid_stride_[kOne] + h * grid_stride_[kTwo] + w * grid_stride_[kThree];
        T x = grid_addr[grid_ptr_NDHW];
        T y = grid_addr[grid_ptr_NDHW + grid_stride_[kFour]];
        T z = grid_addr[grid_ptr_NDHW + kTwo * grid_stride_[kFour]];
        T gx_mult, gy_mult, gz_mult;
        x = grid_sampler_compute_source_index_set_grad(x, x_shape_[kFour], padding_mode, align_corners, &gx_mult);
        y = grid_sampler_compute_source_index_set_grad(y, x_shape_[kThree], padding_mode, align_corners, &gy_mult);
        z = grid_sampler_compute_source_index_set_grad(z, x_shape_[kTwo], padding_mode, align_corners, &gz_mult);
        if (interpolation_mode == "bilinear") {
          size_t grad_ptr_NCDHW =
            n * grad_stride_[kZero] + d * grad_stride_[kTwo] + h * grad_stride_[kThree] + w * grad_stride_[kFour];
          size_t dx_ptr_NC = n * dx_stride_[kZero], x_ptr_NC = n * x_stride_[kZero];
          std::vector<T *> addr = {grad_addr, x_addr, dx_addr, dgrid_addr};
          std::vector<T> location = {x, y, z};
          std::vector<T> mult = {gx_mult, gy_mult, gz_mult};
          std::vector<size_t> ptr = {grad_ptr_NCDHW, x_ptr_NC, dx_ptr_NC, dgrid_ptr_NDHW};
          BilinearKernel<T>(addr, location, mult, ptr);
        } else if (interpolation_mode == "nearest") {
          int64_t x_nearest = static_cast<int64_t>(std::round(x));
          int64_t y_nearest = static_cast<int64_t>(std::round(y));
          int64_t z_nearest = static_cast<int64_t>(std::round(z));
          size_t grad_ptr_NCDHW =
            n * grad_stride_[kZero] + d * grad_stride_[kTwo] + h * grad_stride_[kThree] + w * grad_stride_[kFour];
          size_t dx_ptr_NC = n * dx_stride_[kZero];
          for (size_t c = kZero; c < x_shape_[kOne];
               c++, grad_ptr_NCDHW += grad_stride_[kOne], dx_ptr_NC += dx_stride_[kOne]) {
            safe_add_3d(&dx_addr[dx_ptr_NC], z_nearest, y_nearest, x_nearest, dx_stride_[kTwo], dx_stride_[kThree],
                        dx_stride_[kFour], x_shape_[kTwo], x_shape_[kThree], x_shape_[kFour],
                        grad_addr[grad_ptr_NCDHW]);
          }
          dgrid_addr[dgrid_ptr_NDHW] = static_cast<T>(kZero);
          dgrid_addr[dgrid_ptr_NDHW + kOne] = static_cast<T>(kZero);
          dgrid_addr[dgrid_ptr_NDHW + kTwo] = static_cast<T>(kZero);
        }
      }
    }
  }
}

template <typename T>
void GridSampler3DGradCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &outputs) {
  auto grad_data_addr = reinterpret_cast<T *>(inputs[kZero]->addr);
  auto x_data_addr = reinterpret_cast<T *>(inputs[kOne]->addr);
  auto grid_data_addr = reinterpret_cast<T *>(inputs[kTwo]->addr);
  auto dx_data_addr = reinterpret_cast<T *>(outputs[kZero]->addr);
  auto dgrid_data_addr = reinterpret_cast<T *>(outputs[kOne]->addr);
  size_t loop_count = x_shape_[kZero];
  for (size_t i = kZero; i < dx_size_; i++) {
    dx_data_addr[i] = static_cast<T>(kZero);
  }
  auto task = [this, &grad_data_addr, &x_data_addr, &grid_data_addr, &dx_data_addr, &dgrid_data_addr](size_t start,
                                                                                                      size_t end) {
    for (size_t n = start; n < end; n++) {
      ComputeTask<T>(grad_data_addr, x_data_addr, grid_data_addr, dx_data_addr, dgrid_data_addr, n);
    }
  };
  if (grid_size_ * sizeof(T) < kDataSizeThreshold) {
    task(kZero, loop_count);
  } else {
    CPUKernelUtils::ParallelFor(task, loop_count);
  }
}

template <typename T>
T GridSampler3DGradCpuKernelMod::grid_sampler_compute_source_index_set_grad(T coord, size_t size,
                                                                            const std::string &padding_mode,
                                                                            bool align_corners, T *grad_x) {
  T grad_clip, grad_refl;
  if (align_corners) {
    *grad_x = static_cast<T>(size - kOne) / kTwo;
    coord = ((coord + kOne) / kTwo) * (size - kOne);
  } else {
    *grad_x = static_cast<T>(size) / kTwo;
    coord = ((coord + kOne) * size - kOne) / kTwo;
  }
  if (padding_mode == "border") {
    coord = clip_coordinates_set_grad(coord, static_cast<int64_t>(size), &grad_clip);
    *grad_x = (*grad_x) * grad_clip;
  } else if (padding_mode == "reflection") {
    if (align_corners) {
      coord = reflect_coordinates_set_grad(coord, static_cast<int64_t>(kZero), kTwo * (size - kOne), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -static_cast<int64_t>(kOne), kTwo * size - kOne, &grad_refl);
    }
    coord = clip_coordinates_set_grad(coord, static_cast<int64_t>(size), &grad_clip);
    *grad_x = (*grad_x) * grad_refl * grad_clip;
  }
  return coord;
}

template <typename T>
T GridSampler3DGradCpuKernelMod::clip_coordinates_set_grad(T x, int64_t clip_limit, T *grad_x) {
  if (x <= static_cast<T>(kZero)) {
    *grad_x = static_cast<T>(kZero);
    return static_cast<T>(kZero);
  } else {
    T max = static_cast<T>(clip_limit - kOne);
    if (x >= max) {
      *grad_x = static_cast<T>(kZero);
      return max;
    } else {
      *grad_x = static_cast<T>(kOne);
      return x;
    }
  }
}

template <typename T>
T GridSampler3DGradCpuKernelMod::reflect_coordinates_set_grad(T x, int64_t twice_low, int64_t twice_high, T *grad_x) {
  if (twice_low == twice_high) {
    *grad_x = static_cast<T>(kZero);
    return static_cast<T>(kZero);
  }
  size_t grad_x_mult_;
  T min = static_cast<T>(twice_low) / kTwo;
  T span = static_cast<T>(twice_high - twice_low) / kTwo;
  x = x - min;
  if (x < static_cast<T>(kZero)) {
    grad_x_mult_ = -kOne;
    x = -x;
  } else {
    grad_x_mult_ = kOne;
  }
  T extra = std::fmod(x, span);
  int64_t flips = static_cast<int64_t>(std::floor(x / span));
  if (flips % kTwo == static_cast<int64_t>(kZero)) {
    *grad_x = static_cast<T>(grad_x_mult_);
    return extra + min;
  } else {
    *grad_x = static_cast<T>(-grad_x_mult_);
    return (span - extra) + min;
  }
}

template <typename T>
void GridSampler3DGradCpuKernelMod::safe_add_3d(T *data, int64_t d, int64_t h, int64_t w, size_t sD, size_t sH,
                                                size_t sW, size_t D, size_t H, size_t W, T delta) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    data[d * sD + h * sH + w * sW] += static_cast<T>(delta);
  }
}

bool GridSampler3DGradCpuKernelMod::within_bounds_3d(int64_t d, int64_t h, int64_t w, size_t D, size_t H, size_t W) {
  int64_t iD = static_cast<int64_t>(D);
  int64_t iH = static_cast<int64_t>(H);
  int64_t iW = static_cast<int64_t>(W);
  return d >= 0 && d < iD && h >= 0 && h < iH && w >= 0 && w < iW;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GridSampler3DGrad, GridSampler3DGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
