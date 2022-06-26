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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_grad_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace luojianet_ms {
namespace kernel {
constexpr int kMaxShapeSize = 4;
template <typename T>
class BroadcastOpGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastOpGradGpuKernelMod() { ResetResource(); }
  ~BroadcastOpGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x1 = GetDeviceAddress<T>(inputs, 0);
    T *x2 = GetDeviceAddress<T>(inputs, 1);
    T *dy = GetDeviceAddress<T>(inputs, 2);
    T *dx1 = GetDeviceAddress<T>(outputs, 0);
    T *dx2 = GetDeviceAddress<T>(outputs, 1);

    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(dx1, 0, outputs[0]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemSet Failed");
    CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                               cudaMemsetAsync(dx2, 0, outputs[1]->size, reinterpret_cast<cudaStream_t>(stream_ptr)),
                               "cudaMemSet Failed");
    if (need_broadcast_) {
      BroadcastGrad(x1_shape_[0], x1_shape_[1], x1_shape_[2], x1_shape_[3], x2_shape_[0], x2_shape_[1], x2_shape_[2],
                    x2_shape_[3], dy_shape_[0], dy_shape_[1], dy_shape_[2], dy_shape_[3], grad_x_, grad_y_, op_type_,
                    x1, x2, dy, dx1, dx2, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      NoBroadcastGrad(output_num_, grad_x_, grad_y_, op_type_, x1, x2, dy, dx1, dx2,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    GetOpType(kernel_node);
    auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape3 = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    is_null_input_ = CHECK_SHAPE_NULL(shape1, kernel_name_, "input_1") ||
                     CHECK_SHAPE_NULL(shape2, kernel_name_, "input_2") ||
                     CHECK_SHAPE_NULL(shape3, kernel_name_, "input_3");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(shape1, shape2);
    if (need_broadcast_ && shape1.size() > kMaxShapeSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than "
                        << kMaxShapeSize << ", but got " << shape1.size();
    }

    for (size_t i = 0; i < shape3.size(); i++) {
      if (need_broadcast_) {
        dy_shape_[i] = shape3[i];
      }
      output_num_ *= shape3[i];
    }
    int x1_offset = shape3.size() - shape1.size();
    for (size_t i = 0; i < shape1.size(); i++) {
      if (need_broadcast_) {
        if ((i + x1_offset) >= 0 && (i + x1_offset) < kMaxShapeSize) {
          x1_shape_[i + x1_offset] = shape1[i];
        } else {
          auto index = i + x1_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than "
                            << kMaxShapeSize << ", but got " << (index + 1);
        }
      }
      input1_num_ *= shape1[i];
    }
    int x2_offset = shape3.size() - shape2.size();
    for (size_t i = 0; i < shape2.size(); i++) {
      if (need_broadcast_) {
        if ((i + x2_offset) >= 0 && (i + x2_offset) < kMaxShapeSize) {
          x2_shape_[i + x2_offset] = shape2[i];
        } else {
          auto index = i + x2_offset;
          MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than "
                            << kMaxShapeSize << ", but got " << (index + 1);
        }
      }
      input2_num_ *= shape2[i];
    }

    grad_x_ = GetAttr<bool>(kernel_node, "grad_x");
    grad_y_ = GetAttr<bool>(kernel_node, "grad_y");

    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    op_type_ = BROADCAST_GRAD_TYPE_INVALID;
    need_broadcast_ = false;
    is_null_input_ = false;
    input1_num_ = 1;
    input2_num_ = 1;
    output_num_ = 1;
    std::fill(x1_shape_, x1_shape_ + kMaxShapeSize, 1);
    std::fill(x2_shape_, x2_shape_ + kMaxShapeSize, 1);
    std::fill(dy_shape_, dy_shape_ + kMaxShapeSize, 1);
    grad_x_ = false;
    grad_y_ = false;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { return; }
  void InitSizeLists() override {
    input_size_list_.push_back(input1_num_ * sizeof(T));
    input_size_list_.push_back(input2_num_ * sizeof(T));
    input_size_list_.push_back(output_num_ * sizeof(T));
    output_size_list_.push_back(input1_num_ * sizeof(T));
    output_size_list_.push_back(input2_num_ * sizeof(T));
  }

 private:
  void GetOpType(const CNodePtr &kernel_node) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);

    static std::map<std::string, BroadcastGradOpType> kBroadcastTypeMap = {
      {"MaximumGrad", BROADCAST_GRAD_TYPE_MAXIMUM},
      {"MinimumGrad", BROADCAST_GRAD_TYPE_MINIMUM},
    };

    auto iter = kBroadcastTypeMap.find(kernel_name);
    if (iter == kBroadcastTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << ", only support these types: MaximumGrad or MinimumGrad currently, but got " << kernel_name;
    } else {
      op_type_ = iter->second;
    }
  }

  BroadcastGradOpType op_type_;
  bool need_broadcast_;
  bool is_null_input_;
  size_t input1_num_;
  size_t input2_num_;
  size_t output_num_;
  size_t x1_shape_[kMaxShapeSize] = {1, 1, 1, 1};
  size_t x2_shape_[kMaxShapeSize] = {1, 1, 1, 1};
  size_t dy_shape_[kMaxShapeSize] = {1, 1, 1, 1};
  bool grad_x_;
  bool grad_y_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_BROADCAST_GRAD_GPU_KERNEL_H_
