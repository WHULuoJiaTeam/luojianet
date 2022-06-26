/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_COMPLEX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_COMPLEX_GPU_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <complex>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr int MAX_DIMS = 7;
template <typename T, typename S, typename G>
class BroadcastComplexOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  BroadcastComplexOpGpuKernelMod() { ResetResource(); }
  ~BroadcastComplexOpGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *lhs = GetDeviceAddress<T>(inputs, 0);
    S *rhs = GetDeviceAddress<S>(inputs, 1);

    G *output = GetDeviceAddress<G>(outputs, 0);
    if (need_broadcast_) {
      BroadcastComplexArith(lhs_shape_, rhs_shape_, output_shape_, op_type_, lhs, rhs, output,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseComplexArith(output_num_, op_type_, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    GetOpType(kernel_node);

    auto shape1 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 0);
    auto shape2 = AnfAlgo::GetInputDeviceShapeAdaptively(kernel_node, 1);
    auto shape3 = AnfAlgo::GetOutputDeviceShapeAdaptively(kernel_node, 0);
    need_broadcast_ = common::AnfAlgo::IsTensorBroadcast(shape1, shape2);
    if (need_broadcast_ && shape1.size() > MAX_DIMS) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than " << MAX_DIMS
                        << ", but got " << shape1.size();
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    for (size_t i = 0; i < shape3.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = shape3[i];
      }
      output_num_ *= shape3[i];
    }
    int lhs_offset = shape3.size() - shape1.size();
    for (size_t j = 0; j < shape1.size(); j++) {
      if (need_broadcast_) {
        lhs_shape_[j + lhs_offset] = shape1[j];
      }
      input1_num_ *= shape1[j];
    }
    int rhs_offset = shape3.size() - shape2.size();
    for (size_t k = 0; k < shape2.size(); k++) {
      if (need_broadcast_) {
        rhs_shape_[k + rhs_offset] = shape2[k];
      }
      input2_num_ *= shape2[k];
    }

    InitSizeLists();
    return true;
  }
  void ResetResource() noexcept override {
    op_type_ = BROADCAST_TYPE_INVALID;
    need_broadcast_ = false;
    input1_num_ = 1;
    input2_num_ = 1;
    output_num_ = 1;
    lhs_shape_.clear();
    rhs_shape_.clear();
    output_shape_.clear();
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override { return; }
  void InitSizeLists() override {
    input_size_list_.push_back(input1_num_ * sizeof(T));
    input_size_list_.push_back(input2_num_ * sizeof(S));
    output_size_list_.push_back(output_num_ * sizeof(G));
  }

 private:
  void GetOpType(const CNodePtr &kernel_node) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    static std::map<std::string, BroadcastOpType> kBroadcastArithmetricTypeMap = {
      {"RealDiv", BROADCAST_TYPE_REALDIV}, {"Mul", BROADCAST_TYPE_MUL}, {"Sub", BROADCAST_TYPE_SUB},
      {"Add", BROADCAST_TYPE_ADD},         {"Div", BROADCAST_TYPE_DIV}, {"Complex", BROADCAST_TYPE_COMPLEX}};

    auto iter = kBroadcastArithmetricTypeMap.find(kernel_name);
    if (iter != kBroadcastArithmetricTypeMap.end()) {
      op_type_ = iter->second;
      return;
    }

    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", only support these types: RealDiv, Mul, Sub, Add, Div or Complex currently, but got "
                      << kernel_name;
  }

  BroadcastOpType op_type_;
  bool need_broadcast_;
  size_t input1_num_;
  size_t input2_num_;
  size_t output_num_;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
};  // namespace kernel
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_BROADCAST_COMPLEX_GPU_KERNEL_H_
