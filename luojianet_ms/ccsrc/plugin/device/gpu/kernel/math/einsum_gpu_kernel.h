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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_KERNEL_H_

#include <ctype.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <set>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/math/einsum_helper.h"
namespace luojianet_ms {
namespace kernel {
constexpr int IDX_NAME = 0;
constexpr int IDX_INP_SHAPE = 1;
constexpr int IDX_PARAM = 2;
constexpr int IDX_OUT_SHAPE = 3;
template <typename T>
class EinsumGpuKernelMod : public NativeGpuKernelMod {
 public:
  EinsumGpuKernelMod() { ResetResource(); }
  ~EinsumGpuKernelMod() = default;
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  // workspace[2] : res; workspace[1]:src workspace[2]: dst
  void RunSingleOpProcess(const OpStruct &op_info, T *src_ptr, T *dst_ptr, void *stream_ptr) {
    auto name = std::get<IDX_NAME>(op_info);
    auto inp_shape = std::get<IDX_INP_SHAPE>(op_info);
    auto op_param = std::get<IDX_PARAM>(op_info);
    func_helper_.SingleElementProcess(name, src_ptr, dst_ptr, inp_shape, op_param, stream_ptr);
  }
  void RunSingleOpVecProcess(T *input_ptr, const std::vector<OpStruct> &op_info_vec, void *stream_ptr, T **src_ptr,
                             T **dst_ptr) {
    RunSingleOpProcess(op_info_vec[0], input_ptr, (*dst_ptr), stream_ptr);
    T *temp = (*src_ptr);
    (*src_ptr) = (*dst_ptr);
    (*dst_ptr) = temp;
    for (size_t idx = 1; idx < op_info_vec.size(); ++idx) {
      RunSingleOpProcess(op_info_vec[idx], (*src_ptr), (*dst_ptr), stream_ptr);
      temp = (*src_ptr);
      (*src_ptr) = (*dst_ptr);
      (*dst_ptr) = temp;
    }
  }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *src_ptr = GetDeviceAddress<T>(workspace, 0);
    T *dst_ptr = GetDeviceAddress<T>(workspace, 1);
    T *res_ptr = GetDeviceAddress<T>(workspace, 2);
    std::vector<void *> workspace_ptr_list(workspace.size());
    for (size_t idx = 0; idx < workspace.size(); ++idx) {
      workspace_ptr_list[idx] = GetDeviceAddress<void>(workspace, idx);
    }
    func_helper_.Init(workspace_ptr_list, type_id_);

    T *input_ptr = GetDeviceAddress<T>(inputs, 0);
    RunSingleOpVecProcess(input_ptr, single_op_[0], stream_ptr, &src_ptr, &dst_ptr);
    auto back_shape = std::get<IDX_OUT_SHAPE>(single_op_[0].back());
    size_t size = func_helper_.GetShapeSize(back_shape);
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(res_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", cudaMemcpyAsync failed.");

    size_t count = 0;
    for (size_t idx = 1; idx < single_op_.size(); ++idx) {
      input_ptr = GetDeviceAddress<T>(inputs, idx);
      RunSingleOpVecProcess(input_ptr, single_op_[idx], stream_ptr, &src_ptr, &dst_ptr);
      auto name = std::get<IDX_NAME>(res_op_[count]);
      auto shape_a = std::get<IDX_INP_SHAPE>(res_op_[count]);
      auto shape_b = std::get<IDX_PARAM>(res_op_[count]);
      auto shape_c = std::get<IDX_OUT_SHAPE>(res_op_[count]);

      func_helper_.TwoElementProcess(name, res_ptr, src_ptr, dst_ptr, shape_a, shape_b, shape_c, stream_ptr);
      T *temp = res_ptr;
      res_ptr = dst_ptr;
      dst_ptr = temp;
      ++count;
      if (count >= res_op_.size()) {
        break;
      }
      name = std::get<IDX_NAME>(res_op_[count]);
      while (single_op_func_.count(name) != 0) {
        shape_a = std::get<IDX_INP_SHAPE>(res_op_[count]);
        shape_b = std::get<IDX_PARAM>(res_op_[count]);
        func_helper_.SingleElementProcess(name, res_ptr, dst_ptr, shape_a, shape_b, stream_ptr);
        temp = res_ptr;
        res_ptr = dst_ptr;
        dst_ptr = temp;
        ++count;
        if (count == res_op_.size()) {
          break;
        }
        name = std::get<IDX_NAME>(res_op_[count]);
      }
    }
    T *out_ptr = GetDeviceAddress<T>(outputs, 0);
    size = output_size_list_[0];
    CHECK_CUDA_RET_WITH_EXCEPT(
      kernel_node_,
      cudaMemcpyAsync(out_ptr, res_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
      "For " + node_name_ + ", cudaMemcpyAsync failed.");
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    auto node_name = common::AnfAlgo::GetCNodeName(kernel_node);
    node_name_ = node_name;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num < 1) {
      MS_LOG(ERROR) << "For " << node_name << ", input number can not be less than 1, but got " << input_num;
      return false;
    }
    type_id_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
    for (size_t idx = 0; idx < input_num; ++idx) {
      TypeId cur_type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, idx);
      if (cur_type_id != type_id_) {
        MS_LOG(ERROR) << "For " << node_name << ", input types should be the same, but it does not.";
        return false;
      }
      std::vector<size_t> in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, idx);
      input_shapes_.push_back(in_shape);
    }
    std::string equation = GetAttr<std::string>(kernel_node, "equation");
    single_op_ = std::vector<std::vector<OpStruct>>(input_shapes_.size());
    bool flag = func_helper_.Preprocess(equation, node_name, input_shapes_, &out_shape_, &single_op_, &res_op_);
    if (!flag) {
      return false;
    }
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_shapes_.clear();
    out_shape_.clear();
    single_op_.clear();
    res_op_.clear();
  }

 protected:
  void InitSizeLists() override {
    size_t work_size = 0;
    size_t shape_size = 0;
    // if (T == float16) { reduce_sum_work_size = size * 2; } else { reduce_sum_work_size = size; }
    size_t mul_val = (type_id_ == kNumberTypeFloat16) ? 2 : 1;
    size_t reduce_sum_wrok_size = 0;
    for (auto &op_vec : single_op_) {
      auto &inp_shape = std::get<IDX_INP_SHAPE>(op_vec[0]);
      size_t size = func_helper_.GetShapeSize(inp_shape);
      work_size = work_size > size ? work_size : size;
      shape_size = inp_shape.size() > shape_size ? inp_shape.size() : shape_size;
      for (auto &cur_op : op_vec) {
        auto name = std::get<IDX_NAME>(cur_op);
        auto shape = std::get<IDX_INP_SHAPE>(cur_op);
        size = func_helper_.GetShapeSize(shape);
        if (name == "ReduceSum") {
          reduce_sum_wrok_size = reduce_sum_wrok_size > size * mul_val ? reduce_sum_wrok_size : size * mul_val;
        }
        work_size = work_size > size ? work_size : size;
        shape_size = shape.size() > shape_size ? shape.size() : shape_size;
      }
    }
    for (auto &op_info : res_op_) {
      auto name = std::get<IDX_NAME>(op_info);
      auto shape = std::get<IDX_OUT_SHAPE>(op_info);
      size_t size = func_helper_.GetShapeSize(shape);
      if (name == "ReduceSum") {
        reduce_sum_wrok_size = reduce_sum_wrok_size > size * mul_val ? reduce_sum_wrok_size : size * mul_val;
      }
      work_size = work_size > size ? work_size : size;
      shape_size = shape.size() > shape_size ? shape.size() : shape_size;
    }
    workspace_size_list_.emplace_back(work_size);
    workspace_size_list_.emplace_back(work_size);
    workspace_size_list_.emplace_back(work_size);
    if (reduce_sum_wrok_size > 0) {
      workspace_size_list_.emplace_back(reduce_sum_wrok_size);
    }
    workspace_size_list_.emplace_back(shape_size * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_size * sizeof(size_t));
    workspace_size_list_.emplace_back(shape_size * sizeof(size_t));
    size_t output_size = func_helper_.GetShapeSize(out_shape_);
    output_size_list_.push_back(output_size);
  }

 private:
  EinsumHelper<T> func_helper_;
  std::string node_name_;
  TypeId type_id_;
  std::vector<std::vector<size_t>> input_shapes_;
  std::vector<size_t> out_shape_;
  std::vector<std::vector<OpStruct>> single_op_;
  std::vector<OpStruct> res_op_;
  std::set<std::string> single_op_func_ = {"ReduceSum", "Diagonal", "Transpose"};
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_EINSUM_KERNEL_H_
