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

#include "plugin/device/cpu/kernel/mkldnn/lstm_grad_cpu_kernel.h"
#include <cstring>
#include <string>
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace luojianet_ms {
namespace kernel {
namespace {
constexpr size_t kLstmGradInputsNum = 11;
constexpr size_t kLstmGradOutputsNum = 4;
constexpr int kMaxLSTMLayer = 100;
constexpr int kInputWorkSpaceIndex = 10;
constexpr int kInputWeightIndex = 3;
constexpr int kOutputWeightIndex = 3;

constexpr int kSrcLayerIdx = 0;
constexpr int kSrcIterIdx = 1;
constexpr int kSrcIterCIdx = 2;
constexpr int kDstLayerIdx = 4;
constexpr int kDstIterIdx = 5;
constexpr int kDstIterCIdx = 6;
constexpr int kDiffDstLayerIdx = 7;
constexpr int kDiffDstIterIdx = 8;
constexpr int kDiffDstIterCIdx = 9;
constexpr int kWorkspaceIdx = 10;

using tag = dnnl::memory::format_tag;
using dim = dnnl::memory::dims;
using dt = dnnl::memory::data_type;
}  // namespace

void LSTMGradCpuKernelMod::InitInputOutputSize(const CNodePtr &kernel_node) {
  NativeCpuKernelMod::InitInputOutputSize(kernel_node);
  input_size_list_[kInputWorkSpaceIndex] = reserve_size_;
}

void LSTMGradCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  CheckParam(kernel_node);
  auto eng = engine_;
  dnnl::rnn_direction direction = dnnl::rnn_direction::unidirectional;
  if (bidirectional_) {
    direction = dnnl::rnn_direction::bidirectional_concat;
  }
  dim src_dims = {seq_len_, batch_size_, input_size_};
  dim src_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim src_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  weights_dims_ = {num_layers_, num_directions_, input_size_, 4, hidden_size_};
  weights_h_dims_ = {num_layers_, num_directions_, hidden_size_, 4, hidden_size_};
  bias_dims_ = {num_layers_, num_directions_, 4, hidden_size_};
  dim dst_dims = {seq_len_, batch_size_, static_cast<int64_t>(hidden_size_) * num_directions_};
  dim dst_h_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dim dst_c_dims = {num_layers_, num_directions_, batch_size_, hidden_size_};
  dnnl::memory::desc src_desc = formatted_md(src_dims, tag::tnc);
  dnnl::memory::desc src_h_desc = formatted_md(src_h_dims, tag::ldnc);
  dnnl::memory::desc src_c_desc = formatted_md(src_c_dims, tag::ldnc);
  dnnl::memory::desc bias_desc = formatted_md(bias_dims_, tag::ldgo);
  dnnl::memory::desc dst_desc = formatted_md(dst_dims, tag::tnc);
  dnnl::memory::desc dst_h_desc = formatted_md(dst_h_dims, tag::ldnc);
  dnnl::memory::desc dst_c_desc = formatted_md(dst_c_dims, tag::ldnc);
  auto weights_desc = formatted_md(weights_dims_, tag::any);
  auto weights_h_desc = formatted_md(weights_h_dims_, tag::any);

  auto forward_desc = CreatePrimitive<dnnl::lstm_forward::desc>(dnnl::prop_kind::forward_training, direction, src_desc,
                                                                src_h_desc, src_c_desc, weights_desc, weights_h_desc,
                                                                bias_desc, dst_desc, dst_h_desc, dst_c_desc);
  auto prim_forward_desc = CreateDesc<dnnl::lstm_forward::primitive_desc>(*forward_desc, eng);
  auto backward_desc = CreatePrimitive<dnnl::lstm_backward::desc>(
    dnnl::prop_kind::backward, direction, src_desc, src_h_desc, src_c_desc, weights_desc, weights_h_desc, bias_desc,
    dst_desc, dst_h_desc, dst_c_desc, src_desc, src_h_desc, src_c_desc, weights_desc, weights_h_desc, bias_desc,
    dst_desc, dst_h_desc, dst_c_desc);
  prim_backward_desc_ = CreateDesc<dnnl::lstm_backward::primitive_desc>(*backward_desc, eng, prim_forward_desc);
  primitive_ = CreatePrimitive<dnnl::lstm_backward>(prim_backward_desc_);
  auto wksp_desc = GetWorkspaceDesc(prim_forward_desc);
  reserve_size_ = GetSize(wksp_desc);
  AddArgument(DNNL_ARG_WORKSPACE, wksp_desc);
  AddArgumentOp(src_desc, src_h_desc, src_c_desc, bias_desc, dst_desc, dst_h_desc, dst_c_desc);

  // construct fw memory
  weights_layer_desc_ = GetWeightsLayerDesc(prim_backward_desc_);
  weights_iter_desc_ = GetWeightsIterDesc(prim_backward_desc_);
  bias_desc_ = GetBiasDesc(prim_backward_desc_);
  auto weights_mem_desc = CreateDesc<dnnl::memory::desc>(weights_dims_, dt::f32, tag::ldgoi);
  auto weights_h_mem_desc = CreateDesc<dnnl::memory::desc>(weights_h_dims_, dt::f32, tag::ldgoi);
  user_weights_memory_ = CreateDesc<dnnl::memory>(weights_mem_desc, eng);
  user_weights_h_memory_ = CreateDesc<dnnl::memory>(weights_h_mem_desc, eng);
  weights_memory_ = CreateDesc<dnnl::memory>(weights_layer_desc_, eng);
  weights_h_memory_ = CreateDesc<dnnl::memory>(weights_iter_desc_, eng);
  bias_memory_ = CreateDesc<dnnl::memory>(bias_desc_, eng);

  // construct bw memory
  diff_weights_layer_desc_ = GetDiffWeightsLayerDesc(prim_backward_desc_);
  diff_weights_iter_desc_ = GetDiffWeightsIterDesc(prim_backward_desc_);
  diff_bias_desc_ = GetDiffBiasDesc(prim_backward_desc_);
  diff_weights_memory_ = CreateDesc<dnnl::memory>(diff_weights_layer_desc_, eng);
  diff_weights_h_memory_ = CreateDesc<dnnl::memory>(diff_weights_iter_desc_, eng);
  diff_bias_memory_ = CreateDesc<dnnl::memory>(diff_bias_desc_, eng);
  user_diff_weights_memory_ = CreateDesc<dnnl::memory>(weights_mem_desc, eng);
  user_diff_weights_h_memory_ = CreateDesc<dnnl::memory>(weights_h_mem_desc, eng);
}

void LSTMGradCpuKernelMod::AddArgumentOp(const dnnl::memory::desc &src_desc, const dnnl::memory::desc &src_h_desc,
                                         const dnnl::memory::desc &src_c_desc, const dnnl::memory::desc &bias_desc,
                                         const dnnl::memory::desc &dst_desc, const dnnl::memory::desc &dst_h_desc,
                                         const dnnl::memory::desc &dst_c_desc) {
  AddArgument(DNNL_ARG_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_SRC_ITER_C, src_c_desc);
  AddArgument(DNNL_ARG_WEIGHTS_LAYER, weights_layer_desc_);
  AddArgument(DNNL_ARG_WEIGHTS_ITER, weights_iter_desc_);
  AddArgument(DNNL_ARG_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DST_ITER_C, dst_c_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_LAYER, src_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_ITER, src_h_desc);
  AddArgument(DNNL_ARG_DIFF_SRC_ITER_C, src_c_desc);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_LAYER, diff_weights_layer_desc_);
  AddArgument(DNNL_ARG_DIFF_WEIGHTS_ITER, diff_weights_iter_desc_);
  AddArgument(DNNL_ARG_DIFF_BIAS, bias_desc);
  AddArgument(DNNL_ARG_DIFF_DST_LAYER, dst_desc);
  AddArgument(DNNL_ARG_DIFF_DST_ITER, dst_h_desc);
  AddArgument(DNNL_ARG_DIFF_DST_ITER_C, dst_c_desc);
}

void LSTMGradCpuKernelMod::CheckParam(const CNodePtr &kernel_node) {
  std::vector<size_t> src_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  std::vector<size_t> src_h_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
  std::vector<size_t> src_c_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
  if (src_shape.size() != 3 || src_h_shape.size() != 3 || src_c_shape.size() != 3) {
    MS_LOG(EXCEPTION) << "Lstm only support 3-D input!";
  }
  bidirectional_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "bidirectional");
  input_size_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "input_size");
  hidden_size_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "hidden_size");
  num_layers_ = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "num_layers");
  has_bias_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "has_bias");
  batch_size_ = SizeToInt(src_shape[1]);
  seq_len_ = SizeToInt(src_shape[0]);
  num_directions_ = 1;
  if (bidirectional_) {
    num_directions_ = 2;
  }
  const int64_t gate_size = 4 * hidden_size_;
  if (num_layers_ <= 0) {
    MS_LOG(EXCEPTION) << "Layers must be greater than zero!";
  }
  if (num_layers_ > kMaxLSTMLayer) {
    MS_LOG(EXCEPTION) << "Layers must be lower than 100!";
  }
  for (int64_t i = 0; i < num_layers_; ++i) {
    weight_size_ += gate_size * (i == 0 ? input_size_ : hidden_size_ * num_directions_);
    weight_h_size_ += gate_size * hidden_size_;
  }
  weight_size_ = weight_size_ * num_directions_;
  weight_h_size_ = weight_h_size_ * num_directions_;
  if (num_directions_ * num_layers_ != SizeToLong(src_h_shape[0])) {
    MS_LOG(EXCEPTION) << "Error iteration shape!";
  }
}

void LSTMGradCpuKernelMod::SetArgumentHandleOp(const std::vector<kernel::AddressPtr> &inputs,
                                               const std::vector<kernel::AddressPtr> &outputs) {
  SetArgumentHandle(DNNL_ARG_SRC_LAYER, inputs[kSrcLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER, inputs[kSrcIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_SRC_ITER_C, inputs[kSrcIterCIdx]->addr);
  SetArgumentHandle(DNNL_ARG_WEIGHTS_LAYER, GetDataHandle(weights_memory_));
  SetArgumentHandle(DNNL_ARG_WEIGHTS_ITER, GetDataHandle(weights_h_memory_));
  SetArgumentHandle(DNNL_ARG_BIAS, GetDataHandle(bias_memory_));
  SetArgumentHandle(DNNL_ARG_DST_LAYER, inputs[kDstLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER, inputs[kDstIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DST_ITER_C, inputs[kDstIterCIdx]->addr);
  SetArgumentHandle(DNNL_ARG_WORKSPACE, inputs[kWorkspaceIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_LAYER, outputs[kSrcLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_ITER, outputs[kSrcIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_SRC_ITER_C, outputs[kSrcIterCIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_LAYER, GetDataHandle(diff_weights_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_WEIGHTS_ITER, GetDataHandle(diff_weights_h_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_BIAS, GetDataHandle(diff_bias_memory_));
  SetArgumentHandle(DNNL_ARG_DIFF_DST_LAYER, inputs[kDiffDstLayerIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST_ITER, inputs[kDiffDstIterIdx]->addr);
  SetArgumentHandle(DNNL_ARG_DIFF_DST_ITER_C, inputs[kDiffDstIterCIdx]->addr);
}

void LSTMGradCpuKernelMod::ResetMemory(const dnnl::memory &mem, const string name) const {
  auto dst_ptr = GetDataHandle(mem);
  auto mem_desc = GetMemDesc(mem);
  auto size = GetSize(mem_desc);
  if (memset_s(dst_ptr, size, 0, size)) {
    MS_LOG(EXCEPTION) << name << " memset error";
  }
}

bool LSTMGradCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                  const std::vector<kernel::AddressPtr> &,
                                  const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLstmGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLstmGradOutputsNum, kernel_name_);
  SetDataHandle(user_weights_memory_, inputs[kInputWeightIndex]->addr);
  SetDataHandle(user_weights_h_memory_, reinterpret_cast<float *>(inputs[kInputWeightIndex]->addr) + weight_size_);
  Reorder(&user_weights_memory_, &weights_memory_);
  Reorder(&user_weights_h_memory_, &weights_h_memory_);
  if (has_bias_) {
    SetDataHandle(bias_memory_,
                  reinterpret_cast<float *>(inputs[kInputWeightIndex]->addr) + weight_size_ + weight_h_size_);
  } else {
    auto dst_ptr = GetDataHandle(bias_memory_);
    auto size = GetSize(bias_desc_);
    if (memset_s(dst_ptr, size, 0, size)) {
      MS_LOG(EXCEPTION) << "Bias memset error";
    }
  }

  SetDataHandle(user_diff_weights_memory_, outputs[kOutputWeightIndex]->addr);
  SetDataHandle(user_diff_weights_h_memory_,
                reinterpret_cast<float *>(outputs[kOutputWeightIndex]->addr) + weight_size_);
  ResetMemory(user_diff_weights_memory_, "user weights grad");
  ResetMemory(user_diff_weights_h_memory_, "user weights iter grad");
  ResetMemory(diff_weights_memory_, "weights grad");
  ResetMemory(diff_weights_h_memory_, "weights iter grad");
  if (has_bias_) {
    SetDataHandle(diff_bias_memory_,
                  reinterpret_cast<float *>(outputs[kOutputWeightIndex]->addr) + weight_size_ + weight_h_size_);
  }
  auto dst_ptr = GetDataHandle(diff_bias_memory_);
  auto size = GetSize(diff_bias_desc_);
  if (memset_s(dst_ptr, size, 0, size)) {
    MS_LOG(EXCEPTION) << "Bias grad memset error";
  }
  SetArgumentHandleOp(inputs, outputs);
  ExecutePrimitive();
  Reorder(&diff_weights_memory_, &user_diff_weights_memory_);
  Reorder(&diff_weights_h_memory_, &user_diff_weights_h_memory_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LSTMGrad, LSTMGradCpuKernelMod);
}  // namespace kernel
}  // namespace luojianet_ms
