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

#include "src/runtime/runtime_pass.h"
#include "nnacl/conv_parameter.h"

namespace {
const constexpr int kMaxDepth = 2048;
}

namespace luojianet_ms::lite {
void Nc4hw4PassReplace(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors, size_t index) {
  kernel::LiteKernel *conv_kernel = kernels->at(index);
  kernel::LiteKernel *transpose_kernel = conv_kernel->out_kernels().front();
  kernel::LiteKernel *c4_kernel = transpose_kernel->out_kernels().front();
  kernel::LiteKernel *transpose2_kernel = c4_kernel->out_kernels().front();
  std::vector<kernel::LiteKernel *> end_kernels = transpose2_kernel->out_kernels();

  /* tensor */
  {
    /* transpose_kernel */
    Tensor *transpose_param_tensor = transpose_kernel->in_tensors().at(1);
    (void)VectorSetNull(tensors, transpose_param_tensor);
    delete transpose_param_tensor;
    transpose_param_tensor = nullptr;

    Tensor *conv_out_tensor = conv_kernel->out_tensors().front();
    conv_out_tensor->set_format(NC4HW4);
    Tensor *c4_input_tensor = c4_kernel->in_tensors().front();
    c4_kernel->set_in_tensor(conv_out_tensor, 0);
    (void)VectorSetNull(tensors, c4_input_tensor);
    delete c4_input_tensor;
    c4_input_tensor = nullptr;
  }
  {
    /* transpose2_kernel */
    Tensor *transpose_param_tensor = transpose2_kernel->in_tensors().at(1);
    (void)VectorSetNull(tensors, transpose_param_tensor);
    delete transpose_param_tensor;
    transpose_param_tensor = nullptr;

    Tensor *nwhc_tensor = c4_kernel->out_tensors().front();
    std::vector<int> nhwc_shape = {nwhc_tensor->Batch(), nwhc_tensor->Height(), nwhc_tensor->Width(),
                                   nwhc_tensor->Channel()};
    nwhc_tensor->set_format(NHWC);
    nwhc_tensor->set_shape(nhwc_shape);
    for (auto end : end_kernels) {
      end->set_in_tensor(nwhc_tensor, 0);
    }
    Tensor *trans_out = transpose2_kernel->out_tensors().front();
    (void)VectorSetNull(tensors, trans_out);
    delete trans_out;
    trans_out = nullptr;
  }

  /* kernel */
  (void)VectorErase(kernels, transpose_kernel);
  delete transpose_kernel;
  transpose_kernel = nullptr;
  conv_kernel->set_out_kernels({c4_kernel});
  c4_kernel->set_in_kernels({conv_kernel});

  c4_kernel->set_out_kernels(transpose2_kernel->out_kernels());
  for (auto end : end_kernels) {
    end->set_in_kernels({c4_kernel});
  }
  (void)VectorErase(kernels, transpose2_kernel);
  delete transpose2_kernel;
  transpose2_kernel = nullptr;

  return;
}

bool Nc4hw4PassMatch(const std::vector<kernel::LiteKernel *> *kernels, size_t index) {
  kernel::LiteKernel *start_kernel = kernels->at(index);
  if (IsContain(Nc4hw4FormatOutOpList, start_kernel->type()) == false) {
    return false;
  }
  if (start_kernel->out_kernels().size() != 1) {
    return false;
  }
  MS_CHECK_TRUE_MSG(start_kernel->op_parameter() != nullptr, false, "kernel->op_parameter() is nullptr.");
  if (reinterpret_cast<ConvParameter *>(start_kernel->op_parameter())->group_ != 1) {
    /* conv-depthwise and group-conv */
    return false;
  }

  kernel::LiteKernel *traspose_nhwc2nchw_kernel = start_kernel->out_kernels().front();
  if (traspose_nhwc2nchw_kernel->type() != Nc4hw4FormatTransposeOp) {
    return false;
  }
  if (traspose_nhwc2nchw_kernel->out_kernels().size() != 1) {
    return false;
  }

  kernel::LiteKernel *end_kernel = traspose_nhwc2nchw_kernel->out_kernels().front();
  if (IsContain(Nc4hw4FormatInOpList, end_kernel->type()) == false) {
    return false;
  }
  if (end_kernel->out_kernels().size() != 1) {
    return false;
  }

  kernel::LiteKernel *transpose_nchw2nhwc_kernel = end_kernel->out_kernels().front();
  if (transpose_nchw2nhwc_kernel->type() != Nc4hw4FormatTransposeOp) {
    return false;
  }

  /* double check ops topological sorted in kernel-list */
  auto start_iter = find(kernels->begin(), kernels->end(), start_kernel);
  auto start_index = std::distance(kernels->begin(), start_iter);
  auto traspose_nhwc2nchw_iter = find(kernels->begin(), kernels->end(), traspose_nhwc2nchw_kernel);
  auto traspose_nhwc2nchw_index = std::distance(kernels->begin(), traspose_nhwc2nchw_iter);
  auto end_iter = find(kernels->begin(), kernels->end(), end_kernel);
  auto end_index = std::distance(kernels->begin(), end_iter);
  auto transpose_nchw2nhwc_iter = find(kernels->begin(), kernels->end(), transpose_nchw2nhwc_kernel);
  auto transpose_nchw2nhwc_index = std::distance(kernels->begin(), transpose_nchw2nhwc_iter);
  if (start_index > traspose_nhwc2nchw_index || traspose_nhwc2nchw_index > end_index ||
      end_index > transpose_nchw2nhwc_index) {
    return false;
  }

  return true;
}

bool RuntimePassValid(kernel::SubGraphKernel *subgraph) {
  if (subgraph->desc().arch != kernel::KERNEL_ARCH::kCPU) {
    return false;
  }

#if !defined(ENABLE_ARM64) && !defined(ENABLE_AVX)
  return false;
#endif

  auto kernels = subgraph->nodes();

  for (auto kernel : kernels) {
    MS_CHECK_TRUE_MSG(kernel != nullptr, false, "kernel is nullptr.");
    if (kernel->op_parameter() != nullptr) {
      if (kernel->op_parameter()->quant_type_ == schema::QuantType_AwareTraining ||
          kernel->op_parameter()->quant_type_ == schema::QuantType_PostTraining) {
        return false;
      }
    }
  }
  return true;
}

void Nc4hw4PassAct(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors, int i) {
  if (i > kMaxDepth) {
    MS_LOG(ERROR) << "exceed max depth 2048, i " << i;
    return;
  }
  i++;
  size_t kernel_size = kernels->size();
  size_t index = 0;
  for (; index + 3 < kernel_size; index++) {
    kernel::LiteKernel *kernel = kernels->at(index);

    if (kernel->subgraph_type() != kernel::kNotSubGraph) {
      kernel::SubGraphKernel *subgraph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      std::vector<kernel::LiteKernel *> &particial_nodes = subgraph->nodes();
      Nc4hw4PassAct(&particial_nodes, tensors, i);
    }

    if (Nc4hw4PassMatch(kernels, index)) {
      Nc4hw4PassReplace(kernels, tensors, index);
      index += 1;
    }
    kernel_size = kernels->size();
  }
  return;
}

void ConvNormC4PassActReplace(const kernel::LiteKernel *conv_op, const kernel::LiteKernel *in_op) {
  conv_op->out_tensors().front()->set_format(NC4HW4);
  in_op->in_tensors().front()->set_format(NC4HW4);
}

void ConvNormC4PassActIndex(std::vector<kernel::LiteKernel *> *kernels, size_t index) {
  kernel::LiteKernel *start_kernel = kernels->at(index);
  if (start_kernel->type() != ConvNormC4OpConv2DFusion) {
    return;
  }
  if (start_kernel->out_kernels().size() != 1) {
    return;
  }
  CHECK_NULL_RETURN_VOID(start_kernel->op_parameter());
  if (reinterpret_cast<ConvParameter *>(start_kernel->op_parameter())->group_ != 1) {
    /* conv-depthwise and group-conv */
    return;
  }

  kernel::LiteKernel *after_kernel = start_kernel->out_kernels().front();
  if (after_kernel->type() == ConvNormC4OpActivation) {
    if (after_kernel->out_kernels().size() != 1) {
      return;
    }
    kernel::LiteKernel *end_kernel = after_kernel->out_kernels().front();
    if (end_kernel->type() == ConvNormC4OpInstanceNorm) {
      ConvNormC4PassActReplace(start_kernel, end_kernel);
      return;
    }
    return;
  }

  if (after_kernel->type() == ConvNormC4OpInstanceNorm) {
    ConvNormC4PassActReplace(start_kernel, after_kernel);
    return;
  }

  return;
}

void ConvNormC4PassAct(std::vector<kernel::LiteKernel *> *kernels) {
  size_t kernel_size = kernels->size();
  size_t index = 0;
  for (; index < kernel_size; index++) {
    ConvNormC4PassActIndex(kernels, index);
  }
  return;
}

STATUS DeleteRedundantTrans(std::vector<kernel::LiteKernel *> *kernels) {
  for (auto *pre_kernel : *kernels) {
    if (pre_kernel->subgraph_type() != kernel::kNotSubGraph) {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(pre_kernel);
      auto &partial = sub_graph->nodes();
      if (DeleteRedundantTrans(&partial) != RET_OK) {
        MS_LOG(ERROR) << "DeleteRedundantTrans failed in subgraph.";
        return RET_ERROR;
      }
    }
    if (pre_kernel->type() != schema::PrimitiveType_Transpose) {
      continue;
    }
    if (pre_kernel->in_tensors().size() < 1 || pre_kernel->out_tensors().size() < 1) {
      MS_LOG(ERROR) << "kernel input or output is empty.";
      return RET_ERROR;
    }
    auto pre_kernel_in_tensor_shape = pre_kernel->in_tensors().at(0)->shape();
    auto pre_kernel_out_tensor_shape = pre_kernel->out_tensors().at(0)->shape();
    for (size_t i = 0; i < pre_kernel_out_tensor_shape.size(); i++) {
      if (pre_kernel_out_tensor_shape[i] == -1) {
        MS_LOG(DEBUG) << " input need do resize.";
        return RET_OK;
      }
      if (pre_kernel_out_tensor_shape[i] != pre_kernel_in_tensor_shape[i] && pre_kernel_out_tensor_shape[i] != 1) {
        MS_LOG(DEBUG) << "transpose do not delete.";
        return RET_OK;
      }
    }
    auto post_kernel_size = pre_kernel->out_kernels().size();
    if (post_kernel_size != 1) {
      continue;
    }
    auto post_kernel = pre_kernel->out_kernels().front();
    if (post_kernel->type() != schema::PrimitiveType_Reshape) {
      continue;
    }
    if (pre_kernel->in_kernels().size() != 1) {
      continue;
    }
    auto pre_in_kernel = pre_kernel->in_kernels().front();
    pre_in_kernel->set_out_kernels({post_kernel});
    std::vector<kernel::LiteKernel *> post_in_kernels = {pre_in_kernel};
    if (post_kernel->in_kernels().size() == kInputSize1) {
      post_in_kernels.push_back(post_kernel->in_kernels()[1]);
    }
    post_kernel->set_in_kernels(post_in_kernels);
    post_kernel->set_in_tensor(pre_kernel->in_tensors()[0], 0);
    kernels->erase(find(kernels->begin(), kernels->end(), pre_kernel));
    delete pre_kernel;
  }
  return RET_OK;
}

STATUS RuntimePass(std::vector<kernel::LiteKernel *> *subgraphs, std::vector<Tensor *> *tensors) {
  for (auto subgraph : *subgraphs) {
    auto sub = reinterpret_cast<kernel::SubGraphKernel *>(subgraph);
    if (RuntimePassValid(sub) == false) {
      continue;
    }

    int i = 0;
    auto &kernels = sub->nodes();
    Nc4hw4PassAct(&kernels, tensors, i);
    ConvNormC4PassAct(&kernels);
    auto status = DeleteRedundantTrans(&kernels);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DeleteRedundantTrans failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

STATUS GraphOptimizePass(std::vector<kernel::LiteKernel *> *sub_graphs) {
  for (auto subgraph : *sub_graphs) {
    auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(subgraph);
    if (RuntimePassValid(sub_graph) == false) {
      continue;
    }
    auto &kernels = sub_graph->nodes();
    auto status = DeleteRedundantTrans(&kernels);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "DeleteRedundantTrans failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace luojianet_ms::lite
