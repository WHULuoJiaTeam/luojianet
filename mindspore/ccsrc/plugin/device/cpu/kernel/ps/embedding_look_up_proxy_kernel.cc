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

#include "plugin/device/cpu/kernel/ps/embedding_look_up_proxy_kernel.h"
#include <vector>
#include <algorithm>
#include "ps/worker.h"
#include "ps/util.h"

namespace mindspore {
namespace kernel {
namespace ps {
constexpr size_t kEmbeddingLookUpProxyInputsNum = 2;
constexpr size_t kEmbeddingLookUpProxyOutputsNum = 1;

void EmbeddingLookUpProxyKernel::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  EmbeddingLookUpCpuKernelMod::InitKernel(kernel_node);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto indices_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  auto output_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  size_t axis = kShape2dDims - input_shape.size();
  if (input_shape.empty() || input_shape.size() > kShape2dDims) {
    MS_LOG(EXCEPTION) << "Input shape should not empty or greater than " << kShape2dDims << "-D, but got "
                      << input_shape.size();
  }

  for (auto dim : input_shape) {
    input_dims_ *= dim;
  }
  if (input_dims_ * sizeof(float) > INT_MAX) {
    MS_LOG(EXCEPTION) << "PS mode embedding lookup max embedding table size is " << INT_MAX << ", current shape "
                      << input_shape << " is too large.";
  }

  if (mindspore::ps::PSContext::instance()->is_worker()) {
    key_ = common::AnfAlgo::GetNodeAttr<size_t>(kernel_node, kAttrPsKey);
  }
  std::vector<float> values;
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(values),
                       [](size_t dim) -> float { return SizeToFloat(dim); });
  (void)std::transform(indices_shape.begin(), indices_shape.end(), std::back_inserter(values),
                       [](size_t dim) -> float { return SizeToFloat(dim); });
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(values),
                       [](size_t dim) -> float { return SizeToFloat(dim); });
  MS_LOG(INFO) << "Init embedding lookup proxy kernel, input shape:" << input_shape
               << ", indices_shape:" << indices_shape << ", output_shape:" << output_shape;
  if (mindspore::ps::PSContext::instance()->is_worker()) {
    mindspore::ps::Worker::GetInstance().AddEmbeddingTable(key_, input_shape[axis]);
    mindspore::ps::ParamInitInfoMessage info;
    if (!mindspore::ps::Worker::GetInstance().InitPSEmbeddingTable(key_, input_shape, indices_shape, output_shape,
                                                                   info)) {
      MS_LOG(EXCEPTION) << "InitPSEmbeddingTable failed.";
    }
  }
}

bool EmbeddingLookUpProxyKernel::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kEmbeddingLookUpProxyInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kEmbeddingLookUpProxyOutputsNum, kernel_name_);
  auto indices_addr = reinterpret_cast<int *>(inputs[1]->addr);
  auto output_addr = reinterpret_cast<float *>(outputs[0]->addr);
  size_t input_size = inputs[1]->size;
  size_t output_size = outputs[0]->size;

  size_t size = input_size / sizeof(int);
  std::vector<int> lookup_ids(size, 0);
  std::vector<float> lookup_result(output_size / sizeof(float), 0);
  auto ret = memcpy_s(lookup_ids.data(), lookup_ids.size() * sizeof(int), indices_addr, input_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "Lookup id memcpy failed.";
  }
  if (!mindspore::ps::Worker::GetInstance().DoPSEmbeddingLookup(key_, lookup_ids, &lookup_result,
                                                                mindspore::ps::kEmbeddingLookupCmd)) {
    MS_LOG(EXCEPTION) << "DoPSEmbeddingLookup failed.";
  }

  auto ret2 = memcpy_s(output_addr, outputs[0]->size, lookup_result.data(), output_size);
  if (ret2 != EOK) {
    MS_LOG(EXCEPTION) << "Lookup result memcpy failed.";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EmbeddingLookupProxy, EmbeddingLookUpProxyKernel);
}  // namespace ps
}  // namespace kernel
}  // namespace mindspore
