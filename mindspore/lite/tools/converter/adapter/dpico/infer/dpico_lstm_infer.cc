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

#include "infer/dpico_lstm_infer.h"
#include <vector>
#include <memory>
#include <iostream>
#include "common/op_enum.h"
#include "utils/log_adapter.h"
#include "common/infer_util.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kHiddenLayerSize = 4;
constexpr size_t kInputSize5 = 5;
constexpr size_t kInputSize6 = 6;
}  // namespace
std::shared_ptr<KernelInterface> DpicoLstmInferCreater() {
  std::shared_ptr<KernelInterface> infer = std::make_shared<DpicoLstmInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "make shared failed, infer is nullptr.";
    return nullptr;
  }
  return infer;
}
Status DpicoLstmInterface::Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
                                 const schema::Primitive *primitive, const kernel::Kernel *kernel) {
  auto status = dpico::CheckCustomInputOutput(inputs, outputs, primitive);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Check custom input output failed.";
    return kLiteError;
  }
  if (inputs->size() != kInputSize5 && inputs->size() != kInputSize6) {
    MS_LOG(ERROR) << "inputs size is invalid: " << inputs->size();
    return kLiteError;
  }
  auto param = primitive->value_as_Custom();
  if (dpico::CheckCustomParam(param, "Lstm") != RET_OK) {
    MS_LOG(ERROR) << "custom param is invalid.";
    return kLiteError;
  }

  const auto &input = (*inputs)[0];
  const auto &hidden_weight = (*inputs)[dpico::kInputIndex2];
  auto &output = (*outputs)[0];
  std::vector<int64_t> output_shape(input.Shape());
  if (output_shape.size() != dpico::kDims3) {
    MS_LOG(ERROR) << "output_shape should be 3 dims, which is " << output_shape.size();
    return kLiteError;
  }
  output_shape[0] = 1;  // bidirectional ? 2 : 1
  output_shape[dpico::kInputIndex2] = hidden_weight.Shape().at(0) / kHiddenLayerSize;
  output.SetShape(output_shape);
  return kSuccess;
}
REGISTER_CUSTOM_KERNEL_INTERFACE(DPICO, Lstm, DpicoLstmInferCreater)
}  // namespace kernel
}  // namespace mindspore
