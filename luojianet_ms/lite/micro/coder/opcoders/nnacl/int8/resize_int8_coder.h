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
#ifndef LUOJIANET_MS_LITE_MICRO_CODER_SLICE_INT8_CODER_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_SLICE_INT8_CODER_H_

#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/base/resize_base_coder.h"
#include "nnacl/op_base.h"

namespace luojianet_ms::lite::micro::nnacl {
class ResizeInt8Coder final : public ResizeBaseCoder {
 public:
  ResizeInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : ResizeBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ResizeInt8Coder() override;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int ReSize();
  void FreeArgs();
  ResizeParameter *param_{nullptr};
  ::QuantArg *quant_in_{nullptr};
  ::QuantArg *quant_out_{nullptr};
  QuantMulArg *multiplier_{nullptr};
};

}  // namespace luojianet_ms::lite::micro::nnacl
#endif  // LUOJIANET_MS_LITE_MICRO_CODER_SLICE_INT8_CODER_H_
