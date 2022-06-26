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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_BASE_QUANT_DTYPE_CAST_CODER_H
#define LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_BASE_QUANT_DTYPE_CAST_CODER_H

#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/int8/quant_dtype_cast_int8.h"

namespace luojianet_ms::lite::micro {
class QuantDTypeCastCoder final : public OperatorCoder {
 public:
  QuantDTypeCastCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                      const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~QuantDTypeCastCoder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  TypeId src_dtype{kTypeUnknown};
  TypeId dst_dtype{kTypeUnknown};
  int thread_num_{0};
  int thread_n_num_{0};
  int thread_n_stride_{0};
  int num_unit_{0};
};
}  // namespace luojianet_ms::lite::micro
#endif  // LUOJIANET_MS_LITE_MICRO_CODER_OPCODERS_BASE_QUANT_DTYPE_CAST_CODER_H
