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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_INT8_MATMUL_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_INT8_MATMUL_INT8_CODER_H_

#include <vector>
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/nnacl/int8/matmul_base_int8_coder.h"
#include "nnacl/matmul_parameter.h"
namespace mindspore::lite::micro::nnacl {
class MatMulInt8Coder final : public MatMulBaseInt8Coder {
 public:
  MatMulInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : MatMulBaseInt8Coder(in_tensors, out_tensors, node, node_index, target) {}

  ~MatMulInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int ReSize(CoderContext *const context) override;
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_INT8_MATMUL_INT8_CODER_H_
