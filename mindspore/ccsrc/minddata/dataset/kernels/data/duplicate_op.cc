/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/data/duplicate_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

Status DuplicateOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1,
                               "Duplicate: only supports transform one column each time, got column num: " +
                                 std::to_string(input.size()) + ", check 'input_columns' when call this operator.");
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(input[0], &out));
  output->push_back(input[0]);
  output->push_back(out);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
