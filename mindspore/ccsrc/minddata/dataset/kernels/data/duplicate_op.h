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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_DUPLICATE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DATA_DUPLICATE_OP_H_

#include <vector>
#include <memory>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

class DuplicateOp : public TensorOp {
 public:
  DuplicateOp() = default;

  ~DuplicateOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  uint32_t NumOutput() override { return 2; }

  std::string Name() const override { return kDuplicateOp; }
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_DUPLICATE_OP_H_
