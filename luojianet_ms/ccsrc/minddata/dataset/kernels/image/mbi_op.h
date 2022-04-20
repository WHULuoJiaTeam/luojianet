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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MBI_OP_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MBI_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

// 定义命名空间
namespace luojianet_ms {
namespace dataset {
// 算子类继承 TensorOp 基类 
class MBIOp : public TensorOp {
 public:
    // Default values, also used by python_bindings.cc
    static const int32_t kDefS_min;
    static const int32_t kDefS_max;
    static const int32_t kDefDelta_s;

    // Constructor
    MBIOp(int32_t s_min = kDefS_min, int32_t s_max = kDefS_max, int32_t delta_s = kDefDelta_s);
    
    ~MBIOp() override = default;

    Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

    std::string Name() const override { return kMBIOp; }

 private:
  int32_t s_min_;
  int32_t s_max_;
  int32_t delta_s_;
};

} // namespace dataset
} // namespace mindspore 
#endif // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_MBI_OP_H_