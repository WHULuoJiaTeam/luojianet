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
 
// 宏定义，与文件路径保持一致
#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_OSAVI_OP_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_OSAVI_OP_H_

// TensorOp 基类的定义
#include "minddata/dataset/kernels/tensor_op.h"

// 定义命名空间
namespace luojianet_ms {
namespace dataset {
// 算子类继承 TensorOp 基类 
class OSAVIOp : public TensorOp {
 public:
    // Default values, also used by python_bindings.cc
    static const float kDeftheta;

    // Constructor
    OSAVIOp(float theta = kDeftheta);
    
    ~OSAVIOp() override = default;

    Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

    // （必选）声明函数 Name，用作表示当前函数的名称
    std::string Name() const override { return kOSAVIOp; }

 private:
  float theta_;
};

} // namespace dataset
} // namespace luojianet_ms
#endif // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_OSAVI_OP_H_