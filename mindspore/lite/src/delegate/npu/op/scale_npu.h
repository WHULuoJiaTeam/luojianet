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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_SCALE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_SCALE_NPU_H_
#include <vector>
#include <string>
#include "include/graph/op/all_ops.h"
#include "include/graph/op/nn_defs.h"
#include "src/delegate/npu/op/npu_op.h"

namespace mindspore {
class ScaleNPUOp : public NPUOp {
 public:
  ScaleNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
             const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, name) {}

  ~ScaleNPUOp() override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  int Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
           const std::vector<mindspore::MSTensor> &out_tensors) override;

  int SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors,
                   const std::vector<ge::Operator *> &npu_inputs) override;

  ge::Operator *GetNPUOp() override;

  int GetAxis() { return axis_; }

 private:
  int SetActivation();

  int ConvertScaleToMul(const std::vector<ge::Operator *> &npu_inputs,
                        const std::vector<mindspore::MSTensor> &in_tensors);

  int Adopt4DScale(const std::vector<ge::Operator *> &npu_inputs, const std::vector<mindspore::MSTensor> &in_tensors);

  ge::Operator *ChangeDims(const ge::Operator *input, std::vector<int64_t> dst_shape, std::string name,
                           bool need_expand_4d = false);

  int axis_ = 0;
  bool use_mul_ = false;
  bool need_expand_ = false;
  schema::ActivationType act_type_ = schema::ActivationType_NO_ACTIVATION;
  hiai::op::Reshape *out_reshape_ = nullptr;
  hiai::op::Scale *scale_ = nullptr;
  hiai::op::Mul *mul_ = nullptr;
  hiai::op::Activation *act_ = nullptr;
  std::vector<ge::Operator *> scale_ops_ = {};
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_SCALE_NPU_H_
