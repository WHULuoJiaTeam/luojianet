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

#include "nnacl/infer/binary_cross_entropy_infer.h"
#include "nnacl/infer/infer_register.h"

int BinaryCrossEntropyInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                 size_t outputs_size, OpParameter *parameter) {
  int ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (ret != NNACL_OK) {
    return ret;
  }
  const TensorC *x = inputs[0];
  TensorC *out = outputs[0];
  SetDataTypeFormat(out, x);
  BinaryCrossEntropyParameter *param = (BinaryCrossEntropyParameter *)parameter;
  int reduction = param->reduction;
  if (reduction == 1 || reduction == 2) {
    out->shape_size_ = 1;
    out->shape_[0] = 1;
  } else {
    SetShapeTensor(out, x);
  }
  return NNACL_OK;
}

REG_INFER(BinaryCrossEntropy, PrimType_BinaryCrossEntropy, BinaryCrossEntropyInferShape)
