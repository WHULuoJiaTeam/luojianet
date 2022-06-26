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

#include "nnacl/infer/assign_add_infer.h"
#include "nnacl/infer/infer_register.h"

int AssignAddInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                        OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *x = inputs[0];
  const TensorC *y = inputs[1];
  TensorC *out = outputs[0];
  if (x->data_type_ != y->data_type_) {
    return NNACL_ERR;
  }
  SetDataTypeFormat(out, x);
  SetShapeTensor(out, x);
  return NNACL_OK;
}

REG_INFER(AssignAdd, PrimType_AssignAdd, AssignAddInferShape)
