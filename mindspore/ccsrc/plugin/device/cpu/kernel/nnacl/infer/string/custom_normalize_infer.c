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

#include "nnacl/infer/string/custom_normalize_infer.h"
#include "nnacl/infer/infer_register.h"

int CustomNormalizeInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];

  SetDataTypeFormat(output, input);

  if (input->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (GetElementNum(input) < 1) {
    return NNACL_ERR;
  }
  if (input->data_type_ != kNumberTypeUInt32 && input->data_type_ != kObjectTypeString) {
    return NNACL_ERR;
  }
  int string_num = *((const int32_t *)(input->data_));  // also look custom_extract_features
  output->shape_size_ = 1;
  output->shape_[0] = (string_num == 0 ? 1 : string_num);
  return NNACL_OK;
}

REG_INFER(CustomNormalize, PrimType_CustomNormalize, CustomNormalizeInferShape)
