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

#ifndef MINDSPORE_NNACL_INT8_PAD_INT8_H_
#define MINDSPORE_NNACL_INT8_PAD_INT8_H_

#include <string.h>
#include "nnacl/op_base.h"
#include "nnacl/pad_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int PadConstant4D(const int8_t *in_data, int8_t *out_data, const int32_t *in_dims, const int32_t *out_dims,
                  const int32_t *paddings, const int tid, const int thread_num);
void MirrorPadInt8(const int8_t *input_data, int8_t *output_data, const int *input_shape, const PadParameter *pad_param,
                   int begin, int end);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_PAD_INT8_H_
