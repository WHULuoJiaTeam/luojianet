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

#ifndef MINDSPORE_NNACL_FP32_CONCAT_BASE_H_
#define MINDSPORE_NNACL_FP32_CONCAT_BASE_H_

#include <string.h>
#include "nnacl/op_base.h"

#ifdef __cplusplus
extern "C" {
#endif
void Concat(void **input, int input_num, int axis, int **inputs_output_shape, size_t shape_size, void *output,
            int task_id, int thread_num, int data_size);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_CONCAT_BASE_H_
