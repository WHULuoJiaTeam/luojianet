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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_SQUEEZE_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_SQUEEZE_INT8_H_

#include "nnacl/squeeze_parameter.h"
#include "nnacl/int8/quantize.h"

#ifdef __cplusplus
extern "C" {
#endif
void SqueezeInt8(const int8_t *input_ptr, int8_t *output_ptr, const SqueezeQuantArg *quant_Squeeze_parm, int num,
                 int task_id, int thread_count);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_OPCLIB_INT8_SQUEEZE_INT8_H_
