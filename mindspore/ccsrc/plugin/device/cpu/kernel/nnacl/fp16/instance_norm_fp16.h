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
#ifndef MINDSPORE_NNACL_FP16_INSTANCE_NORM_H_
#define MINDSPORE_NNACL_FP16_INSTANCE_NORM_H_

#include "nnacl/instance_norm_parameter.h"
#ifdef __cplusplus
extern "C" {
#endif

int InstanceNormFp16(const float16_t *src_data, float16_t *dst_data, const float16_t *gamma_data,
                     const float16_t *beta_data, const InstanceNormParameter *param, size_t task_id);
int InstanceNormNC8HW8Fp16(const float16_t *src_data, float16_t *dst_data, const float16_t *gamma_data,
                           const float16_t *beta_data, const InstanceNormParameter *param, size_t task_id);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP16_INSTANCE_NORM_H_
