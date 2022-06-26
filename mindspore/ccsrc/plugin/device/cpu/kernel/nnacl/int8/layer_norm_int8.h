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
#ifndef MINDSPORE_NNACL_INT8_LAYER_NORM_H_
#define MINDSPORE_NNACL_INT8_LAYER_NORM_H_

#include "nnacl/errorcode.h"
#include "nnacl/layer_norm_parameter.h"
#include "nnacl/int8/fixed_point.h"

#ifdef __cplusplus
extern "C" {
#endif

int LayerNormInt8(const int8_t *src_data, const float *gamma_data, const float *beta_data, int8_t *dst_data,
                  const LayerNormParameter *param, const LayerNormQuantArg *quant, int task_id);

#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_INT8_LAYER_NORM_H_
