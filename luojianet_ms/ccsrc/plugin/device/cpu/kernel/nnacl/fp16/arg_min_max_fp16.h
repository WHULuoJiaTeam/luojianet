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
#ifndef LUOJIANET_MS_NNACL_FP16_ARG_MIN_MAX_H_
#define LUOJIANET_MS_NNACL_FP16_ARG_MIN_MAX_H_

#include <float.h>
#include "nnacl/arg_min_max_parameter.h"
#include "nnacl/nnacl_common.h"

#ifdef __cplusplus
extern "C" {
#endif
void ArgMinMaxFp16(const float16_t *input, void *output, float16_t *output_value, const int *in_shape,
                   const ArgMinMaxParameter *param);
#ifdef __cplusplus
}
#endif

#endif  // LUOJIANET_MS_NNACL_FP16_ARG_MIN_MAX_H_
