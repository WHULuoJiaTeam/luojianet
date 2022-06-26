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
#ifndef LUOJIANET_MS_NNACL_FP16_GRU_H_
#define LUOJIANET_MS_NNACL_FP16_GRU_H_
#include "nnacl/gru_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void GruFp16(float16_t *output, const float16_t *input, const float16_t *weight_g, const float16_t *weight_r,
             const float16_t *input_bias, const float16_t *state_bias, float16_t *hidden_state, float16_t *buffer[4],
             int check_seq_len, const GruParameter *gru_param);
#ifdef __cplusplus
}
#endif

#endif  // LUOJIANET_MS_NNACL_FP16_GRU_H_
