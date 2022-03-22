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
#ifndef LUOJIANET_MS_NNACL_INT8_SPACE_TO_BATCH_INT8_H_
#define LUOJIANET_MS_NNACL_INT8_SPACE_TO_BATCH_INT8_H_

#include "nnacl/op_base.h"
#include "nnacl/fp32/space_to_batch_fp32.h"

#ifdef __cplusplus
extern "C" {
#endif
void DoSpaceToBatchNHWCInt8(const int8_t *input, int8_t *output, const int *block_sizes, const int *in_shape,
                            const int *out_shape);
void DoSpaceToBatchPaddingNHWCInt8(const int8_t *input, int8_t *output, SpaceToBatchParameter *param, int32_t zp);
#ifdef __cplusplus
}
#endif

#endif  // LUOJIANET_MS_NNACL_INT8_SPACE_TO_BATCH_INT8_H_
