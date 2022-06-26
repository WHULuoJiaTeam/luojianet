/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_BASE_SPACE_TO_DEPTH_BASE_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_BASE_SPACE_TO_DEPTH_BASE_H_

#include "nnacl/op_base.h"
#include "nnacl/space_to_depth_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
int SpaceToDepthForNHWC(const void *input, void *output, const int *in_shape, const int *out_shape, int shape_size,
                        SpaceToDepthParameter *param, int task_id);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_NNACL_BASE_SPACE_TO_DEPTH_BASE_H_
