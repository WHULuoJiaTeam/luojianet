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

#include "wrapper/int8/batchnorm_int8_wrapper.h"
#include "nnacl/int8/batchnorm_int8.h"
#include "nnacl/errorcode.h"

int BatchNormInt8Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  BatchNormArgs *args = (BatchNormArgs *)(cdata);
  BatchNormInt8(args->out_addr_, args->in_addr_, args->alpha_addr_, args->beta_addr_, task_id, args->batchnorm_param_);
  return NNACL_OK;
}
