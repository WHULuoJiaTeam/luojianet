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

#ifndef FFTS_PLUS_QOS_UPDATE_H_
#define FFTS_PLUS_QOS_UPDATE_H_

#include "runtime/rt_ffts_plus_define.h"
#include "graph/utils/node_utils.h"
namespace fe {

bool UpdateAicAivCtxQos(rtFftsPlusAicAivCtx_t *ctx, int label, int device_id);
bool UpdateMixAicAivCtxQos(rtFftsPlusMixAicAivCtx_t *ctx, int label, int device_id);
bool UpdateDataCtxQos(rtFftsPlusDataCtx_t *ctx, int device_id);

}

#endif