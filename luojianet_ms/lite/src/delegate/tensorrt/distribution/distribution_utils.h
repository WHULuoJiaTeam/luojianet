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
#ifndef LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_UTILS_H_
#define LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_UTILS_H_

#include <nccl.h>
#include "include/errorcode.h"
#include "NvInfer.h"
#include "schema/ops_types_generated.h"

using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;

namespace luojianet_ms::lite {
ncclDataType_t ConvertNCCLDataType(nvinfer1::DataType type_id);

ncclRedOp_t ConvertNCCLReduceMode(schema::ReduceMode mode);
}  // namespace luojianet_ms::lite
#endif  // LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_DISTRIBUTION_DISTRIBUTION_UTILS_H_
