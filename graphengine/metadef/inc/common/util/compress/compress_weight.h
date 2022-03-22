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

#ifndef COMPRESS_WEIGHT_H
#define COMPRESS_WEIGHT_H

#include "compress.h"

const int SHAPE_SIZE_WEIGHT = 4;

struct CompressOpConfig {
    int64_t wShape[SHAPE_SIZE_WEIGHT];
    size_t compressTilingK;
    size_t compressTilingN;
    struct CompressConfig compressConfig;
};

extern "C" CmpStatus CompressWeightsConv2D(const char *const input,
                                           char *const zipBuffer,
                                           char *const infoBuffer,
                                           CompressOpConfig *const param);
#endif // COMPRESS_WEIGHT_H
