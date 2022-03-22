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

#ifndef COMPRESS_H
#define COMPRESS_H

#include <uchar.h>

enum CmpStatus {
    RET_SUCCESS = 0,
    RET_ERROR = -1
};

struct CompressConfig {
    size_t inputSize; // length of data to compress
    size_t engineNum; // how many decompress engines
    size_t maxRatio; // how much size of a basic compression block, only 64 supported now (8x: 64 4x: 32)
    size_t channel; // channels of L2 or DDR. For load balance
    size_t fractalSize; // size of compressing block
    bool isTight; // whether compose compressed data tightly
    size_t init_offset;
};

CmpStatus CompressWeights(char* input,
                          const CompressConfig& compressConfig,
                          char* indexs,
                          char* output,
                          size_t& compressedLength);


#endif  // COMPRESS_H
