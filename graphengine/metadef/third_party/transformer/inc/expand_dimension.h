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

#ifndef COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
#define COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_

#include <memory.h>
#include <functional>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include "graph/types.h"
#include "axis_util.h"

namespace transformer {

const int32_t CHWN_DIM_C = 0;
const int32_t CHWN_DIM_H = 1;
const int32_t CHWN_DIM_W = 2;
const int32_t CHWN_DIM_N = 3;

const size_t DIMENSION_NUM_FOUR = 4;
const size_t DIMENSION_NUM_FIVE = 5;
const size_t DIMENSION_NUM_TWO = 2;
const std::string RESHAPE_TYPE_FORBIDDEN = "FORBIDDEN";

const std::map<ge::Format, size_t> FULL_SIZE_OF_FORMAT {
    {ge::FORMAT_NCHW, DIMENSION_NUM_FOUR},
    {ge::FORMAT_NHWC, DIMENSION_NUM_FOUR},
    {ge::FORMAT_HWCN, DIMENSION_NUM_FOUR},
    {ge::FORMAT_CHWN, DIMENSION_NUM_FOUR},
    {ge::FORMAT_NDHWC, DIMENSION_NUM_FIVE},
    {ge::FORMAT_NCDHW, DIMENSION_NUM_FIVE},
    {ge::FORMAT_DHWCN, DIMENSION_NUM_FIVE},
    {ge::FORMAT_ND, DIMENSION_NUM_FOUR}
};

inline uint32_t GenerateFormatKey(ge::Format format) {
  return ((static_cast<uint32_t>(format) & 0xff) << 8);
}

inline uint32_t GenerateReshapeTypeKey(ge::Format format, size_t size) {
  return ((static_cast<uint32_t>(format) & 0xff) << 8) | (static_cast<uint32_t>(size) & 0xff);
}

inline uint32_t GenerateAxisIndexKey(ge::Format format, char ch) {
  return ((static_cast<uint32_t>(format) & 0xff) << 8) | (static_cast<uint32_t>(ch) & 0xff);
}

const std::unordered_map<uint32_t, std::string> DEFAULT_RESHAPE_TYPE {
    {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 0), ""},
    {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 0), ""},

    {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 1), "C"},
    {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 1), "C"},

    {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 2), "CH"},
    {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 2), "HW"},
    {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 2), "CN"},
    {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 2), "WN"},
    {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 2), "WC"},
    {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 2), "HW"},
    {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 2), "CN"},

    {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 3), "CHW"},
    {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 3), "HWC"},
    {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 3), "WCN"},
    {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 3), "HWN"},
    {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 3), "HWC"},
    {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 3), "DHW"},
    {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 3), "WCN"},

    {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 4), "DHWC"},
    {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 4), "CDHW"},
    {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 4), "HWCN"}
};

const std::unordered_map<uint32_t, int32_t> AXIS_INDEX_OF_FORMAT {
    {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'N'), AXIS_NCHW_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'C'), AXIS_NCHW_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'H'), AXIS_NCHW_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'W'), AXIS_NCHW_DIM_W},

    {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'N'), AXIS_HWCN_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'C'), AXIS_HWCN_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'H'), AXIS_HWCN_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'W'), AXIS_HWCN_DIM_W},

    {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'N'), AXIS_NHWC_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'C'), AXIS_NHWC_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'H'), AXIS_NHWC_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'W'), AXIS_NHWC_DIM_W},

    {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'N'), CHWN_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'C'), CHWN_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'H'), CHWN_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'W'), CHWN_DIM_W},

    {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'N'), NDHWC_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'C'), NDHWC_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'H'), NDHWC_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'W'), NDHWC_DIM_W},
    {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'D'), NDHWC_DIM_D},

    {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'N'), NCDHW_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'C'), NCDHW_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'H'), NCDHW_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'W'), NCDHW_DIM_W},
    {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'D'), NCDHW_DIM_D},

    {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'N'), DHWCN_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'C'), DHWCN_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'H'), DHWCN_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'W'), DHWCN_DIM_W},
    {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'D'), DHWCN_DIM_D},

    {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'N'), DHWNC_DIM_N},
    {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'C'), DHWNC_DIM_C},
    {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'H'), DHWNC_DIM_H},
    {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'W'), DHWNC_DIM_W},
    {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'D'), DHWNC_DIM_D}
};

/* Pad dimension according to reshape type */
bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, ge::GeShape &shape);

bool ExpandRangeDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                          const uint32_t &tensor_index, const std::string &reshape_type,
                          std::vector<std::pair<int64_t, int64_t>> &ranges);
} // namespace transformer

#endif //COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
