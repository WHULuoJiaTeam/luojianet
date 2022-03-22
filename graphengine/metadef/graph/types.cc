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
#include "external/graph/types.h"
#include <cmath>
#include "graph/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "inc/common/util/error_manager/error_manager.h"

namespace ge {
const char_t *GetFormatName(Format format) {
  static const char_t *names[FORMAT_END] = {
      "NCHW",
      "NHWC",
      "ND",
      "NC1HWC0",
      "FRACTAL_Z",
      "NC1C0HWPAD", // 5
      "NHWC1C0",
      "FSR_NCHW",
      "FRACTAL_DECONV",
      "C1HWNC0",
      "FRACTAL_DECONV_TRANSPOSE",  // 10
      "FRACTAL_DECONV_SP_STRIDE_TRANS",
      "NC1HWC0_C04",
      "FRACTAL_Z_C04",
      "CHWN",
      "DECONV_SP_STRIDE8_TRANS", // 15
      "HWCN",
      "NC1KHKWHWC0",
      "BN_WEIGHT",
      "FILTER_HWCK",
      "LOOKUP_LOOKUPS", // 20
      "LOOKUP_KEYS",
      "LOOKUP_VALUE",
      "LOOKUP_OUTPUT",
      "LOOKUP_HITS",
      "C1HWNCoC0", // 25
      "MD",
      "NDHWC",
      "UNKNOWN", // FORMAT_FRACTAL_ZZ
      "FRACTAL_NZ",
      "NCDHW", // 30
      "DHWCN",
      "NDC1HWC0",
      "FRACTAL_Z_3D",
      "CN",
      "NC", // 35
      "DHWNC",
      "FRACTAL_Z_3D_TRANSPOSE",
      "FRACTAL_ZN_LSTM",
      "FRACTAL_Z_G",
      "UNKNOWN", // 40, FORMAT_RESERVED
      "UNKNOWN", // FORMAT_ALL
      "UNKNOWN", // FORMAT_NULL
      "ND_RNN_BIAS",
      "FRACTAL_ZN_RNN",
  };
  if (format >= FORMAT_END) {
    return "UNKNOWN";
  }
  return names[format];
}

static int64_t CeilDiv(const int64_t n1, const int64_t n2) {
  if (n1 == 0) {
    return 0;
  }
  return (n2 != 0) ? ((n1 - 1) / n2 + 1) : 0;
}

///
/// @ingroup math_util
/// @brief check whether int64 int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
static Status CheckInt64MulOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

int64_t GetSizeInBytes(int64_t element_count, DataType data_type) {
  if (element_count < 0) {
    REPORT_INNER_ERROR("E19999", "GetSizeInBytes failed, element_count:%ld less than 0.", element_count);
    GELOGE(GRAPH_FAILED, "[Check][param]GetSizeInBytes failed, element_count:%ld less than 0.", element_count);
    return -1;
  }
  const auto type_size = GetSizeByDataType(data_type);
  if (type_size < 0) {
    GELOGE(GRAPH_FAILED, "[Check][DataType]GetSizeInBytes failed, data_type:%d not support.", data_type);
    return -1;
  } else if (type_size > kDataTypeSizeBitOffset) {
    const auto bit_size = type_size - kDataTypeSizeBitOffset;
    if (CheckInt64MulOverflow(element_count, bit_size) == FAILED) {
      REPORT_INNER_ERROR("E19999", "GetSizeInBytes failed, int64 mul overflow %ld, %d.",
                         element_count, bit_size);
      GELOGE(GRAPH_FAILED, "[Check][overflow]GetSizeInBytes failed, when multiplying %ld and %d.",
             element_count, bit_size);
      return -1;
    }
    return CeilDiv(element_count * bit_size, kBitNumOfOneByte);
  } else {
    if (CheckInt64MulOverflow(element_count, type_size) == FAILED) {
      REPORT_INNER_ERROR("E19999", "GetSizeInBytes failed, int64 mul overflow %ld, %d.",
                         element_count, type_size);
      GELOGE(GRAPH_FAILED, "[Check][overflow]GetSizeInBytes failed, when multiplying %ld and %d.",
             element_count, type_size);
      return -1;
    }
    return element_count * type_size;
  }
}
}