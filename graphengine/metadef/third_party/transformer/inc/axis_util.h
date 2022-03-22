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

#ifndef COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
#define COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_

#include <memory.h>
#include <functional>
#include <vector>

#include "external/graph/ge_error_codes.h"
#include "external/graph/types.h"
#include "graph/ge_tensor.h"

using namespace ge;
namespace transformer {

const size_t DIM_DEFAULT_SIZE = 4;
const size_t DIM_SIZE_FIVE = 5;
const size_t DIM_SIZE_SIX = 6;
const uint32_t NCHW_DIMENSION_NUM = 4;

const int32_t AXIS_NCHW_DIM_N = 0;
const int32_t AXIS_NCHW_DIM_C = 1;
const int32_t AXIS_NCHW_DIM_H = 2;
const int32_t AXIS_NCHW_DIM_W = 3;

const int32_t AXIS_NHWC_DIM_N = 0;
const int32_t AXIS_NHWC_DIM_H = 1;
const int32_t AXIS_NHWC_DIM_W = 2;
const int32_t AXIS_NHWC_DIM_C = 3;

const int32_t AXIS_NC1HWC0_DIM_N = 0;
const int32_t AXIS_NC1HWC0_DIM_C1 = 1;
const int32_t AXIS_NC1HWC0_DIM_C0 = 4;
const int32_t AXIS_NC1HWC0_DIM_H = 2;
const int32_t AXIS_NC1HWC0_DIM_W = 3;

const int32_t AXIS_HWCN_DIM_H = 0;
const int32_t AXIS_HWCN_DIM_W = 1;
const int32_t AXIS_HWCN_DIM_C = 2;
const int32_t AXIS_HWCN_DIM_N = 3;

const int32_t AXIS_C1HWNCoC0_DIM_C1 = 0;
const int32_t AXIS_C1HWNCoC0_DIM_H = 1;
const int32_t AXIS_C1HWNCoC0_DIM_W = 2;
const int32_t AXIS_C1HWNCoC0_DIM_N = 3;
const int32_t AXIS_C1HWNCoC0_DIM_Co = 4;
const int32_t AXIS_C1HWNCoC0_DIM_C0 = 5;

const int32_t NDHWC_DIM_N = 0;
const int32_t NDHWC_DIM_D = 1;
const int32_t NDHWC_DIM_H = 2;
const int32_t NDHWC_DIM_W = 3;
const int32_t NDHWC_DIM_C = 4;

const int32_t NCDHW_DIM_N = 0;
const int32_t NCDHW_DIM_C = 1;
const int32_t NCDHW_DIM_D = 2;
const int32_t NCDHW_DIM_H = 3;
const int32_t NCDHW_DIM_W = 4;

const int32_t DHWCN_DIM_D = 0;
const int32_t DHWCN_DIM_H = 1;
const int32_t DHWCN_DIM_W = 2;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;

const int32_t DHWNC_DIM_D = 0;
const int32_t DHWNC_DIM_H = 1;
const int32_t DHWNC_DIM_W = 2;
const int32_t DHWNC_DIM_N = 3;
const int32_t DHWNC_DIM_C = 4;

inline bool CheckInt64MulOverflow(int64_t m, int64_t n) {
  if (m > 0) {
    if (n > 0) {
      if (m > ((int64_t)INT64_MAX / n)) {
        return false;
      }
    } else {
      if (n < ((int64_t)INT64_MIN / m)) {
        return false;
      }
    }
  } else {
    if (n > 0) {
      if (m < ((int64_t)INT64_MIN / n)) {
        return false;
      }
    } else {
      if ((m != 0) && (n < ((int64_t)INT64_MAX / m))) {
        return false;
      }
    }
  }
  return true;
}

#define INT64_MULCHECK(a, b)                                                                      \
  if (CheckInt64MulOverflow((a), (b)) != true) {                                                  \
    return false;                                                                                 \
  }

#define CHECK_NOTNULL(val)                                       \
  do {                                                           \
    if ((val) == nullptr) {                                      \
      GELOGE(GRAPH_FAILED, "[ERROR]Parameter[%s] must not be null.", #val); \
      return false;                                              \
    }                                                            \
  } while (0)

#define CHECK(cond, log_func, return_expr) \
  do {                                     \
    if (cond) {                            \
      log_func;                            \
      return_expr;                         \
    }                                      \
  } while (0)

#define INT64_ZEROCHECK(a)                                                                            \
  if (a == 0) {                                                                                       \
    return false;                                                                                     \
  }
enum AxisValueType {
  AXIS_N = 0,
  AXIS_C = 1,
  AXIS_H = 2,
  AXIS_W = 3,
  AXIS_C1 = 4,
  AXIS_C0 = 5,
  AXIS_Co = 6,
  AXIS_D = 7,
  AXIS_G = 8,
  AXIS_INPUT_SIZE = 9,
  AXIS_HIDEEN_SIZE = 10,
  AXIS_BOTTOM = 11
};

inline int64_t DivisionCeiling(int64_t dividend, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  } else if (dividend <= 0) {
    return dividend;
  } else {
    return (dividend + divisor - 1) / divisor;
  }
}

/* Axis value is arranged as {N,C,H,W,C1,C0,...} */
/* The first parameter is old shape's dimension,
 * second is c0 and third is axis value. */
using GetAxisValueInfoByFormat = std::function<bool(const ge::GeShape&, const uint32_t&, std::vector<int64_t>&)>;

using GetAxisValueInfoByFormatPtr = std::shared_ptr<GetAxisValueInfoByFormat>;

class AxisUtil {
 public:
  AxisUtil();
  ~AxisUtil(){};
  bool GetAxisValueByOriginFormat(const ge::Format& format, const ge::GeShape &shape, const uint32_t& c0,
                                  std::vector<int64_t>& axisValue);
  bool HasAxisValueFunc(const ge::Format& format);

  static bool CheckParams(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByNCHW(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByNHWC(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByNC1HWC0(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByFz(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByHWCN(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByND(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByC1HWNCoC0(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axisValue);

  static bool GetAxisValueByNDHWC(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axis_value);

  static bool GetAxisValueByNCDHW(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axis_value);

  static bool GetAxisValueByDHWCN(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axis_value);

  static bool GetAxisValueByDHWNC(const ge::GeShape &shape, const uint32_t& c0, std::vector<int64_t>& axis_value);
};
} // namespace transformer

#endif // COMMON_UTILS_TRANSFORMER_INC_AXIS_UTIL_H_
