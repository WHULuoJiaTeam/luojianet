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

#ifndef GE_COMMON_MATH_MATH_UTIL_H_
#define GE_COMMON_MATH_MATH_UTIL_H_

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "common/fp16_t.h"
#include "framework/common/debug/log.h"
#include "framework/common/fmk_error_codes.h"

namespace ge {
/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckIntAddOverflow(int a, int b) {
  if (((b > 0) && (a > (INT_MAX - b))) || ((b < 0) && (a < (INT_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int8 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt8AddOverflow(int8_t a, int8_t b) {
  if (((b > 0) && (a > (INT8_MAX - b))) || ((b < 0) && (a < (INT8_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt16AddOverflow(int16_t a, int16_t b) {
  if (((b > 0) && (a > (INT16_MAX - b))) || ((b < 0) && (a < (INT16_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt32AddOverflow(int32_t a, int32_t b) {
  if (((b > 0) && (a > (INT32_MAX - b))) || ((b < 0) && (a < (INT32_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int64 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckInt64AddOverflow(int64_t a, int64_t b) {
  if (((b > 0) && (a > (INT64_MAX - b))) || ((b < 0) && (a < (INT64_MIN - b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint8AddOverflow(uint8_t a, uint8_t b) {
  if (a > (UINT8_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint16AddOverflow(uint16_t a, uint16_t b) {
  if (a > (UINT16_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint32AddOverflow(uint32_t a, uint32_t b) {
  if (a > (UINT32_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckUint64AddOverflow(uint64_t a, uint64_t b) {
  if (a > (UINT64_MAX - b)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16AddOverflow(fp16_t a, fp16_t b) {
  fp16_t result = static_cast<fp16_t>(a) + static_cast<fp16_t>(b);
  if (FP16_IS_INVALID(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatAddOverflow(float a, float b) {
  if (std::isfinite(static_cast<float>(a) + static_cast<float>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double addition can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleAddOverflow(double a, double b) {
  if (std::isfinite(static_cast<double>(a) + static_cast<double>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckIntSubOverflow(int a, int b) {
  if (((b > 0) && (a < (INT_MIN + b))) || ((b < 0) && (a > (INT_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int8 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt8SubOverflow(int8_t a, int8_t b) {
  if (((b > 0) && (a < (INT8_MIN + b))) || ((b < 0) && (a > (INT8_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt16SubOverflow(int16_t a, int16_t b) {
  if (((b > 0) && (a < (INT16_MIN + b))) || ((b < 0) && (a > (INT16_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt32SubOverflow(int32_t a, int32_t b) {
  if (((b > 0) && (a < (INT32_MIN + b))) || ((b < 0) && (a > (INT32_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int64 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckInt64SubOverflow(int64_t a, int64_t b) {
  if (((b > 0) && (a < (INT64_MIN + b))) || ((b < 0) && (a > (INT64_MAX + b)))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint8SubOverflow(uint8_t a, uint8_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint16SubOverflow(uint16_t a, uint16_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint32SubOverflow(uint32_t a, uint32_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 subtraction can result in overflow
/// @param [in] a  subtrahend
/// @param [in] b  minuend
/// @return Status
inline Status CheckUint64SubOverflow(uint64_t a, uint64_t b) {
  if (a < b) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16SubOverflow(fp16_t a, fp16_t b) {
  fp16_t result = static_cast<fp16_t>(a) - static_cast<fp16_t>(b);
  if (FP16_IS_INVALID(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatSubOverflow(float a, float b) {
  if (std::isfinite(static_cast<float>(a) - static_cast<float>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double subtraction can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleSubOverflow(double a, double b) {
  if (std::isfinite(static_cast<double>(a) - static_cast<double>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckIntMulOverflow(int a, int b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int8 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt8MulOverflow(int8_t a, int8_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT8_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT8_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT8_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT8_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int16 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt16MulOverflow(int16_t a, int16_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT16_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT16_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT16_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT16_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt32MulOverflow(int32_t a, int32_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT32_MAX / b)) {
        return FAILED;
      }
    } else {
      if (b < (INT32_MIN / a)) {
        return FAILED;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT32_MIN / b)) {
        return FAILED;
      }
    } else {
      if ((a != 0) && (b < (INT32_MAX / a))) {
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

///
/// @ingroup math_util
/// @brief check whether int64 int32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
///
inline Status CheckInt64Int32MulOverflow(int64_t a, int32_t b) {
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

/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status Int64MulCheckOverflow(int64_t a, int64_t b) {
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
/// @ingroup math_util
/// @brief check whether int64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckInt64Uint32MulOverflow(int64_t a, uint32_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }
  if (a > 0) {
    if (a > (INT64_MAX / b)) {
      return FAILED;
    }
  } else {
    if (a < (INT64_MIN / b)) {
      return FAILED;
    }
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint8 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint8MulOverflow(uint8_t a, uint8_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }

  if (a > (UINT8_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint16 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint16MulOverflow(uint16_t a, uint16_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }

  if (a > (UINT16_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint32 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint32MulOverflow(uint32_t a, uint32_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }

  if (a > (UINT32_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether uint64 multiplication can result in overflow
/// @param [in] a  multiplicator
/// @param [in] b  multiplicator
/// @return Status
inline Status CheckUint64MulOverflow(uint64_t a, uint64_t b) {
  if (a == 0 || b == 0) {
    return SUCCESS;
  }

  if (a > (UINT64_MAX / b)) {
    return FAILED;
  }

  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether fp16_t multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFp16MulOverflow(fp16_t a, fp16_t b) {
  fp16_t result = static_cast<fp16_t>(a) * static_cast<fp16_t>(b);
  if (FP16_IS_INVALID(result.val)) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether float multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckFloatMulOverflow(float a, float b) {
  if (std::isfinite(static_cast<float>(a) * static_cast<float>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether double multiplication can result in overflow
/// @param [in] a  addend
/// @param [in] b  addend
/// @return Status
inline Status CheckDoubleMulOverflow(double a, double b) {
  if (std::isfinite(static_cast<double>(a) * static_cast<double>(b)) == false) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
inline Status CheckIntDivOverflow(int a, int b) {
  if ((b == 0) || ((a == INT_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}
/// @ingroup math_util
/// @brief check whether int32 division can result in overflow
/// @param [in] a  dividend
/// @param [in] b  divisor
/// @return Status
inline Status CheckInt32DivOverflow(int32_t a, int32_t b) {
  if ((b == 0) || ((a == INT32_MIN) && (b == -1))) {
    return FAILED;
  }
  return SUCCESS;
}

#define FMK_INT_ADDCHECK(a, b)                                                                    \
  if (ge::CheckIntAddOverflow((a), (b)) != SUCCESS) {                                             \
    GELOGW("Int %d and %d addition can result in overflow!", static_cast<int>(a), \
           static_cast<int>(b));                                                                  \
    return INTERNAL_ERROR;                                                                        \
  }

#define FMK_INT8_ADDCHECK(a, b)                                                                       \
  if (ge::CheckInt8AddOverflow((a), (b)) != SUCCESS) {                                                \
    GELOGW("Int8 %d and %d addition can result in overflow!", static_cast<int8_t>(a),                 \
           static_cast<int8_t>(b));                                                                   \
    return INTERNAL_ERROR;                                                                            \
  }

#define FMK_INT16_ADDCHECK(a, b)                                                                        \
  if (ge::CheckInt16AddOverflow((a), (b)) != SUCCESS) {                                                 \
    GELOGW("Int16 %d and %d addition can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                                    \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_INT32_ADDCHECK(a, b)                                                                        \
  if (ge::CheckInt32AddOverflow((a), (b)) != SUCCESS) {                                                 \
    GELOGW("Int32 %d and %d addition can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                                    \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_INT64_ADDCHECK(a, b)                                                                          \
  if (ge::CheckInt64AddOverflow((a), (b)) != SUCCESS) {                                                   \
    GELOGW("Int64 %ld and %ld addition can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                                      \
    return INTERNAL_ERROR;                                                                                \
  }

#define FMK_UINT8_ADDCHECK(a, b)                                                                        \
  if (ge::CheckUint8AddOverflow((a), (b)) != SUCCESS) {                                                 \
    GELOGW("Uint8 %u and %u addition can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                                    \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_UINT16_ADDCHECK(a, b)                                                                         \
  if (ge::CheckUint16AddOverflow((a), (b)) != SUCCESS) {                                                  \
    GELOGW("UINT16 %u and %u addition can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                                     \
    return INTERNAL_ERROR;                                                                                \
  }

#define FMK_UINT32_ADDCHECK(a, b)                                                                         \
  if (ge::CheckUint32AddOverflow((a), (b)) != SUCCESS) {                                                  \
    GELOGW("Uint32 %u and %u addition can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                                     \
    return INTERNAL_ERROR;                                                                                \
  }

#define FMK_UINT64_ADDCHECK(a, b)                                                                           \
  if (ge::CheckUint64AddOverflow((a), (b)) != SUCCESS) {                                                    \
    GELOGW("Uint64 %lu and %lu addition can result in overflow!", static_cast<uint64_t>(a), \
           static_cast<uint64_t>(b));                                                                       \
    return INTERNAL_ERROR;                                                                                  \
  }

#define FMK_FP16_ADDCHECK(a, b)                                                                      \
  if (ge::CheckFp16AddOverflow((a), (b)) != SUCCESS) {                                               \
    GELOGW("Fp16 %f and %f addition can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                   \
    return INTERNAL_ERROR;                                                                           \
  }

#define FMK_FLOAT_ADDCHECK(a, b)                                                                      \
  if (ge::CheckFloatAddOverflow((a), (b)) != SUCCESS) {                                               \
    GELOGW("Float %f and %f addition can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                    \
    return INTERNAL_ERROR;                                                                            \
  }

#define FMK_DOUBLE_ADDCHECK(a, b)                                                                         \
  if (ge::CheckDoubleAddOverflow((a), (b)) != SUCCESS) {                                                  \
    GELOGW("Double %lf and %lf addition can result in overflow!", static_cast<double>(a), \
           static_cast<double>(b));                                                                       \
    return INTERNAL_ERROR;                                                                                \
  }

#define FMK_INT_SUBCHECK(a, b)                                                                       \
  if (ge::CheckIntSubOverflow((a), (b)) != SUCCESS) {                                                \
    GELOGW("Int %d and %d subtraction can result in overflow!", static_cast<int>(a), \
           static_cast<int>(b));                                                                     \
    return INTERNAL_ERROR;                                                                           \
  }

#define FMK_INT8_SUBCHECK(a, b)                                                                          \
  if (ge::CheckInt8SubOverflow((a), (b)) != SUCCESS) {                                                   \
    GELOGW("Int8 %d and %d subtraction can result in overflow!", static_cast<int8_t>(a), \
           static_cast<int8_t>(b));                                                                      \
    return INTERNAL_ERROR;                                                                               \
  }

#define FMK_INT16_SUBCHECK(a, b)                                                                           \
  if (ge::CheckInt16SubOverflow((a), (b)) != SUCCESS) {                                                    \
    GELOGW("Int16 %d and %d subtraction can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                                       \
    return INTERNAL_ERROR;                                                                                 \
  }

#define FMK_INT32_SUBCHECK(a, b)                                                                           \
  if (ge::CheckInt32SubOverflow((a), (b)) != SUCCESS) {                                                    \
    GELOGW("Int32 %d and %d subtraction can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                                       \
    return INTERNAL_ERROR;                                                                                 \
  }

#define FMK_INT64_SUBCHECK(a, b)                                                                             \
  if (ge::CheckInt64SubOverflow((a), (b)) != SUCCESS) {                                                      \
    GELOGW("Int64 %ld and %ld subtraction can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                                         \
    return INTERNAL_ERROR;                                                                                   \
  }

#define FMK_UINT8_SUBCHECK(a, b)                                                                           \
  if (ge::CheckUint8SubOverflow((a), (b)) != SUCCESS) {                                                    \
    GELOGW("Uint8 %u and %u subtraction can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                                       \
    return INTERNAL_ERROR;                                                                                 \
  }

#define FMK_UINT16_SUBCHECK(a, b)                                                                            \
  if (ge::CheckUint16SubOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Uint16 %u and %u subtraction can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                                        \
    return INTERNAL_ERROR;                                                                                   \
  }

#define FMK_UINT32_SUBCHECK(a, b)                                                                            \
  if (ge::CheckUint32SubOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Uint32 %u and %u subtraction can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                                        \
    return INTERNAL_ERROR;                                                                                   \
  }

#define FMK_UINT64_SUBCHECK(a, b)                                                                              \
  if (ge::CheckUint64SubOverflow((a), (b)) != SUCCESS) {                                                       \
    GELOGW("Uint64 %lu and %lu subtraction can result in overflow!", static_cast<uint64_t>(a), \
           static_cast<uint64_t>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                     \
  }

#define FMK_FP16_SUBCHECK(a, b)                                                                         \
  if (ge::CheckFp16SubOverflow((a), (b)) != SUCCESS) {                                                  \
    GELOGW("Fp16 %f and %f subtraction can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                      \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_FLOAT_SUBCHECK(a, b)                                                                         \
  if (ge::CheckFloatSubOverflow((a), (b)) != SUCCESS) {                                                  \
    GELOGW("Float %f and %f subtraction can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                       \
    return INTERNAL_ERROR;                                                                               \
  }

#define FMK_DOUBLE_SUBCHECK(a, b)                                                                            \
  if (ge::CheckDoubleSubOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Double %lf and %lf subtraction can result in overflow!", static_cast<double>(a), \
           static_cast<double>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                   \
  }

#define FMK_INT_MULCHECK(a, b)                                                                          \
  if (ge::CheckIntMulOverflow((a), (b)) != SUCCESS) {                                                   \
    GELOGW("Int %d and %d multiplication can result in overflow!", static_cast<int>(a), \
           static_cast<int>(b));                                                                        \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_INT8_MULCHECK(a, b)                                                                             \
  if (ge::CheckInt8MulOverflow((a), (b)) != SUCCESS) {                                                      \
    GELOGW("Int8 %d and %d multiplication can result in overflow!", static_cast<int8_t>(a), \
           static_cast<int8_t>(b));                                                                         \
    return INTERNAL_ERROR;                                                                                  \
  }

#define FMK_INT16_MULCHECK(a, b)                                                                              \
  if (ge::CheckInt16MulOverflow((a), (b)) != SUCCESS) {                                                       \
    GELOGW("Int16 %d and %d multiplication can result in overflow!", static_cast<int16_t>(a), \
           static_cast<int16_t>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                    \
  }

#define FMK_INT32_MULCHECK(a, b)                                                                              \
  if (ge::CheckInt32MulOverflow((a), (b)) != SUCCESS) {                                                       \
    GELOGW("Int32 %d and %d multiplication can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                    \
  }

#define FMK_INT64_MULCHECK(a, b)                                                                                \
  if (ge::Int64MulCheckOverflow((a), (b)) != SUCCESS) {                                                         \
    GELOGW("Int64 %ld and %ld multiplication can result in overflow!", static_cast<int64_t>(a), \
           static_cast<int64_t>(b));                                                                            \
    return INTERNAL_ERROR;                                                                                      \
  }

#define FMK_UINT8_MULCHECK(a, b)                                                                              \
  if (ge::CheckUint8MulOverflow((a), (b)) != SUCCESS) {                                                       \
    GELOGW("Uint8 %u and %u multiplication can result in overflow!", static_cast<uint8_t>(a), \
           static_cast<uint8_t>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                    \
  }

#define FMK_UINT16_MULCHECK(a, b)                                                                               \
  if (ge::CheckUint16MulOverflow((a), (b)) != SUCCESS) {                                                        \
    GELOGW("Uint16 %u and %u multiplication can result in overflow!", static_cast<uint16_t>(a), \
           static_cast<uint16_t>(b));                                                                           \
    return INTERNAL_ERROR;                                                                                      \
  }

#define FMK_UINT32_MULCHECK(a, b)                                                                               \
  if (ge::CheckUint32MulOverflow((a), (b)) != SUCCESS) {                                                        \
    GELOGW("Uint32 %u and %u multiplication can result in overflow!", static_cast<uint32_t>(a), \
           static_cast<uint32_t>(b));                                                                           \
    return INTERNAL_ERROR;                                                                                      \
  }

#define FMK_UINT64_MULCHECK(a, b)                                                                                 \
  if (ge::CheckUint64MulOverflow((a), (b)) != SUCCESS) {                                                          \
    GELOGW("Uint64 %lu and %lu multiplication can result in overflow!", static_cast<uint64_t>(a),               \
           static_cast<uint64_t>(b));                                                                             \
    return INTERNAL_ERROR;                                                                                        \
  }

#define FMK_FP16_MULCHECK(a, b)                                                                            \
  if (ge::CheckFp16MulOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Fp16 %f and %f multiplication can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                         \
    return INTERNAL_ERROR;                                                                                 \
  }

#define FMK_FLOAT_MULCHECK(a, b)                                                                            \
  if (ge::CheckFloatMulOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Float %f and %f multiplication can result in overflow!", static_cast<float>(a), \
           static_cast<float>(b));                                                                          \
    return INTERNAL_ERROR;                                                                                  \
  }

#define FMK_DOUBLE_MULCHECK(a, b)                                                                               \
  if (ge::CheckDoubleMulOverflow((a), (b)) != SUCCESS) {                                                        \
    GELOGW("Double %lf and %lf multiplication can result in overflow!", static_cast<double>(a),               \
           static_cast<double>(b));                                                                             \
    return INTERNAL_ERROR;                                                                                      \
  }

#define FMK_INT_DIVCHECK(a, b)                                                                    \
  if (CheckIntDivOverflow((a), (b)) != SUCCESS) {                                                 \
    GELOGW("Int %d and %d division can result in overflow!", static_cast<int>(a), \
           static_cast<int>(b));                                                                  \
    return INTERNAL_ERROR;                                                                        \
  }

#define FMK_INT32_DIVCHECK(a, b)                                                                        \
  if (CheckInt32DivOverflow((a), (b)) != SUCCESS) {                                                     \
    GELOGW("Int32 %d and %d division can result in overflow!", static_cast<int32_t>(a), \
           static_cast<int32_t>(b));                                                                    \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_INT64_UINT32_MULCHECK(a, b)                                                                 \
  if (ge::CheckInt64Uint32MulOverflow((a), (b)) != SUCCESS) {                                           \
    GELOGW("Int64 %ld and Uint32 %u multiplication can result in overflow!", static_cast<int64_t>(a),   \
           static_cast<uint32_t>(b));                                                                   \
    return INTERNAL_ERROR;                                                                              \
  }

#define FMK_FP16_ZEROCHECK(a)                                                                                          \
  if (fabs(a) < DBL_EPSILON || a < 0) {                                                                                \
    GELOGW("Fp16 %f can not less than or equal to zero! ", a);                                                      \
    return INTERNAL_ERROR;                                                                                             \
  }

#define FMK_FLOAT_ZEROCHECK(a)                                                                                         \
  if (fabs(a) < FLT_EPSILON || a < 0) {                                                                                \
    GELOGW("Float %f can not less than or equal to zero! ", a);                                                      \
    return INTERNAL_ERROR;                                                                                             \
  }

#define FMK_DOUBLE_ZEROCHECK(a)                                                                                        \
  if (fabs(a) < DBL_EPSILON || a < 0) {                                                                                \
    GELOGW("Double %lf can not less than or equal to zero! ", a);                                                    \
    return INTERNAL_ERROR;                                                                                             \
  }
}  // namespace ge
#endif  // GE_COMMON_MATH_MATH_UTIL_H_
