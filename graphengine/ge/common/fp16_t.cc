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

#include "common/fp16_t.h"

#include "external/register/register_types.h"

namespace {
constexpr uint16_t kManBitLength = 11;
}
namespace ge {
/// @ingroup fp16_t global filed
/// @brief   round mode of last valid digital
enum TagFp16RoundMode g_round_mode = kRoundToNearest;

void ExtractFp16(const uint16_t &val, uint16_t &s, int16_t &e, uint16_t &m) {
  // 1.Extract
  s = static_cast<uint16_t>(FP16_EXTRAC_SIGN(val));
  e = static_cast<int16_t>(FP16_EXTRAC_EXP(val));
  m = static_cast<uint16_t>(FP16_EXTRAC_MAN(val));
  // Denormal
  if (e == 0) {
    e = 1;
  }
}
/// @ingroup fp16_t static method
/// @param [in] man       truncated mantissa
/// @param [in] shift_out left shift bits based on ten bits
/// @brief   judge whether to add one to the result while converting fp16_t to other datatype
/// @return  Return true if add one, otherwise false
static bool IsRoundOne(uint64_t man, uint16_t trunc_len) {
  uint64_t mask0 = 0x4;
  uint64_t mask1 = 0x2;
  uint64_t mask2;
  uint16_t shift_out = static_cast<uint16_t>(trunc_len - kDim2);
  mask0 = mask0 << shift_out;
  mask1 = mask1 << shift_out;
  mask2 = mask1 - 1;

  bool last_bit = ((man & mask0) > 0);
  bool trunc_high = false;
  bool trunc_left = false;
  if (g_round_mode == kRoundToNearest) {
    trunc_high = ((man & mask1) > 0);
    trunc_left = ((man & mask2) > 0);
  }
  return (trunc_high && (trunc_left || last_bit));
}
/// @ingroup fp16_t public method
/// @param [in] exp       exponent of fp16_t value
/// @param [in] man       exponent of fp16_t value
/// @brief   normalize fp16_t value
/// @return
static void Fp16Normalize(int16_t &exp, uint16_t &man) {
  // set to invalid data
  if (exp >= kFp16MaxExp) {
    exp = static_cast<int16_t>(kFp16MaxExp);
    man = static_cast<uint16_t>(kFp16MaxMan);
  } else if (exp == 0 && man == kFp16ManHideBit) {
    exp++;
    man = 0;
  }
}

/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to float/fp32
/// @return  Return float/fp32 value of fp_val which is the value of fp16_t object
static float Fp16ToFloat(const uint16_t &fp_val) {
  uint16_t hf_sign;
  uint16_t hf_man;
  int16_t hf_exp;
  ExtractFp16(fp_val, hf_sign, hf_exp, hf_man);

  while (hf_man && !(hf_man & kFp16ManHideBit)) {
    hf_man <<= 1;
    hf_exp--;
  }

  uint32_t e_ret, m_ret;
  uint32_t s_ret = hf_sign;
  if (hf_man == 0) {
    e_ret = 0;
    m_ret = 0;
  } else {
    e_ret = hf_exp - kFp16ExpBias + kFp32ExpBias;
    m_ret = hf_man & kFp16ManMask;
    m_ret = m_ret << (kFp32ManLen - kFp16ManLen);
  }
  uint32_t f_val = FP32_CONSTRUCTOR(s_ret, e_ret, m_ret);
  auto p_ret_v = reinterpret_cast<float *>(&f_val);

  return *p_ret_v;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to double/fp64
/// @return  Return double/fp64 value of fp_val which is the value of fp16_t object
static double Fp16ToDouble(const uint16_t &fp_val) {
  uint16_t hf_sign;
  uint16_t hf_man;
  int16_t hf_exp;
  ExtractFp16(fp_val, hf_sign, hf_exp, hf_man);

  while (hf_man && !(hf_man & kFp16ManHideBit)) {
    hf_man <<= 1;
    hf_exp--;
  }

  uint64_t e_ret;
  uint64_t m_ret;
  uint64_t s_ret = hf_sign;
  if (!hf_man) {
    e_ret = 0;
    m_ret = 0;
  } else {
    e_ret = hf_exp - kFp16ExpBias + kFp64ExpBias;
    m_ret = hf_man & kFp16ManMask;
    m_ret = m_ret << (kFp64ManLen - kFp16ManLen);
  }
  uint64_t f_val = (s_ret << kFp64SignIndex) | (e_ret << kFp64ManLen) | (m_ret);
  auto p_ret_v = reinterpret_cast<double *>(&f_val);

  return *p_ret_v;
}
/// @ingroup fp16_t static method
/// @param [in] s_ret       sign of fp16_t value
/// @param [in] long_int_m   man uint64_t value of fp16_t object
/// @param [in] shift_out   shift offset
/// @brief   calculate uint8 value by sign,man and shift offset
/// @return Return uint8 value of fp16_t object
static uint8_t GetUint8ValByMan(uint8_t s_ret, const uint64_t &long_int_m, const uint16_t &shift_out) {
  bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
  auto m_ret = static_cast<uint8_t>((long_int_m >> (kFp16ManLen + shift_out)) & kBitLen8Max);
  need_round = need_round && ((s_ret == 0 && m_ret < kInt8Max) || (s_ret == 1 && m_ret <= kInt8Max));
  if (need_round) {
    m_ret++;
  }
  if (s_ret) {
    m_ret = (~m_ret) + 1;
  }
  if (m_ret == 0) {
    s_ret = 0;
  }
  return static_cast<uint8_t>((s_ret << kBitShift7) | (m_ret));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int8_t
/// @return  Return int8_t value of fp_val which is the value of fp16_t object
static int8_t Fp16ToInt8(const uint16_t &fp_val) {
  int8_t ret;
  uint8_t ret_v;
  // 1.get s_ret and shift it to bit0.
  uint8_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_DENORM(fp_val)) {  // Denormalized number
    ret_v = 0;
    ret = *(reinterpret_cast<uint8_t *>(&ret_v));
    return ret;
  }

  uint64_t long_int_m = hf_m;
  uint8_t overflow_flag = 0;
  uint16_t shift_out = 0;
  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    overflow_flag = 1;
  } else {
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
        if (s_ret == 1 && long_int_m >= 0x20000u) {  // sign=1,negative number(<0)
          long_int_m = 0x20000u;                     // 10 0000 0000 0000 0000  10(fp16_t-man)+7(int8)=17bit
          overflow_flag = 1;
          break;
        } else if (s_ret != 1 && long_int_m >= 0x1FFFFu) {  // sign=0,positive number(>0)
          long_int_m = 0x1FFFFu;                            // 01 1111 1111 1111 1111  10(fp16_t-man)+7(int8)
          overflow_flag = 1;
          break;
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
  }
  if (overflow_flag) {
    ret_v = kInt8Max + s_ret;
  } else {
    // Generate final result
    ret_v = GetUint8ValByMan(s_ret, long_int_m, shift_out);
  }

  ret = *(reinterpret_cast<uint8_t *>(&ret_v));
  return ret;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint8_t
/// @return  Return uint8_t value of fp_val which is the value of fp16_t object
static uint8_t Fp16ToUInt8(const uint16_t &fp_val) {
  uint8_t m_ret = 0;
  // 1.get s_ret and shift it to bit0.
  uint8_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_DENORM(fp_val)) {  // Denormalized number
    return 0;
  }

  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    m_ret = ~0;
  } else {
    uint64_t long_int_m = hf_m;
    uint8_t overflow_flag = 0;
    uint16_t shift_out = 0;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
        if (long_int_m >= 0x40000Lu) {  // overflow 0100 0000 0000 0000 0000
          long_int_m = 0x3FFFFLu;       // 11 1111 1111 1111 1111   10(fp16_t-man)+8(uint8)=18bit
          overflow_flag = 1;
          m_ret = ~0;
          break;
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
    if (!overflow_flag) {
      bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
      m_ret = static_cast<uint8_t>((long_int_m >> (kFp16ManLen + shift_out)) & kBitLen8Max);
      if (need_round && m_ret != kBitLen8Max) {
        m_ret++;
      }
    }
  }

  if (s_ret == 1) {  // Negative number
    m_ret = 0;
  }
  // m_ret equal to final result
  return m_ret;
}
/// @ingroup fp16_t static method
/// @param [in] s_ret       sign of fp16_t value
/// @param [in] long_int_m   man uint64_t value of fp16_t object
/// @param [in] shift_out   shift offset
/// @brief   calculate uint16 value by sign,man and shift offset
/// @return Return uint16 value of fp16_t object
static uint16_t GetUint16ValByMan(uint16_t s_ret, const uint64_t &long_int_m, const uint16_t &shift_out) {
  bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
  auto m_ret = static_cast<uint16_t>((long_int_m >> (kFp16ManLen + shift_out)) & kBitLen16Max);
  if (need_round && m_ret < kInt16Max) {
    m_ret++;
  }
  if (s_ret) {
    m_ret = (~m_ret) + 1;
  }
  if (m_ret == 0) {
    s_ret = 0;
  }
  return static_cast<uint16_t>((s_ret << kBitShift15) | (m_ret));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int16_t
/// @return  Return int16_t value of fp_val which is the value of fp16_t object
static int16_t Fp16ToInt16(const uint16_t &fp_val) {
  int16_t ret;
  uint16_t ret_v;
  // 1.get s_ret and shift it to bit0.
  uint16_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_DENORM(fp_val)) {  // Denormalized number
    ret_v = 0;
    ret = *(reinterpret_cast<uint8_t *>(&ret_v));
    return ret;
  }

  uint64_t long_int_m = hf_m;
  uint8_t overflow_flag = 0;
  uint16_t shift_out = 0;
  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    overflow_flag = 1;
  } else {
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
        if (s_ret == 1 && long_int_m > 0x2000000Lu) {  // sign=1,negative number(<0)
          long_int_m = 0x2000000Lu;                    // 10(fp16_t-man)+15(int16)=25bit
          overflow_flag = 1;
          break;
        } else if (s_ret != 1 && long_int_m >= 0x1FFFFFFLu) {  // sign=0,positive number(>0) Overflow
          long_int_m = 0x1FFFFFFLu;                            // 10(fp16_t-man)+15(int16)=25bit
          overflow_flag = 1;
          break;
        }
      } else {
        hf_e++;
        shift_out++;
      }
    }
  }
  if (overflow_flag) {
    ret_v = kInt16Max + s_ret;
  } else {
    // Generate final result
    ret_v = GetUint16ValByMan(s_ret, long_int_m, shift_out);
  }
  ret = *(reinterpret_cast<int16_t *>(&ret_v));
  return ret;
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint16_t
/// @return  Return uint16_t value of fp_val which is the value of fp16_t object
static uint16_t Fp16ToUInt16(const uint16_t &fp_val) {
  uint16_t m_ret = 0;
  // 1.get s_ret and shift it to bit0.
  uint16_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_DENORM(fp_val)) {  // Denormalized number
    return 0;
  }

  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    m_ret = ~0;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    m_ret = static_cast<uint16_t>((long_int_m >> (kFp16ManLen + shift_out)) & kBitLen16Max);
    if (need_round && m_ret != kBitLen16Max) {
      m_ret++;
    }
  }

  if (s_ret == 1) {  // Negative number
    m_ret = 0;
  }
  // m_ret equal to final result
  return m_ret;
}
/// @ingroup fp16_t math convertion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to int32_t
/// @return  Return int32_t value of fp_val which is the value of fp16_t object
static int32_t Fp16ToInt32(const uint16_t &fp_val) {
  uint32_t ret_v;
  // 1.get s_ret and shift it to bit0.
  uint32_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    ret_v = kInt32Max + s_ret;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0;

    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    auto m_ret = static_cast<uint32_t>((long_int_m >> (kFp16ManLen + shift_out)) & kBitLen32Max);
    if (need_round && m_ret < kInt32Max) {
      m_ret++;
    }

    if (s_ret == 1) {
      m_ret = (~m_ret) + 1;
    }
    if (m_ret == 0) {
      s_ret = 0;
    }
    // Generate final result
    ret_v = (s_ret << kBitShift31) | (m_ret);
  }

  return *(reinterpret_cast<int32_t *>(&ret_v));
}
/// @ingroup fp16_t math conversion static method
/// @param [in] fp_val uint16_t value of fp16_t object
/// @brief   Convert fp16_t to uint32_t
/// @return  Return uint32_t value of fp_val which is the value of fp16_t object
static uint32_t Fp16ToUInt32(const uint16_t &fp_val) {
  uint32_t m_ret;
  // 1.get s_ret and shift it to bit0.
  uint32_t s_ret = FP16_EXTRAC_SIGN(fp_val);
  // 2.get hf_e and hf_m
  uint16_t hf_e = FP16_EXTRAC_EXP(fp_val);
  uint16_t hf_m = FP16_EXTRAC_MAN(fp_val);

  if (FP16_IS_DENORM(fp_val)) {  // Denormalized number
    return 0u;
  }

  if (FP16_IS_INVALID(fp_val)) {  // Inf or NaN
    m_ret = ~0u;
  } else {
    uint64_t long_int_m = hf_m;
    uint16_t shift_out = 0;
    while (hf_e != kFp16ExpBias) {
      if (hf_e > kFp16ExpBias) {
        hf_e--;
        long_int_m = long_int_m << 1;
      } else {
        hf_e++;
        shift_out++;
      }
    }
    bool need_round = IsRoundOne(long_int_m, shift_out + kFp16ManLen);
    m_ret = static_cast<uint32_t>(long_int_m >> (kFp16ManLen + shift_out)) & kBitLen32Max;
    if (need_round && m_ret != kBitLen32Max) {
      m_ret++;
    }
  }

  if (s_ret == 1) {  // Negative number
    m_ret = 0;
  }
  // m_ret equal to final result
  return m_ret;
}
static uint16_t Fp16AddCalVal(uint16_t &s_ret, int16_t e_ret, uint16_t m_ret, uint32_t m_trunc, uint16_t shift_out) {
  uint16_t m_min = kFp16ManHideBit << shift_out;
  uint16_t m_max = m_min << 1;
  // Denormal
  while (m_ret < m_min && e_ret > 0) {  // the value of m_ret should not be smaller than 2^23
    m_ret = m_ret << 1;
    m_ret += (kFp32SignMask & m_trunc) >> kFp32SignIndex;
    m_trunc = m_trunc << 1;
    e_ret = e_ret - 1;
  }
  while (m_ret >= m_max) {  // the value of m_ret should be smaller than 2^24
    m_trunc = m_trunc >> 1;
    m_trunc = m_trunc | (kFp32SignMask * (m_ret & 1));
    m_ret = m_ret >> 1;
    e_ret = e_ret + 1;
  }

  bool b_last_bit = ((m_ret & 1) > 0);
  bool b_trunc_high = 0;
  bool b_trunc_left = 0;
  b_trunc_high = (kRoundToNearest == g_round_mode) && ((m_trunc & kFp32SignMask) > 0);
  b_trunc_left = (kRoundToNearest == g_round_mode) && ((m_trunc & kFp32AbsMax) > 0);
  m_ret = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_ret, shift_out);
  while (m_ret >= m_max) {
    m_ret = m_ret >> 1;
    e_ret = e_ret + 1;
  }

  if (e_ret == 0 && m_ret <= m_max) {
    m_ret = m_ret >> 1;
  }
  Fp16Normalize(e_ret, m_ret);
  uint16_t ret = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return ret;
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t addition
/// @return  Return fp16_t result of adding this and fp
static uint16_t Fp16Add(uint16_t v_1, uint16_t v_2) {
  uint16_t s_a;
  uint16_t s_b;
  int16_t e_a;
  int16_t e_b;
  uint32_t m_a;
  uint32_t m_b;
  uint16_t m_a_tmp;
  uint16_t m_b_tmp;
  uint16_t shift_out = 0;
  // 1.Extract
  ExtractFp16(v_1, s_a, e_a, m_a_tmp);
  ExtractFp16(v_2, s_b, e_b, m_b_tmp);
  m_a = m_a_tmp;
  m_b = m_b_tmp;

  uint16_t sum;
  uint16_t s_ret;
  if (s_a != s_b) {
    ReverseMan(s_a > 0, m_a);
    ReverseMan(s_b > 0, m_b);
    sum = static_cast<uint16_t>(GetManSum(e_a, m_a, e_b, m_b));
    s_ret = (sum & kFp16SignMask) >> kFp16SignIndex;
    ReverseMan(s_ret > 0, m_a);
    ReverseMan(s_ret > 0, m_b);
  } else {
    sum = static_cast<uint16_t>(GetManSum(e_a, m_a, e_b, m_b));
    s_ret = s_a;
  }

  if (sum == 0) {
    shift_out = 3;  // shift to left 3 bits
    m_a = m_a << shift_out;
    m_b = m_b << shift_out;
  }

  uint32_t m_trunc = 0;
  int16_t e_ret = std::max(e_a, e_b);
  int16_t e_tmp = std::abs(e_a - e_b);
  if (e_a > e_b) {
    m_trunc = (m_b << (kBitShift32 - static_cast<uint16_t>(e_tmp)));
    m_b = RightShift(m_b, e_tmp);
  } else if (e_a < e_b) {
    m_trunc = (m_a << (kBitShift32 - static_cast<uint16_t>(e_tmp)));
    m_a = RightShift(m_a, e_tmp);
  }
  // calculate mantissav
  auto m_ret = static_cast<uint16_t>(m_a + m_b);
  return Fp16AddCalVal(s_ret, e_ret, m_ret, m_trunc, shift_out);
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t subtraction
/// @return  Return fp16_t result of subtraction fp from this
static uint16_t Fp16Sub(uint16_t v_1, uint16_t v_2) {
  // Reverse
  uint16_t tmp = ((~(v_2)) & kFp16SignMask) | (v_2 & kFp16AbsMax);
  return Fp16Add(v_1, tmp);
}
/// @ingroup fp16_t math operator
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t multiplication
/// @return  Return fp16_t result of multiplying this and fp
static uint16_t Fp16Mul(uint16_t v_1, uint16_t v_2) {
  uint16_t s_a, s_b;
  int16_t e_a, e_b;
  uint32_t m_a, m_b;
  uint16_t s_ret, m_ret;
  int16_t e_ret;
  uint32_t mul_m;
  uint16_t m_a_tmp, m_b_tmp;
  // 1.Extract
  ExtractFp16(v_1, s_a, e_a, m_a_tmp);
  ExtractFp16(v_2, s_b, e_b, m_b_tmp);
  m_a = m_a_tmp;
  m_b = m_b_tmp;

  e_ret = e_a + e_b - kFp16ExpBias - kDim10;
  mul_m = m_a * m_b;
  s_ret = s_a ^ s_b;

  uint32_t m_min = kFp16ManHideBit;
  uint32_t m_max = m_min << 1;
  uint32_t m_trunc = 0;
  // the value of m_ret should not be smaller than 2^23
  while (mul_m < m_min && e_ret > 1) {
    mul_m = mul_m << 1;
    e_ret = e_ret - 1;
  }
  while (mul_m >= m_max || e_ret < 1) {
    m_trunc = m_trunc >> 1;
    m_trunc = m_trunc | (kFp32SignMask * (mul_m & 1));
    mul_m = mul_m >> 1;
    e_ret = e_ret + 1;
  }
  bool b_last_bit = ((mul_m & 1) > 0);
  bool b_trunc_high = 0;
  bool b_trunc_left = 0;
  b_trunc_high = (kRoundToNearest == g_round_mode) && ((m_trunc & kFp32SignMask) > 0);
  b_trunc_left = (kRoundToNearest == g_round_mode) && ((m_trunc & kFp32AbsMax) > 0);
  mul_m = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, mul_m);

  while (mul_m >= m_max || e_ret < 0) {
    mul_m = mul_m >> 1;
    e_ret = e_ret + 1;
  }

  if (e_ret == 1 && mul_m < kFp16ManHideBit) {
    e_ret = 0;
  }
  m_ret = static_cast<uint16_t>(mul_m);

  Fp16Normalize(e_ret, m_ret);

  uint16_t ret = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return ret;
}
/// @ingroup fp16_t math operator divided
/// @param [in] v_1 left operator value of fp16_t object
/// @param [in] v_2 right operator value of fp16_t object
/// @brief   Performing fp16_t division
/// @return  Return fp16_t result of division this by fp
static uint16_t Fp16Div(uint16_t v_1, uint16_t v_2) {
  uint16_t ret;
  if (FP16_IS_ZERO(v_2)) {  // result is inf
    // throw "fp16_t division by zero.";
    uint16_t s_a, s_b;
    uint16_t s_ret;
    s_a = FP16_EXTRAC_SIGN(v_1);
    s_b = FP16_EXTRAC_SIGN(v_2);
    s_ret = s_a ^ s_b;
    ret = FP16_CONSTRUCTOR(s_ret, kFp16MaxExp, 0u);
  } else if (FP16_IS_ZERO(v_1)) {
    ret = 0u;
  } else {
    uint16_t s_a, s_b;
    int16_t e_a, e_b;
    uint64_t m_a, m_b;
    float m_div;
    uint16_t m_a_tmp, m_b_tmp;
    // 1.Extract
    ExtractFp16(v_1, s_a, e_a, m_a_tmp);
    ExtractFp16(v_2, s_b, e_b, m_b_tmp);
    m_a = m_a_tmp;
    m_b = m_b_tmp;

    uint64_t m_tmp;
    if (e_a > e_b) {
      m_tmp = m_a;
      uint16_t tmp;
      tmp = e_a - e_b;
      for (int i = 0; i < tmp; i++) {
        m_tmp = m_tmp << 1;
      }
      m_a = m_tmp;
    } else if (e_a < e_b) {
      m_tmp = m_b;
      uint16_t tmp = e_b - e_a;
      for (int i = 0; i < tmp; i++) {
        m_tmp = m_tmp << 1;
      }
      m_b = m_tmp;
    }
    m_div = static_cast<float>(m_a * 1.0f / m_b);
    fp16_t fp_div;
    fp_div = m_div;
    ret = fp_div.val;
    if (s_a != s_b) {
      ret |= kFp16SignMask;
    }
  }
  return ret;
}

// operate
fp16_t fp16_t::operator+(const fp16_t fp) {
  uint16_t ret_val = Fp16Add(val, fp.val);
  fp16_t ret(ret_val);
  return ret;
}
fp16_t fp16_t::operator-(const fp16_t fp) {
  uint16_t ret_val = Fp16Sub(val, fp.val);
  fp16_t ret(ret_val);
  return ret;
}
fp16_t fp16_t::operator*(const fp16_t fp) {
  uint16_t ret_val = Fp16Mul(val, fp.val);
  fp16_t ret(ret_val);
  return ret;
}
fp16_t fp16_t::operator/(const fp16_t fp) {
  uint16_t ret_val = Fp16Div(val, fp.val);
  fp16_t ret(ret_val);
  return ret;
}

fp16_t fp16_t::operator+=(const fp16_t fp) {
  val = Fp16Add(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator-=(const fp16_t fp) {
  val = Fp16Sub(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator*=(const fp16_t fp) {
  val = Fp16Mul(val, fp.val);
  return *this;
}
fp16_t fp16_t::operator/=(const fp16_t fp) {
  val = Fp16Div(val, fp.val);
  return *this;
}

// compare
bool fp16_t::operator==(const fp16_t &fp) const {
  bool result = true;
  if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
    result = true;
  } else {
    result = ((val & kBitLen16Max) == (fp.val & kBitLen16Max));  // bit compare
  }
  return result;
}
bool fp16_t::operator!=(const fp16_t &fp) const {
  bool result = true;
  if (FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val)) {
    result = false;
  } else {
    result = ((val & kBitLen16Max) != (fp.val & kBitLen16Max));  // bit compare
  }
  return result;
}
bool fp16_t::operator>(const fp16_t &fp) const {
  uint16_t s_a, s_b;
  uint16_t e_a, e_b;
  uint16_t m_a, m_b;
  bool result = true;

  // 1.Extract
  s_a = FP16_EXTRAC_SIGN(val);
  s_b = FP16_EXTRAC_SIGN(fp.val);
  e_a = FP16_EXTRAC_EXP(val);
  e_b = FP16_EXTRAC_EXP(fp.val);
  m_a = FP16_EXTRAC_MAN(val);
  m_b = FP16_EXTRAC_MAN(fp.val);

  // Compare
  if ((s_a == 0) && (s_b > 0)) {  // +  -
    // -0=0
    result = !(FP16_IS_ZERO(val) && FP16_IS_ZERO(fp.val));
  } else if ((s_a == 0) && (s_b == 0)) {  // + +
    if (e_a > e_b) {                      // e_a - e_b >= 1; Va always larger than Vb
      result = true;
    } else if (e_a == e_b) {
      result = m_a > m_b;
    } else {
      result = false;
    }
  } else if ((s_a > 0) && (s_b > 0)) {  // - -    opposite to  + +
    if (e_a < e_b) {
      result = true;
    } else if (e_a == e_b) {
      result = m_a < m_b;
    } else {
      result = false;
    }
  } else {  // -  +
    result = false;
  }

  return result;
}
bool fp16_t::operator>=(const fp16_t &fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = true;
  } else if ((*this) == fp) {
    result = true;
  } else {
    result = false;
  }

  return result;
}
bool fp16_t::operator<(const fp16_t &fp) const {
  bool result = true;
  if ((*this) >= fp) {
    result = false;
  } else {
    result = true;
  }

  return result;
}
bool fp16_t::operator<=(const fp16_t &fp) const {
  bool result = true;
  if ((*this) > fp) {
    result = false;
  } else {
    result = true;
  }

  return result;
}

// evaluation
fp16_t &fp16_t::operator=(const fp16_t &fp) {
  if (&fp == this) {
    return *this;
  }
  val = fp.val;
  return *this;
}
fp16_t &fp16_t::operator=(const float &f_val) {
  uint16_t s_ret, m_ret;
  int16_t e_ret;
  uint32_t e_f, m_f;
  const uint32_t ui32_v = *(reinterpret_cast<const uint32_t *>(&f_val));  // 1:8:23bit sign:exp:man
  uint32_t m_len_delta;

  s_ret = static_cast<uint16_t>((ui32_v & kFp32SignMask) >> kFp32SignIndex);  // 4Byte->2Byte
  e_f = (ui32_v & kFp32ExpMask) >> kFp32ManLen;                               // 8 bit exponent
  m_f = (ui32_v & kFp32ManMask);  // 23 bit mantissa dont't need to care about denormal
  m_len_delta = kFp32ManLen - kFp16ManLen;

  bool need_round = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (e_f > 0x8Fu) {  // 0x8Fu:142=127+15
    e_ret = kFp16MaxExp - 1;
    m_ret = kFp16MaxMan;
  } else if (e_f <= 0x70u) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
    e_ret = 0;
    if (e_f >= 0x67) {  // 0x67:103=127-24 Denormal
      m_f = (m_f | kFp32ManHideBit);
      uint16_t shift_out = kFp32ManLen;
      uint64_t m_tmp = (static_cast<uint64_t>(m_f)) << (e_f - 0x67);

      need_round = IsRoundOne(m_tmp, shift_out);
      m_ret = static_cast<uint16_t>(m_tmp >> shift_out);
      if (need_round) {
        m_ret++;
      }
    } else if (e_f == 0x66 && m_f > 0) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
      m_ret = 1;
    } else {
      m_ret = 0;
    }
  } else {  // Regular case with no overflow or underflow
    e_ret = static_cast<int16_t>(e_f - 0x70u);

    need_round = IsRoundOne(m_f, static_cast<uint16_t>(m_len_delta));
    m_ret = static_cast<uint16_t>(m_f >> m_len_delta);
    if (need_round) {
      m_ret++;
    }
    if (m_ret & kFp16ManHideBit) {
      e_ret++;
    }
  }

  Fp16Normalize(e_ret, m_ret);
  val = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return *this;
}
fp16_t &fp16_t::operator=(const int8_t &i_val) {
  uint16_t s_ret, e_ret, m_ret;

  s_ret = static_cast<uint16_t>(((static_cast<uint8_t>(i_val)) & 0x80) >> kDim7);
  m_ret = static_cast<uint16_t>(((static_cast<uint8_t>(i_val)) & kInt8Max));

  if (m_ret == 0) {
    e_ret = 0;
  } else {
    if (s_ret) {                                       // negative number(<0)
      m_ret = static_cast<uint16_t>(std::abs(i_val));  // complement
    }

    e_ret = kFp16ManLen;
    while ((m_ret & kFp16ManHideBit) == 0) {
      m_ret = m_ret << 1;
      e_ret = e_ret - 1;
    }
    e_ret = e_ret + kFp16ExpBias;
  }

  val = FP16_CONSTRUCTOR(s_ret, e_ret, m_ret);
  return *this;
}
fp16_t &fp16_t::operator=(const uint8_t &ui_val) {
  uint16_t s_ret, e_ret, m_ret;
  s_ret = 0;
  e_ret = 0;
  m_ret = ui_val;
  if (m_ret) {
    e_ret = kFp16ManLen;
    while ((m_ret & kFp16ManHideBit) == 0) {
      m_ret = m_ret << 1;
      e_ret = e_ret - 1;
    }
    e_ret = e_ret + kFp16ExpBias;
  }

  val = FP16_CONSTRUCTOR(s_ret, e_ret, m_ret);
  return *this;
}
static void SetValByUint16Val(const uint16_t &input_val, const uint16_t &sign, uint16_t &ret_val) {
  uint32_t m_tmp = (input_val & kFp32AbsMax);
  uint16_t m_min = kFp16ManHideBit;
  uint16_t m_max = m_min << 1;
  uint16_t len = static_cast<uint16_t>(GetManBitLength(m_tmp));
  if (m_tmp) {
    int16_t e_ret;
    if (len > kDim11) {
      e_ret = kFp16ExpBias + kFp16ManLen;
      uint16_t e_tmp = len - kDim11;
      uint32_t trunc_mask = 1;
      for (int i = 1; i < e_tmp; i++) {
        trunc_mask = (trunc_mask << 1) + 1;
      }
      uint32_t m_trunc = (m_tmp & trunc_mask) << (kBitShift32 - e_tmp);
      for (int i = 0; i < e_tmp; i++) {
        m_tmp = (m_tmp >> 1);
        e_ret = e_ret + 1;
      }
      bool b_last_bit = ((m_tmp & 1) > 0);
      bool b_trunc_high = 0;
      bool b_trunc_left = 0;
      if (kRoundToNearest == g_round_mode) {  // trunc
        b_trunc_high = ((m_trunc & kFp32SignMask) > 0);
        b_trunc_left = ((m_trunc & kFp32AbsMax) > 0);
      }
      m_tmp = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_tmp);
      while (m_tmp >= m_max || e_ret < 0) {
        m_tmp = m_tmp >> 1;
        e_ret = e_ret + 1;
      }
    } else {
      e_ret = kFp16ExpBias;
      m_tmp = m_tmp << (kManBitLength - len);
      e_ret = e_ret + (len - 1);
    }
    auto m_ret = static_cast<uint16_t>(m_tmp);
    ret_val = FP16_CONSTRUCTOR(sign, static_cast<uint16_t>(e_ret), m_ret);
  }
}

fp16_t &fp16_t::operator=(const int16_t &i_val) {
  if (i_val == 0) {
    val = 0;
  } else {
    uint16_t ui_val = *(reinterpret_cast<const uint16_t *>(&i_val));
    auto s_ret = static_cast<uint16_t>(ui_val >> kBitShift15);
    if (s_ret) {
      int16_t iValM = -i_val;
      ui_val = *(reinterpret_cast<uint16_t *>(&iValM));
    }
    SetValByUint16Val(ui_val, s_ret, val);
  }
  return *this;
}
fp16_t &fp16_t::operator=(const uint16_t &ui_val) {
  if (ui_val == 0) {
    val = 0;
  } else {
    int16_t e_ret;
    uint16_t m_ret = ui_val;
    uint16_t m_min = kFp16ManHideBit;
    uint16_t m_max = m_min << 1;
    uint16_t len = static_cast<uint16_t>(GetManBitLength(m_ret));
    if (len > kManBitLength) {
      e_ret = kFp16ExpBias + kFp16ManLen;
      uint32_t m_trunc;
      uint32_t trunc_mask = 1;
      uint16_t e_tmp = len - kManBitLength;
      for (int i = 1; i < e_tmp; i++) {
        trunc_mask = (trunc_mask << 1) + 1;
      }
      m_trunc = (m_ret & trunc_mask) << (kBitShift32 - e_tmp);
      for (int i = 0; i < e_tmp; i++) {
        m_ret = (m_ret >> 1);
        e_ret = e_ret + 1;
      }
      bool b_last_bit = ((m_ret & 1) > 0);
      bool b_trunc_high = 0;
      bool b_trunc_left = 0;
      if (kRoundToNearest == g_round_mode) {  // trunc
        b_trunc_high = ((m_trunc & kFp32SignMask) > 0);
        b_trunc_left = ((m_trunc & kFp32AbsMax) > 0);
      }
      m_ret = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_ret);
      while (m_ret >= m_max || e_ret < 0) {
        m_ret = m_ret >> 1;
        e_ret = e_ret + 1;
      }
      if (FP16_IS_INVALID(val)) {
        val = kFp16Max;
      }
    } else {
      e_ret = kFp16ExpBias;
      m_ret = m_ret << (kDim11 - len);
      e_ret = e_ret + (len - 1);
    }
    val = FP16_CONSTRUCTOR(0u, static_cast<uint16_t>(e_ret), m_ret);
  }
  return *this;
}
static void SetValByUint32Val(const uint32_t &input_val, const uint16_t &sign, uint16_t &ret_val) {
  int16_t e_ret;
  uint32_t m_tmp = (input_val & kFp32AbsMax);
  uint32_t m_min = kFp16ManHideBit;
  uint32_t m_max = m_min << 1;
  uint16_t len = static_cast<uint16_t>(GetManBitLength(m_tmp));
  if (len > kDim11) {
    e_ret = kFp16ExpBias + kFp16ManLen;
    uint32_t m_trunc = 0;
    uint32_t trunc_mask = 1;
    uint16_t e_tmp = len - kDim11;
    for (int i = 1; i < e_tmp; i++) {
      trunc_mask = (trunc_mask << 1) + 1;
    }
    m_trunc = (m_tmp & trunc_mask) << (kBitShift32 - e_tmp);
    for (int i = 0; i < e_tmp; i++) {
      m_tmp = (m_tmp >> 1);
      e_ret = e_ret + 1;
    }
    bool b_last_bit = ((m_tmp & 1) > 0);
    bool b_trunc_high = 0;
    bool b_trunc_left = 0;
    if (kRoundToNearest == g_round_mode) {  // trunc
      b_trunc_high = ((m_trunc & kFp32SignMask) > 0);
      b_trunc_left = ((m_trunc & kFp32AbsMax) > 0);
    }
    m_tmp = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_tmp);
    while (m_tmp >= m_max || e_ret < 0) {
      m_tmp = m_tmp >> 1;
      e_ret = e_ret + 1;
    }
    if (e_ret >= kFp16MaxExp) {
      e_ret = kFp16MaxExp - 1;
      m_tmp = kFp16MaxMan;
    }
  } else {
    e_ret = kFp16ExpBias;
    m_tmp = m_tmp << (kDim11 - len);
    e_ret = e_ret + (len - 1);
  }
  auto m_ret = static_cast<uint16_t>(m_tmp);
  ret_val = FP16_CONSTRUCTOR(sign, static_cast<uint16_t>(e_ret), m_ret);
}
fp16_t &fp16_t::operator=(const int32_t &i_val) {
  if (i_val == 0) {
    val = 0;
  } else {
    uint32_t ui_val = *(reinterpret_cast<const uint32_t *>(&i_val));
    auto s_ret = static_cast<uint16_t>(ui_val >> kBitShift31);
    if (s_ret) {
      int32_t iValM = -i_val;
      ui_val = *(reinterpret_cast<uint32_t *>(&iValM));
    }
    SetValByUint32Val(ui_val, s_ret, val);
  }
  return *this;
}
fp16_t &fp16_t::operator=(const uint32_t &ui_val) {
  if (ui_val == 0) {
    val = 0;
  } else {
    int16_t e_ret;
    uint32_t m_tmp = ui_val;
    uint32_t m_min = kFp16ManHideBit;
    uint32_t m_max = m_min << 1;
    uint16_t len = static_cast<uint16_t>(GetManBitLength(m_tmp));
    if (len > kDim11) {
      e_ret = kFp16ExpBias + kFp16ManLen;
      uint32_t m_trunc = 0;
      uint32_t trunc_mask = 1;
      uint16_t e_tmp = len - kDim11;
      for (int i = 1; i < e_tmp; i++) {
        trunc_mask = (trunc_mask << 1) + 1;
      }
      m_trunc = (m_tmp & trunc_mask) << static_cast<uint32_t>(kBitShift32 - e_tmp);
      for (uint16_t i = 0; i < e_tmp; i++) {
        m_tmp = (m_tmp >> 1);
        e_ret = e_ret + 1;
      }
      bool b_last_bit = ((m_tmp & 1) > 0);
      bool b_trunc_high = false;
      bool b_trunc_left = false;
      if (g_round_mode == kRoundToNearest) {  // trunc
        b_trunc_high = ((m_trunc & kFp32SignMask) > 0);
        b_trunc_left = ((m_trunc & kFp32AbsMax) > 0);
      }
      m_tmp = ManRoundToNearest(b_last_bit, b_trunc_high, b_trunc_left, m_tmp);
      while (m_tmp >= m_max || e_ret < 0) {
        m_tmp = m_tmp >> 1;
        e_ret = e_ret + 1;
      }
      if (e_ret >= kFp16MaxExp) {
        e_ret = kFp16MaxExp - 1;
        m_tmp = kFp16MaxMan;
      }
    } else {
      e_ret = kFp16ExpBias;
      m_tmp = m_tmp << (kDim11 - len);
      e_ret = e_ret + (len - 1);
    }
    auto m_ret = static_cast<uint16_t>(m_tmp);
    val = FP16_CONSTRUCTOR(0u, static_cast<uint16_t>(e_ret), m_ret);
  }
  return *this;
}
fp16_t &fp16_t::operator=(const double &d_val) {
  uint16_t s_ret;
  uint16_t m_ret;
  int16_t e_ret;
  uint64_t e_d;
  uint64_t m_d;
  uint64_t ui64_v = *(reinterpret_cast<const uint64_t *>(&d_val));  // 1:11:52bit sign:exp:man
  uint32_t m_len_delta;

  s_ret = static_cast<uint16_t>((ui64_v & kFp64SignMask) >> kFp64SignIndex);  // 4Byte
  e_d = (ui64_v & kFp64ExpMask) >> kFp64ManLen;                               // 10 bit exponent
  m_d = (ui64_v & kFp64ManMask);                                              // 52 bit mantissa
  m_len_delta = kFp64ManLen - kFp16ManLen;

  bool need_round = false;
  // Exponent overflow/NaN converts to signed inf/NaN
  if (e_d >= 0x410u) {  // 0x410:1040=1023+16
    e_ret = kFp16MaxExp - 1;
    m_ret = kFp16MaxMan;
    val = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  } else if (e_d <= 0x3F0u) {  // Exponent underflow converts to denormalized half or signed zero
    // 0x3F0:1008=1023-15
    // Signed zeros, denormalized floats, and floats with small
    // exponents all convert to signed zero half precision.
    e_ret = 0;
    if (e_d >= 0x3E7u) {  // 0x3E7u:999=1023-24 Denormal
      // Underflows to a denormalized value
      m_d = (kFp64ManHideBit | m_d);
      uint16_t shift_out = kFp64ManLen;
      uint64_t m_tmp = (static_cast<uint64_t>(m_d)) << (e_d - 0x3E7u);

      need_round = IsRoundOne(m_tmp, shift_out);
      m_ret = static_cast<uint16_t>(m_tmp >> shift_out);
      if (need_round) {
        m_ret++;
      }
    } else if (e_d == 0x3E6u && m_d > 0) {
      m_ret = 1;
    } else {
      m_ret = 0;
    }
  } else {  // Regular case with no overflow or underflow
    e_ret = static_cast<int16_t>(e_d - 0x3F0u);

    need_round = IsRoundOne(m_d, m_len_delta);
    m_ret = static_cast<uint16_t>(m_d >> m_len_delta);
    if (need_round) {
      m_ret++;
    }
    if (m_ret & kFp16ManHideBit) {
      e_ret++;
    }
  }

  Fp16Normalize(e_ret, m_ret);
  val = FP16_CONSTRUCTOR(s_ret, static_cast<uint16_t>(e_ret), m_ret);
  return *this;
}

// convert
fp16_t::operator float() const {
  return Fp16ToFloat(val);
}
fp16_t::operator double() const {
  return Fp16ToDouble(val);
}
fp16_t::operator int8_t() const {
  return Fp16ToInt8(val);
}
fp16_t::operator uint8_t() const {
  return Fp16ToUInt8(val);
}
fp16_t::operator int16_t() const {
  return Fp16ToInt16(val);
}
fp16_t::operator uint16_t() const {
  return Fp16ToUInt16(val);
}
fp16_t::operator int32_t() const {
  return Fp16ToInt32(val);
}
fp16_t::operator uint32_t() const {
  return Fp16ToUInt32(val);
}
// Cannot be used, just in order to solve the compile error
fp16_t::operator int64_t() const {
  return 0;
}
// Cannot be used, just in order to solve the compile error
fp16_t::operator uint64_t() const {
  return 0;
}

int fp16_t::IsInf() {
  if ((val & kFp16AbsMax) == kFp16ExpMask) {
    if (val & kFp16SignMask) {
      return -1;
    } else {
      return 1;
    }
  } else {
    return 0;
  }
}

float fp16_t::ToFloat() const {
  return Fp16ToFloat(val);
}
double fp16_t::ToDouble() const {
  return Fp16ToDouble(val);
}
int8_t fp16_t::ToInt8() const {
  return Fp16ToInt8(val);
}
uint8_t fp16_t::ToUInt8() const {
  return Fp16ToUInt8(val);
}
int16_t fp16_t::ToInt16() const {
  return Fp16ToInt16(val);
}
uint16_t fp16_t::ToUInt16() const {
  return Fp16ToUInt16(val);
}
int32_t fp16_t::ToInt32() const {
  return Fp16ToInt32(val);
}
uint32_t fp16_t::ToUInt32() const {
  return Fp16ToUInt32(val);
}
}  // namespace ge
