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

#ifndef GE_COMMON_FP16_T_H_
#define GE_COMMON_FP16_T_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ge {
using DimIndex = enum {
  kDim0 = 0,
  kDim1,
  kDim2,
  kDim3,
  kDim4,
  kDim5,
  kDim6,
  kDim7,
  kDim8,
  kDim9,
  kDim10,
  kDim11,
  kDim12,
  kDim13,
  kDim14,
  kDim15,
  kDim16,
};

using BitShift = enum {
  kBitShift2 = 2,
  kBitShift3 = 3,
  kBitShift4 = 4,
  kBitShift5 = 5,
  kBitShift6 = 6,
  kBitShift7 = 7,
  kBitShift8 = 8,
  kBitShift9 = 9,
  kBitShift10 = 10,
  kBitShift11 = 11,
  kBitShift12 = 12,
  kBitShift13 = 13,
  kBitShift14 = 14,
  kBitShift15 = 15,
  kBitShift16 = 16,
  kBitShift20 = 20,
  kBitShift24 = 24,
  kBitShift27 = 27,
  kBitShift28 = 28,
  kBitShift31 = 31,
  kBitShift32 = 32,
  kBitShift36 = 36,
  kBitShift40 = 40,
  kBitShift44 = 44,
  kBitShift48 = 48,
  kBitShift52 = 52,
  kBitShift56 = 56,
  kBitShift59 = 59,
  kBitShift60 = 60,
  kBitShift63 = 63,
  kBitShift64 = 64,
  kBitShift128 = 128,
  kBitShift255 = 255,
  kBitShift256 = 256,
  kBitShift512 = 512,
  kBitShift768 = 768,
  kBitShift784 = 784,
  kBitShift1020 = 1020,
  kBitShift1024 = 1024,
  kBitShift3136 = 3136,
  kBitShift4096 = 4096,
  kBitShift6144 = 6144,
  kBitShift10240 = 10240,
  kBitShift65536 = 65536
};
/// @ingroup fp16 basic parameter
/// @brief   fp16 exponent bias
constexpr uint16_t kFp16ExpBias = 15;
/// @ingroup fp16 basic parameter
/// @brief   the exponent bit length of fp16 is 5
constexpr uint16_t kFp16ExpLen = 5;
/// @ingroup fp16 basic parameter
/// @brief   the mantissa bit length of fp16 is 10
constexpr uint16_t kFp16ManLen = 10;
/// @ingroup fp16 basic parameter
/// @brief   bit index of sign in fp16
constexpr uint16_t kFp16SignIndex = 15;
/// @ingroup fp16 basic parameter
/// @brief   sign mask of fp16         (1 00000 00000 00000)
constexpr uint16_t kFp16SignMask = 0x8000;
/// @ingroup fp16 basic parameter
/// @brief   exponent mask of fp16     (  11111 00000 00000)
constexpr uint16_t kFp16ExpMask = 0x7C00;
/// @ingroup fp16 basic parameter
/// @brief   mantissa mask of fp16     (        11111 11111)
constexpr uint16_t kFp16ManMask = 0x03FF;
/// @ingroup fp16 basic parameter
/// @brief   hide bit of mantissa of fp16(   1 00000 00000)
constexpr uint16_t kFp16ManHideBit = 0x0400;
/// @ingroup fp16 basic parameter
/// @brief   maximum value            (0111 1011 1111 1111)
constexpr uint16_t kFp16Max = 0x7BFF;
/// @ingroup fp16 basic parameter
/// @brief   minimum value            (1111 1011 1111 1111)
constexpr uint16_t kFp16Min = 0xFBFF;
/// @ingroup fp16 basic parameter
/// @brief   absolute maximum value   (0111 1111 1111 1111)
constexpr uint16_t kFp16AbsMax = 0x7FFF;
/// @ingroup fp16 basic parameter
/// @brief   maximum exponent value of fp16 is 15(11111)
constexpr uint16_t kFp16MaxExp = 0x001F;
/// @ingroup fp16 basic parameter
/// @brief   maximum valid exponent value of fp16 is 14(11110)
constexpr uint16_t kFp16MaxValidExp = 0x001E;
/// @ingroup fp16 basic parameter
/// @brief   maximum mantissa value of fp16(11111 11111)
constexpr uint16_t kFp16MaxMan = 0x03FF;
/// @ingroup fp16 basic parameter
/// @brief   absolute minimum normal value of fp16
///          (E=1,M=0 D=2^(-14)=0.00006103515625)
constexpr uint16_t kFp16MinNormal = 1.0f / (2 << 14);
/// @ingroup fp16 basic operator
/// @brief   get sign of fp16
#define FP16_EXTRAC_SIGN(x) (((x) >> 15) & 1)
/// @ingroup fp16 basic operator
/// @brief   get exponent of fp16
#define FP16_EXTRAC_EXP(x) (((x) >> 10) & kFp16MaxExp)
/// @ingroup fp16 basic operator
/// @brief   get mantissa of fp16
#define FP16_EXTRAC_MAN(x) ((((x) >> 0) & 0x3FF) | (((((x) >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400))
/// @ingroup fp16 basic operator
/// @brief   constructor of fp16 from sign exponent and mantissa
#define FP16_CONSTRUCTOR(s, e, m) (((s) << kFp16SignIndex) | ((e) << kFp16ManLen) | ((m)&kFp16MaxMan))
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is zero
#define FP16_IS_ZERO(x) (((x)&kFp16AbsMax) == 0)
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is a denormalized value
#define FP16_IS_DENORM(x) ((((x)&kFp16ExpMask) == 0))
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is infinite
#define FP16_IS_INF(x) (((x)&kFp16AbsMax) == kFp16ExpMask)
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is NaN
#define FP16_IS_NAN(x) (((x & kFp16ExpMask) == kFp16ExpMask) && (x & kFp16ManMask))
/// @ingroup fp16 special value judgment
/// @brief   whether a fp16 is invalid
#define FP16_IS_INVALID(x) ((x & kFp16ExpMask) == kFp16ExpMask)
/// @ingroup fp32 basic parameter
/// @brief   fp32 exponent bias
constexpr uint16_t kFp32ExpBias = 127;
/// @ingroup fp32 basic parameter
/// @brief   the exponent bit length of float/fp32 is 8
constexpr uint16_t kFp32ExpLen = 8;
/// @ingroup fp32 basic parameter
/// @brief   the mantissa bit length of float/fp32 is 23
constexpr uint16_t kFp32ManLen = 23;
/// @ingroup fp32 basic parameter
/// @brief   bit index of sign in float/fp32
constexpr uint16_t kFp32SignIndex = 31;
/// @ingroup fp32 basic parameter
/// @brief   sign mask of fp32         (1 0000 0000  0000 0000 0000 0000 000)
constexpr uint32_t kFp32SignMask = 0x80000000u;
/// @ingroup fp32 basic parameter
/// @brief   exponent mask of fp32     (  1111 1111  0000 0000 0000 0000 000)
constexpr uint32_t kFp32ExpMask = 0x7F800000u;
/// @ingroup fp32 basic parameter
/// @brief   mantissa mask of fp32     (             1111 1111 1111 1111 111)
constexpr uint32_t kFp32ManMask = 0x007FFFFFu;
/// @ingroup fp32 basic parameter
/// @brief   hide bit of mantissa of fp32      (  1  0000 0000 0000 0000 000)
constexpr uint32_t kFp32ManHideBit = 0x00800000u;
/// @ingroup fp32 basic parameter
/// @brief   absolute maximum value    (0 1111 1111  1111 1111 1111 1111 111)
constexpr uint32_t kFp32AbsMax = 0x7FFFFFFFu;
/// @ingroup fp32 basic parameter
/// @brief   maximum exponent value of fp32 is 255(1111 1111)
constexpr uint32_t kFp32MaxExp = 0xFF;
/// @ingroup fp32 basic parameter
/// @brief   maximum mantissa value of fp32    (1111 1111 1111 1111 1111 111)
constexpr uint32_t kFp32MaxMan = 0x7FFFFF;
/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is NaN
#define FP32_IS_NAN(x) (((x & kFp32ExpMask) == kFp32ExpMask) && (x & kFp32ManMask))
/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is infinite
#define FP32_IS_INF(x) (((x & kFp32ExpMask) == kFp32ExpMask) && (!(x & kFp32ManMask)))
/// @ingroup fp32 special value judgment
/// @brief   whether a fp32 is a denormalized value
#define FP32_IS_DENORM(x) ((((x)&kFp32ExpMask) == 0))
/// @ingroup fp32 basic operator
/// @brief   get sign of fp32
#define FP32_EXTRAC_SIGN(x) (((x) >> kFp32SignIndex) & 1)
/// @ingroup fp32 basic operator
/// @brief   get exponent of fp16
#define FP32_EXTRAC_EXP(x) (((x)&kFp32ExpMask) >> kFp32ManLen)
/// @ingroup fp32 basic operator
/// @brief   get mantissa of fp16
#define FP32_EXTRAC_MAN(x) (((x)&kFp32ManMask) | (((((x) >> kFp32ManLen) & kFp32MaxExp) > 0 ? 1 : 0) * kFp32ManHideBit))
/// @ingroup fp32 basic operator
/// @brief   constructor of fp32 from sign exponent and mantissa
#define FP32_CONSTRUCTOR(s, e, m) (((s) << kFp32SignIndex) | ((e) << kFp32ManLen) | ((m)&kFp32MaxMan))
/// @ingroup fp64 basic parameter
/// @brief   fp64 exponent bias
constexpr uint16_t kFp64ExpBias = 1023;
/// @ingroup fp64 basic parameter
/// @brief   the exponent bit length of double/fp64 is 11
constexpr uint16_t kFp64ExpLen = 11;
/// @ingroup fp64 basic parameter
/// @brief   the mantissa bit length of double/fp64 is 52
constexpr uint16_t kFp64ManLen = 52;
/// @ingroup fp64 basic parameter
/// @brief   bit index of sign in double/fp64 is 63
constexpr uint16_t kFp64SignIndex = 63;
/// @ingroup fp64 basic parameter
/// @brief   sign mask of fp64                (1 000                   (total 63bits 0))
constexpr uint64_t kFp64SignMask = 0x8000000000000000LLu;
/// @ingroup fp64 basic parameter
/// @brief   exponent mask of fp64            (0 1 11111 11111  0000?-?-(total 52bits 0))
constexpr uint64_t kFp64ExpMask = 0x7FF0000000000000LLu;
/// @ingroup fp64 basic parameter
/// @brief   mantissa mask of fp64            (                 1111?-?-(total 52bits 1))
constexpr uint64_t kFp64ManMask = 0x000FFFFFFFFFFFFFLLu;
/// @ingroup fp64 basic parameter
/// @brief   hide bit of mantissa of fp64     (               1 0000?-?-(total 52bits 0))
constexpr uint64_t kFp64ManHideBit = 0x0010000000000000LLu;
/// @ingroup fp64 basic parameter
/// @brief   absolute maximum value           (0 111?-?-(total 63bits 1))
constexpr uint64_t kFp64AbsMax = 0x7FFFFFFFFFFFFFFFLLu;
/// @ingroup fp64 basic parameter
/// @brief   maximum exponent value of fp64 is 2047(1 11111 11111)
constexpr uint64_t kFp64MaxExp = 0x07FF;
/// @ingroup fp64 basic parameter
/// @brief   maximum mantissa value of fp64  (111?-?-(total 52bits 1))
constexpr uint64_t kFp64MaxMan = 0xFFFFFFFFFFFLLu;
/// @ingroup fp64 special value judgment
/// @brief   whether a fp64 is NaN
#define FP64_IS_NAN(x) (((x & kFp64ExpMask) == kFp64ExpMask) && (x & kFp64ManMask))
/// @ingroup fp64 special value judgment
/// @brief   whether a fp64 is infinite
#define FP64_IS_INF(x) (((x & kFp64ExpMask) == kFp64ExpMask) && (!(x & kFp64ManMask)))
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int8_t            (0111 1111)
constexpr int8_t kInt8Max = 0x7F;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 8 bits length  (1111 111)
constexpr uint8_t kBitLen8Max = 0xFF;
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int16_t           (0111 1111 1111 1111)
constexpr int16_t kInt16Max = 0x7FFF;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 16 bits length (1111 1111 1111 1111)
constexpr uint16_t kBitLen16Max = 0xFFFF;
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int32_t           (0111 1111 1111 1111 1111 1111 1111 1111)
constexpr int32_t kInt32Max = 0x7FFFFFFFu;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 32 bits length (1111 1111 1111 1111 1111 1111 1111 1111)
constexpr uint32_t kBitLen32Max = 0xFFFFFFFFu;
/// @ingroup integer special value judgment
/// @brief   maximum positive value of int64_t
/// (0111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111)
constexpr int64_t kInt64Max = 0x7FFFFFFFFFFFFFFFu;
/// @ingroup integer special value judgment
/// @brief   maximum value of a data with 64 bits length
/// (1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111 1111)
constexpr uint64_t kBitLen64Max = 0xFFFFFFFFFFFFFFFFu;

/// @ingroup fp16_t enum
/// @brief   round mode of last valid digital
enum TagFp16RoundMode {
  kRoundToNearest = 0,  // < round to nearest even
  kRoundByTruncated,    // < round by truncated
  kRoundModeReserved,
};

/// @ingroup fp16_t
/// @brief   Half precision float
///          bit15:       1 bit SIGN      +---+-----+------------+
///          bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
///          bit0-9:      10bit MAN       +---+-----+------------+
using fp16_t = struct TagFp16 {
  uint16_t val;

 public:
  /// @ingroup fp16_t constructor
  /// @brief   Constructor without any param(default constructor)
  TagFp16(void) { val = 0x0u; }
  /// @ingroup fp16_t constructor
  /// @brief   Constructor with an uint16_t value
  TagFp16(const uint16_t &ui_val) : val(ui_val) {}
  /// @ingroup fp16_t constructor
  /// @brief   Constructor with a fp16_t object(copy constructor)
  TagFp16(const TagFp16 &fp) : val(fp.val) {}

  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be added
  /// @brief   Override addition operator to performing fp16_t addition
  /// @return  Return fp16_t result of adding this and fp
  TagFp16 operator+(const TagFp16 fp);
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be subtracted
  /// @brief   Override addition operator to performing fp16_t subtraction
  /// @return  Return fp16_t result of subtraction fp from this
  TagFp16 operator-(const TagFp16 fp);
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be multiplied
  /// @brief   Override multiplication operator to performing fp16_t multiplication
  /// @return  Return fp16_t result of multiplying this and fp
  TagFp16 operator*(const TagFp16 fp);
  /// @ingroup fp16_t math operator divided
  /// @param [in] fp fp16_t object to be divided
  /// @brief   Override division operator to performing fp16_t division
  /// @return  Return fp16_t result of division this by fp
  TagFp16 operator/(const TagFp16 fp);
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be added
  /// @brief   Override addition operator to performing fp16_t addition
  /// @return  Return fp16_t result of adding this and fp
  TagFp16 operator+=(const TagFp16 fp);
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be subtracted
  /// @brief   Override addition operator to performing fp16_t subtraction
  /// @return  Return fp16_t result of subtraction fp from this
  TagFp16 operator-=(const TagFp16 fp);
  /// @ingroup fp16_t math operator
  /// @param [in] fp fp16_t object to be multiplied
  /// @brief   Override multiplication operator to performing fp16_t multiplication
  /// @return  Return fp16_t result of multiplying this and fp
  TagFp16 operator*=(const TagFp16 fp);
  /// @ingroup fp16_t math operator divided
  /// @param [in] fp fp16_t object to be divided
  /// @brief   Override division operator to performing fp16_t division
  /// @return  Return fp16_t result of division this by fp
  TagFp16 operator/=(const TagFp16 fp);

  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t if-equal comparison
  /// @return  Return boolean result of if-equal comparison of this and fp.
  bool operator==(const TagFp16 &fp) const;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t not-equal comparison
  /// @return  Return boolean result of not-equal comparison of this and fp.
  bool operator!=(const TagFp16 &fp) const;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t greater-than comparison
  /// @return  Return boolean result of greater-than comparison of this and fp.
  bool operator>(const TagFp16 &fp) const;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t greater-equal comparison
  /// @return  Return boolean result of greater-equal comparison of this and fp.
  bool operator>=(const TagFp16 &fp) const;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t less-than comparison
  /// @return  Return boolean result of less-than comparison of this and fp.
  bool operator<(const TagFp16 &fp) const;
  /// @ingroup fp16_t math compare operator
  /// @param [in] fp fp16_t object to be compared
  /// @brief   Override basic comparison operator to performing fp16_t less-equal comparison
  /// @return  Return boolean result of less-equal comparison of this and fp.
  bool operator<=(const TagFp16 &fp) const;

  /// @ingroup fp16_t math evaluation operator
  /// @param [in] fp fp16_t object to be copy to fp16_t
  /// @brief   Override basic evaluation operator to copy fp16_t to a new fp16_t
  /// @return  Return fp16_t result from fp
  TagFp16 &operator=(const TagFp16 &fp);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] f_val float object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert float to fp16_t
  /// @return  Return fp16_t result from f_val
  TagFp16 &operator=(const float &f_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] d_val double object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert double to fp16_t
  /// @return  Return fp16_t result from d_val
  TagFp16 &operator=(const double &d_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] i_val float object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert float to fp16_t
  /// @return  Return fp16_t result from i_val
  TagFp16 &operator=(const int8_t &i_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] ui_val uint8_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert uint8_t to fp16_t
  /// @return  Return fp16_t result from ui_val
  TagFp16 &operator=(const uint8_t &ui_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] i_val int16_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert int16_t to fp16_t
  /// @return  Return fp16_t result from i_val
  TagFp16 &operator=(const int16_t &i_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] ui_val uint16_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert uint16_t to fp16_t
  /// @return  Return fp16_t result from ui_val
  TagFp16 &operator=(const uint16_t &ui_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] i_val int32_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert int32_t to fp16_t
  /// @return  Return fp16_t result from i_val
  TagFp16 &operator=(const int32_t &i_val);
  /// @ingroup fp16_t math evaluation operator
  /// @param [in] ui_val uint32_t object to be converted to fp16_t
  /// @brief   Override basic evaluation operator to convert uint32_t to fp16_t
  /// @return  Return fp16_t result from ui_val
  TagFp16 &operator=(const uint32_t &ui_val);
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  operator float() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to double/fp64
  /// @return  Return double/fp64 value of fp16_t
  operator double() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int8_t
  /// @return  Return int8_t value of fp16_t
  operator int8_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint8_t
  /// @return  Return uint8_t value of fp16_t
  operator uint8_t() const;
  /// @ingroup fp16_t conversion
  /// @brief   Override convert operator to convert fp16_t to int16_t
  /// @return  Return int16_t value of fp16_t
  operator int16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  operator uint16_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int32_t
  /// @return  Return int32_t value of fp16_t
  operator int32_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint32_t
  /// @return  Return uint32_t value of fp16_t
  operator uint32_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to int64_t
  /// @return  Return int64_t value of fp16_t
  operator int64_t() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Override convert operator to convert fp16_t to uint64_t
  /// @return  Return uint64_t value of fp16_t
  operator uint64_t() const;
  /// @ingroup fp16_t judgment method
  /// @param [in] fp fp16_t object to be judgement
  /// @brief   whether a fp16_t is inifinite
  /// @return  Returns 1:+INF -1:-INF 0:not INF
  int IsInf();
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to float/fp32
  /// @return  Return float/fp32 value of fp16_t
  float ToFloat() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to double/fp64
  /// @return  Return double/fp64 value of fp16_t
  double ToDouble() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to int8_t
  /// @return  Return int8_t value of fp16_t
  int8_t ToInt8() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint8_t
  /// @return  Return uint8_t value of fp16_t
  uint8_t ToUInt8() const;
  /// @ingroup fp16_t conversion
  /// @brief   Convert fp16_t to int16_t
  /// @return  Return int16_t value of fp16_t
  int16_t ToInt16() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint16_t
  /// @return  Return uint16_t value of fp16_t
  uint16_t ToUInt16() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to int32_t
  /// @return  Return int32_t value of fp16_t
  int32_t ToInt32() const;
  /// @ingroup fp16_t math conversion
  /// @brief   Convert fp16_t to uint32_t
  /// @return  Return uint32_t value of fp16_t
  uint32_t ToUInt32() const;
};

/// @ingroup fp16_t public method
/// @param [in]     val signature is negative
/// @param [in|out] s   sign of fp16_t object
/// @param [in|out] e   exponent of fp16_t object
/// @param [in|out] m   mantissa of fp16_t object
/// @brief   Extract the sign, exponent and mantissa of a fp16_t object
void ExtractFp16(const uint16_t &val, uint16_t &s, int16_t &e, uint16_t &m);
/// @ingroup fp16_t public method
/// @param [in]     negative sign is negative
/// @param [in|out] man      mantissa to be reverse
/// @brief   Calculate a mantissa's complement (add ont to it's radix-minus-one complement)
/// @return  Return complement of man
template <typename T>
void ReverseMan(bool negative, T &man) {
  if (negative) {
    man = (~(man)) + 1;
  }
}
/// @ingroup fp16_t public method
/// @param [in] e_a exponent of one fp16_t/float number
/// @param [in] m_a mantissa of one fp16_t/float number
/// @param [in] e_b exponent of another fp16_t/float number
/// @param [in] m_b mantissa of another fp16_t/float number
/// @brief   choose mantissa to be shift right whoes exponent is less than another one
/// @return  Return mantissawhoes exponent is less than another one
template <typename T>
T MinMan(const int16_t &e_a, T &m_a, const int16_t &e_b, T &m_b) {
  return (e_a > e_b) ? m_b : m_a;
}
/// @ingroup fp16_t public method
/// @param [in] man   mantissa to be operate
/// @param [in] shift right shift bits
/// @brief   right shift a mantissa
/// @return  Return right-shift mantissa
template <typename T>
T RightShift(T man, int16_t shift) {
  int bits = sizeof(T) * 8;  // one byte have 8 bits
  T mask = (((T)1u) << ((unsigned int)(bits - 1)));
  for (int i = 0; i < shift; i++) {
    man = ((man & mask) | (man >> 1));
  }
  return man;
}
/// @ingroup fp16_t public method
/// @param [in] e_a exponent of one temp fp16_t number
/// @param [in] m_a mantissa of one temp fp16_t number
/// @param [in] e_b exponent of another temp fp16_t number
/// @param [in] m_b mantissa of another temp fp16_t number
/// @brief   Get mantissa sum of two temp fp16_t numbers, T support types: uint16_t/uint32_t/uint64_t
/// @return  Return mantissa sum
template <typename T>
T GetManSum(int16_t e_a, const T &m_a, int16_t e_b, const T &m_b) {
  T sum = 0;
  if (e_a != e_b) {
    T m_tmp = 0;
    int16_t e_tmp = std::abs(e_a - e_b);
    if (e_a > e_b) {
      m_tmp = m_b;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = m_a + m_tmp;
    } else {
      m_tmp = m_a;
      m_tmp = RightShift(m_tmp, e_tmp);
      sum = m_tmp + m_b;
    }
  } else {
    sum = m_a + m_b;
  }
  return sum;
}
/// @ingroup fp16_t public method
/// @param [in] bit0    whether the last preserved bit is 1 before round
/// @param [in] bit1    whether the abbreviation's highest bit is 1
/// @param [in] bitLeft whether the abbreviation's bits which not contain highest bit grater than 0
/// @param [in] man     mantissa of a fp16_t or float number, support types: uint16_t/uint32_t/uint64_t
/// @param [in] shift   abbreviation bits
/// @brief    Round fp16_t or float mantissa to nearest value
/// @return   Returns true if round 1,otherwise false;
template <typename T>
T ManRoundToNearest(bool bit0, bool bit1, bool bitLeft, T man, uint16_t shift = 0) {
  man = (man >> shift) + ((bit1 && (bitLeft || bit0)) ? 1 : 0);
  return man;
}
/// @ingroup fp16_t public method
/// @param [in] man    mantissa of a float number, support types: uint16_t/uint32_t/uint64_t
/// @brief   Get bit length of a uint32_t number
/// @return  Return bit length of man
template <typename T>
int16_t GetManBitLength(T man) {
  int16_t len = 0;
  while (man) {
    man >>= 1;
    len++;
  }
  return len;
}
}  // namespace ge
#endif  // GE_COMMON_FP16_T_H_
