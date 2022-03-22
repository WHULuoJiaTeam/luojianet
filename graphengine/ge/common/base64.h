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

#ifndef GE_COMMON_BASE64_H_
#define GE_COMMON_BASE64_H_

#include <algorithm>
#include <string>

#include "framework/common/debug/ge_log.h"
#include "external/ge/ge_error_codes.h"

namespace ge {
namespace {
const char *kBase64Chars =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
  "abcdefghijklmnopqrstuvwxyz"
  "0123456789+/";
const char kEqualSymbol = '=';
const size_t kBase64CharsNum = 64;
const size_t kThreeByteOneGroup = 3;
const size_t kFourByteOneGroup = 4;
const size_t kThreeByteOneGroupIndex0 = 0;
const size_t kThreeByteOneGroupIndex1 = 1;
const size_t kThreeByteOneGroupIndex2 = 2;
const size_t kFourByteOneGroupIndex0 = 0;
const size_t kFourByteOneGroupIndex1 = 1;
const size_t kFourByteOneGroupIndex2 = 2;
const size_t kFourByteOneGroupIndex3 = 3;
}  // namespace

namespace base64 {
static inline bool IsBase64Char(const char &c) { return (isalnum(c) || (c == '+') || (c == '/')); }

static std::string EncodeToBase64(const std::string &raw_data) {
  size_t encode_length = raw_data.size() / kThreeByteOneGroup * kFourByteOneGroup;
  encode_length += raw_data.size() % kThreeByteOneGroup == 0 ? 0 : kFourByteOneGroup;
  size_t raw_data_index = 0;
  size_t encode_data_index = 0;
  std::string encode_data;
  encode_data.resize(encode_length);

  for (; raw_data_index + kThreeByteOneGroup <= raw_data.size(); raw_data_index += kThreeByteOneGroup) {
    auto char_1 = static_cast<uint8_t>(raw_data[raw_data_index]);
    auto char_2 = static_cast<uint8_t>(raw_data[raw_data_index + kThreeByteOneGroupIndex1]);
    auto char_3 = static_cast<uint8_t>(raw_data[raw_data_index + kThreeByteOneGroupIndex2]);
    encode_data[encode_data_index++] = kBase64Chars[char_1 >> 2u];
    encode_data[encode_data_index++] = kBase64Chars[((char_1 << 4u) & 0x30) | (char_2 >> 4u)];
    encode_data[encode_data_index++] = kBase64Chars[((char_2 << 2u) & 0x3c) | (char_3 >> 6u)];
    encode_data[encode_data_index++] = kBase64Chars[char_3 & 0x3f];
  }

  if (raw_data_index < raw_data.size()) {
    auto tail = raw_data.size() - raw_data_index;
    auto char_1 = static_cast<uint8_t>(raw_data[raw_data_index]);
    if (tail == 1) {
      encode_data[encode_data_index++] = kBase64Chars[char_1 >> 2u];
      encode_data[encode_data_index++] = kBase64Chars[(char_1 << 4u) & 0x30];
      encode_data[encode_data_index++] = kEqualSymbol;
      encode_data[encode_data_index++] = kEqualSymbol;
    } else {
      auto char_2 = static_cast<uint8_t>(raw_data[raw_data_index + 1]);
      encode_data[encode_data_index++] = kBase64Chars[char_1 >> 2u];
      encode_data[encode_data_index++] = kBase64Chars[((char_1 << 4u) & 0x30) | (char_2 >> 4u)];
      encode_data[encode_data_index++] = kBase64Chars[(char_2 << 2u) & 0x3c];
      encode_data[encode_data_index++] = kEqualSymbol;
    }
  }
  return encode_data;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
static Status DecodeFromBase64(const std::string &base64_data, std::string &decode_data) {
  if (base64_data.size() % kFourByteOneGroup != 0) {
    GELOGE(PARAM_INVALID, "base64 data size must can be divided by 4, but given data size is %zu", base64_data.size());
    return PARAM_INVALID;
  }
  decode_data.clear();
  size_t base64_data_len = base64_data.size();
  uint8_t byte_4[kFourByteOneGroup];
  auto FindCharInBase64Chars = [&](const char &raw_char) -> uint8_t {
    auto char_pos = std::find(kBase64Chars, kBase64Chars + kBase64CharsNum, raw_char);
    return static_cast<uint8_t>(std::distance(kBase64Chars, char_pos)) & 0xff;
  };

  for (std::size_t input_data_index = 0; input_data_index < base64_data_len; input_data_index += kFourByteOneGroup) {
    for (size_t i = 0; i < kFourByteOneGroup; ++i) {
      if (base64_data[input_data_index + i] == kEqualSymbol &&
          input_data_index >= base64_data_len - kFourByteOneGroup && i > 1) {
        byte_4[i] = kBase64CharsNum;
      } else if (IsBase64Char(base64_data[input_data_index + i])) {
        byte_4[i] = FindCharInBase64Chars(base64_data[input_data_index + i]);
      } else {
        GELOGE(PARAM_INVALID, "given base64 data is illegal");
        return PARAM_INVALID;
      }
    }
    decode_data +=
      static_cast<char>((byte_4[kFourByteOneGroupIndex0] << 2u) + ((byte_4[kFourByteOneGroupIndex1] & 0x30) >> 4u));
    if (byte_4[kFourByteOneGroupIndex2] >= kBase64CharsNum) {
      break;
    } else if (byte_4[kFourByteOneGroupIndex3] >= kBase64CharsNum) {
      decode_data += static_cast<char>(((byte_4[kFourByteOneGroupIndex1] & 0x0f) << 4u) +
                                       ((byte_4[kFourByteOneGroupIndex2] & 0x3c) >> 2u));
      break;
    }
    decode_data += static_cast<char>(((byte_4[kFourByteOneGroupIndex1] & 0x0f) << 4u) +
                                     ((byte_4[kFourByteOneGroupIndex2] & 0x3c) >> 2u));
    decode_data +=
      static_cast<char>(((byte_4[kFourByteOneGroupIndex2] & 0x03) << 6u) + byte_4[kFourByteOneGroupIndex3]);
  }
  return SUCCESS;
}
#pragma GCC diagnostic pop
}  // namespace base64
}  // namespace ge
#endif  // GE_COMMON_BASE64_H_