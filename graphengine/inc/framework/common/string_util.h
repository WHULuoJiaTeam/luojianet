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

#ifndef INC_FRAMEWORK_COMMON_STRING_UTIL_H_
#define INC_FRAMEWORK_COMMON_STRING_UTIL_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <cctype>
#include <securec.h>

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace ge {
class GE_FUNC_VISIBILITY StringUtils {
 public:
  static std::string &Ltrim(std::string &s) {
#if __cplusplus >= 201103L
    (void)s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int32_t c) { return std::isspace(c) == 0; }));
#else
    (void)s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int32_t, int32_t>(std::isspace))));
#endif
    return s;
  }
  // lint -esym(551,*)
  static std::string &Rtrim(std::string &s) { /*lint !e618*/
#if __cplusplus >= 201103L
    (void)s.erase(std::find_if(s.rbegin(), s.rend(), [](int32_t c) { return std::isspace(c) == 0; }).base(), s.end());
#else
    (void)s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int32_t, int32_t>(std::isspace))).base(),
                  s.end());
#endif
    return s;
  }
  // lint -esym(551,*)
  ///
  ///  @ingroup domi_common
  ///  @brief delete spaces at the beginning and end of a string
  ///  @param [in] string to be trimmed
  ///  @return string after trim
  ///
  static std::string &Trim(std::string &s) {
    return Ltrim(Rtrim(s));
  }

  ///
  ///  @ingroup domi_common
  ///  @brief string splitting
  ///  @param [in] str string to be trimmed
  ///  @param [in] delim  separator
  ///  @return string array after segmentation
  ///
  static std::vector<std::string, std::allocator<std::string>> Split(const std::string &str, char delim) {
    std::vector<std::string, std::allocator<std::string>> elems;

    if (str.empty()) {
      elems.emplace_back("");
      return elems;
    }

    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, delim)) {
      elems.push_back(item);
    }

    auto str_size = str.size();
    if ((str_size > 0) && (str[str_size - 1] == delim)) {
      elems.emplace_back("");
    }

    return elems;
  }
  ///
  ///  @ingroup domi_common
  ///  @brief obtain the file name
  ///  @param [in] s path name
  ///  @return file name
  ///
  static std::string GetFileName(std::string &s) {
    if (s.empty()) {
      return "";
    }
    std::vector<std::string> files = StringUtils::Split(s, '/');

    return files.empty() ? "" : files[files.size() - 1];
  }
  ///
  ///  @ingroup domi_common
  ///  @brief full replacement
  ///  @link
  ///  @param [in] str str string to be replaced
  ///  @param [in] old_value  old Characters Before Replacement
  ///  @param [in] new_value  new Characters Before Replacement
  ///  @return string after replacement
  ///
  static std::string ReplaceAll(std::string str, const std::string &old_value, const std::string &new_value) {
    std::string::size_type cur_pos = 0;
    std::string::size_type old_length = old_value.length();
    std::string::size_type new_length = new_value.length();
    // cycle replace
    for (; cur_pos != std::string::npos; cur_pos += new_length) {
      if ((cur_pos = str.find(old_value, cur_pos)) != std::string::npos) {
        (void)str.replace(cur_pos, old_length, new_value);
      } else {
        break;
      }
    }
    return str;
  }

  ///
  ///  @ingroup domi_common
  ///  @brief checks whether a character string starts with a character string (prefix)
  ///  @link
  ///  @param [in] str string to be compared
  ///  @param [in] str_x prefix
  ///  @return if the value is a prefix, true is returned. Otherwise, false is returned
  ///
  static bool StartWith(const std::string &str, const std::string str_x) {
    return ((str.size() >= str_x.size()) && (str.compare(0, str_x.size(), str_x) == 0));
  }

  ///
  ///  @ingroup domi_common
  ///  @brief format string
  ///  @link
  ///  @param [in] format specifies the character string format
  ///  @param [in] ... format Filling Content
  ///  @return formatted string
  ///
  static std::string FormatString(const char *format, ...) {
    const uint32_t MAX_BUFFER_LEN = 1024;  // the stack memory plint check result must be less than 1024
    va_list args;
    va_start(args, format);
    char buffer[MAX_BUFFER_LEN] = {0};
    int32_t ret = vsnprintf_s(buffer, MAX_BUFFER_LEN, MAX_BUFFER_LEN - 1, format, args);
    va_end(args);
    return ret > 0 ? buffer : "";
  }
};
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_STRING_UTIL_H_
