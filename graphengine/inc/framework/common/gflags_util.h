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

#ifndef INC_FRAMEWORK_COMMON_GFLAGS_UTIL_H_
#define INC_FRAMEWORK_COMMON_GFLAGS_UTIL_H_

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

#include <gflags/gflags.h>
#include <string>

namespace ge {
class GE_FUNC_VISIBILITY GflagsUtils {
 public:
  static bool IsSetCommandTrue(const char *name) {
    std::string out;
    return gflags::GetCommandLineOption(name, &out) && out == "true";
  }

  ///
  /// @brief Determines whether the parameter is empty
  /// @param name name parameter name
  /// @return true if empty otherwise false
  ///
  static bool IsSetCommandNotEmpty(const char *name) {
    std::string out;
    return gflags::GetCommandLineOption(name, &out) && !out.empty();
  }

  ///
  /// @brief Determines whether the parameter is not default
  /// @param flag_name name parameter name
  /// @return true if not default otherwise false
  ///
  static bool IsCommandLineNotDefault(const char *flag_name) {
    google::CommandLineFlagInfo info;
    return GetCommandLineFlagInfo(flag_name, &info) && !info.is_default;
  }

  ///
  /// @brief Modify gflags to print help information
  /// @param flags_h Pass in the self-defined help parameter, it is recommended to be FLAGS_h
  /// @return void
  ///
  static void ChangeHelpFlags(bool flags_h) {
    if (flags_h || IsSetCommandTrue("help") || IsSetCommandTrue("helpfull") || IsSetCommandNotEmpty("helpon") ||
        IsSetCommandNotEmpty("helpmatch") || IsSetCommandTrue("helppackage") || IsSetCommandTrue("helpxml")) {
      gflags::SetCommandLineOption("help", "false");
      gflags::SetCommandLineOption("helpfull", "false");
      gflags::SetCommandLineOption("helpon", "");
      gflags::SetCommandLineOption("helpmatch", "");
      gflags::SetCommandLineOption("helppackage", "false");
      gflags::SetCommandLineOption("helpxml", "false");
      gflags::SetCommandLineOption("helpshort", "true");
    }
  }
};
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_GFLAGS_UTIL_H_
