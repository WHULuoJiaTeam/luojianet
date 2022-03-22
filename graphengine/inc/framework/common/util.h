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

#ifndef AIR_INC_FRAMEWORK_COMMON_UTIL_H_
#define AIR_INC_FRAMEWORK_COMMON_UTIL_H_

#include <climits>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <google/protobuf/text_format.h>
#include "external/graph/types.h"
#include "framework/common/debug/log.h"
#include "framework/common/scope_guard.h"
#include "framework/common/ge_inner_error_codes.h"

#define GE_CHECK_POSITIVE_SIZE_RANGE(size)                             \
  do {                                                                 \
    if ((size) <= 0) {                                                 \
      GELOGE(ge::FAILED, "param[%s] is not a positive number", #size); \
      return PARAM_INVALID;                                            \
    }                                                                  \
  } while (false)

#define CHECK_FALSE_EXEC(expr, exec_expr, ...) \
  {                                            \
    const bool b = (expr);                     \
    if (!b) {                                  \
      exec_expr;                               \
    }                                          \
  }

// new ge marco
// Encapsulate common resource releases
#define GE_MAKE_GUARD_RTMEM(var)         \
  GE_MAKE_GUARD(var, [&] {               \
    if (var) GE_CHK_RT(rtFreeHost(var)); \
  });

#define GE_MAKE_GUARD_RTSTREAM(var)           \
  GE_MAKE_GUARD(var, [&] {                    \
    if (var) GE_CHK_RT(rtStreamDestroy(var)); \
  });

// For propagating errors when calling a function.
#define GE_RETURN_IF_ERROR(expr)           \
  do {                                     \
    const ge::Status _chk_status = (expr); \
    if (_chk_status != ge::SUCCESS) {      \
      return _chk_status;                  \
    }                                      \
  } while (false)

#define GE_RETURN_WITH_LOG_IF_ERROR(expr, ...) \
  do {                                         \
    const ge::Status _chk_status = (expr);     \
    if (_chk_status != ge::SUCCESS) {          \
      GELOGE(ge::FAILED, __VA_ARGS__);         \
      return _chk_status;                      \
    }                                          \
  } while (false)

// check whether the parameter is true. If it is, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                             \
    if (condition) {                               \
      GELOGE(ge::FAILED, __VA_ARGS__);             \
      return ge::FAILED;                           \
    }                                              \
  } while (false)

// Check if the parameter is false. If yes, return FAILED and record the error log
#define GE_RETURN_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                              \
    const bool _condition = (condition);            \
    if (!_condition) {                              \
      GELOGE(ge::FAILED, __VA_ARGS__);              \
      return ge::FAILED;                            \
    }                                               \
  } while (false)

// Checks whether the parameter is true. If so, returns PARAM_INVALID and records the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_TRUE(condition, ...) \
  do {                                                       \
    if (condition) {                                         \
      GELOGE(ge::FAILED, __VA_ARGS__);                       \
      return ge::PARAM_INVALID;                              \
    }                                                        \
  } while (false)

// Check if the parameter is false. If yes, return PARAM_INVALID and record the error log
#define GE_RT_PARAM_INVALID_WITH_LOG_IF_FALSE(condition, ...) \
  do {                                                        \
    const bool _condition = (condition);                      \
    if (!_condition) {                                        \
      GELOGE(ge::FAILED, __VA_ARGS__);                        \
      return ge::PARAM_INVALID;                               \
    }                                                         \
  } while (false)

// Check if the parameter is null. If yes, return PARAM_INVALID and record the error
#define GE_CHECK_NOTNULL(val)                                                   \
  do {                                                                          \
    if ((val) == nullptr) {                                                     \
      REPORT_INNER_ERROR("E19999", "Param:%s is nullptr, check invalid", #val); \
      GELOGE(ge::FAILED, "[Check][Param:%s]null is invalid.", #val);            \
      return ge::PARAM_INVALID;                                                 \
    }                                                                           \
  } while (false)

// Check if the parameter is null. If yes, just return and record the error
#define GE_CHECK_NOTNULL_JUST_RETURN(val)                      \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val); \
      return;                                                  \
    }                                                          \
  } while (false)

// Check whether the parameter is null. If so, execute the exec_expr expression and record the error log
#define GE_CHECK_NOTNULL_EXEC(val, exec_expr)                  \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val); \
      exec_expr;                                               \
    }                                                          \
  } while (false)

// Check whether the parameter is null. If yes, return directly and record the error log
#define GE_RT_VOID_CHECK_NOTNULL(val)                          \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val); \
      return;                                                  \
    }                                                          \
  } while (false)

// Check if the parameter is null. If yes, return false and record the error log
#define GE_RT_FALSE_CHECK_NOTNULL(val)                         \
  do {                                                         \
    if ((val) == nullptr) {                                    \
      GELOGE(ge::FAILED, "param[%s] must not be null.", #val); \
      return false;                                            \
    }                                                          \
  } while (false)

// Check if the parameter is out of bounds
#define GE_CHECK_SIZE(size)                                   \
  do {                                                        \
    if ((size) == 0) {                                        \
      GELOGE(ge::FAILED, "param[%s] is out of range", #size); \
      return ge::PARAM_INVALID;                               \
    }                                                         \
  } while (false)

// Check if the value on the left is greater than or equal to the value on the right
#define GE_CHECK_GE(lhs, rhs)                                       \
  do {                                                              \
    if ((lhs) < (rhs)) {                                            \
      GELOGE(ge::FAILED, "param[%s] is less than[%s]", #lhs, #rhs); \
      return ge::PARAM_INVALID;                                     \
    }                                                               \
  } while (false)

// Check if the value on the left is less than or equal to the value on the right
#define GE_CHECK_LE(lhs, rhs)                                          \
  do {                                                                 \
    if ((lhs) > (rhs)) {                                               \
      GELOGE(ge::FAILED, "param[%s] is greater than[%s]", #lhs, #rhs); \
      return ge::PARAM_INVALID;                                        \
    }                                                                  \
  } while (false)

#define GE_DELETE_NEW_SINGLE(var) \
  do {                            \
    if ((var) != nullptr) {       \
      delete (var);               \
      (var) = nullptr;            \
    }                             \
  } while (false)

#define GE_DELETE_NEW_ARRAY(var) \
  do {                           \
    if ((var) != nullptr) {      \
      delete[](var);             \
      (var) = nullptr;           \
    }                            \
  } while (false)

#define GE_FREE_RT_LOG(addr)                                        \
  do {                                                              \
    if ((addr) != nullptr) {                                        \
      const rtError_t error = rtFree(addr);                         \
      if (error != RT_ERROR_NONE) {                                 \
        GELOGE(RT_FAILED, "Call rtFree failed, error: %#x", error); \
      }                                                             \
      (addr) = nullptr;                                             \
    }                                                               \
  } while (false)

/**
 * Finding an Integer Ceiling Value Without Precision Loss
 */
#define CEIL(N, n) (((N) + (n)-1) / (n))

///
/// @ingroup domi_common
/// @brief onverts Vector of a number to a string.
/// @param [in] v  Vector of a number
/// @return string
///
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(std::vector<T> &v) {
  std::stringstream ss;
  ss << "[";
  for (const T x : v) {
    ss << x;
    ss << ", ";
  }
  // Delete the two extra characters at the end of the line.
  std::string str = ss.str().substr(0U, ss.str().length() - 2U);
  str += "]";
  return str;
}

///
/// @ingroup domi_common
/// @brief Converts RepeatedField to String.
/// @param [in] rpd_field  RepeatedField
/// @return string
///
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(const google::protobuf::RepeatedField<T> &rpd_field) {
  std::stringstream ss;
  ss << "[";
  for (const T x : rpd_field) {
    ss << x;
    ss << ", ";
  }
  // Delete the two extra characters at the end of the line.
  std::string str = ss.str().substr(0U, ss.str().length() - 2U);
  str += "]";
  return str;
}

///
///  @ingroup ge_ir_utils
///  @brief RepeatedPtrField->String
///  @param [in] const rpd_field  RepeatedPtrField
///  @return String
///
template <typename T>
GE_FUNC_VISIBILITY std::string ToString(const google::protobuf::RepeatedPtrField<T> &rpd_ptr_field) {
  std::stringstream ss;
  ss << "[";
  for (const T &x : rpd_ptr_field) {
    ss << x;
    ss << ", ";
  }
  std::string str_ret = ss.str().substr(0U, ss.str().length() - 2U);
  str_ret += "]";
  return str_ret;
}

namespace ge {
/**
 * @ingroup domi_common
 * @brief version of om.proto file
 */
constexpr int32_t OM_PROTO_VERSION = 2;

///
/// @ingroup domi_common
/// @brief Reads the proto structure from an array.
/// @param [in] data proto data to be read
/// @param [in] size proto data size
/// @param [out] proto Memory for storing the proto file
/// @return true success
/// @return false fail
///
GE_FUNC_VISIBILITY bool ReadProtoFromArray(const void *const data, const int32_t size,
                                           google::protobuf::Message *const proto);

///
/// @ingroup domi_proto
/// @brief Reads the proto file in the text format.
/// @param [in] file path of proto file
/// @param [out] message Memory for storing the proto file
/// @return true success
/// @return false fail
///
GE_FUNC_VISIBILITY bool ReadProtoFromText(const char_t *const file, google::protobuf::Message *const message);

///
/// @ingroup: domi_common
/// @brief: get length of file
/// @param [in] input_file: path of file
/// @return int64_t： File length. If the file length fails to be obtained, the value -1 is returned.
///
GE_FUNC_VISIBILITY extern int64_t GetFileLength(const std::string &input_file);

///
/// @ingroup domi_common
/// @brief Reads all data from a binary file.
/// @param [in] file_name  path of file
/// @param [out] buffer  Output memory address, which needs to be released by the caller.
/// @param [out] length  Output memory size
/// @return false fail
/// @return true success
///
GE_FUNC_VISIBILITY bool ReadBytesFromBinaryFile(const char_t *const file_name, char_t **const buffer, int32_t &length);

GE_FUNC_VISIBILITY bool ReadBytesFromBinaryFile(const char_t *file_name, std::vector<char_t> &buffer);

///
/// @ingroup domi_common
/// @brief Recursively Creating a Directory
/// @param [in] directory_path  Path, which can be a multi-level directory.
/// @return 0 success
/// @return -1 fail
///
GE_FUNC_VISIBILITY extern int32_t CreateDirectory(const std::string &directory_path);

///
/// @ingroup domi_common
/// @brief Obtains the current time string.
/// @return Time character string in the format ： %Y%m%d%H%M%S, eg: 20171011083555
///
GE_FUNC_VISIBILITY std::string CurrentTimeInStr();

///
/// @ingroup domi_common
/// @brief Obtains the absolute time (timestamp) of the current system.
/// @return Timestamp, in microseconds (US)
///
///
GE_FUNC_VISIBILITY uint64_t GetCurrentTimestamp();

///
/// @ingroup domi_common
/// @brief Obtains the absolute time (timestamp) of the current system.
/// @return Timestamp, in seconds (US)
///
///
GE_FUNC_VISIBILITY uint32_t GetCurrentSecondTimestap();

///
/// @ingroup domi_common
/// @brief Check whether the product of two int64 numbers exceeds the int64 range.
/// @param [in] a
/// @param [in] b
/// @return false: true: The result is within the normal int64 range.
///
GE_FUNC_VISIBILITY bool CheckInt64MulOverflow(const int64_t a, const int64_t b);

///
/// @ingroup domi_common
/// @brief Absolute path for obtaining files.
/// @param [in] path of input file
/// @param [out] Absolute path of a file. If the absolute path cannot be obtained, an empty string is returned
///
GE_FUNC_VISIBILITY std::string RealPath(const char_t *path);

///
/// @ingroup domi_common
/// @brief Check whether the specified input file path is valid.
/// 1.  The specified path cannot be empty.
/// 2.  The path can be converted to an absolute path.
/// 3.  The file path exists and is readable.
/// @param [in] file_path path of input file
/// @param [out] result
///
GE_FUNC_VISIBILITY bool CheckInputPathValid(const std::string &file_path, const std::string &atc_param = "");

///
/// @ingroup domi_common
/// @brief Checks whether the specified output file path is valid.
/// @param [in] file_path path of output file
/// @param [out] result
///
GE_FUNC_VISIBILITY bool CheckOutputPathValid(const std::string &file_path, const std::string &atc_param = "");

///
/// @ingroup domi_common
/// @brief Check whether the file path meets the whitelist verification requirements.
/// @param [in] str file path
/// @param [out] result
///
GE_FUNC_VISIBILITY bool ValidateStr(const std::string &file_path, const std::string &mode);

///
/// @ingroup domi_common
/// @brief Check path invalid
/// @param [in] path, path to be checked
/// @param [in] length, length of path
/// @return 0 success
/// @return -1 fail
///
GE_FUNC_VISIBILITY Status CheckPath(const char_t *path, size_t length);
}  // namespace ge

#endif  // AIR_INC_FRAMEWORK_COMMON_UTIL_H_
