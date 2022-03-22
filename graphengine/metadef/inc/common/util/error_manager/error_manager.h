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

#ifndef ERROR_MANAGER_H_
#define ERROR_MANAGER_H_

#include <map>
#include <set>
#include <string>
#include <vector>
#include <mutex>
#include <cstring>

namespace error_message {
using char_t = char;
#ifdef __GNUC__
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max,
                           const char_t *format, ...)__attribute__((format(printf, 3, 4)));
#define TRIM_PATH(x) (((x).find_last_of('/') != std::string::npos) ? (x).substr((x).find_last_of('/') + 1U) : (x))
#else
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...);
#define TRIM_PATH(x) (((x).find_last_of('\\') != std::string::npos) ? (x).substr((x).find_last_of('\\') + 1U) : (x))
#endif
}

constexpr size_t const LIMIT_PER_MESSAGE = 512U;

///
/// @brief Report error message
/// @param [in] key: vector parameter key
/// @param [in] value: vector parameter value
///
#define REPORT_INPUT_ERROR(error_code, key, value)                                          \
  ErrorManager::GetInstance().ATCReportErrMessage(error_code, key, value)

///
/// @brief Report error message
/// @param [in] key: vector parameter key
/// @param [in] value: vector parameter value
///
#define REPORT_ENV_ERROR(error_code, key, value)                                            \
  ErrorManager::GetInstance().ATCReportErrMessage(error_code, key, value)

#define REPORT_INNER_ERROR(error_code, fmt, ...)                                                                     \
do {                                                                                                                 \
  std::vector<char> error_string(LIMIT_PER_MESSAGE, '\0');                                                           \
  if (error_message::FormatErrorMessage(error_string.data(), error_string.size(), fmt, ##__VA_ARGS__) > 0) {         \
    if (error_message::FormatErrorMessage(error_string.data(), error_string.size(), "%s[FUNC:%s][FILE:%s][LINE:%d]", \
        error_string.data(), &__FUNCTION__[0], TRIM_PATH(std::string(__FILE__)).c_str(), __LINE__) > 0) {            \
      (void)ErrorManager::GetInstance().ReportInterErrMessage(error_code, std::string(error_string.data()));         \
    }                                                                                                                \
  }                                                                                                                  \
} while (false)

#define REPORT_CALL_ERROR REPORT_INNER_ERROR

namespace error_message {
  // first stage
  constexpr char_t const *kInitialize   = "INIT";
  constexpr char_t const *kModelCompile = "COMP";
  constexpr char_t const *kModelLoad    = "LOAD";
  constexpr char_t const *kModelExecute = "EXEC";
  constexpr char_t const *kFinalize     = "FINAL";

  // SecondStage
  // INITIALIZE
  constexpr char_t const *kParser               = "PARSER";
  constexpr char_t const *kOpsProtoInit         = "OPS_PRO";
  constexpr char_t const *kSystemInit           = "SYS";
  constexpr char_t const *kEngineInit           = "ENGINE";
  constexpr char_t const *kOpsKernelInit        = "OPS_KER";
  constexpr char_t const *kOpsKernelBuilderInit = "OPS_KER_BLD";
  // MODEL_COMPILE
  constexpr char_t const *kPrepareOptimize    = "PRE_OPT";
  constexpr char_t const *kOriginOptimize     = "ORI_OPT";
  constexpr char_t const *kSubGraphOptimize   = "SUB_OPT";
  constexpr char_t const *kMergeGraphOptimize = "MERGE_OPT";
  constexpr char_t const *kPreBuild           = "PRE_BLD";
  constexpr char_t const *kStreamAlloc        = "STM_ALLOC";
  constexpr char_t const *kMemoryAlloc        = "MEM_ALLOC";
  constexpr char_t const *kTaskGenerate       = "TASK_GEN";
  // COMMON
  constexpr char_t const *kOther = "DEFAULT";

  struct Context {
    uint64_t work_stream_id;
    std::string first_stage;
    std::string second_stage;
    std::string log_header;
  };
}

class ErrorManager {
 public:
  ///
  /// @brief Obtain  ErrorManager instance
  /// @return ErrorManager instance
  ///
  static ErrorManager &GetInstance();

  ///
  /// @brief init
  /// @return int 0(success) -1(fail)
  ///
  int32_t Init();

  ///
  /// @brief init
  /// @param [in] path: current so path
  /// @return int 0(success) -1(fail)
  ///
  int32_t Init(const std::string path);

  int32_t ReportInterErrMessage(const std::string error_code, const std::string &error_msg);

  ///
  /// @brief Report error message
  /// @param [in] error_code: error code
  /// @param [in] args_map: parameter map
  /// @return int 0(success) -1(fail)
  ///
  int32_t ReportErrMessage(const std::string error_code, const std::map<std::string, std::string> &args_map);

  ///
  /// @brief output error message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int32_t OutputErrMessage(int32_t handle);

  ///
  /// @brief output  message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int32_t OutputMessage(int32_t handle);

  std::string GetErrorMessage();

  std::string GetWarningMessage();

  ///
  /// @brief Report error message
  /// @param [in] key: vector parameter key
  /// @param [in] value: vector parameter value
  ///
  void ATCReportErrMessage(const std::string error_code, const std::vector<std::string> &key = {},
                           const std::vector<std::string> &value = {});

  ///
  /// @brief report graph compile failed message such as error code and op_name in mstune case
  /// @param [in] graph_name: root graph name
  /// @param [in] msg: failed message map, key is error code, value is op_name
  /// @return int 0(success) -1(fail)
  ///
  int32_t ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
                                       const std::map<std::string, std::string> &msg);

  ///
  /// @brief get graph compile failed message in mstune case
  /// @param [in] graph_name: graph name
  /// @param [out] msg_map: failed message map, key is error code, value is op_name list
  /// @return int 0(success) -1(fail)
  ///
  int32_t GetMstuneCompileFailedMsg(const std::string &graph_name,
                                std::map<std::string,
                                std::vector<std::string>> &msg_map);

  // @brief generate work_stream_id by current pid and tid, clear error_message stored by same work_stream_id
  // used in external api entrance, all sync api can use
  void GenWorkStreamIdDefault();

  // @brief generate work_stream_id by args sessionid and graphid, clear error_message stored by same work_stream_id
  // used in external api entrance
  void GenWorkStreamIdBySessionGraph(const uint64_t session_id, const uint64_t graph_id);

  const std::string &GetLogHeader();

  error_message::Context &GetErrorManagerContext();

  void SetErrorContext(const error_message::Context error_context);

  void SetStage(const std::string &first_stage, const std::string &second_stage);

  void SetStage(const error_message::char_t *first_stage, const size_t first_len,
                const error_message::char_t *second_stage, const size_t second_len);

 private:
  struct ErrorInfoConfig {
    std::string error_id;
    std::string error_message;
    std::vector<std::string> arg_list;
  };

  struct ErrorItem {
    std::string error_id;
    std::string error_message;

    bool operator==(const ErrorItem &rhs) const {
      return (error_id == rhs.error_id) && (error_message == rhs.error_message);
    }
  };

  ErrorManager() {}
  ~ErrorManager() {}

  ErrorManager(const ErrorManager &) = delete;
  ErrorManager(ErrorManager &&) = delete;
  ErrorManager &operator=(const ErrorManager &) = delete;
  ErrorManager &operator=(ErrorManager &&) = delete;

  int32_t ParseJsonFile(const std::string path);

  static int32_t ReadJsonFile(const std::string &file_path, void *const handle);

  void ClassifyCompileFailedMsg(const std::map<std::string, std::string> &msg,
                                std::map<std::string,
                                std::vector<std::string>> &classified_msg);

  bool IsInnerErrorCode(const std::string &error_code) const;

  inline bool IsValidErrorCode(const std::string &error_codes) const {
    const uint32_t kErrorCodeValidLength = 6U;
    return error_codes.size() == kErrorCodeValidLength;
  }

  std::vector<ErrorItem> &GetErrorMsgContainerByWorkId(uint64_t work_id);
  std::vector<ErrorItem> &GetWarningMsgContainerByWorkId(uint64_t work_id);

  void ClearErrorMsgContainerByWorkId(const uint64_t work_stream_id);
  void ClearWarningMsgContainerByWorkId(const uint64_t work_stream_id);

  bool is_init_ = false;
  std::mutex mutex_;
  std::map<std::string, ErrorInfoConfig> error_map_;
  std::map<std::string, std::map<std::string, std::vector<std::string>>> compile_failed_msg_map_;

  std::map<uint64_t, std::vector<ErrorItem>> error_message_per_work_id_;
  std::map<uint64_t, std::vector<ErrorItem>> warning_messages_per_work_id_;

  thread_local static error_message::Context error_context_;
};
#endif  // ERROR_MANAGER_H_
