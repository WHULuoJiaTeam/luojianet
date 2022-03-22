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

#include "common/util/error_manager/error_manager.h"

using namespace error_message;

thread_local Context ErrorManager::error_context_ = {0, "", "", ""};

namespace error_message {
int FormatErrorMessage(char *str_dst, size_t dst_max, const char *format, ...) {
  return 1;
}
}

  ErrorManager &ErrorManager::GetInstance() {
    static ErrorManager instance;
    return instance;
  }

  ///
  /// @brief init
  /// @param [in] path: current so path
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::Init(std::string path) { return 0; }

  ///
  /// @brief Report error message
  /// @param [in] error_code: error code
  /// @param [in] args_map: parameter map
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::ReportErrMessage(std::string error_code, const std::map<std::string, std::string> &args_map) {
    return 0;
  }

  std::string ErrorManager::GetErrorMessage() {
    return std::string();
  }

  std::string ErrorManager::GetWarningMessage() {
    return std::string();
  }

  int ErrorManager::ReportInterErrMessage(std::string error_code, const std::string &error_msg) {
    return 0;
  }

  ///
  /// @brief output error message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::OutputErrMessage(int handle) { return 0; }

  ///
  /// @brief output  message
  /// @param [in] handle: print handle
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::OutputMessage(int handle) { return 0; }

  ///
  /// @brief Report error message
  /// @param [in] key: vector parameter key
  /// @param [in] value: vector parameter value
  ///
  void ErrorManager::ATCReportErrMessage(std::string error_code, const std::vector<std::string> &key,
                                         const std::vector<std::string> &value) {
  }

  ///
  /// @brief report graph compile failed message such as error code and op_name in mstune case
  /// @param [in] msg: failed message map, key is error code, value is op_name
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
		                                 const std::map<std::string, std::string> &msg) { return 0; }

  ///
  /// @brief get graph compile failed message in mstune case
  /// @param [in] graph_name: graph name
  /// @param [out] msg_map: failed message map, key is error code, value is op_name list
  /// @return int 0(success) -1(fail)
  ///
  int ErrorManager::GetMstuneCompileFailedMsg(const std::string &graph_name, std::map<std::string, std::vector<std::string>> &msg_map) { return 0; }


  void ErrorManager::GenWorkStreamIdDefault() {}

  void ErrorManager::GenWorkStreamIdBySessionGraph(uint64_t session_id, uint64_t graph_id) {}

  const std::string &ErrorManager::GetLogHeader() { return error_context_.log_header; }

  struct error_message::Context &ErrorManager::GetErrorManagerContext() {
    static struct error_message::Context error_context;
    return error_context;
  }

void ErrorManager::SetErrorContext(struct error_message::Context error_context) {}

void ErrorManager::SetStage(const std::string &first_stage, const std::string &second_stage) {}
