/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include <fstream>
#include <iostream>
#include <algorithm>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <cstdarg>
#include <securec.h>

#include "mmpa/mmpa_api.h"
#include "toolchain/slog.h"

#define GE_MODULE_NAME static_cast<int32_t>(GE)

class GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    thread_local static const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    thread_local static const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

#define GELOGE(fmt, ...)                                                      \
  dlog_error(GE_MODULE_NAME, "%lu %s: %s" fmt, GeLog::GetTid(), __FUNCTION__, \
             ErrorManager::GetInstance().GetLogHeader().c_str(), ##__VA_ARGS__)
#define GELOGW(fmt, ...)                      \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) \
  dlog_warn(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GELOGI(fmt, ...)                      \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) \
  dlog_info(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)
#define GELOGD(fmt, ...)                       \
  if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) \
  dlog_debug(GE_MODULE_NAME, "%lu %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__)

namespace error_message {
int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...) {
  int32_t ret;
  va_list arg_list;

  va_start(arg_list, format);
  ret = vsprintf_s(str_dst, dst_max, format, arg_list);
  va_end(arg_list);
  (void)arg_list;

  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, pattern:%s", ret, format);
  }
  return ret;
}
}

namespace {

#ifdef __GNUC__
const error_message::char_t *const kErrorCodePath = "../conf/error_manager/error_code.json";
const error_message::char_t *const kSeparator = "/";
#else
const error_message::char_t *const kErrorCodePath = "..\\conf\\error_manager\\error_code.json";
const error_message::char_t *const kSeparator = "\\";
#endif

const error_message::char_t *const kErrorList = "error_info_list";
const error_message::char_t *const kErrCode = "ErrCode";
const error_message::char_t *const kErrMessage = "ErrMessage";
const error_message::char_t *const kArgList = "Arglist";
const uint64_t kLength = 2UL;

std::string &Ltrim(std::string &s) {
  (void)s.erase(s.begin(),
                std::find_if(s.begin(), s.end(), [](const error_message::char_t c) { return !std::isspace(c); }));
  return s;
}

std::string &Rtrim(std::string &s) {
  (void)s.erase(
      std::find_if(s.rbegin(), s.rend(), [](const error_message::char_t c) { return !std::isspace(c); }).base(),
      s.end());
  return s;
}

/// @ingroup domi_common
/// @brief trim space
std::string &Trim(std::string &s) { return Ltrim(Rtrim(s)); }

///
/// @brief Obtain error manager self library path
/// @return store liberror_manager.so path
///
std::string GetSelfLibraryDir(void) {
  mmDlInfo dl_info;
  if (mmDladdr(reinterpret_cast<void *>(GetSelfLibraryDir), &dl_info) != EN_OK) {
    const char *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    GELOGW("Failed to read the shared library file path! reason:%s", error);
    return std::string();
  } else {
    std::string so_path = dl_info.dli_fname;
    error_message::char_t path[MMPA_MAX_PATH] = {'0'};
    if (so_path.length() >= MMPA_MAX_PATH) {
        GELOGW("The shared library file path is too long!");
        return std::string();
    }
    if (mmRealPath(so_path.c_str(), path, MMPA_MAX_PATH) != EN_OK) {
      GELOGW("Failed to get realpath of %s, reason:%s", so_path.c_str(), strerror(errno));
      return std::string();
    }

    so_path = path;
    so_path = so_path.substr(0UL, so_path.rfind(kSeparator) + 1UL);
    return so_path;
  }
}

// split string
std::vector<std::string> Split(const std::string &str, error_message::char_t delim) {
  std::vector<std::string> elems;

  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    elems.push_back(Trim(item));
  }
  const auto str_size = str.size();
  if ((str_size > 0UL) && (str[str_size - 1UL] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}

}  // namespace

using namespace error_message;

thread_local Context ErrorManager::error_context_ = {0UL, "", "", ""};

///
/// @brief Obtain ErrorManager instance
/// @return ErrorManager instance
///
ErrorManager &ErrorManager::GetInstance() {
  static ErrorManager instance;
  return instance;
}

///
/// @brief init
/// @param [in] path: current so path
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init(const std::string path) {
  if (is_init_) {
    return 0;
  }
  const std::string file_path = path + kErrorCodePath;
  const int32_t ret = ParseJsonFile(file_path);
  if (ret != 0) {
    GELOGW("[Parse][File]Parser config file:%s failed", file_path.c_str());
    return -1;
  }
  is_init_ = true;
  return 0;
}

///
/// @brief init
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::Init() {
    return Init(GetSelfLibraryDir());
}

int32_t ErrorManager::ReportInterErrMessage(const std::string error_code, const std::string &error_msg) {
  const uint64_t kMaxWorkSize = 1000UL;
  if (!is_init_) {
    GELOGI("ErrorManager has not inited, can't report error message");
    return -1;
  }
  if (!IsInnerErrorCode(error_code)) {
    GELOGE("[Report][Error]error_code %s is not internal error code", error_code.c_str());
    return -1;
  }

  if (error_context_.work_stream_id == 0UL) {
    if (error_message_per_work_id_.size() > kMaxWorkSize) {
      GELOGE("[Report][Error]error_code %s, error work_stream total size exceed %lu, skip record",
             error_code.c_str(), kMaxWorkSize);
      return -1;
    }
    GenWorkStreamIdDefault();
  }

  const std::unique_lock<std::mutex> lock(mutex_);
  auto& error_messages = GetErrorMsgContainerByWorkId(error_context_.work_stream_id);
  auto& warning_messages = GetWarningMsgContainerByWorkId(error_context_.work_stream_id);

  if (error_messages.size() > kMaxWorkSize) {
    GELOGE("[Report][Error]error_code %s, error work_stream_id:%u item size exceed %lu, skip record",
            error_code.c_str(), error_context_.work_stream_id, kMaxWorkSize);
    return -1;
  }

  ErrorManager::ErrorItem item = {error_code, error_msg};
  if (error_code[0UL] == 'W') {
    const auto it = find(warning_messages.begin(), warning_messages.end(), item);
    if (it == warning_messages.end()) {
      warning_messages.emplace_back(item);
    }
  } else {
    const auto it = find(error_messages.begin(), error_messages.end(), item);
    if (it == error_messages.end()) {
      error_messages.emplace_back(item);
    }
  }
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] args_map: parameter map
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportErrMessage(const std::string error_code,
                                       const std::map<std::string, std::string> &args_map) {
  if (!is_init_) {
    GELOGI("ErrorManager has not inited, can't report error message");
    return 0;
  }

  if (error_context_.work_stream_id == 0UL) { GenWorkStreamIdDefault(); }

  const auto it = error_map_.find(error_code);
  if (it == error_map_.end()) {
    GELOGE("[Report][Error]error_code %s is not registered", error_code.c_str());
    return -1;
  }
  const ErrorInfoConfig &error_info = it->second;
  std::string error_message = error_info.error_message;
  for (const std::string &arg : error_info.arg_list) {
    if (arg.empty()) {
      GELOGI("arg is null");
      break;
    }
    const auto arg_it = args_map.find(arg);
    if (arg_it == args_map.end()) {
      GELOGE("[Report][Error]error_code: %s, arg %s is not existed in map", error_code.c_str(), arg.c_str());
      return -1;
    }
    const std::string &arg_value = arg_it->second;
    const auto index = error_message.find("%s");
    if (index == std::string::npos) {
      GELOGE("[Report][Error]error_code: %s, %s location in error_message is not found",
             error_code.c_str(), arg.c_str());
      return -1;
    }
    (void)error_message.replace(index, kLength, arg_value);
  }

  if (error_context_.work_stream_id == 0UL) {
    GELOGW("work_id in this work stream is zero, work_id set action maybe forgeted in some externel api.");
  }

  const std::unique_lock<std::mutex> lock(mutex_);
  auto& error_messages = GetErrorMsgContainerByWorkId(error_context_.work_stream_id);
  auto& warning_messages = GetWarningMsgContainerByWorkId(error_context_.work_stream_id);

  ErrorManager::ErrorItem error_item = {error_code, error_message};
  if (error_code[0UL] == 'W') {
    const auto it = find(warning_messages.begin(), warning_messages.end(), error_item);
    if (it == warning_messages.end()) { warning_messages.emplace_back(error_item); }
  } else {
    const auto it = find(error_messages.begin(), error_messages.end(), error_item);
    if (it == error_messages.end()) { error_messages.emplace_back(error_item); }
  }
  return 0;
}

std::string ErrorManager::GetErrorMessage() {
  const std::unique_lock<std::mutex> lock(mutex_);
  auto& error_messages = GetErrorMsgContainerByWorkId(error_context_.work_stream_id);

  if (error_messages.empty()) {
    return "";
  }

  std::stringstream err_stream;
  std::string first_code = error_messages[0UL].error_id;
  for (auto &item : error_messages) {
    if (!IsInnerErrorCode(item.error_id)) {
      first_code = item.error_id;
      err_stream << first_code << ": " << item.error_message << std::endl;
      break;
    }
  }
  if (IsInnerErrorCode(first_code)) {
    err_stream << first_code << ": Inner Error!" << std::endl;
    for (auto &item : error_messages) {
      err_stream << "        " << item.error_message << std::endl;
    }
  } else {
    for (auto &item : error_messages) {
      if (first_code == item.error_id) {
        continue;
      }
      err_stream << "        " << item.error_message << std::endl;
    }
  }
  ClearErrorMsgContainerByWorkId(error_context_.work_stream_id);
  return err_stream.str();
}

std::string ErrorManager::GetWarningMessage() {
  const std::unique_lock<std::mutex> lock(mutex_);
  auto& warning_messages = GetWarningMsgContainerByWorkId(error_context_.work_stream_id);

  std::stringstream warning_stream;
  for (auto &item : warning_messages) {
    warning_stream << item.error_id << ": " << item.error_message << std::endl;
  }
  ClearWarningMsgContainerByWorkId(error_context_.work_stream_id);
  return warning_stream.str();
}

///
/// @brief output error message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputErrMessage(int32_t handle) {
  std::string err_msg = GetErrorMessage();

  if (err_msg.empty()) {
    std::stringstream err_stream;
    err_stream << "E19999: Inner Error!" << std::endl;
    err_stream << "        " << "Unknown error occurred. Please check the log." << std::endl;
    err_msg = err_stream.str();
  }

  if (handle <= fileno(stderr)) {
    std::cout << err_msg << std::endl;
  } else {
    const mmSsize_t ret =
        mmWrite(handle, const_cast<error_message::char_t *>(err_msg.c_str()), static_cast<uint32_t>(err_msg.length()));
    if (ret == -1) {
      GELOGE("[Write][File]fail, reason:%s",  strerror(errno));
      return -1;
    }
  }
  return 0;
}

///
/// @brief output message
/// @param [in] handle: print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::OutputMessage(int32_t handle) {
  const std::string warning_msg = GetWarningMessage();
  std::cout << warning_msg << std::endl;
  handle = 0;
  return handle;
}

///
/// @brief parse json file
/// @param [in] path: json path
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ParseJsonFile(const std::string path) {
  GELOGI("Begin to parser json file");
  nlohmann::json json_file;
  const int32_t status = ReadJsonFile(path, &json_file);
  if (status != 0) {
    GELOGW("[Read][JsonFile]file path is %s", path.c_str());
    return -1;
  }

  try {
    const nlohmann::json &error_list_json = json_file[kErrorList];
    if (error_list_json.is_null()) {
      GELOGW("[Check][Config]The message of error_info_list is not found in %s", path.c_str());
      return -1;
    }
    if (!error_list_json.is_array()) {
      GELOGW("[Check][Config]The message of error_info_list is not array in %s", path.c_str());
      return -1;
    }

    for (size_t i = 0UL; i < error_list_json.size(); i++) {
      ErrorInfoConfig error_info;
      error_info.error_id = error_list_json[i][kErrCode];
      error_info.error_message = error_list_json[i][kErrMessage];
      error_info.arg_list = Split(error_list_json[i][kArgList], ',');
      const auto it = error_map_.find(error_info.error_id);
      if (it != error_map_.end()) {
        GELOGW("[Check][Config]There are the same error code %s in %s",
               error_info.error_id.c_str(), path.c_str());
        return -1;
      }
      (void)error_map_.emplace(error_info.error_id, error_info);
    }
  } catch (const nlohmann::json::exception &e) {
    GELOGW("[Parse][JsonFile]the file path is %s, exception message: %s", path.c_str(), e.what());
    return -1;
  }

  GELOGI("Parse json file success");
  return 0;
}

///
/// @brief read json file
/// @param [in] file_path: json path
/// @param [in] handle:  print handle
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReadJsonFile(const std::string &file_path, void *const handle) {
  GELOGI("Begin to read json file");
  if (file_path.empty()) {
    GELOGW("[Read][JsonFile]path %s is not valid", file_path.c_str());
    return -1;
  }
  nlohmann::json *const json_file = reinterpret_cast<nlohmann::json *const>(handle);
  if (json_file == nullptr) {
    GELOGW("[Check][Param]JsonFile is nullptr");
    return -1;
  }
  const error_message::char_t *file = file_path.data();
  if ((mmAccess2(file, M_F_OK)) != EN_OK) {
    GELOGW("[Read][JsonFile] %s is not exist, error %s", file_path.c_str(), strerror(errno));
    return -1;
  }

  std::ifstream ifs(file_path);
  if (!ifs.is_open()) {
    GELOGW("[Read][JsonFile]Open %s failed", file_path.c_str());
    return -1;
  }

  try {
    ifs >> *json_file;
  } catch (const nlohmann::json::exception &e) {
    GELOGW("[Read][JsonFile]ifstream to json fail. path: %s, exception message: %s.", file_path.c_str(), e.what());
    ifs.close();
    return -1;
  }

  ifs.close();
  GELOGI("Read json file success");
  return 0;
}

///
/// @brief report error message
/// @param [in] error_code: error code
/// @param [in] vector parameter key, vector parameter value
/// @return int 0(success) -1(fail)
///
void ErrorManager::ATCReportErrMessage(const std::string error_code, const std::vector<std::string> &key,
                                       const std::vector<std::string> &value) {
  if (!is_init_) {
    GELOGI("ErrorManager has not inited, can't report error message");
    return;
  }
  std::map<std::string, std::string> args_map;
  if (key.empty()) {
    (void)ErrorManager::GetInstance().ReportErrMessage(error_code, args_map);
  } else if (key.size() == value.size()) {
    for (size_t i = 0UL; i < key.size(); ++i) {
      (void)args_map.insert(std::make_pair(key[i], value[i]));
    }
    (void)ErrorManager::GetInstance().ReportErrMessage(error_code, args_map);
  } else {
    GELOGW("ATCReportErrMessage wrong, vector key and value size is not equal");
  }
}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @param [out] classified_msg: classified_msg message map, key is error code, value is op_name vector
///
void ErrorManager::ClassifyCompileFailedMsg(const std::map<std::string, std::string> &msg,
                                            std::map<std::string,
                                            std::vector<std::string>> &classified_msg) {
  for (const auto &itr : msg) {
    const std::string &error_code = itr.first;
    const std::string &op_name = itr.second;
    GELOGD("msg is error_code:%s, op_name:%s", error_code.c_str(), op_name.c_str());
    const auto err_code_itr = classified_msg.find(error_code);
    if (err_code_itr == classified_msg.end()) {
      (void)classified_msg.emplace(error_code, std::vector<std::string>{op_name});
    } else {
      std::vector<std::string> &op_name_list = err_code_itr->second;
      op_name_list.emplace_back(op_name);
    }
  }
}

///
/// @brief report graph compile failed message such as error code and op_name in mustune case
/// @param [in] root_graph_name: root graph name
/// @param [in] msg: failed message map, key is error code, value is op_name
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::ReportMstuneCompileFailedMsg(const std::string &root_graph_name,
                                                   const std::map<std::string, std::string> &msg) {
  if (!is_init_) {
    GELOGI("ErrorManager has not inited, can't report compile message");
    return 0;
  }
  if (msg.empty() || root_graph_name.empty()) {
    GELOGW("Msg or root graph name is empty, msg size is %zu, root graph name is %s",
           msg.size(), root_graph_name.c_str());
    return -1;
  }
  GELOGD("Report graph:%s compile failed msg", root_graph_name.c_str());
  const std::unique_lock<std::mutex> lock(mutex_);
  const auto itr = compile_failed_msg_map_.find(root_graph_name);
  if (itr != compile_failed_msg_map_.end()) {
    std::map<std::string, std::vector<std::string>> &classified_msg = itr->second;
    ClassifyCompileFailedMsg(msg, classified_msg);
  } else {
    std::map<std::string, std::vector<std::string>> classified_msg;
    ClassifyCompileFailedMsg(msg, classified_msg);
    (void)compile_failed_msg_map_.emplace(root_graph_name, classified_msg);
  }
  return 0;
}

///
/// @brief get graph compile failed message in mustune case
/// @param [in] graph_name: graph name
/// @param [out] msg_map: failed message map, key is error code, value is op_name list
/// @return int 0(success) -1(fail)
///
int32_t ErrorManager::GetMstuneCompileFailedMsg(const std::string &graph_name, std::map<std::string,
                                            std::vector<std::string>> &msg_map) {
  if (!is_init_) {
    GELOGI("ErrorManager has not inited, can't report compile failed message");
    return 0;
  }
  if (!msg_map.empty()) {
    GELOGW("msg_map is not empty, exist msg");
    return -1;
  }
  const std::unique_lock<std::mutex> lock(mutex_);
  const auto iter = compile_failed_msg_map_.find(graph_name);
  if (iter == compile_failed_msg_map_.end()) {
    GELOGW("can not find graph, name is:%s", graph_name.c_str());
    return -1;
  } else {
    auto &compile_failed_msg = iter->second;
    msg_map.swap(compile_failed_msg);
    (void)compile_failed_msg_map_.erase(graph_name);
  }
  GELOGI("get graph:%s compile failed msg success", graph_name.c_str());

  return 0;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetErrorMsgContainerByWorkId(uint64_t work_id) {
  auto iter = error_message_per_work_id_.find(work_id);
  if (iter == error_message_per_work_id_.end()) {
    (void)error_message_per_work_id_.emplace(work_id, std::vector<ErrorItem>());
    iter = error_message_per_work_id_.find(work_id);
  }
  return iter->second;
}

std::vector<ErrorManager::ErrorItem> &ErrorManager::GetWarningMsgContainerByWorkId(uint64_t work_id) {
  auto iter = warning_messages_per_work_id_.find(work_id);
  if (iter == warning_messages_per_work_id_.end()) {
    (void)warning_messages_per_work_id_.emplace(work_id, std::vector<ErrorItem>());
    iter = warning_messages_per_work_id_.find(work_id);
  }
  return iter->second;
}

void ErrorManager::GenWorkStreamIdDefault() {
  // system getpid and gettid is always successful
  const int32_t pid = mmGetPid();
  const int32_t tid = mmGetTid();

  const uint64_t kPidOffset = 100000UL;
  const uint64_t work_stream_id = static_cast<uint64_t>((pid * kPidOffset) + tid);
  error_context_.work_stream_id = work_stream_id;

  ClearErrorMsgContainerByWorkId(work_stream_id);
  ClearWarningMsgContainerByWorkId(work_stream_id);
}

void ErrorManager::GenWorkStreamIdBySessionGraph(const uint64_t session_id, const uint64_t graph_id) {
  const uint64_t kSessionIdOffset = 100000UL;
  uint64_t work_stream_id = (session_id * kSessionIdOffset) + graph_id;
  error_context_.work_stream_id = work_stream_id;

  ClearErrorMsgContainerByWorkId(work_stream_id);
  ClearWarningMsgContainerByWorkId(work_stream_id);
}

void ErrorManager::ClearErrorMsgContainerByWorkId(const uint64_t work_stream_id) {
  const auto err_iter = error_message_per_work_id_.find(work_stream_id);
  if (err_iter != error_message_per_work_id_.end()) {
    (void)error_message_per_work_id_.erase(err_iter);
  }
}

void ErrorManager::ClearWarningMsgContainerByWorkId(const uint64_t work_stream_id) {
  const auto warn_iter = warning_messages_per_work_id_.find(work_stream_id);
  if (warn_iter != warning_messages_per_work_id_.end()) {
    (void)warning_messages_per_work_id_.erase(warn_iter);
  }
}


const std::string &ErrorManager::GetLogHeader() {
  if ((error_context_.first_stage == "") && (error_context_.second_stage == "")) {
    error_context_.log_header = "";
  } else {
    error_context_.log_header = "[" + error_context_.first_stage + "][" + error_context_.second_stage + "]";
  }
  return error_context_.log_header;
}

Context &ErrorManager::GetErrorManagerContext() {
  return error_context_;
}

void ErrorManager::SetErrorContext(const Context error_context) {
  error_context_.work_stream_id = error_context.work_stream_id;
  error_context_.first_stage = move(error_context.first_stage);
  error_context_.second_stage = move(error_context.second_stage);
  error_context_.log_header = move(error_context.log_header);
}

void ErrorManager::SetStage(const std::string &first_stage, const std::string &second_stage) {
  error_context_.first_stage = first_stage;
  error_context_.second_stage = second_stage;
}

void ErrorManager::SetStage(const error_message::char_t *first_stage, const size_t first_len,
                            const error_message::char_t *second_stage, const size_t second_len) {
  if (first_stage != nullptr) {
    (void)error_context_.first_stage.assign(first_stage, first_len);
  }
  if (second_stage != nullptr) {
    (void)error_context_.second_stage.assign(second_stage, second_len);
  }
}

bool ErrorManager::IsInnerErrorCode(const std::string &error_code) const {
  const std::string kInterErrorCodePrefix = "9999";
  if (!IsValidErrorCode(error_code)) {
    return false;
  } else {
    return error_code.substr(2U, 4U) == kInterErrorCodePrefix;
  }
}


