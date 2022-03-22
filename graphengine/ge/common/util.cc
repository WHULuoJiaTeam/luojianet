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

#include "framework/common/util.h"

#include <sys/stat.h>
#ifdef __GNUC__
#include <regex.h>
#else
#include <regex>
#endif
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/fmk_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "mmpa/mmpa_api.h"

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

namespace {
/*
 * kProtoReadBytesLimit and kWarningThreshold are real arguments of CodedInputStream::SetTotalBytesLimit.
 * In order to prevent integer overflow and excessive memory allocation during protobuf processing,
 * it is necessary to limit the length of proto message (call SetTotalBytesLimit function).
 * In theory, the minimum message length that causes an integer overflow is 512MB, and the default is 64MB.
 * If the limit of warning_threshold is exceeded, the exception information will be printed in stderr.
 * If such an exception is encountered during operation,
 * the proto file can be divided into several small files or the limit value can be increased.
 */
const int kFileSizeOutLimitedOrOpenFailed = -1;
const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.
const int kWarningThreshold = 1073741824;  // 536870912 * 2 536870912 represent 512M

/// The maximum length of the file.
const uint32_t kMaxFileSizeLimit = UINT32_MAX;  // 4G for now
const int kMaxBuffSize = 256;
const char *const kPathValidReason = "The path can only contain 'a-z' 'A-Z' '0-9' '-' '.' '_' and chinese character";
constexpr uint32_t kMaxConfigFileByte = 10485760;  // 10 * 1024 * 1024
}  // namespace

namespace ge {
static bool ReadProtoFromCodedInputStream(CodedInputStream &coded_stream, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(proto == nullptr, return false, "incorrect parameter. nullptr == proto");

  coded_stream.SetTotalBytesLimit(kProtoReadBytesLimit, kWarningThreshold);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromArray(const void *data, int size, Message *proto) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((proto == nullptr || data == nullptr || size == 0), return false,
                                 "incorrect parameter. proto is nullptr || data is nullptr || size is 0");

  google::protobuf::io::CodedInputStream coded_stream(reinterpret_cast<uint8_t *>(const_cast<void *>(data)), size);
  return ReadProtoFromCodedInputStream(coded_stream, proto);
}

// Get file length
long GetFileLength(const std::string &input_file) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(input_file.empty(), return -1, "input_file path is null.");

  std::string real_path = RealPath(input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return -1, "input_file path '%s' not valid", input_file.c_str());
  unsigned long long file_length = 0;
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    mmGetFileSize(input_file.c_str(), &file_length) != EN_OK,
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {input_file, strerror(errno)});
    return kFileSizeOutLimitedOrOpenFailed, "Open file[%s] failed. errmsg:%s", input_file.c_str(), strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_length == 0),
                                 REPORT_INNER_ERROR("E19999", "file:%s size is 0, not valid", input_file.c_str());
                                 return -1, "File[%s] size is 0, not valid.", input_file.c_str());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
      file_length > kMaxFileSizeLimit,
      REPORT_INNER_ERROR("E19999", "file:%s size:%lld is out of limit: %d.", input_file.c_str(), file_length,
                         kMaxFileSizeLimit);
    return kFileSizeOutLimitedOrOpenFailed, "File[%s] size %lld is out of limit: %d.", input_file.c_str(), file_length,
           kMaxFileSizeLimit);
  return static_cast<long>(file_length);
}

/** @ingroup domi_common
 *  @brief Read all data from binary file
 *  @param [in] file_name  File path
 *  @param [out] buffer  The address of the output memory, which needs to be released by the caller
 *  @param [out] length  Output memory size
 *  @return false fail
 *  @return true success
 */
bool ReadBytesFromBinaryFile(const char *file_name, char **buffer, int &length) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr), return false, "incorrect parameter. file is nullptr");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((buffer == nullptr), return false, "incorrect parameter. buffer is nullptr");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "file path '%s' not valid", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "[Read][File]Failed, file %s", file_name);
    REPORT_CALL_ERROR("E19999", "Read file %s failed", file_name);
    return false;
  }

  length = static_cast<int>(file.tellg());

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((length <= 0), file.close(); return false, "file length <= 0");

  file.seekg(0, std::ios::beg);

  *buffer = new (std::nothrow) char[length]();
  GE_CHK_BOOL_TRUE_EXEC_RET_STATUS(*buffer == nullptr, false, file.close(), "new an object failed.");

  file.read(*buffer, length);
  file.close();
  return true;
}

bool ReadBytesFromBinaryFile(const char *file_name, std::vector<char> &buffer) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file_name == nullptr), return false, "incorrect parameter. file path is null");

  std::string real_path = RealPath(file_name);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), return false, "file path '%s' not valid", file_name);

  std::ifstream file(real_path.c_str(), std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    GELOGE(ge::FAILED, "[Read][File]Failed, file %s", file_name);
    REPORT_CALL_ERROR("E19999", "Read file %s failed", file_name);
    return false;
  }

  std::streamsize size = file.tellg();

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((size <= 0), file.close(); return false, "file length <= 0, not valid.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(size > static_cast<int64_t>(kMaxFileSizeLimit), file.close();
                                 return false, "file size %ld is out of limit: %d.", size, kMaxFileSizeLimit);

  file.seekg(0, std::ios::beg);  // [no need to check value]

  buffer.resize(static_cast<uint64_t>(size));  // [no need to check value]
  file.read(&buffer[0], size);                 // [no need to check value]
  file.close();
  GELOGI("Read size:%ld", size);
  return true;
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(), return -1, "directory path is empty.");
  auto dir_path_len = directory_path.length();
  if (dir_path_len >= MMPA_MAX_PATH) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {directory_path, std::to_string(MMPA_MAX_PATH)});
    GELOGW("Path[%s] len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return -1;
  }
  char tmp_dir_path[MMPA_MAX_PATH] = {0};
  for (size_t i = 0; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
        int32_t ret = mmMkdir(tmp_dir_path, M_IRUSR | M_IWUSR | M_IXUSR);  // 700
        if (ret != 0) {
          if (errno != EEXIST) {
            REPORT_CALL_ERROR("E19999",
                              "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                              directory_path.c_str(), strerror(errno));
            GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                   directory_path.c_str(), strerror(errno));
            return ret;
          }
        }
      }
    }
  }
  int32_t ret = mmMkdir(const_cast<char *>(directory_path.c_str()), M_IRUSR | M_IWUSR | M_IXUSR);  // 700
  if (ret != 0) {
    if (errno != EEXIST) {
      REPORT_CALL_ERROR("E19999",
                        "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                        directory_path.c_str(), strerror(errno));
      GELOGW("Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
             directory_path.c_str(), strerror(errno));
      return ret;
    }
  }
  return 0;
}

std::string CurrentTimeInStr() {
  std::time_t now = std::time(nullptr);
  std::tm *ptm = std::localtime(&now);
  if (ptm == nullptr) {
    GELOGE(ge::FAILED, "[Check][Param]Localtime incorrect, errmsg %s", strerror(errno));
    REPORT_CALL_ERROR("E19999", "Localtime incorrect, errmsg %s", strerror(errno));
    return "";
  }

  const int kTimeBufferLen = 32;
  char buffer[kTimeBufferLen + 1] = {0};
  // format: 20171122042550
  std::strftime(buffer, kTimeBufferLen, "%Y%m%d%H%M%S", ptm);
  return std::string(buffer);
}

bool ReadProtoFromText(const char *file, google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((file == nullptr || message == nullptr), return false,
                                 "incorrect parameter. nullptr == file || nullptr == message");

  std::string real_path = RealPath(file);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(real_path.empty(), ErrorManager::GetInstance().ATCReportErrMessage(
                                                      "E19000", {"path", "errmsg"}, {file, strerror(errno)});
                                 return false, "Path[%s]'s realpath is empty, errmsg[%s]", file, strerror(errno));

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(GetFileLength(real_path) == -1, return false, "file size not valid.");

  std::ifstream fs(real_path.c_str(), std::ifstream::in);

  if (!fs.is_open()) {
    REPORT_INNER_ERROR("E19999", "open file:%s failed", real_path.c_str());
    GELOGE(ge::FAILED, "[Open][ProtoFile]Failed, real path %s, orginal file path %s",
           real_path.c_str(), file);
    return false;
  }

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(!ret, ErrorManager::GetInstance().ATCReportErrMessage("E19018", {"protofile"}, {file});
                  GELOGE(ret, "[Parse][File]Through [google::protobuf::TextFormat::Parse] failed, "
                         "file %s", file));
  fs.close();

  return ret;
}

bool ReadProtoFromMem(const char *data, int size, google::protobuf::Message *message) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((data == nullptr || message == nullptr), return false,
                                 "incorrect parameter. data is nullptr || message is nullptr");
  std::string str(data, static_cast<size_t>(size));
  std::istringstream fs(str);

  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, message);
  GE_IF_BOOL_EXEC(
    !ret, GELOGE(ret, "Call [google::protobuf::TextFormat::Parse] func ret fail, please check your text file."));

  return ret;
}

uint64_t GetCurrentTimestamp() {
  mmTimeval tv{};
  int ret = mmGetTimeOfDay(&tv, nullptr);
  GE_LOGE_IF(ret != EN_OK, "Func gettimeofday may failed, ret:%d, errmsg:%s", ret, strerror(errno));
  auto total_use_time = tv.tv_usec + tv.tv_sec * 1000000;  // 1000000: seconds to microseconds
  return static_cast<uint64_t>(total_use_time);
}

uint32_t GetCurrentSecondTimestap() {
  mmTimeval tv{};
  int ret = mmGetTimeOfDay(&tv, nullptr);
  GE_LOGE_IF(ret != EN_OK, "Func gettimeofday may failed, ret:%d, errmsg:%s", ret, strerror(errno));
  auto total_use_time = tv.tv_sec;  // seconds
  return static_cast<uint32_t>(total_use_time);
}

bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }
  return true;
}

std::string RealPath(const char *path) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path == nullptr, return "", "path pointer is NULL.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(path) >= MMPA_MAX_PATH,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                                                 {path, std::to_string(MMPA_MAX_PATH)});
                                 return "", "Path[%s] len is too long, it must be less than %d", path, MMPA_MAX_PATH);

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char resolved_path[MMPA_MAX_PATH] = {0};
  if (mmRealPath(path, resolved_path, MMPA_MAX_PATH) == EN_OK) {
    res = resolved_path;
  }

  return res;
}

void PathValidErrReport(const std::string &file_path, const std::string &atc_param, const std::string &reason) {
  if (!atc_param.empty()) {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({atc_param, file_path, reason}));
  } else {
    REPORT_INNER_ERROR("E19999", "Path[%s] invalid, reason:%s", file_path.c_str(), reason.c_str());
  }
}

bool CheckInputPathValid(const std::string &file_path, const std::string &atc_param) {
  // The specified path is empty
  std::map<std::string, std::string> args_map;
  if (file_path.empty()) {
    if (!atc_param.empty()) {
      REPORT_INPUT_ERROR("E10004", std::vector<std::string>({"parameter"}), std::vector<std::string>({atc_param}));
    } else {
      REPORT_INNER_ERROR("E19999", "Param file_path is empty, check invalid.");
    }
    GELOGW("Input parameter %s is empty.", file_path.c_str());
    return false;
  }
  std::string real_path = RealPath(file_path.c_str());
  // Unable to get absolute path (does not exist or does not have permission to access)
  if (real_path.empty()) {
    std::string reason = "realpath error, errmsg:" + std::string(strerror(errno));
    PathValidErrReport(file_path, atc_param, reason);
    GELOGW("Path[%s]'s realpath is empty, errmsg[%s]", file_path.c_str(), strerror(errno));
    return false;
  }

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
#ifdef __GNUC__
  std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";
#else
  std::string mode = "^[a-zA-Z]:([\\\\/][^\\s\\\\/:*?<>\"|][^\\\\/:*?<>\"|]*)*([/\\\\][^\\s\\\\/:*?<>\"|])?$";
#endif

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    !ValidateStr(real_path, mode),
    PathValidErrReport(file_path, atc_param, kPathValidReason);
    return false, "Invalid value for %s[%s], %s.", atc_param.c_str(), real_path.c_str(), kPathValidReason);

  // The absolute path points to a file that is not readable
  if (mmAccess2(real_path.c_str(), M_R_OK) != EN_OK) {
    PathValidErrReport(file_path, atc_param, "cat not access, errmsg:" + std::string(strerror(errno)));
    GELOGW("Read file[%s] failed, errmsg[%s]", file_path.c_str(), strerror(errno));
    return false;
  }

  return true;
}

bool CheckOutputPathValid(const std::string &file_path, const std::string &atc_param) {
  // The specified path is empty
  if (file_path.empty()) {
    if (!atc_param.empty()) {
      REPORT_INPUT_ERROR("E10004", std::vector<std::string>({"parameter"}), std::vector<std::string>({atc_param}));
    } else {
      REPORT_INNER_ERROR("E19999", "Param file_path is empty, check invalid.");
    }
    ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {atc_param});
    GELOGW("Input parameter's value is empty.");
    return false;
  }

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(file_path.c_str()) >= MMPA_MAX_PATH,
                                 std::string reason = "len is too long, it must be less than " +
                                                      std::to_string(MMPA_MAX_PATH);
                                 PathValidErrReport(file_path, atc_param, reason);
                                 return false, "Path[%s] len is too long, it must be less than %d", file_path.c_str(),
                                        MMPA_MAX_PATH);

  // A regular matching expression to verify the validity of the input file path
  // Path section: Support upper and lower case letters, numbers dots(.) chinese and underscores
  // File name section: Support upper and lower case letters, numbers, underscores chinese and dots(.)
#ifdef __GNUC__
  std::string mode = "^[\u4e00-\u9fa5A-Za-z0-9./_-]+$";
#else
  std::string mode = "^[a-zA-Z]:([\\\\/][^\\s\\\\/:*?<>\"|][^\\\\/:*?<>\"|]*)*([/\\\\][^\\s\\\\/:*?<>\"|])?$";
#endif

  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
    !ValidateStr(file_path, mode),
    PathValidErrReport(file_path, atc_param, kPathValidReason);
    return false, "Invalid value for %s[%s], %s.", atc_param.c_str(), file_path.c_str(), kPathValidReason);

  std::string real_path = RealPath(file_path.c_str());
  // Can get absolute path (file exists)
  if (!real_path.empty()) {
    // File is not readable or writable
    if (mmAccess2(real_path.c_str(), M_W_OK | M_F_OK) != EN_OK) {
      PathValidErrReport(file_path, atc_param, "cat not access, errmsg:" + std::string(strerror(errno)));
      GELOGW("Write file[%s] failed, errmsg[%s]", real_path.c_str(), strerror(errno));
      return false;
    }
  } else {
    // Find the last separator
    int path_split_pos = static_cast<int>(file_path.size() - 1);
    for (; path_split_pos >= 0; path_split_pos--) {
      if (file_path[path_split_pos] == '\\' || file_path[path_split_pos] == '/') {
        break;
      }
    }
    if (path_split_pos == 0) {
      return true;
    }
    if (path_split_pos != -1) {
      std::string prefix_path = std::string(file_path).substr(0, static_cast<size_t>(path_split_pos));
      // Determine whether the specified path is valid by creating the path
      if (CreateDirectory(prefix_path) != 0) {
        PathValidErrReport(file_path, atc_param, "Can not create directory");
        GELOGW("Can not create directory[%s].", file_path.c_str());
        return false;
      }
    }
  }

  return true;
}

FMK_FUNC_HOST_VISIBILITY bool ValidateStr(const std::string &str, const std::string &mode) {
#ifdef __GNUC__
  char ebuff[kMaxBuffSize];
  regex_t reg;
  int cflags = REG_EXTENDED | REG_NOSUB;
  int ret = regcomp(&reg, mode.c_str(), cflags);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGW("regcomp failed, reason: %s", ebuff);
    regfree(&reg);
    return true;
  }

  ret = regexec(&reg, str.c_str(), 0, NULL, 0);
  if (ret) {
    regerror(ret, &reg, ebuff, kMaxBuffSize);
    GELOGE(ge::PARAM_INVALID, "[Rgexec][Param]Failed, reason %s", ebuff);
    REPORT_CALL_ERROR("E19999", "Rgexec failed, reason %s", ebuff);
    regfree(&reg);
    return false;
  }

  regfree(&reg);
  return true;
#else
  std::wstring wstr(str.begin(), str.end());
  std::wstring wmode(mode.begin(), mode.end());
  std::wsmatch match;
  bool res = false;

  try {
    std::wregex reg(wmode, std::regex::icase);
    // Matching string part
    res = regex_match(wstr, match, reg);
    res = regex_search(str, std::regex("[`!@#$%^&*()|{}';',<>?]"));
  } catch (std::exception &ex) {
    GELOGW("The directory %s is invalid, error: %s.", str.c_str(), ex.what());
    return false;
  }
  return !(res) && (str.size() == match.str().size());
#endif
}

FMK_FUNC_HOST_VISIBILITY bool IsValidFile(const char *file_path) {
  if (file_path == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param]Config path is null");
    REPORT_INNER_ERROR("E19999", "Config path is null");
    return false;
  }
  if (!CheckInputPathValid(file_path)) {
    GELOGE(PARAM_INVALID, "[Check][Param]Config path %s is invalid", file_path);
    REPORT_CALL_ERROR("E19999", "Config path %s is invalid", file_path);
    return false;
  }
  // Normalize the path
  std::string resolved_file_path = RealPath(file_path);
  if (resolved_file_path.empty()) {
    GELOGE(PARAM_INVALID, "[Check][Param]Invalid input file path %s, errmsg %s", file_path, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Invalid input file path %s, errmsg %s", file_path, strerror(errno));
    return false;
  }

  mmStat_t stat = {0};
  int32_t ret = mmStatGet(resolved_file_path.c_str(), &stat);
  if (ret != EN_OK) {
    GELOGE(PARAM_INVALID, "[Get][FileStatus]Failed, which path %s maybe not exist, "
           "return %d, errcode %d", resolved_file_path.c_str(), ret, mmGetErrorCode());
    REPORT_CALL_ERROR("E19999", "Get config file status failed, which path %s maybe not exist, "
                      "return %d, errcode %d", resolved_file_path.c_str(), ret, mmGetErrorCode());
    return false;
  }
  if ((stat.st_mode & S_IFMT) != S_IFREG) {
    GELOGE(PARAM_INVALID, "[Check][Param]Config file is not a common file, which path is %s, "
           "mode is %u", resolved_file_path.c_str(), stat.st_mode);
    REPORT_CALL_ERROR("E19999", "Config file is not a common file, which path is %s, "
                      "mode is %u", resolved_file_path.c_str(), stat.st_mode);
    return false;
  }
  if (stat.st_size > kMaxConfigFileByte) {
    GELOGE(PARAM_INVALID, "[Check][Param]Config file %s size %ld is larger than max config "
           "file Bytes %u", resolved_file_path.c_str(), stat.st_size, kMaxConfigFileByte);
    REPORT_CALL_ERROR("E19999", "Config file %s size %ld is larger than max config file Bytes %u",
                      resolved_file_path.c_str(), stat.st_size, kMaxConfigFileByte);
    return false;
  }
  return true;
}

Status CheckPath(const char *path, size_t length) {
  if (path == nullptr) {
    GELOGE(PARAM_INVALID, "[Check][Param]Config path is invalid");
    REPORT_CALL_ERROR("E19999", "Config path is invalid");
    return PARAM_INVALID;
  }

  if (strlen(path) != length) {
    GELOGE(PARAM_INVALID, "[Check][Param]Path %s is invalid or length %zu "
           "not equal to given length %zu", path, strlen(path), length);
    REPORT_CALL_ERROR("E19999", "Path %s is invalid or length %zu "
                      "not equal to given length %zu", path, strlen(path), length);
    return PARAM_INVALID;
  }

  if (length == 0 || length > MMPA_MAX_PATH) {
    GELOGE(PARAM_INVALID, "[Check][Param]Length of config path %zu is invalid", length);
    REPORT_INNER_ERROR("E19999", "Length of config path %zu is invalid", length);
    return PARAM_INVALID;
  }

  INT32 is_dir = mmIsDir(path);
  if (is_dir != EN_OK) {
    GELOGE(PATH_INVALID, "[Open][Directory]Failed, directory path %s, errmsg %s",
           path, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Open directory %s failed, errmsg %s", path, strerror(errno));
    return PATH_INVALID;
  }

  if (mmAccess2(path, M_R_OK) != EN_OK) {
    GELOGE(PATH_INVALID, "[Read][Path]Failed, path %s, errmsg %s", path, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Read path %s failed, errmsg %s", path, strerror(errno));
    return PATH_INVALID;
  }
  return SUCCESS;
}
}  //  namespace ge
