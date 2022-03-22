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

#include "common/model_saver.h"

#include <securec.h>
#include <cstdlib>
#include <fstream>
#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/util/error_manager/error_manager.h"

namespace ge {
const uint32_t kInteval = 2;

Status ModelSaver::SaveJsonToFile(const char *file_path, const Json &model) {
  Status ret = SUCCESS;
  if (file_path == nullptr || SUCCESS != CheckPath(file_path)) {
    GELOGE(FAILED, "[Check][OutputFile]Failed, file %s", file_path);
    REPORT_CALL_ERROR("E19999", "Output file %s check invalid", file_path);
    return FAILED;
  }
  std::string model_str;
  try {
    model_str = model.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, reason: %s, savefile:%s.", e.what(), file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s, reason %s",
           file_path, e.what());
    return FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Failed to convert JSON to string, savefile:%s.", file_path);
    GELOGE(FAILED, "[Convert][File]Failed to convert JSON to string, file %s", file_path);
    return FAILED;
  }

  char real_path[MMPA_MAX_PATH] = {0};
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(file_path) >= MMPA_MAX_PATH, return FAILED, "file path is too long!");
  GE_IF_BOOL_EXEC(mmRealPath(file_path, real_path, MMPA_MAX_PATH) != EN_OK,
                  GELOGI("File %s does not exit, it will be created.", file_path));

  // Open file
  mmMode_t mode = M_IRUSR | M_IWUSR;
  int32_t fd = mmOpen2(real_path, M_RDWR | M_CREAT | O_TRUNC, mode);
  if (fd == EN_ERROR || fd == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19001", {"file", "errmsg"}, {file_path, strerror(errno)});
    GELOGE(FAILED, "[Open][File]Failed, file %s, errmsg %s", file_path, strerror(errno));
    return FAILED;
  }
  const char *model_char = model_str.c_str();
  uint32_t len = static_cast<uint32_t>(model_str.length());
  // Write data to file
  mmSsize_t mmpa_ret = mmWrite(fd, const_cast<void *>((const void *)model_char), len);
  if (mmpa_ret == EN_ERROR || mmpa_ret == EN_INVALID_PARAM) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E19004", {"file", "errmsg"}, {file_path, strerror(errno)});
    // Need to both print the error info of mmWrite and mmClose, so return ret after mmClose
    GELOGE(FAILED, "[Write][Data]To file %s failed. errno %ld, errmsg %s",
           file_path, mmpa_ret, strerror(errno));
    ret = FAILED;
  }
  // Close file
  if (mmClose(fd) != EN_OK) {
    GELOGE(FAILED, "[Close][File]Failed, file %s, errmsg %s", file_path, strerror(errno));
    REPORT_CALL_ERROR("E19999", "Close file %s failed, errmsg %s", file_path, strerror(errno));
    ret = FAILED;
  }
  return ret;
}
}  // namespace ge
