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
#include "debug/rdr/string_recorder.h"
#include <sys/stat.h>
#include <fstream>
#include <utility>
#include "include/common/utils/utils.h"
#include "luojianet_ms/core/utils/log_adapter.h"
#include "include/common/debug/rdr/recorder_manager.h"

namespace luojianet_ms {
void StringRecorder::Export() {
  auto realpath = GetFileRealPath();
  if (!realpath.has_value()) {
    return;
  }
  std::string file_path = realpath.value() + ".txt";
  ChangeFileMode(file_path, S_IRWXU);
  std::ofstream fout(file_path, std::ofstream::app);
  if (!fout.is_open()) {
    MS_LOG(WARNING) << "Open file for saving string failed. File path: '" << file_path << "'.";
    return;
  }
  fout << data_;
  fout.close();
  // set file mode to read only by user
  ChangeFileMode(file_path, S_IRUSR);
}

namespace RDR {
bool RecordString(SubModuleId module, const std::string &name, const std::string &data) {
  if (!luojianet_ms::RecorderManager::Instance().RdrEnable()) {
    return false;
  }
  std::string submodule_name = std::string(GetSubModuleName(module));
  StringRecorderPtr string_recorder = std::make_shared<StringRecorder>(submodule_name, name, data);
  bool ans = luojianet_ms::RecorderManager::Instance().RecordObject(std::move(string_recorder));
  return ans;
}
}  // namespace RDR
}  // namespace luojianet_ms
