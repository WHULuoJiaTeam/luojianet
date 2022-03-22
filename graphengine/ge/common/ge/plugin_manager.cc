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

#include "common/ge/plugin_manager.h"

#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "framework/common/debug/log.h"
#include "framework/common/util.h"

namespace {
const int kMaxNumOfSo = 64;
const int kMaxSizeOfSo = 209100800;        // = 200M(unit is Byte)
const int kMaxSizeOfLoadedSo = 522752000;  // = 500M(unit is Byte)
const char *const kExt = ".so";            // supported extension of shared object
}  // namespace
namespace ge {
void PluginManager::ClearHandles_() noexcept {
  for (const auto &handle : handles_) {
    if (mmDlclose(handle.second) != 0) {
      const char *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed to close handle of %s, errmsg:%s", handle.first.c_str(), error);
    }
  }
  handles_.clear();
}

PluginManager::~PluginManager() { ClearHandles_(); }

string PluginManager::GetPath() {
  mmDlInfo dl_info;
  if (mmDladdr(reinterpret_cast<void *>(&PluginManager::GetPath), &dl_info) != EN_OK) {
    const char *error = mmDlerror();
    GE_IF_BOOL_EXEC(error == nullptr, error = "");
    GELOGW("Failed to read the shared library file path! errmsg:%s", error);
    return string();
  } else {
    GE_IF_BOOL_EXEC(dl_info.dli_fname == nullptr, return string());
    std::string so_path = dl_info.dli_fname;
    char path[MMPA_MAX_PATH] = {0};
    if (so_path.length() >= MMPA_MAX_PATH) {
      GELOGW("The shared library file path is too long!");
      return string();
    }
    if (mmRealPath(so_path.c_str(), path, MMPA_MAX_PATH) != EN_OK) {
      GELOGW("Failed to get realpath of %s, errmsg:%s", so_path.c_str(), strerror(errno));
      return string();
    }

    so_path = path;
    so_path = so_path.substr(0, so_path.rfind('/') + 1);
    return so_path;
  }
}

void PluginManager::SplitPath(const string &mutil_path, vector<string> &path_vec) {
  std::string tmp_string = mutil_path + ":";
  std::string::size_type start_pos = 0;
  std::string::size_type cur_pos = tmp_string.find(':', 0);
  while (cur_pos != std::string::npos) {
    std::string path = tmp_string.substr(start_pos, cur_pos - start_pos);
    if (!path.empty()) {
      path_vec.push_back(path);
    }
    start_pos = cur_pos + 1;
    cur_pos = tmp_string.find(':', start_pos);
  }
}

Status PluginManager::LoadSo(const string &path, const vector<string> &func_check_list) {
  uint32_t num_of_loaded_so = 0;
  int64_t size_of_loaded_so = 0;
  so_list_.clear();
  ClearHandles_();

  std::vector<std::string> path_vec;
  SplitPath(path, path_vec);
  for (const auto &single_path : path_vec) {
    GE_IF_BOOL_EXEC(single_path.length() >= MMPA_MAX_PATH, GELOGE(ACL_ERROR_GE_PLGMGR_PATH_INVALID,
                    "The shared library file path is too long!");
                    continue);
    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= kMaxNumOfSo) {
      GELOGW("The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
             " and only the first %d shared libraries will be loaded.", kMaxNumOfSo);
      break;
    }

    std::string file_name = single_path.substr(single_path.rfind('/') + 1, string::npos);
    string file_path_dlopen = RealPath(single_path.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("Failed to get realpath of %s!", single_path.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", file_path_dlopen.c_str());
      continue;
    }

    GELOGI("dlopen the shared library path name: %s.", file_path_dlopen.c_str());

    // load continue when dlopen is failed
    auto handle = mmDlopen(file_path_dlopen.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
    if (handle == nullptr) {
      const char *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      REPORT_INNER_ERROR("E19999", "DLOpen SharedLibraryPath failed, path[%s]. Errormessage[%s]!",
                         file_path_dlopen.c_str(), error);
      GELOGE(ACL_ERROR_GE_PLGMGR_PATH_INVALID,
             "[DLOpen][SharedLibraryPath]Failed, path[%s]. Errormessage[%s]!",
             file_path_dlopen.c_str(), error);
      continue;
    }

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      auto real_fn = (void (*)())mmDlsym(handle, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        REPORT_INNER_ERROR("E19999", "[Check][So]%s is skipped since function %s is not existed! errmsg:%s",
                           func_name.c_str(), func_name.c_str(), error);
        GELOGE(ACL_ERROR_GE_PLGMGR_PATH_INVALID,
               "[Check][So]%s is skipped since function %s is not existed! errmsg:%s",
               func_name.c_str(), func_name.c_str(), error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    so_list_.emplace_back(file_name);
    handles_[string(file_name)] = handle;
    num_of_loaded_so++;
  }

  GELOGI("The total number of shared libraries loaded: %u", num_of_loaded_so);
  for (auto name : so_list_) {
    GELOGI("load shared library %s successfully", name.c_str());
  }

  if (num_of_loaded_so == 0) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

Status PluginManager::ValidateSo(const string &file_path, int64_t size_of_loaded_so, int64_t &file_size) const {
  // read file size
  struct stat stat_buf;
  if (stat(file_path.c_str(), &stat_buf) != 0) {
    GELOGW("The shared library file check failed: %s", file_path.c_str());
    return FAILED;
  }

  // load continue when the size itself reaches maximum
  file_size = stat_buf.st_size;
  if (stat_buf.st_size > kMaxSizeOfSo) {
    GELOGW("The %s is skipped since its size exceeds maximum! (size: %ldB, maximum: %dB)", file_path.c_str(), file_size,
           kMaxSizeOfSo);
    return FAILED;
  }

  // load continue if the total size of so reaches maximum when it is loaded
  if (size_of_loaded_so + file_size > kMaxSizeOfLoadedSo) {
    GELOGW(
        "%s is skipped because the size of loaded share library reaches maximum if it is loaded! "
        "(size: %ldB, size of loaded share library: %ldB, maximum: %dB)",
        file_path.c_str(), file_size, size_of_loaded_so, kMaxSizeOfLoadedSo);
    return FAILED;
  }

  return SUCCESS;
}

Status PluginManager::Load(const string &path, const vector<string> &func_check_list) {
  uint32_t num_of_loaded_so = 0;
  int64_t size_of_loaded_so = 0;
  const unsigned char is_folder = 0x4;
  const std::string ext = kExt;
  so_list_.clear();
  ClearHandles_();

  char canonical_path[MMPA_MAX_PATH] = {0};
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path.length() >= MMPA_MAX_PATH, GELOGW("File path is too long!");
                                 return FAILED, "File path is too long!");
  if (mmRealPath(path.c_str(), canonical_path, MMPA_MAX_PATH) != EN_OK) {
    GELOGW("Failed to get realpath of %s, errmsg:%s", path.c_str(), strerror(errno));
    return SUCCESS;
  }

  INT32 is_dir = mmIsDir(canonical_path);
  // Lib plugin path not exist
  if (is_dir != EN_OK) {
      GELOGW("Invalid path for load: %s, errmsg:%s", path.c_str(), strerror(errno));
      return SUCCESS;
  }

  mmDirent **entries = nullptr;
  auto ret = mmScandir(canonical_path, &entries, nullptr, nullptr);
  if (ret < EN_OK) {
      GELOGW("scan dir failed. path = %s, ret = %d, errmsg = %s", canonical_path, ret, strerror(errno));
      return FAILED;
  }
  for (int i = 0; i < ret; ++i) {
    mmDirent *entry = entries[i];
    // read fileName and fileType
    std::string file_name = entry->d_name;
    unsigned char file_type = entry->d_type;

    // ignore folder
    bool invalid_file = (file_type == is_folder ||
                         // ignore file whose name length is less than 3
                         file_name.size() <= ext.size() ||
                         // ignore file whose extension is not so
                         file_name.compare(file_name.size() - ext.size(), ext.size(), ext) != 0);
    if (invalid_file) {
      continue;
    }

    // load break when number of loaded so reach maximum
    if (num_of_loaded_so >= kMaxNumOfSo) {
      GELOGW("The number of dynamic libraries loaded exceeds the kMaxNumOfSo,"
             " and only the first %d shared libraries will be loaded.", kMaxNumOfSo);
      break;
    }

    std::string canonical_path_str = (std::string(canonical_path) + "/" + file_name);
    string file_path_dlopen = RealPath(canonical_path_str.c_str());
    if (file_path_dlopen.empty()) {
      GELOGW("failed to get realpath of %s", canonical_path_str.c_str());
      continue;
    }

    int64_t file_size = 0;
    if (ValidateSo(file_path_dlopen, size_of_loaded_so, file_size) != SUCCESS) {
      GELOGW("Failed to validate the shared library: %s", canonical_path_str.c_str());
      continue;
    }

    GELOGI("Dlopen so path name: %s. ", file_path_dlopen.c_str());

    // load continue when dlopen is failed
    auto handle = mmDlopen(file_path_dlopen.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
    if (handle == nullptr) {
      const char *error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed in dlopen %s!", error);
      continue;
    }

    GELOGW("The shared library will not be checked. Please ensure that the source of the shared library is trusted.");

    // load continue when so is invalid
    bool is_valid = true;
    for (const auto &func_name : func_check_list) {
      auto real_fn = (void (*)())mmDlsym(handle, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGW("The %s is skipped since function %s is not existed! errmsg:%s",
               file_name.c_str(), func_name.c_str(), error);
        is_valid = false;
        break;
      }
    }
    if (!is_valid) {
      if (mmDlclose(handle) != 0) {
        const char *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGE(FAILED, "[DLClose][Handle]Failed. errmsg:%s", error);
      }
      continue;
    }

    // add file to list
    size_of_loaded_so += file_size;
    so_list_.emplace_back(file_name);
    handles_[string(file_name)] = handle;
    num_of_loaded_so++;
  }
  mmScandirFree(entries, ret);
  if (num_of_loaded_so == 0) {
    GELOGW("No loadable shared library found in the path: %s", path.c_str());
    return SUCCESS;
  }

  return SUCCESS;
}

const vector<string> &PluginManager::GetSoList() const { return so_list_; }
}  // namespace ge
