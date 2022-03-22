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

#ifndef GE_COMMON_GE_PLUGIN_MANAGER_H_
#define GE_COMMON_GE_PLUGIN_MANAGER_H_

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/engine/dnnengine.h"
#include "framework/common/debug/ge_log.h"
#include "mmpa/mmpa_api.h"

namespace ge {
using SoToHandleMap = std::map<std::string, void *>;
using std::function;
using std::map;
using std::string;
using std::vector;

class PluginManager {
 public:
  PluginManager() = default;

  ~PluginManager();

  static string GetPath();

  void SplitPath(const string &mutil_path, vector<string> &path_vec);

  Status LoadSo(const string &path, const vector<string> &func_check_list = vector<string>());

  Status Load(const string &path, const vector<string> &func_check_list = vector<string>());

  const vector<string> &GetSoList() const;

  template <typename R, typename... Types>
  Status GetAllFunctions(const string &func_name, map<string, function<R(Types... args)>> &funcs) {
    for (const auto &handle : handles_) {
      auto real_fn = (R(*)(Types...))mmDlsym(handle.second, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to get function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_FUNC_NOT_EXIST;
      } else {
        funcs[handle.first] = real_fn;
      }
    }
    return SUCCESS;
  }

  template <typename... Types>
  Status InvokeAll(const string &func_name, Types... args) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      auto real_fn = (void (*)(Types...))mmDlsym(handle.second, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        real_fn(args...);
      }
    }
    return SUCCESS;
  }

  template <typename T>
  Status InvokeAll(const string &func_name, T arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      auto real_fn = (void (*)(T))mmDlsym(handle.second, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      }
      typename std::remove_reference<T>::type arg_temp;
      real_fn(arg_temp);

      if (std::is_same<typename std::remove_reference<T>::type, map<std::string, std::shared_ptr<DNNEngine>>>::value) {
        for (const auto &val : arg_temp) {
          if (arg.find(val.first) != arg.end()) {
            GELOGW("FuncName %s in so %s find the same key: %s, will replace it", func_name.c_str(),
                   handle.first.c_str(), val.first.c_str());
            arg[val.first] = val.second;
          }
        }
      }
      arg.insert(arg_temp.begin(), arg_temp.end());
    }
    return SUCCESS;
  }
  template <typename T1, typename T2>
  Status InvokeAll(const string &func_name, T1 arg) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      auto real_fn = (T2(*)(T1))mmDlsym(handle.second, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        T2 res = real_fn(arg);
        if (res != SUCCESS) {
          return FAILED;
        }
      }
    }
    return SUCCESS;
  }

  template <typename T>
  Status InvokeAll(const string &func_name) {
    for (const auto &handle : handles_) {
      // If the funcName is existed, signature of realFn can be casted to any type
      auto real_fn = (T(*)())mmDlsym(handle.second, const_cast<char *>(func_name.c_str()));
      if (real_fn == nullptr) {
        const char *error = mmDlerror();
        if (error == nullptr) {
          error = "";
        }
        GELOGW("Failed to invoke function %s in %s! errmsg:%s", func_name.c_str(), handle.first.c_str(), error);
        return GE_PLGMGR_INVOKE_FAILED;
      } else {
        T res = real_fn();
        if (res != SUCCESS) {
          return FAILED;
        }
      }
    }
    return SUCCESS;
  }

 private:
  void ClearHandles_() noexcept;
  Status ValidateSo(const string &file_path, int64_t size_of_loaded_so, int64_t &file_size) const;

  vector<string> so_list_;
  SoToHandleMap handles_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_PLUGIN_MANAGER_H_
