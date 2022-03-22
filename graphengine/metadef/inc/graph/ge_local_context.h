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
#ifndef INC_GRAPH_GE_LOCAL_CONTEXT_H_
#define INC_GRAPH_GE_LOCAL_CONTEXT_H_

#include <map>
#include <string>
#include <vector>
#include "graph/ge_error_codes.h"

namespace ge {
class GEThreadLocalContext {
 public:
  graphStatus GetOption(const std::string &key, std::string &option);
  void SetGraphOption(std::map<std::string, std::string> options_map);
  void SetSessionOption(std::map<std::string, std::string> options_map);
  void SetGlobalOption(std::map<std::string, std::string> options_map);

  std::map<std::string, std::string> GetAllGraphOptions() const;
  std::map<std::string, std::string> GetAllSessionOptions() const;
  std::map<std::string, std::string> GetAllGlobalOptions() const;
  std::map<std::string, std::string> GetAllOptions() const;

 private:
  std::map<std::string, std::string> graph_options_;
  std::map<std::string, std::string> session_options_;
  std::map<std::string, std::string> global_options_;
};  // class GEThreadLocalContext

GEThreadLocalContext &GetThreadLocalContext();
}  // namespace ge
#endif  // INC_GRAPH_GE_LOCAL_CONTEXT_H_
