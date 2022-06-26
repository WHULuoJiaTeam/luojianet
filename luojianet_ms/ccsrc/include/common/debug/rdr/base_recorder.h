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
#ifndef LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_RDR_BASE_RECORDER_H_
#define LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_RDR_BASE_RECORDER_H_
#include <memory>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include "include/common/debug/common.h"
#include "include/common/visible.h"
#include "utils/log_adapter.h"

namespace luojianet_ms {
class COMMON_EXPORT BaseRecorder {
 public:
  BaseRecorder();
  BaseRecorder(const std::string &module, const std::string &name);
  virtual ~BaseRecorder() {}

  std::string GetModule() const { return module_; }
  std::string GetName() const { return name_; }
  std::string GetTimeStamp() const { return timestamp_; }
  std::optional<std::string> GetFileRealPath(const std::string &suffix = "") const;

  virtual void Export() {}

 protected:
  std::string module_;
  std::string name_;
  std::string directory_;
  std::string filename_;
  std::string timestamp_;  // year,month,day,hour,minute,second
  std::string delimiter_{"."};
};
using BaseRecorderPtr = std::shared_ptr<BaseRecorder>;

class CNode;
using CNodePtr = std::shared_ptr<CNode>;
class FuncGraph;
using FuncGraphPtr = std::shared_ptr<FuncGraph>;
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_RDR_BASE_RECORDER_H_
