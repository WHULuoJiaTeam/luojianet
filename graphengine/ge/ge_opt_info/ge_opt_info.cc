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

#include "ge_opt_info/ge_opt_info.h"

#include <string>
#include <map>
#include "graph/ge_local_context.h"
#include "ge/ge_api_types.h"
#include "common/debug/ge_log.h"
#include "opt_info.h"

namespace ge {
Status GeOptInfo::SetOptInfo() {
  std::string soc_ver;
  graphStatus ret = GetThreadLocalContext().GetOption(SOC_VERSION, soc_ver);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get soc version failed.");
    GELOGE(FAILED, "[Get][SocVersion]Get soc version failed.");
    return FAILED;
  }
  GELOGD("Soc version:%s.", soc_ver.c_str());
  std::map<std::string, std::string> opt_info;
  // the first arg does not work at present.
  if (gelc::GetOptInfo(gelc::kOffline, soc_ver, opt_info) != gelc::SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get optional information failed, is_offline:%d, soc version:%s",
                      gelc::kOffline, soc_ver.c_str());
    GELOGE(FAILED, "[Get][OptInfo]Get optional information failed, is_offline:%d, soc version:%s",
           gelc::kOffline, soc_ver.c_str());
    return FAILED;
  }
  // do nothing if get empty information
  if (opt_info.empty()) {
    GELOGI("Optional information is empty.");
    return SUCCESS;
  }
  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  for (const auto &itr : opt_info) {
    graph_options.emplace(itr.first, itr.second);
    GELOGI("Get optional information success, key:%s, value:%s.", itr.first.c_str(), itr.second.c_str());
  }
  GetThreadLocalContext().SetGraphOption(graph_options);
  return SUCCESS;
}
}  // namespace ge
