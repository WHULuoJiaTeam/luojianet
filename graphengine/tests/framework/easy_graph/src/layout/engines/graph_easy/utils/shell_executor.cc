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

#include <sys/types.h>
#include <sys/wait.h>
#include <cstdlib>
#include <string>
#include "easy_graph/infra/log.h"
#include "layout/engines/graph_easy/utils/shell_executor.h"

EG_NS_BEGIN

Status ShellExecutor::execute(const std::string &script) {
  EG_DBG("%s", script.c_str());

  pid_t status = system(script.c_str());
  if (-1 == status) {
    EG_ERR("system execute return error!");
    return EG_FAILURE;
  }

  if (WIFEXITED(status) && (0 == WEXITSTATUS(status)))
    return EG_SUCCESS;

  EG_ERR("system execute {%s} exit status value = [0x%x], exit code: %d\n", script.c_str(), status,
         WEXITSTATUS(status));
  return EG_FAILURE;
}

EG_NS_END
