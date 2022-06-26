/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/pass/npu_pass_manager.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore {
void NPUPassManager::AddPass(NPUBasePass *pass) { all_pass_.push_back(pass); }

int NPUPassManager::RunPass(NPUGraph *subgraph) {
  for (auto pass : all_pass_) {
    auto ret = pass->Run(subgraph);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "NPU Pass Run failed. Pass name is:" << pass->name() << " for subgraph " << subgraph->name();
      return ret;
    }
  }
  return RET_OK;
}

void NPUPassManager::Clear() {
  for (auto pass : all_pass_) {
    delete pass;
  }
  all_pass_.clear();
}
}  // namespace mindspore
