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

#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "framework/common/debug/log.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"

namespace ge {
namespace hybrid {
namespace {
const size_t kNumTaskWithAtomicAddrCleanTask = 2;
}
const char *AiCoreKernelRegistry::GetUnique(const string &stub_key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = unique_stubs_.find(stub_key);
  if (it != unique_stubs_.end()) {
    return it->c_str();
  }
  it = unique_stubs_.insert(unique_stubs_.end(), stub_key);
  return it->c_str();
}

AiCoreTaskBuilder::AiCoreTaskBuilder(const OpDescPtr &op_desc, const std::vector<domi::TaskDef> &task_defs)
    : op_desc_(op_desc), task_defs_(task_defs) {
}

Status AiCoreTaskBuilder::BuildTask(std::unique_ptr<AiCoreNodeTask> &node_task,
                                    bool ignore_failure_on_atomic,
                                    bool is_single_op) {
  GE_CHECK_NOTNULL(op_desc_);
  if (task_defs_.size() > kNumTaskWithAtomicAddrCleanTask) {
    GELOGE(INTERNAL_ERROR, "[Check][Size][%s] At most %zu task was supported, but got %zu",
           op_desc_->GetName().c_str(), kNumTaskWithAtomicAddrCleanTask, task_defs_.size());
    REPORT_INNER_ERROR("E19999", "[%s] At most %zu task was supported, but got %zu, check invalid.",
                       op_desc_->GetName().c_str(), kNumTaskWithAtomicAddrCleanTask, task_defs_.size());
    return INTERNAL_ERROR;
  }

  std::vector<std::unique_ptr<AiCoreOpTask>> op_tasks;
  if (ExpectAtomicAddrCleanTask()) {
    if (task_defs_.size() != kNumTaskWithAtomicAddrCleanTask) {
      if (ignore_failure_on_atomic) {
        GELOGI("[%s] AtomicAddrClean task was expected, but got %zu task_defs",
               op_desc_->GetName().c_str(),
               task_defs_.size());
        return SUCCESS;
      } else {
        GELOGE(INTERNAL_ERROR, "[Check][Size][%s] AtomicAddrClean task was expected:%zu, but got %zu task_defs",
               op_desc_->GetName().c_str(), kNumTaskWithAtomicAddrCleanTask, task_defs_.size());
        REPORT_INNER_ERROR("E19999", "[%s] AtomicAddrClean task was expected:%zu, but got %zu task_defs,",
                           op_desc_->GetName().c_str(), kNumTaskWithAtomicAddrCleanTask, task_defs_.size());
        return INTERNAL_ERROR;
      }
    }

    GELOGD("[%s] Build AtomicAddrClean task.", op_desc_->GetName().c_str());
    auto atomic_task =
        std::unique_ptr<AtomicAddrCleanOpTask>(new(std::nothrow)AtomicAddrCleanOpTask());
    GE_CHECK_NOTNULL(atomic_task);
    atomic_task->SetSingleOp(is_single_op);
    GE_CHK_STATUS_RET(atomic_task->Init(*op_desc_, task_defs_.front()),
                      "[Invoke][AtomicAddrCleanOpTask::Init] failed for [%s].",
                      op_desc_->GetName().c_str());
    op_tasks.emplace_back(std::move(atomic_task));
  }

  // build aicore task
  auto aicore_task = std::unique_ptr<AiCoreOpTask>(new(std::nothrow)AiCoreOpTask());
  GE_CHECK_NOTNULL(aicore_task);
  aicore_task->SetSingleOp(is_single_op);
  GE_CHK_STATUS_RET(aicore_task->Init(*op_desc_, task_defs_.back()),
                    "[Invoke][AiCoreOpTask::Init] failed for [%s].",
                    op_desc_->GetName().c_str());
  op_tasks.emplace_back(std::move(aicore_task));

  node_task.reset(new(std::nothrow)AiCoreNodeTask(std::move(op_tasks)));
  GE_CHECK_NOTNULL(node_task);
  return SUCCESS;
}

bool AiCoreTaskBuilder::ExpectAtomicAddrCleanTask() {
  if (op_desc_->HasAttr(ATOMIC_ATTR_OUTPUT_INDEX)) {
    GELOGD("[%s] Node has ATOMIC_ATTR_OUTPUT_INDEX", op_desc_->GetName().c_str());
    return true;
  }
  map<string, map<int64_t, int64_t>> workspace_info;
  workspace_info = op_desc_->TryGetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);

  return !workspace_info.empty();
}
}  // namespace hybrid
}  // namespace ge
