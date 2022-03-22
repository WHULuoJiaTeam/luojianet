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

#include "graph/manager/util/rt_context_util.h"

#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
  const int64_t kDefaultGraphId = -1;
}

void RtContextUtil::AddRtContext(uint64_t session_id, rtContext_t context) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  rt_contexts_[session_id][kDefaultGraphId].emplace_back(context);
}

void RtContextUtil::AddRtContext(uint64_t session_id, uint32_t graph_id, rtContext_t context) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  rt_contexts_[session_id][static_cast<int64_t>(graph_id)].emplace_back(context);
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  auto &session_ctxs = rt_contexts_[session_id];
  for (auto &graph_ctx_pair : session_ctxs) {
    DestroyRtContexts(session_id, graph_ctx_pair.first, graph_ctx_pair.second);
  }

  auto iter = rt_contexts_.find(session_id);
  if (iter != rt_contexts_.end()) {
    rt_contexts_.erase(iter);
  }
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id, uint32_t graph_id) {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  auto &session_ctxs = rt_contexts_[session_id];
  auto &graph_ctxs = session_ctxs[graph_id];
  DestroyRtContexts(session_id, static_cast<int64_t>(graph_id), graph_ctxs);

  auto iter = session_ctxs.find(graph_id);
  if (iter != session_ctxs.end()) {
    session_ctxs.erase(iter);
  }
}

void RtContextUtil::DestroyAllRtContexts() {
  std::lock_guard<std::mutex> lock(ctx_mutex_);
  for (auto &session_ctx_pair : rt_contexts_) {
    for (auto &graph_ctx_pair : session_ctx_pair.second) {
      DestroyRtContexts(session_ctx_pair.first, graph_ctx_pair.first, graph_ctx_pair.second);
    }
  }
  rt_contexts_.clear();
}

void RtContextUtil::DestroyRtContexts(uint64_t session_id, int64_t graph_id, std::vector<rtContext_t> &contexts) {
  GELOGI("Destroy %zu rts contexts for graph %ld of session %lu.", contexts.size(), graph_id, session_id);
  for (auto &rtContext : contexts) {
    (void)rtCtxDestroy(rtContext);
  }
  contexts.clear();
}
}  // namespace ge
