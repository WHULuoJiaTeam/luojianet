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

#include "hybrid/executor/rt_callback_manager.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"

namespace ge {
namespace hybrid {
Status CallbackManager::RegisterCallback(rtStream_t stream, rtCallback_t callback, void *user_data) {
  GELOGD("To register callback");
  rtEvent_t event = nullptr;
  GE_CHK_RT_RET(rtEventCreate(&event));
  auto rt_ret = rtEventRecord(event, stream);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Invoke][rtEventRecord] failed, error code = %d", rt_ret);
    REPORT_CALL_ERROR("E19999", "Invoke rtEventRecord failed, error code = %d", rt_ret);
    (void) rtEventDestroy(event);
    return RT_FAILED;
  }

  auto cb = std::pair<rtCallback_t, void *>(callback, user_data);
  auto entry = std::pair<rtEvent_t, std::pair<rtCallback_t, void *>>(event, std::move(cb));
  if (!callback_queue_.Push(entry)) {
    (void) rtEventDestroy(event);
    return INTERNAL_ERROR;
  }

  GELOGD("Registering callback successfully");
  return SUCCESS;
}

Status CallbackManager::Init() {
  rtContext_t ctx = nullptr;
  GE_CHK_RT_RET(rtCtxGetCurrent(&ctx));
  ret_future_ = std::async(std::launch::async, [&](rtContext_t context) ->Status {
    return CallbackProcess(context);
  }, ctx);
  if (!ret_future_.valid()) {
    GELOGE(INTERNAL_ERROR, "[Check][ShareState]Failed to init callback manager.");
    REPORT_INNER_ERROR("E19999", "Failed to init callback manager.");
    return INTERNAL_ERROR;
  }

  return SUCCESS;
}

Status CallbackManager::CallbackProcess(rtContext_t context) {
  GE_CHK_RT_RET(rtCtxSetCurrent(context));
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> entry;
  while (true) {
    if (!callback_queue_.Pop(entry)) {
      GELOGI("CallbackManager stopped");
      return INTERNAL_ERROR;
    }

    auto event = entry.first;
    if (event == nullptr) {
      return SUCCESS;
    }

    auto rt_err = rtEventSynchronize(event);
    if (rt_err != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "[Invoke][rtEventSynchronize] failed. ret = %d", rt_err);
      REPORT_CALL_ERROR("E19999", "Invoke rtEventSynchronize failed, ret = %d.", rt_err);
      GE_CHK_RT(rtEventDestroy(event));
      return RT_FAILED;
    }

    GE_CHK_RT(rtEventDestroy(event));

    auto cb_func = entry.second.first;
    auto cb_args = entry.second.second;
    cb_func(cb_args);
  }
}

Status CallbackManager::Destroy() {
  GELOGI("To destroy callback manager.");
  if (!ret_future_.valid()) {
    GELOGI("CallbackManager not initialized.");
    return SUCCESS;
  }

  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  callback_queue_.Push(eof_entry);

  auto ret = ret_future_.get();
  GELOGI("Callback manager ended. ret = %u", ret);
  return ret;
}

void CallbackManager::RtCallbackFunc(void *data) {
  GELOGD("To invoke callback function");
  auto callback_func = reinterpret_cast<std::function<void()> *>(data);
  (*callback_func)();
  delete callback_func;
}

Status CallbackManager::RegisterCallback(rtStream_t stream, const std::function<void()> &callback) {
  auto func = std::unique_ptr<std::function<void()>>(new(std::nothrow) std::function<void()>(callback));
  GE_CHECK_NOTNULL(func);
  GELOGD("Callback registered");
  return RegisterCallback(stream, RtCallbackFunc, func.release());
}
}  // namespace hybrid
}  // namespace ge
