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

#include "hybrid/executor/hybrid_execution_context.h"
#include <atomic>

namespace ge {
namespace hybrid {
namespace {
const uint32_t kEndOfSequence = 0x0704000a;
const uint32_t kEndOfSequenceNew = 507005;
const int32_t kModelAbortNormal = 0x0704000e;
const int32_t kModelAbortNormalNew = 507024;

std::atomic_ulong context_id_gen {};
}  // namespace

long GraphExecutionContext::profiling_level = 0;

GraphExecutionContext::GraphExecutionContext() {
  context_id = context_id_gen++;
}

void GraphExecutionContext::SetErrorCode(Status error_code) {
  std::lock_guard<std::mutex> lk(mu);
  this->status = error_code;
}

Status GraphExecutionContext::GetStatus() const {
  std::lock_guard<std::mutex> lk(mu);
  return this->status;
}

Status GraphExecutionContext::Synchronize(rtStream_t rt_stream) {
  auto rt_ret = rtStreamSynchronize(rt_stream);
  if (rt_ret == RT_ERROR_NONE) {
    return SUCCESS;
  }

  if (rt_ret == kEndOfSequence || rt_ret == kEndOfSequenceNew) {
    GELOGI("Got end of sequence");
    is_eos_ = true;
    return END_OF_SEQUENCE;
  }

  if (rt_ret == kModelAbortNormal || rt_ret == kModelAbortNormalNew) {
    GELOGI("The model with multiple datasets aborts normally");
    return SUCCESS;
  }

  GELOGE(RT_FAILED, "[Invoke][rtStreamSynchronize] failed, ret = %d", rt_ret);
  REPORT_CALL_ERROR("E19999", "invoke rtStreamSynchronize failed, ret = %d", rt_ret);
  return RT_FAILED;
}

Status GraphExecutionContext::DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos) {
  if (exception_infos.empty()) {
    GELOGI("[Dump][ExceptionInfo] Exception info is null.");
    return SUCCESS;
  }
  GELOGI("[Dump][ExceptionInfo] Start to search dynamic op info and to dump.");
  if (exception_dumper.DumpExceptionInfo(exception_infos) != SUCCESS) {
    GELOGE(FAILED, "[Dump][Exception] Dump dynamic op exception info failed.");
    return FAILED;
  }
  GELOGI("[Dump][ExceptionInfo] Start to search static op info and to dump.");
  for (const auto &iter : davinci_model) {
    if (iter != nullptr) {
      if (iter->DumpExceptionInfo(exception_infos) != SUCCESS) {
        GELOGE(FAILED, "[Dump][ExceptionInfo] Dump static op exception info failed.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace hybrid
}  // namespace ge