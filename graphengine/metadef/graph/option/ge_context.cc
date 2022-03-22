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

#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace {
const int64_t kMinTrainingTraceJobId = 65536;
const int32_t kDecimal = 10;
const char_t *kHostExecPlacement = "HOST";
}
GEContext &GetContext() {
  static GEContext ge_context{};
  return ge_context;
}

thread_local uint64_t GEContext::session_id_ = 0UL;
thread_local uint64_t GEContext::context_id_ = 0UL;
thread_local uint64_t GEContext::work_stream_id_ = 0UL;

graphStatus GEContext::GetOption(const std::string &key, std::string &option) {
  return GetThreadLocalContext().GetOption(key, option);
}

bool GEContext::GetHostExecFlag() {
  std::string exec_placement;
  if (GetThreadLocalContext().GetOption("ge.exec.placement", exec_placement) != GRAPH_SUCCESS) {
    GELOGD("get option ge.exec.placement failed.");
    return false;
  }
  GELOGD("Option ge.exec.placement is %s.", exec_placement.c_str());
  return exec_placement == kHostExecPlacement;
}

std::map<std::string, std::string> &GetMutableGlobalOptions() {
  static std::map<std::string, std::string> global_options{};
  return global_options;
}

void GEContext::Init() {
  std::string session_id;
  (void)GetOption("ge.exec.sessionId", session_id);
  try{
    session_id_ = static_cast<uint64_t>(std::stoi(session_id.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("[Init][GetSessionId] Transform option session_id %s to int failed, as catching invalid_argument exception",
           session_id.c_str());
  } catch (std::out_of_range &) {
    GELOGW("[Init][GetSessionId] Transform option session_id %s to int failed, as catching out_of_range exception",
           session_id.c_str());
  }

  std::string device_id;
  (void)GetOption("ge.exec.deviceId", device_id);
  try{
    device_id_ = static_cast<uint32_t>(std::stoi(device_id.c_str()));
  } catch (std::invalid_argument &) {
    GELOGW("[Init][GetDeviceId] Transform option device_id %s to int failed, as catching invalid_argument exception",
           device_id.c_str());
  } catch (std::out_of_range &) {
    GELOGW("[Init][GetDeviceId] Transform option device_id %s to int failed, as catching out_of_range exception",
           device_id.c_str());
  }

  std::string job_id;
  (void)GetOption("ge.exec.jobId", job_id);
  std::string s_job_id = "";
  for (const auto c : job_id) {
    if ((c >= '0') && (c <= '9')) {
      s_job_id += c;
    }
  }
  if (s_job_id == "") {
    trace_id_ = static_cast<size_t>(kMinTrainingTraceJobId);
    return;
  }
  const int64_t d_job_id = static_cast<int64_t>(std::strtoll(s_job_id.c_str(), nullptr, kDecimal));
  if (d_job_id < kMinTrainingTraceJobId) {
    trace_id_ = static_cast<size_t>(d_job_id + kMinTrainingTraceJobId);
  } else {
    trace_id_ = static_cast<size_t>(d_job_id);
  }
}

uint64_t GEContext::SessionId() { return session_id_; }

uint64_t GEContext::ContextId() { return context_id_; }

uint64_t GEContext::WorkStreamId() { return work_stream_id_; }

uint32_t GEContext::DeviceId() { return device_id_; }

uint64_t GEContext::TraceId() { return trace_id_; }

void GEContext::SetSessionId(const uint64_t session_id) { session_id_ = session_id; }

void GEContext::SetContextId(const uint64_t context_id) { context_id_ = context_id; }

void GEContext::SetWorkStreamId(const uint64_t work_stream_id) { work_stream_id_ = work_stream_id; }

void GEContext::SetCtxDeviceId(const uint32_t device_id) { device_id_ = device_id; }

}  // namespace ge
