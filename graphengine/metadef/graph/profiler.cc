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
#include "graph/profiler.h"
#include <cstring>
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/debug/ge_log.h"

namespace ge {
namespace profiling {
namespace {
constexpr char kVersion[] = "1.0";
int64_t GetThread() {
#ifdef __GNUC__
  thread_local static auto tid = static_cast<int64_t>(syscall(__NR_gettid));
#else
  thread_local static auto tid = static_cast<int64_t>(GetCurrentThreadId());
#endif
  return tid;
}
void DumpEventType(EventType et, std::ostream &out_stream) {
  switch (et) {
    case kEventStart:
      out_stream << "Start";
      break;
    case kEventEnd:
      out_stream << "End";
      break;
    case kEventTimestamp:
      break;
    default:
      out_stream << "UNKNOWN(" << static_cast<int64_t>(et) << ")";
      break;
  }
}
}

void Profiler::RecordCurrentThread(int64_t element, int64_t event, EventType et) {
  Record(element, GetThread(), event, et);
}
void Profiler::RegisterString(int64_t index, const std::string &str) {
  if (index >= kMaxStrIndex) {
    return;
  }

  // can not use strcpy_s, which will copy nothing when the length of str beyond kMaxStrLen
  auto ret = strncpy_s(GetStrings()[index], kMaxStrLen, str.c_str(), kMaxStrLen - 1);
  if (ret != EN_OK) {
    GELOGW("Register string failed, index %ld, str %s", index, str.c_str());
  }
}
void Profiler::Record(int64_t element, int64_t thread, int64_t event, EventType et) {
  auto current_index = record_size_++;
  if (current_index >= kMaxRecordNum) {
    return;
  }
  records_[current_index] = ProfilingRecord({element, thread, event, et, std::chrono::system_clock::now()});
}
void Profiler::Dump(std::ostream &out_stream) const {
  size_t print_size = record_size_;
  out_stream << "Profiler version: " << kVersion << ", dump start, records num: " << print_size << std::endl;
  if (print_size > records_.size()) {
    out_stream << "Too many records(" << print_size << "), the records after "
               << records_.size() << " will be dropped" << std::endl;
    print_size = records_.size();
  }
  for (size_t i = 0; i < print_size; ++i) {
    auto &rec = records_[i];
    // in format: <timestamp> <thread-id> <module-id> <record-type> <event-type>
    out_stream << std::chrono::duration_cast<std::chrono::nanoseconds>(rec.timestamp.time_since_epoch()).count() << ' ';
    out_stream << rec.thread << ' ';
    DumpByIndex(rec.element, out_stream);
    out_stream << ' ';
    DumpByIndex(rec.event, out_stream);
    out_stream << ' ';
    DumpEventType(rec.et, out_stream);
    out_stream << std::endl;
  }
  out_stream << "Profiling dump end" << std::endl;
}
void Profiler::DumpByIndex(int64_t index, std::ostream &out_stream) const {
  if (index < 0 || index >= kMaxStrIndex || strnlen(GetStrings()[index], kMaxStrLen) == 0) {
    out_stream << "UNKNOWN(" << index << ")";
  } else {
    out_stream << '[' << GetStrings()[index] << "]";
  }
}
Profiler::Profiler() : record_size_(0), records_(), indexes_to_str_() {}
void Profiler::Reset() {
  // 不完全reset，indexes_to_str_还是有值的
  record_size_ = 0;
}
std::unique_ptr<Profiler> Profiler::Create() {
  return std::unique_ptr<Profiler>(new(std::nothrow) Profiler());
}
size_t Profiler::GetRecordNum() const noexcept {
  return record_size_;
}
const ProfilingRecord *Profiler::GetRecords() const {
  return &(records_[0]);
}
Profiler::ConstStringsPointer Profiler::GetStrings() const {
  return indexes_to_str_;
}
Profiler::StringsPointer Profiler::GetStrings() {
  return indexes_to_str_;
}
Profiler::~Profiler() = default;
}
}