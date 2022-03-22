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
#ifndef METADEF_CXX_PROFILER_H
#define METADEF_CXX_PROFILER_H
#include <memory>
#include <array>
#include <vector>
#include <ostream>
#include <chrono>
#include <atomic>

namespace ge {
namespace profiling {
constexpr size_t kMaxStrLen = 64;
constexpr int64_t kMaxStrIndex = 1024 * 1024;
constexpr size_t kMaxRecordNum = 10 * 1024 * 1024;
enum EventType {
  kEventStart,
  kEventEnd,
  kEventTimestamp,
  kEventTypeEnd
};
struct ProfilingRecord {
  int64_t element;
  int64_t thread;
  int64_t event;
  EventType et;
  std::chrono::time_point<std::chrono::system_clock> timestamp;
};
class Profiler {
 public:
  static std::unique_ptr<Profiler> Create();
  void RegisterString(int64_t index, const std::string &str);
  void Record(int64_t element, int64_t thread, int64_t event, EventType et);
  void RecordCurrentThread(int64_t element, int64_t event, EventType et);

  void Reset();
  void Dump(std::ostream &out_stream) const;

  size_t GetRecordNum() const noexcept;
  const ProfilingRecord *GetRecords() const;

  using ConstStringsPointer = char const(*)[kMaxStrLen];
  using StringsPointer = char (*)[kMaxStrLen];
  ConstStringsPointer GetStrings() const;
  StringsPointer GetStrings() ;

  ~Profiler();

 private:
  Profiler();
  void DumpByIndex(int64_t index, std::ostream &out_stream) const;

 private:
  std::atomic<size_t> record_size_;
  std::array<ProfilingRecord, kMaxRecordNum> records_;
  char indexes_to_str_[kMaxStrIndex][kMaxStrLen];
};
}
}
#endif  // METADEF_CXX_PROFILER_H
