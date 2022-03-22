/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#include <gtest/gtest.h>
#include "graph/profiler.h"
namespace ge {
namespace profiling {
namespace {
std::string FindNext(const std::string &s, size_t &pos) {
  std::stringstream ss;
  for (; pos < s.size(); ++pos) {
    if (s[pos] == '\r') {
      ++pos;
      if (pos + 1 < s.size() && s[pos + 1] == '\n') {
        ++pos;
      }
      return ss.str();
    }
    if (s[pos] == '\n') {
      ++pos;
      if (pos + 1 < s.size() && s[pos + 1] == '\r') {
        ++pos;
      }
      return ss.str();
    }
    ss << s[pos];
  }
  return ss.str();
}
std::vector<std::string> SplitLines(const std::string &s) {
  std::vector<std::string> strings;
  size_t i = 0;
  while (i < s.size()) {
    strings.emplace_back(FindNext(s, i));
  }
  return strings;
}
std::vector<std::string> Split(const std::string &s, std::string spliter) {
  std::vector<std::string> strings;
  size_t i = 0;
  while (i < s.size()) {
    auto pos = s.find_first_of(spliter, i);
    if (pos == std::string::npos) {
      strings.emplace_back(s, i);
      break;
    } else {
      strings.emplace_back(s, i, pos - i);
      i = pos + spliter.size();
    }
  }

  return strings;
}
}
class ProfilerUt : public testing::Test {};

TEST_F(ProfilerUt, OneRecord) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, kEventStart);
  EXPECT_EQ(p->GetRecordNum(), 1);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, TimeStampRecord) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, kEventTimestamp);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 4);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
}

TEST_F(ProfilerUt, MultipleRecords) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, kEventStart);
  p->Record(0, 1, 2, kEventEnd);
  EXPECT_EQ(p->GetRecordNum(), 2);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 4);

  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");

  elements = Split(lines[2], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "End");
}

TEST_F(ProfilerUt, RecordStr) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, kEventStart);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "[Node1]");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, RecordCurrentThread) {
  auto p = Profiler::Create();
  p->RecordCurrentThread(0, 2, kEventStart);
  p->RecordCurrentThread(0, 2, kEventEnd);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 4);

  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "Start");

  elements = Split(lines[2], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "End");
}

TEST_F(ProfilerUt, Reset) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, kEventStart);
  p->Reset();
  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 2);
}

TEST_F(ProfilerUt, ResetRemainsRegisteredString) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  p->Record(0, 1, 2, kEventStart);
  p->Reset();
  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 2);


  p->Record(0, 1, 2, kEventStart);
  ss = std::stringstream();
  p->Dump(ss);
  lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "[Node1]");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, RegisterStringBeyondMaxSize) {
  auto p = Profiler::Create();
  p->RegisterString(2, "InferShape");
  p->RegisterString(kMaxStrIndex, "[Node1]");
  p->Record(kMaxStrIndex, 1, 2, kEventStart);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(" + std::to_string(kMaxStrIndex) + ")");
  EXPECT_EQ(elements[3], "[InferShape]");
  EXPECT_EQ(elements[4], "Start");
}

TEST_F(ProfilerUt, EventTypeBeyondRange) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, kEventTypeEnd);

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), 3);
  auto elements = Split(lines[1], " ");
  EXPECT_EQ(elements.size(), 5);
  EXPECT_EQ(elements[1], "1");
  EXPECT_EQ(elements[2], "UNKNOWN(0)");
  EXPECT_EQ(elements[3], "UNKNOWN(2)");
  EXPECT_EQ(elements[4], "UNKNOWN(3)");
}

TEST_F(ProfilerUt, GetRecords) {
  auto p = Profiler::Create();
  p->Record(0, 1, 2, kEventTypeEnd);
  auto rec = p->GetRecords();
  EXPECT_EQ(rec->element, 0);
  EXPECT_EQ(rec->thread, 1);
  EXPECT_EQ(rec->event, 2);
  EXPECT_EQ(rec->et, kEventTypeEnd);
}

TEST_F(ProfilerUt, GetStrings) {
  auto p = Profiler::Create();
  p->RegisterString(0, "Node1");
  p->RegisterString(2, "InferShape");
  auto s = p->GetStrings();
  EXPECT_EQ(strcmp(s[0], "Node1"), 0);
  EXPECT_EQ(strcmp(s[2], "InferShape"), 0);
}

TEST_F(ProfilerUt, RegisterTooLongString) {
  auto p = Profiler::Create();
  p->RegisterString(0, "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyz");
  p->RegisterString(2, "InferShape");
  auto s = p->GetStrings();
  EXPECT_EQ(strcmp(s[0], "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijk"), 0);
  EXPECT_EQ(strcmp(s[2], "InferShape"), 0);
}

TEST_F(ProfilerUt, ModifyStrings) {
  auto p = Profiler::Create();
  p->RegisterString(0, "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyz");
  p->RegisterString(2, "InferShape");
  auto s = p->GetStrings();
  strcpy(s[2], "Tiling");
  EXPECT_EQ(strcmp(s[0], "AbcdefghijklmnopqrstuvwxyzAbcdefghijklmnopqrstuvwxyzAbcdefghijk"), 0);
  EXPECT_EQ(strcmp(p->GetStrings()[2], "Tiling"), 0);
}

/* takes very long time
TEST_F(ProfilerUt, BeyondMaxRecordsNum) {
  auto p = Profiler::Create();
  for (int64_t i = 0; i < profiling::kMaxRecordNum; ++i) {
    p->Record(0, 1, i, kEventStart);
    p->Record(0, 1, i, kEventEnd);
  }

  std::stringstream ss;
  p->Dump(ss);
  auto lines = SplitLines(ss.str());
  EXPECT_EQ(lines.size(), profiling::kMaxRecordNum + 3);
}
*/
}
}