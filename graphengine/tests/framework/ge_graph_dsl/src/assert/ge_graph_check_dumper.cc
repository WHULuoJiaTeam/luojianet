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

#include "ge_graph_check_dumper.h"
#include "graph/model.h"
#include "graph/buffer.h"
#include "graph/utils/graph_utils.h"
#include "ge_graph_default_checker.h"

GE_NS_BEGIN

GeGraphCheckDumper::GeGraphCheckDumper() { Reset(); }

bool GeGraphCheckDumper::IsNeedDump(const std::string &suffix) const {
  auto iter = std::find(suffixes_.begin(), suffixes_.end(), suffix);
  return (iter != suffixes_.end());
}

void GeGraphCheckDumper::Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix) {
  if (!IsNeedDump(suffix)) {
    return;
  }
  auto iter = buffers_.find(suffix);
  if (iter != buffers_.end()) {
    DumpGraph(graph, iter->second);
  } else {
    buffers_[suffix] = Buffer();
    DumpGraph(graph, buffers_.at(suffix));
  }
}

bool GeGraphCheckDumper::CheckFor(const GeGraphChecker &checker) {
  auto iter = buffers_.find(checker.PhaseId());
  if (iter == buffers_.end()) {
    return false;
  }
  DoCheck(checker, iter->second);
  return true;
}

void GeGraphCheckDumper::DoCheck(const GeGraphChecker &checker, ::GE_NS::Buffer &buffer) {
  Model model("", "");
  Model::Load(buffer.GetData(), buffer.GetSize(), model);
  auto load_graph = model.GetGraph();
  checker.Check(GraphUtils::GetComputeGraph(load_graph));
}

void GeGraphCheckDumper::DumpGraph(const ge::ComputeGraphPtr &graph, ::GE_NS::Buffer &buffer) {
  Model model("", "");
  buffer.clear();
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  model.Save(buffer, true);
}

void GeGraphCheckDumper::Update(const std::vector<std::string> &new_suffixes_) {
  suffixes_ = new_suffixes_;
  buffers_.clear();
}

void GeGraphCheckDumper::Reset() {
  static std::vector<std::string> default_suffixes_{"PreRunAfterBuild"};
  suffixes_ = default_suffixes_;
  buffers_.clear();
}

GE_NS_END