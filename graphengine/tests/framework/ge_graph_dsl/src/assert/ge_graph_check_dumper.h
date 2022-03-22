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
#ifndef INC_8EFED0015C27464897BF64531355C810
#define INC_8EFED0015C27464897BF64531355C810

#include "ge_graph_dsl/ge.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "ge_dump_filter.h"
#include <string>

GE_NS_BEGIN

struct GeGraphChecker;

struct GeGraphCheckDumper : GeGraphDumper, GeDumpFilter {
  GeGraphCheckDumper();
  virtual void Dump(const ge::ComputeGraphPtr &graph, const std::string &suffix);
  bool CheckFor(const GeGraphChecker &checker);

 private:
  void DoCheck(const GeGraphChecker &checker, ::GE_NS::Buffer &buffer);
  void DumpGraph(const ge::ComputeGraphPtr &graph, ::GE_NS::Buffer &buffer);

 private:
  void Update(const std::vector<std::string> &) override;
  void Reset() override;
  bool IsNeedDump(const std::string &suffix) const;

 private:
  std::map<std::string, ::GE_NS::Buffer> buffers_;
  std::vector<std::string> suffixes_;
};

GE_NS_END

#endif