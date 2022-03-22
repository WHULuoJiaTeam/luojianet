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

#include <map>
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"

EG_NS_BEGIN

namespace {
struct Format {
  const char *format;
  const char *postfix;
};

std::map<LayoutFormat, Format> formats = {{LayoutFormat::ASCII, {"ascii", "txt"}},
                                          {LayoutFormat::BOXART, {"boxart", "txt"}},
                                          {LayoutFormat::SVG, {"svg", "svg"}},
                                          {LayoutFormat::DOT, {"dot", "dot"}},
                                          {LayoutFormat::HTML, {"html", "html"}}};

std::string GetLayoutOutputArg(const GraphEasyOption &options, const std::string &graphName) {
  if (options.output_ == LayoutOutput::CONSOLE)
    return "";
  return std::string(" --output ") + options.output_path_ + graphName + "." + formats[options.format_].postfix;
}

std::string GetLayoutFomartArg(const GraphEasyOption &options) {
  return std::string(" --as=") + formats[options.format_].format;
}
}  // namespace

const GraphEasyOption &GraphEasyOption::GetDefault() {
  static GraphEasyOption option;
  return option;
}

std::string GraphEasyOption::GetLayoutCmdArgs(const std::string &graphName) const {
  return GetLayoutFomartArg(*this) + GetLayoutOutputArg(*this, graphName);
}

EG_NS_END
