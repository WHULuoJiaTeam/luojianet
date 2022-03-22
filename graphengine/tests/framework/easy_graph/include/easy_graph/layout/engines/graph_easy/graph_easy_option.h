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

#ifndef H37156CC2_92BD_44DA_8DA7_A11629E762BE
#define H37156CC2_92BD_44DA_8DA7_A11629E762BE

#include "easy_graph/layout/layout_option.h"
#include <string>

EG_NS_BEGIN

enum class FlowDir {
  LR = 0,
  TB,
};

enum class LayoutType {
  FREE = 0,
  REGULAR,
};

enum class LayoutFormat {
  ASCII = 0,
  BOXART,
  SVG,
  DOT,
  HTML,
};

enum class LayoutOutput {
  CONSOLE = 0,
  FILE,
};

struct GraphEasyOption : LayoutOption {
  static const GraphEasyOption &GetDefault();

  std::string GetLayoutCmdArgs(const std::string &graphName) const;

  LayoutFormat format_{LayoutFormat::BOXART};
  LayoutOutput output_{LayoutOutput::CONSOLE};
  FlowDir dir_{FlowDir::LR};
  LayoutType type_{LayoutType::FREE};
  size_t scale_{1};
  std::string output_path_{"./"};
};

EG_NS_END

#endif
