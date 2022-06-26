/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "include/common/debug/dump_proto.h"
#include "proto/mind_ir.pb.h"

namespace mindspore {

void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix) { return; }

std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph) { return ""; }

std::string GetOnnxProtoString(const FuncGraphPtr &func_graph) { return ""; }

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph) { return ""; }

bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                     const FuncGraphPtr &param_layout_fg) {
  return true;
}
}  // namespace mindspore
