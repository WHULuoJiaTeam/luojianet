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
#ifndef LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_DUMP_PROTO_H_
#define LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_DUMP_PROTO_H_

#include <string>
#include <memory>

#include "ir/func_graph.h"
#include "proto/mind_ir.pb.h"
#include "include/common/debug/common.h"
#include "include/common/visible.h"
#include "proto/anf_ir.pb.h"

namespace luojianet_ms {
COMMON_EXPORT std::string GetFuncGraphProtoString(const FuncGraphPtr &func_graph);

std::string GetOnnxProtoString(const FuncGraphPtr &func_graph);

std::string GetBinaryProtoString(const FuncGraphPtr &func_graph);

bool DumpBinaryProto(const FuncGraphPtr &func_graph, const std::string &file_path,
                     const FuncGraphPtr &param_layout_fg = nullptr);

COMMON_EXPORT void DumpIRProto(const FuncGraphPtr &func_graph, const std::string &suffix);

COMMON_EXPORT void GetFuncGraphProto(const FuncGraphPtr &func_graph, irpb::GraphProto *graph_proto);
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_INCLUDE_COMMON_DEBUG_DUMP_PROTO_H_
