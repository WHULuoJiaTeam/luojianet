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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMMON_COMPONENT_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMMON_COMPONENT_H_

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include "src/tensor.h"
#include "coder/context.h"
#include "coder/config.h"

namespace luojianet_ms::lite::micro {
void CodeSessionCompileGraph(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx, const Configurator *config);
void CodeCreateSessionImplement(std::ofstream &ofs, const Configurator *config);
void CodeCreateSessionDestructor(std::ofstream &ofs, const Configurator *config);

void CodeCopyOutputsState(std::ofstream &ofs);
void CodeCopyOutputsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeGlobalCodeBlocks(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeInputState(std::ofstream &ofs);
void CodeInputImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeGraphQuantArgsState(std::ofstream &ofs);
void CodeGraphQuantArgsImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeManageResourceState(std::ofstream &ofs);
void CodeInitResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeFreeResourceImplement(std::ofstream &ofs, const std::unique_ptr<CoderContext> &ctx);

void CodeInferenceState(std::ofstream &ofs);
}  // namespace luojianet_ms::lite::micro

#endif  // LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMMON_COMPONENT_H_
