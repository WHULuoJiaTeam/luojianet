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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMPONENT_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMPONENT_H_

namespace luojianet_ms::lite::micro {
constexpr auto kInputPrefixName = "g_Input";
constexpr auto kOutputPrefixName = "g_Output";
constexpr auto kWeightPrefixName = "g_Weight";
constexpr auto kPackWeightOffsetName = "w_offset";
constexpr auto kPackWeightSizeName = "w_size";
constexpr auto kBufferPrefixName = "g_Buffer";
constexpr auto kBufferPrefixNameAdd = "g_Buffer + ";

constexpr auto kModelName = "net";

constexpr auto kSourcePath = "/src/";

constexpr auto kBenchmarkPath = "/benchmark/";
constexpr auto kBenchmarkFile = "benchmark.cc";

constexpr auto kSession = "session";
constexpr auto kTensor = "tensor";

constexpr auto kNameSpaceLUOJIANET_MS = "namespace luojianet_ms";
constexpr auto kNameSpaceLite = "namespace lite";

constexpr auto kAlignedString = "  ";

constexpr auto kDebugUtils = "debug_utils.h";

constexpr auto kThreadWrapper = "thread_wrapper.h";

constexpr auto kExternCpp =
  "#ifdef __cplusplus\n"
  "extern \"C\" {\n"
  "#endif\n";

constexpr auto kEndExternCpp =
  "#ifdef __cplusplus\n"
  "}\n"
  "#endif\n";
}  // namespace luojianet_ms::lite::micro
#endif  // LUOJIANET_MS_LITE_MICRO_CODER_GENERATOR_COMPONENT_H_
