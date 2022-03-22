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

#include "register/op_tiling_registry.h"
#include "framework/common/debug/ge_log.h"

namespace optiling {
size_t ByteBufferGetAll(ByteBuffer &buf, char *dest, size_t dest_len) {
  size_t nread = 0;
  size_t rn = 0;
  do {
    rn = buf.readsome(dest + nread, dest_len - nread);
    nread += rn;
  } while (rn > 0 && dest_len > nread);

  return nread;
}

ByteBuffer &ByteBufferPut(ByteBuffer &buf, const uint8_t *data, size_t data_len) {
  buf.write(reinterpret_cast<const char *>(data), data_len);
  buf.flush();
  return buf;
}

CompileInfoCache::CompileInfoCache() {}
CompileInfoCache::~CompileInfoCache() {}

CompileInfoCache& CompileInfoCache::Instance() {
  static CompileInfoCache compile_info_cache_instance;
  return compile_info_cache_instance;
}

bool CompileInfoCache::HasCompileInfo(const std::string &key) {
  return this->compile_info_map_.find(key) != this->compile_info_map_.end();
}

void* CompileInfoCache::GetCompileInfo(const std::string &key) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  auto iter = this->compile_info_map_.find(key);
  if (iter == this->compile_info_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

void CompileInfoCache::SetCompileInfo(const std::string &key, void *value) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  this->compile_info_map_.emplace(key, value);
}

std::unordered_map<std::string, OpTilingFunc> &OpTilingRegistryInterf::RegisteredOpInterf() {
  static std::unordered_map<std::string, OpTilingFunc> interf;
  return interf;
}

OpTilingRegistryInterf::OpTilingRegistryInterf(std::string op_type, OpTilingFunc func) {
  auto &interf = RegisteredOpInterf();
  interf.emplace(op_type, func);
  GELOGI("Register tiling function: op_type:%s, funcPointer:%p, registered count:%zu", op_type.c_str(),
         func.target<OpTilingFuncPtr>(), interf.size());
}

std::unordered_map<std::string, OpTilingFuncV2> &OpTilingRegistryInterf_V2::RegisteredOpInterf() {
  static std::unordered_map<std::string, OpTilingFuncV2> interf;
  GELOGI("Generate interf by new method, registered count: %zu", interf.size());
  return interf;
}

OpTilingRegistryInterf_V2::OpTilingRegistryInterf_V2(const std::string &op_type, OpTilingFuncV2 func) {
  auto &interf = RegisteredOpInterf();
  interf.emplace(op_type, std::move(func));
  GELOGI("Register tiling function by new method: op_type:%s, registered count:%zu", op_type.c_str(), interf.size());
}
namespace utils {
}  // namespace utils

OpTilingFuncInfo::OpTilingFuncInfo(const std::string &op_type)
  : op_type_(op_type), tiling_func_(nullptr), tiling_func_v2_(nullptr), tiling_func_v3_(nullptr),
    parse_func_v3_(nullptr) {}

bool OpTilingFuncInfo::IsFunctionV3() {
  return this->tiling_func_v3_ != nullptr && this->parse_func_v3_ != nullptr;
}
bool OpTilingFuncInfo::IsFunctionV2() {
  return this->tiling_func_v2_ != nullptr;
}
bool OpTilingFuncInfo::IsFunctionV1() {
  return this->tiling_func_ != nullptr;
}
void OpTilingFuncInfo::SetOpTilingFunc(OpTilingFunc &tiling_func) {
  this->tiling_func_ = tiling_func;
}
void OpTilingFuncInfo::SetOpTilingFuncV2(OpTilingFuncV2 &tiling_func) {
  this->tiling_func_v2_ = tiling_func;
}
void OpTilingFuncInfo::SetOpTilingFuncV3(OpTilingFuncV3 &tiling_func, OpParseFuncV3 &parse_func) {
  this->tiling_func_v3_ = tiling_func;
  this->parse_func_v3_ = parse_func;
}
const OpTilingFunc& OpTilingFuncInfo::GetOpTilingFunc() {
  return this->tiling_func_;
}
const OpTilingFuncV2& OpTilingFuncInfo::GetOpTilingFuncV2() {
  return this->tiling_func_v2_;
}
const OpTilingFuncV3& OpTilingFuncInfo::GetOpTilingFuncV3() {
  return this->tiling_func_v3_;
}
const OpParseFuncV3& OpTilingFuncInfo::GetOpParseFuncV3() {
  return this->parse_func_v3_;
}

std::unordered_map<std::string, OpTilingFuncInfo> &OpTilingFuncRegistry::RegisteredOpFuncInfo() {
  static std::unordered_map<std::string, OpTilingFuncInfo> op_func_map;
  return op_func_map;
}

OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type, OpTilingFunc tiling_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFunc(tiling_func);
    op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFunc(tiling_func);
  }
  GELOGI("Register op tiling function V1 for op_type:%s", op_type.c_str());
}
OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type, OpTilingFuncV2 tiling_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFuncV2(tiling_func);
    op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFuncV2(tiling_func);
  }
  GELOGI("Register op tiling function V2 for op_type:%s", op_type.c_str());
}

OpTilingFuncRegistry::OpTilingFuncRegistry(const std::string &op_type,
                                           OpTilingFuncV3 tiling_func, OpParseFuncV3 parse_func) {
  auto &op_func_map = RegisteredOpFuncInfo();
  auto iter = op_func_map.find(op_type);
  if (iter == op_func_map.end()) {
    OpTilingFuncInfo op_func_info(op_type);
    op_func_info.SetOpTilingFuncV3(tiling_func, parse_func);
    op_func_map.emplace(op_type, op_func_info);
  } else {
    iter->second.SetOpTilingFuncV3(tiling_func, parse_func);
  }
  GELOGI("Register op tiling and parse function V3 for op_type:%s", op_type.c_str());
}
}  // namespace optiling
