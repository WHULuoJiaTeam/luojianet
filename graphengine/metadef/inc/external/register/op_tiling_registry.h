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

#ifndef INC_EXTERNAL_REGISTER_OP_TILING_REGISTRY_H_
#define INC_EXTERNAL_REGISTER_OP_TILING_REGISTRY_H_

#include <functional>
#include <unordered_map>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include "external/graph/operator.h"
#include "external/register/register_error_codes.h"
#include "external/register/register_types.h"
#include "external/register/op_tiling_info.h"

#define REGISTER_OP_TILING(optype, opfunc) REGISTER_OP_TILING_UNIQ_HELPER(optype, opfunc, __COUNTER__)

#define REGISTER_OP_TILING_UNIQ_HELPER(optype, opfunc, counter) REGISTER_OP_TILING_UNIQ(optype, opfunc, counter)

#define REGISTER_OP_TILING_UNIQ(optype, opfunc, counter)                                                               \
  static optiling::OpTilingFuncRegistry g_##optype##TilingRegistryInterfV1##counter(#optype, opfunc)

#define REGISTER_OP_TILING_V2(optype, opfunc) REGISTER_OP_TILING_UNIQ_HELPER_V2(optype, opfunc, __COUNTER__)

#define REGISTER_OP_TILING_UNIQ_HELPER_V2(optype, opfunc, counter) REGISTER_OP_TILING_UNIQ_V2(optype, opfunc, counter)

#define REGISTER_OP_TILING_UNIQ_V2(optype, opfunc, counter)                                                            \
  static optiling::OpTilingFuncRegistry g_##optype##TilingRegistryInterfV2##counter(#optype, opfunc)

#define REGISTER_OP_TILING_V3(optype, tilingfunc, parsefunc)                                                           \
  REGISTER_OP_TILING_UNIQ_HELPER_V3(optype, tilingfunc, parsefunc, __COUNTER__)

#define REGISTER_OP_TILING_UNIQ_HELPER_V3(optype, tilingfunc, parsefunc, counter)                                      \
  REGISTER_OP_TILING_UNIQ_V3(optype, tilingfunc, parsefunc, counter)

#define REGISTER_OP_TILING_UNIQ_V3(optype, tilingfunc, parsefunc, counter)                                             \
  static optiling::OpTilingFuncRegistry g_##optype##TilingRegistryInterfV3##counter(#optype, tilingfunc, parsefunc)

using Status = domi::Status;
namespace optiling {
template<class T>
ByteBuffer &ByteBufferPut(ByteBuffer &buf, const T &value) {
  buf.write(reinterpret_cast<const char *>(&value), sizeof(value));
  buf.flush();
  return buf;
}

template<class T>
ByteBuffer &ByteBufferGet(ByteBuffer &buf, T &value) {
  buf.read(reinterpret_cast<char *>(&value), sizeof(value));
  return buf;
}

size_t ByteBufferGetAll(ByteBuffer &buf, char *dest, size_t dest_len);
ByteBuffer &ByteBufferPut(ByteBuffer &buf, const uint8_t *data, size_t dest_len);

class CompileInfoCache {
public:
  CompileInfoCache(const CompileInfoCache &) = delete;
  CompileInfoCache &operator=(const CompileInfoCache &) = delete;
  static CompileInfoCache& Instance();
  bool HasCompileInfo(const std::string &key);
  void* GetCompileInfo(const std::string &key);
  void SetCompileInfo(const std::string &key, void* value);

private:
  CompileInfoCache();
  ~CompileInfoCache();
  mutable std::mutex compile_info_mutex_;
  std::unordered_map<std::string, void *> compile_info_map_;
};

using OpTilingFunc = std::function<bool(const TeOpParas &, const OpCompileInfo &, OpRunInfo &)>;
using OpTilingFuncPtr = std::shared_ptr<OpTilingFunc>;
class FMK_FUNC_HOST_VISIBILITY OpTilingRegistryInterf {
 public:
  OpTilingRegistryInterf(std::string op_type, OpTilingFunc func);
  ~OpTilingRegistryInterf() = default;
  static std::unordered_map<std::string, OpTilingFunc> &RegisteredOpInterf();
};

using OpRunInfoV2 = utils::OpRunInfo;
using OpCompileInfoV2 = utils::OpCompileInfo;
using OpTilingFuncV2 = std::function<bool(const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &)>;
using OpTilingFuncV2Ptr = std::shared_ptr<OpTilingFuncV2>;
class FMK_FUNC_HOST_VISIBILITY OpTilingRegistryInterf_V2 {
public:
  OpTilingRegistryInterf_V2(const std::string &op_type, OpTilingFuncV2 func);
  ~OpTilingRegistryInterf_V2() = default;
  static std::unordered_map<std::string, OpTilingFuncV2> &RegisteredOpInterf();
};
namespace utils {
}  // namespace utils

using OpTilingFuncV3 = std::function<bool(const ge::Operator &, const void *, OpRunInfoV2 &)>;
using OpParseFuncV3 = std::function<void*(const ge::Operator &, const ge::AscendString &)>;

class OpTilingFuncInfo {
public:
  explicit OpTilingFuncInfo(const std::string &op_type);
  OpTilingFuncInfo() = default;
  ~OpTilingFuncInfo() = default;

  bool IsFunctionV3();
  bool IsFunctionV2();
  bool IsFunctionV1();
  void SetOpTilingFunc(OpTilingFunc &tiling_func);
  void SetOpTilingFuncV2(OpTilingFuncV2 &tiling_func);
  void SetOpTilingFuncV3(OpTilingFuncV3 &tiling_func, OpParseFuncV3 &parse_func);
  const OpTilingFunc& GetOpTilingFunc();
  const OpTilingFuncV2& GetOpTilingFuncV2();
  const OpTilingFuncV3& GetOpTilingFuncV3();
  const OpParseFuncV3& GetOpParseFuncV3();
  const std::string& GetOpType() const {
    return op_type_;
  }

private:
  std::string op_type_;
  OpTilingFunc tiling_func_;
  OpTilingFuncV2 tiling_func_v2_;
  OpTilingFuncV3 tiling_func_v3_;
  OpParseFuncV3 parse_func_v3_;
};

class FMK_FUNC_HOST_VISIBILITY OpTilingFuncRegistry {
public:
  OpTilingFuncRegistry(const std::string &op_type, OpTilingFunc tiling_func);
  OpTilingFuncRegistry(const std::string &op_type, OpTilingFuncV2 tiling_func);
  OpTilingFuncRegistry(const std::string &op_type, OpTilingFuncV3 tiling_func, OpParseFuncV3 parse_func);
  ~OpTilingFuncRegistry() = default;
  static std::unordered_map<std::string, OpTilingFuncInfo> &RegisteredOpFuncInfo();
};

}  // namespace optiling
#endif  // INC_EXTERNAL_REGISTER_OP_TILING_REGISTRY_H_
