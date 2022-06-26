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
#ifndef LUOJIANET_MS_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#define LUOJIANET_MS_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <unordered_map>
#include "utils/hash_map.h"
#include "utils/ms_utils.h"
#include "ir/anf.h"

namespace luojianet_ms::pynative {
struct AbsCacheKey {
  std::string prim_name_;
  size_t prim_hash_value_;
  luojianet_ms::HashMap<std::string, ValuePtr> prim_attrs_;
};

struct AbsCacheKeyHasher {
  size_t operator()(const AbsCacheKey &key) const { return key.prim_hash_value_; }
};

struct AbsCacheKeyEqual {
  bool operator()(const AbsCacheKey &lk, const AbsCacheKey &rk) const {
    if (lk.prim_name_ != rk.prim_name_) {
      return false;
    }
    return common::IsAttrsEqual(lk.prim_attrs_, rk.prim_attrs_);
  }
};

struct PrimAbsInfo {
  abstract::AbstractBasePtr abs;
  bool is_dynamic_shape = false;
  luojianet_ms::HashMap<std::string, ValuePtr> attrs;
};
using AbstractListMap = std::unordered_map<abstract::AbstractBasePtrList, PrimAbsInfo,
                                           abstract::AbstractBasePtrListHasher, abstract::AbstractBasePtrListEqual>;
using PrimAbsCache = std::unordered_map<AbsCacheKey, AbstractListMap, AbsCacheKeyHasher, AbsCacheKeyEqual>;

// Used for id
struct PyObjectHasher {
  size_t operator()(const py::handle &key) const { return py::hash(key); }
};

struct PyObjectEqual {
  bool operator()(const py::handle &p1, const py::handle &p2) const { return p1 == p2; }
};
using PyObjectIdCache = std::unordered_map<py::handle, std::string, PyObjectHasher, PyObjectEqual>;

struct PrimSignature {
  bool has_dtype_sig;
  std::vector<SignatureEnumDType> dtypes;
  luojianet_ms::HashMap<SignatureEnumDType, std::vector<size_t>> type_indexes;
};
using ImplicitCastCache = luojianet_ms::HashMap<std::string, PrimSignature>;
}  // namespace luojianet_ms::pynative
#endif  // LUOJIANET_MS_CCSRC_PIPELINE_PYNATIVE_PYNATIVE_ABS_CACHE_H
