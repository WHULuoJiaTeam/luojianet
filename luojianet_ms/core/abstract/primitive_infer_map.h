/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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
#ifndef LUOJIANET_MS_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
#define LUOJIANET_MS_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_

#include <vector>
#include <set>
#include <memory>
#include "utils/hash_map.h"
#include "ir/primitive.h"
#include "ops/primitive_c.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"

namespace luojianet_ms {
namespace abstract {
using InferShapeImpl = AbstractBasePtr (*)(const abstract::AnalysisEnginePtr &, const PrimitivePtr &,
                                           const AbstractBasePtrList &);
using InferValueImpl = ValuePtr (*)(const PrimitivePtr &, const AbstractBasePtrList &);

struct StandardPrimitiveImplReg {
  InferShapeImpl infer_shape_impl_;  // infer shape and type for ops
  InferValueImpl infer_value_impl_;  // infer value for ops
  // in_white_list_ is true means this primitive can be executed by vm backend
  // else will be optimized by frontend
  bool in_white_list_;
};

using PrimitiveEvalImplMap =
  luojianet_ms::HashMap<PrimitivePtr, StandardPrimitiveImplReg, PrimitiveHasher, PrimitiveEqual>;

MS_CORE_API PrimitiveEvalImplMap &GetPrimitiveToEvalImplMap();

MS_CORE_API PrimitiveEvalImplMap &GetPrimitiveToBackendEvalImplMap();

MS_CORE_API StandardPrimitiveImplReg GetPrimitiveInferImpl(const PrimitivePtr &primitive);

MS_CORE_API std::set<int64_t> GetDependsFormMap(const CNodePtr &cnode);

MS_CORE_API void RegisterStandardPrimitiveImpl(const PrimitivePtr &primitive, const StandardPrimitiveImplReg &impl_reg);

class RegisterStandardPrimitiveEvalHelper {
 public:
  RegisterStandardPrimitiveEvalHelper(const PrimitivePtr &primitive, const InferShapeImpl &infer_impl,
                                      const InferValueImpl &infer_value_impl, const bool is_white_list = true) {
    const StandardPrimitiveImplReg impl_reg{infer_impl, infer_value_impl, is_white_list};
    RegisterStandardPrimitiveImpl(primitive, impl_reg);
  }
  ~RegisterStandardPrimitiveEvalHelper() = default;
};

#define REGISTER_PRIMITIVE_EVAL_IMPL(name, primitive, infer_impl, infer_value_impl, is_white_list)         \
  static auto helper_##name =                                                                              \
    abstract::RegisterStandardPrimitiveEvalHelper(primitive, infer_impl, infer_value_impl, is_white_list); \
  std::shared_ptr<ops::PrimitiveC> GetDefaultPrimC##name() {                                               \
    name out;                                                                                              \
    return std::dynamic_pointer_cast<ops::PrimitiveC>(out.impl());                                         \
  }                                                                                                        \
  ops::OpPrimCRegisterHelper primc_gen_##name(#name, GetDefaultPrimC##name);
}  // namespace abstract
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CORE_ABSTRACT_PRIMITIVE_INFER_MAP_H_
