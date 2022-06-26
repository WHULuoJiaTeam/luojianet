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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_ZIP_OPERATION_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_ZIP_OPERATION_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <memory>

#include "utils/hash_map.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"

namespace luojianet_ms {
// namespace to support composite operators definition
namespace prim {
using AbstractBasePtr = abstract::AbstractBasePtr;
using AbstractBasePtrList = abstract::AbstractBasePtrList;
using AbstractTuplePtr = abstract::AbstractTuplePtr;

class ZipOperation : public MetaFuncGraph {
 public:
  explicit ZipOperation(const std::string &name) : MetaFuncGraph(name) {}
  ~ZipOperation() override = default;
  MS_DECLARE_PARENT(ZipOperation, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ZipOperation &op) {
    os << op.name_;
    return os;
  }
  friend bool operator==(const ZipOperation &lhs, const ZipOperation &rhs) { return lhs.name_ == rhs.name_; }
};
using ZipOperationPtr = std::shared_ptr<ZipOperation>;
}  // namespace prim
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_ZIP_OPERATION_H_
