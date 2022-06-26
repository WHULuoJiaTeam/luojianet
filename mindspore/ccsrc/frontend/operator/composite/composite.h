/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <set>
#include <memory>
#include "utils/hash_map.h"
#include "frontend/operator/composite/zip_operation.h"
#include "frontend/operator/composite/list_append_operation.h"
#include "frontend/operator/composite/list_insert_operation.h"
#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/operator/composite/multitype_funcgraph.h"
#include "pipeline/jit/static_analysis/static_analysis.h"
#include "utils/misc.h"
#include "utils/any.h"
#include "ir/dtype.h"
#include "ir/meta_func_graph.h"
#include "base/core_ops.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
using AbstractSlicePtr = abstract::AbstractSlicePtr;
using AbstractScalarPtr = abstract::AbstractScalarPtr;
using AbstractTensorPtr = abstract::AbstractTensorPtr;
using ElemwiseMap = mindspore::HashMap<std::string, PrimitivePtr>;
using ArgsPairList = std::vector<std::pair<AnfNodePtr, TypePtr>>;

class HyperMap : public MetaFuncGraph {
 public:
  explicit HyperMap(bool reverse = false, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr);
  HyperMap(const HyperMap &h);
  void Init();
  HyperMap &operator=(const HyperMap &h) noexcept {
    if (this != &h) {
      fn_leaf_ = h.fn_leaf_;
      reverse_ = h.reverse_;
      nonleaf_ = h.nonleaf_;
      if (fn_leaf_) {
        name_ = "hyper_map[" + fn_leaf_->name() + "]";
      }
    }
    return *this;
  }
  ~HyperMap() override = default;
  MS_DECLARE_PARENT(HyperMap, MetaFuncGraph)

  abstract::AbstractBasePtrList NormalizeArgs(const abstract::AbstractBasePtrList &args_spec_list) const override;
  FuncGraphPtr GenerateFromTypes(const TypePtrList &args_spec_list) override;
  MetaFuncGraphPtr GetFnLeaf() { return fn_leaf_; }

 private:
  AnfNodePtr FullMake(const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map);
  AnfNodePtr FullMake(const std::shared_ptr<List> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map);
  AnfNodePtr FullMake(const std::shared_ptr<Tuple> &type, const FuncGraphPtr &func_graph, const AnfNodePtr &fn_arg,
                      const ArgsPairList &arg_map);
  AnfNodePtr Make(const FuncGraphPtr &graph, const AnfNodePtr &fn_arg, const ArgsPairList &arg_map);
  std::pair<std::string, std::string> GetHyperMapInputIndex(size_t num);

  MultitypeFuncGraphPtr fn_leaf_;
  bool reverse_;
  std::set<TypeId> nonleaf_;
};
using HyperMapPtr = std::shared_ptr<HyperMap>;

class HyperMapPy : public HyperMap {
 public:
  explicit HyperMapPy(bool reverse = false, const std::shared_ptr<MultitypeFuncGraph> &fn_leaf = nullptr)
      : HyperMap(reverse, fn_leaf) {}
  ~HyperMapPy() override = default;
  MS_DECLARE_PARENT(HyperMapPy, HyperMap)
};
using HyperMapPyPtr = std::shared_ptr<HyperMapPy>;

extern ValuePtr kCompositeHyperMap;

enum TailType { kGradAll, kGradFirst, kGradByPosition, kNotGrad };

class Tail : public MetaFuncGraph {
 public:
  explicit Tail(const std::string &name, TailType tail_type = kNotGrad)
      : MetaFuncGraph(name), tail_type_(tail_type), enable_tuple_grad_(false) {}
  ~Tail() override = default;
  MS_DECLARE_PARENT(Tail, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  FuncGraphPtr GenerateSequenceFuncGraph(const abstract::AbstractSequencePtr &sequeue,
                                         const abstract::AbstractSequencePtr &pos = nullptr) const;

  friend bool operator==(const Tail &lhs, const Tail &rhs) { return lhs.name_ == rhs.name_; }
  void set_enable_tuple_grad(bool enable_tuple_grad) { enable_tuple_grad_ = enable_tuple_grad; }

 private:
  TailType tail_type_;
  bool enable_tuple_grad_;
};
using TailPtr = std::shared_ptr<Tail>;

class MakeTupleGradient : public MetaFuncGraph {
 public:
  explicit MakeTupleGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeTupleGradient() override = default;
  MS_DECLARE_PARENT(MakeTupleGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MakeTupleGradient &lhs, const MakeTupleGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeTupleGradientPtr = std::shared_ptr<MakeTupleGradient>;

class MakeListGradient : public MetaFuncGraph {
 public:
  explicit MakeListGradient(const std::string &name) : MetaFuncGraph(name) {}
  ~MakeListGradient() override = default;
  MS_DECLARE_PARENT(MakeListGradient, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const MakeListGradient &lhs, const MakeListGradient &rhs) { return lhs.name_ == rhs.name_; }
};
using MakeListGradientPtr = std::shared_ptr<MakeListGradient>;

class GradOperation : public MetaFuncGraph {
 public:
  explicit GradOperation(const std::string &name, bool get_all = false, bool get_by_list = false,
                         bool sens_param = false, bool get_by_position = false);
  ~GradOperation() override = default;
  MS_DECLARE_PARENT(GradOperation, MetaFuncGraph)

  FuncGraphPtr GetGrad(const AnfNodePtr &j, const AnfNodePtr &weights, const AnfNodePtr &position,
                       const std::vector<AnfNodePtr> &forward_graph_params, bool enable_tuple_grad,
                       const std::vector<AnfNodePtr> &weight_args = {});

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;

  void set_grad_position(const std::string &grad_position) { grad_position_ = grad_position; }
  bool sens_param() const { return sens_param_; }
  bool get_all_;
  bool get_by_list_;
  bool sens_param_;
  bool get_by_position_;
  std::string grad_position_;

 private:
  void GradByParameter(const FuncGraphPtr &k_child, const AnfNodePtr &f_app, const AnfNodePtr &bprop,
                       const AnfNodePtr &weights, const AnfNodePtr &position, bool enable_tuple_grad);
};
using GradOperationPtr = std::shared_ptr<GradOperation>;

class TaylorOperation : public MetaFuncGraph {
 public:
  explicit TaylorOperation(const std::string &name);
  ~TaylorOperation() override = default;
  MS_DECLARE_PARENT(TaylorOperation, MetaFuncGraph);
  FuncGraphPtr GetTaylorGrad(const AnfNodePtr &k, const std::vector<AnfNodePtr> &forward_graph_params);

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
using TaylorOperationPtr = std::shared_ptr<TaylorOperation>;

class ListMap {
 public:
  explicit ListMap(const std::string &name) : name_(name) { cache_.clear(); }
  ~ListMap() = default;
  void MakeCond(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &gnext_ptr, const FuncGraphPtr &graph_ptr);
  void MakeNext(const std::vector<AnfNodePtr> &lists, const FuncGraphPtr &gcond_ptr, const FuncGraphPtr &graph_ptr);
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list);

 private:
  std::string name_;
  std::map<std::vector<AnyPtr>, FuncGraphPtr> cache_;
};

class TupleAdd : public MetaFuncGraph {
 public:
  explicit TupleAdd(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleAdd() override = default;
  MS_DECLARE_PARENT(TupleAdd, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TupleAdd &lhs, const TupleAdd &rhs) { return lhs.name_ == rhs.name_; }
};
using TupleAddPtr = std::shared_ptr<TupleAdd>;

class SequenceSlice : public MetaFuncGraph {
 public:
  explicit SequenceSlice(const std::string &name, const PrimitivePtr &prim, const PrimitivePtr &get_item)
      : MetaFuncGraph(name), prim_(prim), get_item_(get_item) {}
  ~SequenceSlice() override = default;
  MS_DECLARE_PARENT(SequenceSlice, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) final;
  friend bool operator==(const SequenceSlice &lhs, const SequenceSlice &rhs) { return lhs.name_ == rhs.name_; }
  virtual std::pair<abstract::AbstractSequencePtr, abstract::AbstractSlicePtr> CheckArgs(
    const AbstractBasePtrList &args_spec_list) = 0;

 private:
  PrimitivePtr prim_;
  PrimitivePtr get_item_;
};

class TupleSlice : public SequenceSlice {
 public:
  explicit TupleSlice(const std::string &name) : SequenceSlice(name, prim::kPrimMakeTuple, prim::kPrimTupleGetItem) {}
  ~TupleSlice() override = default;
  MS_DECLARE_PARENT(TupleSlice, SequenceSlice)
  std::pair<abstract::AbstractSequencePtr, abstract::AbstractSlicePtr> CheckArgs(
    const AbstractBasePtrList &args_spec_list) override;
};
using TupleSlicePtr = std::shared_ptr<TupleSlice>;

class ListSlice : public SequenceSlice {
 public:
  explicit ListSlice(const std::string &name) : SequenceSlice(name, prim::kPrimMakeList, prim::kPrimListGetItem) {}
  ~ListSlice() override = default;
  MS_DECLARE_PARENT(ListSlice, SequenceSlice)
  std::pair<abstract::AbstractSequencePtr, abstract::AbstractSlicePtr> CheckArgs(
    const AbstractBasePtrList &args_spec_list) override;
};

class TupleGetItemTensor : public MetaFuncGraph {
 public:
  explicit TupleGetItemTensor(const std::string &name) : MetaFuncGraph(name) {}
  ~TupleGetItemTensor() override = default;
  MS_DECLARE_PARENT(TupleGetItemTensor, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
  friend bool operator==(const TupleGetItemTensor &lhs, const TupleGetItemTensor &rhs) {
    return lhs.name_ == rhs.name_;
  }
};
using TupleGetItemTensorPtr = std::shared_ptr<TupleGetItemTensor>;

class Shard : public MetaFuncGraph {
 public:
  explicit Shard(const string &name) : MetaFuncGraph(name) {
    signatures_ =
      // def shard(func:read, weight_list:read, in_axes:read, out_axes:read, device:read, level:read):
      std::vector<Signature>({{"func", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"in_axes", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"out_axes", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"device", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault},
                              {"level", SignatureEnumRW::kRWRead, SignatureEnumKind::kKindDefault}});
  }
  ~Shard() override = default;
  MS_DECLARE_PARENT(Shard, MetaFuncGraph)

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};

class VmapOperation : public MetaFuncGraph {
 public:
  explicit VmapOperation(const std::string &name);
  ~VmapOperation() override = default;
  MS_DECLARE_PARENT(VmapOperation, MetaFuncGraph)

  FuncGraphPtr GetVmap(const AnfNodePtr &k, const std::vector<AnfNodePtr> &forward_graph_params);

  FuncGraphPtr GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) override;
};
using VmapOperationPtr = std::shared_ptr<VmapOperation>;
}  // namespace prim
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPERATOR_COMPOSITE_H_
