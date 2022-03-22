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

#ifndef EXTERNAL_REGISTER_SCOPE_SCOPE_FUSION_PASS_REGISTER_H_
#define EXTERNAL_REGISTER_SCOPE_SCOPE_FUSION_PASS_REGISTER_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "ge/ge_api_error_codes.h"
#include "register/register_error_codes.h"
#include "register/register_types.h"
#include "graph/operator.h"

#define CHECK_INNER_NODE_CONDITION(cond, fusion_rlt)  \
  do {                                                \
    if (!(cond)) {                                    \
      if ((fusion_rlt) != nullptr) {                  \
        (fusion_rlt)->SetType(ge::kScopeInvalidType); \
      }                                               \
      return;                                         \
    }                                                 \
  } while (0)

namespace domi {
class TensorFlowModelParser;
}  // namespace domi
namespace ge {
const int32_t kFusionDisableIndex = 99999;
const char *const kScopeToMultiNodes = "ScopeToMultiNodes";
const char *const kScopeInvalidType = "ScopeInvalidType";
const char *const kInputFromFusionScope = "InputFromFusionScope";
const char *const kOutputToFusionScope = "OutputToFusionScope";
class ScopePattern;
using ScopeFusionPatterns = std::vector<std::vector<ScopePattern *>>;

class ScopePassManager;

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY Scope {
 public:
  Scope();
  ATTRIBUTED_DEPRECATED(Status Init(const char *, const char *, Scope *))
  Status Init(const std::string &name, const std::string &sub_type = "", Scope *father_scope = nullptr);
  Status Init(const char *name, const char *sub_type, Scope *father_scope = nullptr);
  ~Scope();
  ATTRIBUTED_DEPRECATED(Status Name(AscendString &) const)
  const std::string &Name() const;
  Status Name(AscendString &name) const;
  ATTRIBUTED_DEPRECATED(Status SubType(AscendString &) const)
  const std::string &SubType() const;
  Status SubType(AscendString &sub_type) const;
  ATTRIBUTED_DEPRECATED(Status AllNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &) const)
  const std::unordered_map<std::string, ge::OperatorPtr> &AllNodesMap() const;
  Status AllNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &node_map) const;
  ATTRIBUTED_DEPRECATED(Scope *GetSubScope(const char *scope_name) const)
  Scope *GetSubScope(const std::string &scope_name) const;
  Scope *GetSubScope(const char *scope_name) const;
  ATTRIBUTED_DEPRECATED(Status LastName(AscendString &) const)
  const std::string LastName() const;
  Status LastName(AscendString &name) const;
  const std::vector<Scope *> &GetAllSubScopes() const;
  const Scope *GetFatherScope() const;

 private:
  class ScopeImpl;
  std::unique_ptr<ScopeImpl> impl_;
  friend class ScopeBasePass;
  friend class ScopeTree;
  friend class NodeOpTypeFeature;
  friend class NodeAttrFeature;
  friend class ScopeFeature;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY FusionScopesResult {
 public:
  FusionScopesResult();
  Status Init();
  ~FusionScopesResult();
  ATTRIBUTED_DEPRECATED(void SetName(const char *))
  void SetName(const std::string &name);
  void SetName(const char *name);
  ATTRIBUTED_DEPRECATED(void SetType(const char *))
  void SetType(const std::string &type);
  void SetType(const char *type);
  ATTRIBUTED_DEPRECATED(void SetDescription(const char *))
  void SetDescription(const std::string &description);
  void SetDescription(const char *description);
  ATTRIBUTED_DEPRECATED(const Status Name(AscendString &) const)
  const std::string &Name() const;
  const Status Name(AscendString &name) const;
  const std::vector<ge::OperatorPtr> &Nodes() const;
  ATTRIBUTED_DEPRECATED(void InsertInputs(const char *, const std::vector<int32_t> &))
  void InsertInputs(const std::string &inner_op_name, const std::vector<int32_t> &index_map);
  void InsertInputs(const char *inner_op_name, const std::vector<int32_t> &index_map);
  ATTRIBUTED_DEPRECATED(void InsertOutputs(const char *, const std::vector<int32_t> &))
  void InsertOutputs(const std::string &inner_op_name, const std::vector<int32_t> &index_map);
  void InsertOutputs(const char *inner_op_name, const std::vector<int32_t> &index_map);

  class InnerNodeInfo {
   public:
    ATTRIBUTED_DEPRECATED(InnerNodeInfo(const char *))
    explicit InnerNodeInfo(const std::string &fusion_node_name);
    explicit InnerNodeInfo(const char *fusion_node_name);
    ATTRIBUTED_DEPRECATED(InnerNodeInfo(const char *, const char *, const char *))
    InnerNodeInfo(const std::string &fusion_node_name, const std::string &name, const std::string &type);
    InnerNodeInfo(const char *fusion_node_name, const char *name, const char *type);
    InnerNodeInfo(InnerNodeInfo &&other) noexcept;
    InnerNodeInfo &operator=(InnerNodeInfo &&other) noexcept;
    InnerNodeInfo(const InnerNodeInfo &) = delete;
    InnerNodeInfo &operator=(const InnerNodeInfo &) = delete;
    ~InnerNodeInfo();
    ATTRIBUTED_DEPRECATED(InnerNodeInfo &SetName(const char *))
    InnerNodeInfo &SetName(const std::string &name);
    InnerNodeInfo &SetName(const char *name);
    ATTRIBUTED_DEPRECATED(InnerNodeInfo &SetType(const char *))
    InnerNodeInfo &SetType(const std::string &type);
    InnerNodeInfo &SetType(const char *type);
    ATTRIBUTED_DEPRECATED(InnerNodeInfo &InsertInput(const char *, int32_t))
    InnerNodeInfo &InsertInput(const std::string &input_node, int32_t peer_out_idx);
    InnerNodeInfo &InsertInput(const char *input_node, int32_t peer_out_idx);
    ATTRIBUTED_DEPRECATED(InnerNodeInfo &InsertOutput(const char *, int32_t))
    InnerNodeInfo &InsertOutput(const std::string &output_node, int32_t peer_in_idx);
    InnerNodeInfo &InsertOutput(const char *output_node, int32_t peer_in_idx);
    ge::graphStatus BuildInnerNode();
    ATTRIBUTED_DEPRECATED(ge::graphStatus SetInputFormat(const char *, const char *))
    ge::graphStatus SetInputFormat(const std::string &input_name, const std::string &format);
    ge::graphStatus SetInputFormat(const char *input_name, const char *format);
    ATTRIBUTED_DEPRECATED(ge::graphStatus SetOutputFormat(const char *, const char *))
    ge::graphStatus SetOutputFormat(const std::string &output_name, const std::string &format);
    ge::graphStatus SetOutputFormat(const char *output_name, const char *format);
    ATTRIBUTED_DEPRECATED(ge::graphStatus SetDynamicInputFormat(const char *, uint32_t index, const char *))
    ge::graphStatus SetDynamicInputFormat(const std::string &input_name, uint32_t index, const std::string &format);
    ge::graphStatus SetDynamicInputFormat(const char *input_name, uint32_t index, const char *format);
    ATTRIBUTED_DEPRECATED(ge::graphStatus SetDynamicOutputFormat(const char *, uint32_t, const char *))
    ge::graphStatus SetDynamicOutputFormat(const std::string &output_name, uint32_t index, const std::string &format);
    ge::graphStatus SetDynamicOutputFormat(const char *output_name, uint32_t index, const char *format);
    ge::Operator *MutableOperator();
    ATTRIBUTED_DEPRECATED(ge::graphStatus GetName(AscendString &) const)
    std::string GetName() const;
    ge::graphStatus GetName(AscendString &name) const;
    ATTRIBUTED_DEPRECATED(ge::graphStatus GetType(AscendString &) const)
    std::string GetType() const;
    ge::graphStatus GetType(AscendString &type) const;
    ATTRIBUTED_DEPRECATED(ge::graphStatus GetInputs(std::vector<std::pair<AscendString, int32_t>> &) const)
    std::vector<std::pair<std::string, int32_t>> GetInputs() const;
    ge::graphStatus GetInputs(std::vector<std::pair<AscendString, int32_t>> &inputs) const;
    ATTRIBUTED_DEPRECATED(ge::graphStatus GetOutputs(std::vector<std::pair<AscendString, int32_t>> &) const)
    std::vector<std::pair<std::string, int32_t>> GetOutputs() const;
    ge::graphStatus GetOutputs(std::vector<std::pair<AscendString, int32_t>> &outputs) const;
   private:
    class InnerNodeInfoImpl;
    std::unique_ptr<InnerNodeInfoImpl> impl_;
  };
  ATTRIBUTED_DEPRECATED(InnerNodeInfo *AddInnerNode(const char *, const char *))
  InnerNodeInfo *AddInnerNode(const std::string &name, const std::string &type);
  InnerNodeInfo *AddInnerNode(const char *name, const char *type);
  InnerNodeInfo *MutableRecentInnerNode();
  InnerNodeInfo *MutableInnerNode(uint32_t index);
  ge::graphStatus CheckInnerNodesInfo();

 private:
  class FusionScopesResultImpl;
  std::unique_ptr<FusionScopesResultImpl> impl_;
  friend class ScopeGraph;
  friend class ScopeBasePass;
  friend class TensorFlowModelParser;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeTree {
 public:
  ScopeTree();
  Status Init();
  ScopeTree(const ScopeTree &scopetree) = delete;
  ScopeTree &operator=(const ScopeTree &scopetree) = delete;
  ~ScopeTree();

  const std::vector<Scope *> &GetAllScopes() const;

 private:
  class ScopeTreeImpl;
  std::unique_ptr<ScopeTreeImpl> impl_;
  friend class ScopeGraph;
  friend class ScopeBasePass;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeGraph {
 public:
  ScopeGraph();
  Status Init();
  ScopeGraph(const ScopeGraph &scope_graph) = delete;
  ScopeGraph &operator=(const ScopeGraph &scope_graph) = delete;
  ~ScopeGraph();

  const ScopeTree *GetScopeTree() const;
  ATTRIBUTED_DEPRECATED(Status GetNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &) const)
  const std::unordered_map<std::string, ge::OperatorPtr> &GetNodesMap() const;
  Status GetNodesMap(std::unordered_map<AscendString, ge::OperatorPtr> &nodes_map) const;

 private:
  class ScopeGraphImpl;
  std::unique_ptr<ScopeGraphImpl> impl_;
  friend class ScopePassManager;
  friend class ScopeBasePass;
  friend class TensorFlowModelParser;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeAttrValue {
 public:
  ScopeAttrValue();
  ScopeAttrValue(ScopeAttrValue const &attr_value);
  ScopeAttrValue &operator=(ScopeAttrValue const &attr_value);
  ~ScopeAttrValue();

  void SetIntValue(int64_t value);
  void SetFloatValue(float value);
  ATTRIBUTED_DEPRECATED(void SetStringValue(const char *))
  void SetStringValue(std::string value);
  void SetStringValue(const char *value);
  void SetBoolValue(bool value);

 private:
  class ScopeAttrValueImpl;
  std::unique_ptr<ScopeAttrValueImpl> impl_;
  friend class NodeAttrFeature;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeBaseFeature {
 public:
  virtual bool Match(const Scope *scope) = 0;
  ScopeBaseFeature() = default;
  ScopeBaseFeature(const ScopeBaseFeature &) = delete;
  ScopeBaseFeature &operator=(const ScopeBaseFeature &) = delete;
  ScopeBaseFeature(ScopeBaseFeature &&) = delete;
  ScopeBaseFeature &operator=(ScopeBaseFeature &&) = delete;
  virtual ~ScopeBaseFeature(){};
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY NodeOpTypeFeature : ScopeBaseFeature {
 public:
  ATTRIBUTED_DEPRECATED(NodeOpTypeFeature(const char *, int, int))
  NodeOpTypeFeature(std::string nodeType, int num, int step = 0);
  NodeOpTypeFeature(const char *node_type, int num, int step = 0);
  NodeOpTypeFeature(NodeOpTypeFeature const &feature);
  NodeOpTypeFeature &operator=(NodeOpTypeFeature const &feature);
  ~NodeOpTypeFeature() override;
  bool Match(const Scope *scope) override;

 private:
  class NodeOpTypeFeatureImpl;
  std::unique_ptr<NodeOpTypeFeatureImpl> impl_;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY NodeAttrFeature : ScopeBaseFeature {
 public:
  ATTRIBUTED_DEPRECATED(NodeAttrFeature(const char *, const char *, ge::DataType, ScopeAttrValue &))
  NodeAttrFeature(std::string nodeType, std::string attr_name,
                  ge::DataType datatype, ScopeAttrValue &attr_value);
  NodeAttrFeature(const char *node_type, const char *attr_name,
                  ge::DataType datatype, ScopeAttrValue &attr_value);
  NodeAttrFeature(NodeAttrFeature const &feature);
  NodeAttrFeature &operator=(NodeAttrFeature const &feature);
  ~NodeAttrFeature() override;
  bool Match(const Scope *scope) override;

 private:
  class NodeAttrFeatureImpl;
  std::unique_ptr<NodeAttrFeatureImpl> impl_;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeFeature : ScopeBaseFeature {
 public:
  ATTRIBUTED_DEPRECATED(ScopeFeature(const char *, int32_t, const char *, const char *, int))
  ScopeFeature(std::string sub_type, int32_t num, std::string suffix = "",
               std::string sub_scope_mask = "", int step = 0);
  ScopeFeature(const char *sub_type, int32_t num, const char *suffix,
               const char *sub_scope_mask, int step = 0);
  ScopeFeature(ScopeFeature const &feature);
  ScopeFeature &operator=(ScopeFeature const &feature);
  ~ScopeFeature() override;
  bool Match(const Scope *scope) override;

 private:
  class ScopeFeatureImpl;
  std::unique_ptr<ScopeFeatureImpl> impl_;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopePattern {
 public:
  ScopePattern();
  ~ScopePattern();
  ATTRIBUTED_DEPRECATED(ScopePattern &SetSubType(const char *))
  ScopePattern &SetSubType(const std::string &sub_type);
  ScopePattern &SetSubType(const char *sub_type);
  ScopePattern &AddNodeOpTypeFeature(NodeOpTypeFeature feature);
  ScopePattern &AddNodeAttrFeature(NodeAttrFeature feature);
  ScopePattern &AddScopeFeature(ScopeFeature feature);

 private:
  class ScopePatternImpl;
  std::unique_ptr<ScopePatternImpl> impl_;
  friend class ScopeBasePass;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopesResult {
 public:
  ScopesResult();
  ScopesResult(ScopesResult const &result);
  ScopesResult &operator=(ScopesResult const &result);
  ~ScopesResult();

  void SetScopes(std::vector<Scope *> &scopes);
  void SetNodes(std::vector<ge::OperatorPtr> &nodes);

 private:
  class ScopesResultImpl;
  std::unique_ptr<ScopesResultImpl> impl_;
  friend class ScopeBasePass;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeBasePass {
 public:
  ScopeBasePass();
  ScopeBasePass(const ScopeBasePass &) = delete;
  ScopeBasePass &operator=(const ScopeBasePass &) = delete;
  ScopeBasePass(ScopeBasePass &&) = delete;
  ScopeBasePass &operator=(ScopeBasePass &&) = delete;
  virtual ~ScopeBasePass();

 protected:
  // Subclasses implement respective fusion strategies and build the Patterns
  virtual std::vector<ScopeFusionPatterns> DefinePatterns() = 0;
  // Define the name of the scope pass
  virtual std::string PassName() = 0;
  // Subclasses implement respective multi-scope or operator fusion methods across scopes
  virtual Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                                       std::vector<ScopesResult> &results) = 0;
  // Subclasses implement their own results and set the input and output of the final fusion operator
  virtual void GenerateFusionResult(const std::vector<Scope *> &scopes, FusionScopesResult *fusion_rlt) = 0;

 private:
  class ScopeBasePassImpl;
  std::unique_ptr<ScopeBasePassImpl> impl_;
  friend class ge::ScopePassManager;
  friend class ScopeBasePassImpl;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeFusionPassRegistry {
 public:
  using CreateFn = ScopeBasePass *(*)();
  ~ScopeFusionPassRegistry();

  static ScopeFusionPassRegistry& GetInstance();

  ATTRIBUTED_DEPRECATED(void RegisterScopeFusionPass(const char *, CreateFn, bool))
  void RegisterScopeFusionPass(const std::string &pass_name, CreateFn create_fn, bool is_general);

  void RegisterScopeFusionPass(const char *pass_name, CreateFn create_fn, bool is_general);

 private:
  ScopeFusionPassRegistry();
  class ScopeFusionPassRegistryImpl;
  /*lint -e148*/
  std::unique_ptr<ScopeFusionPassRegistryImpl> impl_;
  friend class TensorFlowModelParser;
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeUtil {
 public:
  ATTRIBUTED_DEPRECATED(static AscendString StringReplaceAll(const char *, const char *, const char *))
  static std::string StringReplaceAll(std::string str, const std::string &old_value, const std::string &new_value);
  static AscendString StringReplaceAll(const char *str, const char *old_value, const char *new_value);
  static void FreeScopePatterns(ScopeFusionPatterns &patterns);
  static void FreeOneBatchPattern(std::vector<ScopePattern *> &one_batch_pattern);
};

class GE_FUNC_HOST_VISIBILITY GE_FUNC_DEV_VISIBILITY ScopeFusionPassRegistrar {
 public:
  ScopeFusionPassRegistrar(const char *pass_name, ScopeBasePass *(*create_fn)(), bool is_general);
  ~ScopeFusionPassRegistrar() {}
};
}  // namespace ge
#define REGISTER_SCOPE_FUSION_PASS(pass_name, scope_pass, is_general) \
  REGISTER_SCOPE_FUSION_PASS_UNIQ_HELPER(__COUNTER__, (pass_name), scope_pass, is_general)

#define REGISTER_SCOPE_FUSION_PASS_UNIQ_HELPER(ctr, pass_name, scope_pass, is_general) \
  REGISTER_SCOPE_FUSION_PASS_UNIQ(ctr, (pass_name), scope_pass, is_general)

#define REGISTER_SCOPE_FUSION_PASS_UNIQ(ctr, pass_name, scope_pass, is_general)                   \
  static ::ge::ScopeFusionPassRegistrar register_scope_fusion_pass##ctr __attribute__((unused)) = \
      ::ge::ScopeFusionPassRegistrar(                                                             \
          (pass_name), []() -> ::ge::ScopeBasePass * { return new (std::nothrow) scope_pass(); }, is_general)
#endif  // EXTERNAL_REGISTER_SCOPE_SCOPE_FUSION_PASS_REGISTER_H_

