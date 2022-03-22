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

#ifndef INC_GRAPH_OP_DESC_H_
#define INC_GRAPH_OP_DESC_H_

#include <functional>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <set>
#include <unordered_set>
#include <vector>
#include "detail/attributes_holder.h"
#include "graph/range_vistor.h"

#define DYNAMIN_INPUT_NAME(name, index) (((name)) + std::to_string((index)))
#define DYNAMIN_OUTPUT_NAME(name, index) (((name)) + std::to_string((index)))
namespace ge {
using std::map;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;

class Operator;
class GeTensorDesc;

using GeTensorDescPtr = shared_ptr<GeTensorDesc>;
using ConstGeTensorDescPtr = shared_ptr<const GeTensorDesc>;

class OpDesc;

using OpDescPtr = shared_ptr<OpDesc>;
using ConstOpDescPtr = shared_ptr<const OpDesc>;

using ConstOpDesc = const OpDesc;

class OpDescImpl;
using OpDescImplPtr = std::shared_ptr<OpDescImpl>;

enum SubgraphType {
  kStatic,
  kDynamic,
  kSubgraphTypeEnd
};

class OpDesc : public std::enable_shared_from_this<OpDesc>, public AttrHolder {
 public:
  template <class T>
  using Vistor = RangeVistor<T, shared_ptr<ConstOpDesc>>;

  friend class GraphBuilderImpl;

  friend class OperatorImpl;

  OpDesc(const std::string &name, const std::string &type);

  OpDesc(const OpDesc &op_desc);

  OpDesc(OpDesc &&op_desc);

  explicit OpDesc(const ge::proto::OpDef &op_def);

  OpDesc();

  ~OpDesc() override;

  bool operator==(const OpDesc &r_op_desc) const;
  OpDesc& operator=(OpDesc op_desc);

  std::string GetName() const;

  void SetName(const std::string &name);

  std::string GetType() const;

  void SetType(const std::string &type);

  graphStatus AddInputDesc(const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(const std::string &name, const GeTensorDesc &input_desc);

  graphStatus AddInputDesc(const uint32_t index, const ge::GeTensorDesc &input_desc);

  graphStatus AddInputDescForward(const std::string &name, const uint32_t num);

  graphStatus AddInputDescMiddle(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddOutputDescMiddle(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddOutputDescForward(const std::string &name, const uint32_t num);

  graphStatus AddOptionalInputDesc(const std::string &name, const GeTensorDesc &input_desc);

  graphStatus UpdateInputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateInputDesc(const std::string &name, const GeTensorDesc &tensor_desc);

  bool InputIsSet(const std::string &name) const;

  const GeTensorDesc &GetInputDesc(const uint32_t index) const;

  const GeTensorDesc &GetInputDesc(const std::string &name) const;

  Vistor<string> GetAllInputNames() const;

  GeTensorDescPtr MutableInputDesc(const uint32_t index) const;

  GeTensorDescPtr MutableInputDesc(const std::string &name) const;

  Vistor<GeTensorDesc> GetAllInputsDesc() const;

  Vistor<GeTensorDescPtr> GetAllInputsDescPtr() const;

  size_t GetInputsSize() const;

  size_t GetAllInputsSize() const;

  graphStatus AddOutputDesc(const GeTensorDesc &output_desc);

  graphStatus AddOutputDesc(const std::string &name, const GeTensorDesc &output_desc);

  graphStatus UpdateOutputDesc(const uint32_t index, const GeTensorDesc &tensor_desc);

  graphStatus UpdateOutputDesc(const std::string &name, const GeTensorDesc &tensor_desc);

  const GeTensorDesc &GetOutputDesc(const uint32_t index) const;

  const GeTensorDesc &GetOutputDesc(const std::string &name) const;

  GeTensorDescPtr MutableOutputDesc(const uint32_t index) const;

  GeTensorDescPtr MutableOutputDesc(const std::string &name) const;

  uint32_t GetAllOutputsDescSize() const;

  Vistor<GeTensorDesc> GetAllOutputsDesc() const;

  Vistor<GeTensorDescPtr> GetAllOutputsDescPtr() const;

  size_t GetOutputsSize() const;

  ConstGeTensorDescPtr GetOutputDescPtr(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtr(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtrDfault(const uint32_t index) const;

  ConstGeTensorDescPtr GetInputDescPtr(const std::string &name) const;

  graphStatus AddDynamicInputDesc(const std::string &name, const uint32_t num, const bool is_push_back = true);

  graphStatus AddDynamicInputDescByIndex(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddDynamicOutputDesc(const std::string &name, const uint32_t num, const bool is_push_back = true);

  bool IsOptionalInput(const std::string &name) const;

  bool IsOptionalInput(const uint32_t index) const;

  std::map<std::string, uint32_t> GetAllInputName() const;

  std::map<std::string, uint32_t> GetAllOutputName();

  std::map<std::string, uint32_t>& MutableAllInputName();

  std::map<std::string, uint32_t>& MutableAllOutputName();

  bool UpdateInputName(const std::map<std::string, uint32_t> input_name_idx);

  bool UpdateOutputName(const std::map<std::string, uint32_t> output_name_idx);

  void AddInferFunc(const std::function<graphStatus(Operator &)> &func);

  std::function<graphStatus(Operator &)> GetInferFunc() const;

  graphStatus InferShapeAndType();

  void AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func);

  std::function<graphStatus(Operator &)> GetInferFormatFunc() const;

  graphStatus DefaultInferFormat();

  std::function<graphStatus(Operator &)> GetVerifyFunc() const;

  void AddVerifierFunc(const std::function<graphStatus(Operator &)> &func);

  graphStatus CallInferFormatFunc(Operator &op);

  graphStatus CallInferValueRangeFunc(Operator &op);

  graphStatus OpVerify();

  graphStatus CommonVerify() const;

  graphStatus AddRegisterInputName(const std::string &name);

  graphStatus AddRegisterOutputName(const std::string &name);

  std::vector<std::string> GetRegisterInputName() const;

  std::vector<std::string> GetRegisterOutputName() const;

  using AttrHolder::AddRequiredAttr;
  using AttrHolder::DelAttr;
  using AttrHolder::GetAllAttrNames;
  using AttrHolder::GetAllAttrs;
  using AttrHolder::GetAttr;
  using AttrHolder::HasAttr;
  using AttrHolder::SetAttr;

  void SetId(const int64_t id);
  int64_t GetId() const;
  void SetStreamId(const int64_t stream_id);
  int64_t GetStreamId() const;
  void SetInputName(const std::vector<std::string> &input_name);
  std::vector<std::string> GetInputName() const;
  void SetSrcName(const std::vector<std::string> &src_name);
  std::vector<std::string> GetSrcName() const;
  void SetSrcIndex(const std::vector<int64_t> &src_index);
  std::vector<int64_t> GetSrcIndex() const;
  void SetInputOffset(const std::vector<int64_t> &input);
  std::vector<int64_t> GetInputOffset() const;
  void SetOutputOffset(const std::vector<int64_t> &output);
  std::vector<int64_t> GetOutputOffset() const;
  void SetDstName(const std::vector<std::string> &dst_name);
  std::vector<std::string> GetDstName() const;
  void SetDstIndex(const std::vector<int64_t> &dst_index);
  std::vector<int64_t> GetDstIndex() const;
  void SetWorkspace(const std::vector<int64_t> &workspace);
  std::vector<int64_t> GetWorkspace() const;
  void SetWorkspaceBytes(const std::vector<int64_t> &workspace_bytes);
  std::vector<int64_t> GetWorkspaceBytes() const;
  void SetIsInputConst(const std::vector<bool> &is_input_const);
  std::vector<bool> GetIsInputConst() const;

  void SetOpInferDepends(const std::vector<std::string> &depend_names);
  std::vector<std::string> GetOpInferDepends() const;

  std::string GetInputNameByIndex(const uint32_t index) const;
  std::string GetValidInputNameByIndex(const uint32_t index) const;
  int32_t GetValidInputIndexByName(const std::string &name) const;
  int32_t GetInputIndexByName(const std::string &name) const;

  std::string GetOutputNameByIndex(const uint32_t index) const;

  int32_t GetOutputIndexByName(const std::string &name) const;

  graphStatus RestoreInputNameIdx(const std::string &name, const int32_t &index);

  graphStatus RestoreOutputNameIdx(const std::string &name, const int32_t &index);

  graphStatus CallInferFunc(Operator &op);

  void SetOpKernelLibName(const std::string &name);

  std::string GetOpKernelLibName() const;

  void SetOpEngineName(const std::string &name);

  std::string GetOpEngineName() const;

  void RegisterSubgraphIrName(const std::string &name, const SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;

  graphStatus AddSubgraphName(const std::string &name);
  const std::map<std::string, uint32_t> &GetSubgraphNameIndexes() const;

  std::string GetSubgraphInstanceName(const uint32_t index) const;
  const std::vector<std::string> &GetSubgraphInstanceNames() const;
  /// Does not provide functions `AddSubgraphInstance` or `AppendSubgraphInstance`,
  /// because this kind of functions will only append a new subgraph instance name
  /// at the tail of `subgraph_instance_names_` and ignore the synchronous change of `subgraph_names_to_index_`.
  /// If we want to append a new subgraph instance name, the function `AddSubgraphName` should be called first.
  /// \param index
  /// \param name
  /// \return
  graphStatus SetSubgraphInstanceName(const uint32_t index, const std::string &name);
  void RemoveSubgraphInstanceName(const std::string &name);

  graphStatus GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const;

  graphStatus InferDataSlice();

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  bool OpDescMembersAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescAttrsAreEqual(const OpDesc &r_op_desc) const;
  bool OpDescGenTensorDescsAreEqual(const OpDesc &r_op_desc) const;

  OpDescImplPtr impl_;
  friend class OpDescUtils;
  friend class ModelSerializeImp;
  friend class AttrUtils;
  friend class GeAttrValueImp;
  friend class OnnxUtils;
  friend class GraphUtils;
};
}  // namespace ge
#endif  // INC_GRAPH_OP_DESC_H_
