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

#ifndef GRAPH_OP_DESC_IMPL_H_
#define GRAPH_OP_DESC_IMPL_H_

#include <string>
#include <utility>
#include <vector>
#include "graph/op_desc.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"

namespace ge {
class OpDescImpl;
class MetaDataStore {
 public:
  using SmallIntVector = SmallVector<int64_t, kDefaultMaxInputNum>;
  MetaDataStore() = default;
  ~MetaDataStore() = default;
  MetaDataStore(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)) {}
  const string &GetName() const {return name_;}
  const string &GetType() const {return type_;}
  const vector<std::string> &GetInputs() const {return inputs_;}
  bool HasOutAttr() const {return has_out_attr_;}
  int64_t GetId() const {return id_;}
  int64_t GetStreamId() const {return stream_id_;}
  const vector<std::string> &GetInputNames() const {return input_names_;}
  const vector<std::string> &GetSrcNames() const {return src_names_;}
  const vector<int64_t> &GetSrcIndexes() const {return src_indexes_;}
  const vector<std::string> &GetDstNames() const {return dst_names_;}
  const vector<int64_t> &GetDstIndexes() const {return dst_indexes_;}
  const vector<int64_t> &GetInputOffsets() const {return input_offsets_;}
  const vector<int64_t> &GetOutputOffsets() const {return output_offsets_;}
  const vector<bool> &GetIsInputConsts() const {return is_input_consts_;}
  const vector<std::string> &GetSubgraphNames() const {return subgraph_names_;}
  void AddSubGraphName(const string &name) {subgraph_names_.push_back(name);}

 private:
  friend class OpDescImpl;
  std::string name_;
  std::string type_;
  std::vector<std::string> inputs_;
  bool has_out_attr_{false};
  int64_t id_{0};
  int64_t stream_id_{0};
  std::vector<std::string> input_names_;
  std::vector<std::string> src_names_;
  std::vector<int64_t> src_indexes_;
  std::vector<std::string> dst_names_;
  std::vector<int64_t> dst_indexes_;
  std::vector<int64_t> input_offsets_;
  std::vector<int64_t> output_offsets_;
  SmallIntVector workspaces;
  SmallIntVector workspace_bytes_list_;
  std::vector<bool> is_input_consts_;
  std::vector<std::string> subgraph_names_;
};

class OpDescImpl {
 public:
  OpDescImpl();
  OpDescImpl(const std::string &name, const std::string &type);
  explicit OpDescImpl(const ge::proto::OpDef &op_def);

  ~OpDescImpl() = default;

  std::string GetName() const;
  void SetName(const std::string &name);
  std::string GetType() const;
  void SetType(const std::string &type);

  graphStatus AddInputDesc(const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDesc(const uint32_t index, const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDesc(const std::string &name, const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDescMiddle(const std::string &name, const uint32_t num, const size_t index);
  graphStatus AddOutputDescMiddle(const std::string &name, const uint32_t num, const size_t index);
  graphStatus AddInputDescForward(const std::string &name, const uint32_t num);
  graphStatus AddOutputDescForward(const std::string &name, const uint32_t num);
  graphStatus AddOptionalInputDesc(const std::string &name, const ge::GeTensorDesc &input_desc);

  graphStatus UpdateInputDesc(const uint32_t index, const ge::GeTensorDesc &tensor_Desc);
  graphStatus UpdateInputDesc(const std::string &name, const ge::GeTensorDesc &tensor_Desc);

  bool OpDescMembersAreEqual(const OpDescImpl &r_op_desc) const;
  bool OpDescAttrsAreEqual(const OpDescImpl &r_op_desc) const;
  bool OpDescGenTensorDescsAreEqual(const OpDescImpl &r_op_desc) const;

  bool InputIsSet(const std::string &name) const;

  const GeTensorDesc &GetInputDesc(const uint32_t index) const;
  const GeTensorDesc &GetInputDesc(const std::string &name) const;
  GeTensorDescPtr MutableInputDesc(const uint32_t index) const;
  GeTensorDescPtr MutableInputDesc(const std::string &name) const;
  OpDesc::Vistor<string> GetAllInputNames(const ConstOpDescPtr &op_desc) const;

  void SetOpKernelLibName(const std::string &name);
  std::string GetOpKernelLibName() const;
  void SetOpEngineName(const std::string &name);
  std::string GetOpEngineName() const;

  OpDesc::Vistor<GeTensorDesc> GetAllInputsDesc(const ConstOpDescPtr &op_desc) const;
  OpDesc::Vistor<GeTensorDescPtr> GetAllInputsDescPtr(const ConstOpDescPtr &op_desc) const;

  size_t GetInputsSize() const;
  size_t GetAllInputsSize() const;

  graphStatus AddOutputDesc(const ge::GeTensorDesc &output_desc);
  graphStatus AddOutputDesc(const std::string &name, const ge::GeTensorDesc &output_desc);
  graphStatus UpdateOutputDesc(const uint32_t index, const ge::GeTensorDesc &tensor_Desc);
  graphStatus UpdateOutputDesc(const std::string &name, const ge::GeTensorDesc &tensor_Desc);
  const GeTensorDesc &GetOutputDesc(const uint32_t index) const;
  const GeTensorDesc &GetOutputDesc(const std::string &name) const;
  GeTensorDescPtr MutableOutputDesc(const uint32_t index) const;
  GeTensorDescPtr MutableOutputDesc(const std::string &name) const;

  uint32_t GetAllOutputsDescSize() const;
  OpDesc::Vistor<GeTensorDesc> GetAllOutputsDesc(const ConstOpDescPtr &op_desc) const;
  OpDesc::Vistor<GeTensorDescPtr> GetAllOutputsDescPtr(const ConstOpDescPtr &op_desc) const;
  ConstGeTensorDescPtr GetOutputDescPtr(const uint32_t index) const;
  size_t GetOutputsSize() const;

  ConstGeTensorDescPtr GetInputDescPtr(const uint32_t index) const;
  ConstGeTensorDescPtr GetInputDescPtrDfault(const uint32_t index) const;
  ConstGeTensorDescPtr GetInputDescPtr(const std::string &name) const;

  graphStatus AddRegisterInputName(const std::string &name);
  std::vector<std::string> GetRegisterInputName() const;

  graphStatus AddDynamicInputDesc(const std::string &name, const uint32_t num, const bool is_push_back);
  graphStatus AddDynamicInputDescByIndex(const std::string &name, const uint32_t num, const size_t index);

  graphStatus AddRegisterOutputName(const std::string &name);
  std::vector<std::string> GetRegisterOutputName() const;

  graphStatus AddDynamicOutputDesc(const std::string &name, const uint32_t num, const bool is_push_back);
  bool IsOptionalInput(const std::string &name) const;
  bool IsOptionalInput(const uint32_t index) const;
  std::map<std::string, uint32_t> GetAllInputName() const;
  std::map<std::string, uint32_t> GetAllOutputName();
  std::map<std::string, uint32_t>& MutableAllInputName();
  std::map<std::string, uint32_t>& MutableAllOutputName();
  bool UpdateInputName(std::map<std::string, uint32_t> input_name_idx);
  bool UpdateOutputName(std::map<std::string, uint32_t> output_name_idx);

  std::function<graphStatus(Operator &)> GetInferFunc() const;
  std::function<graphStatus(Operator &)> GetVerifyFunc() const;
  void AddInferFunc(const std::function<graphStatus(Operator &)> &func);
  std::function<graphStatus(Operator &)> GetInferFormatFunc() const;
  void AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func);
  void AddVerifierFunc(const std::function<graphStatus(Operator &)> &func);

  graphStatus InferShapeAndType(const OpDescPtr &op_desc);
  graphStatus DefaultInferFormat(const ConstOpDescPtr &op_desc) const;
  graphStatus OpVerify(const OpDescPtr &op_desc);

  std::string GetInputNameByIndex(const uint32_t index) const;
  int32_t GetInputIndexByName(const std::string &name) const;
  int32_t GetValidInputIndexByName(const std::string &name) const;
  std::string GetValidInputNameByIndex(const uint32_t index) const;

  std::string GetOutputNameByIndex(const uint32_t index) const;
  int32_t GetOutputIndexByName(const std::string &name) const;

  ProtoAttrMap &MutableAttrMap();
  ConstProtoAttrMap &GetAttrMap() const;

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

  graphStatus RestoreInputNameIdx(const std::string &name, const int32_t &index);
  graphStatus RestoreOutputNameIdx(const std::string &name, const int32_t &index);

  graphStatus CallInferFunc(Operator &op, const OpDescPtr &op_desc);
  graphStatus CallInferFormatFunc(Operator &op, const ConstOpDescPtr &op_desc);
  graphStatus CallInferValueRangeFunc(Operator &op, const ConstOpDescPtr &op_desc);

  std::string GetSubgraphInstanceName(const size_t index) const;
  const std::vector<std::string> &GetSubgraphInstanceNames() const;
  void RemoveSubgraphInstanceName(const std::string &name);
  graphStatus AddSubgraphName(const std::string &name);
  const std::map<std::string, uint32_t> &GetSubgraphNameIndexes() const;
  graphStatus SetSubgraphInstanceName(const size_t index, const std::string &name);

  void RegisterSubgraphIrName(const std::string &name, const SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;
  graphStatus GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const;
  graphStatus InferDataSlice(const OpDescPtr &op_desc);

 private:
  void DeSerializeOpDefToMetaData(const proto::OpDef &op_def);
  void SerializeMetaDataToOpDef(proto::OpDef * const op_def);
  friend class AttrUtils;
  friend class OpDescUtils;
  friend class ModelSerializeImp;
  friend class OnnxUtils;
  friend class GraphUtils;
  std::vector<std::string> subgraph_instance_names_;

  // subgraph names to index, for a `if` operator:
  // then_branch: 0
  // else_branch: 1
  // or for a `case` node:
  // branches0: 0
  // branches1: 1
  // branches2: 2
  std::map<std::string, uint32_t> subgraph_names_to_index_;

  // subgraph ir names to type, for a `if` operator:
  // then_branch: static
  // else_branch: static
  // or for a `case` op:
  // branches: dynamic
  std::map<std::string, SubgraphType> subgraph_ir_names_to_type_;

  std::vector<GeTensorDescPtr> inputs_desc_{};
  std::map<std::string, uint32_t> input_name_idx_{};
  std::vector<std::string> register_input_name_{};
  std::set<std::string> optional_input_names_{};
  std::vector<GeTensorDescPtr> outputs_desc_{};
  std::map<std::string, uint32_t> output_name_idx_{};
  std::vector<std::string> register_output_name_{};
  std::function<graphStatus(Operator &)> infer_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_format_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_value_range_func_ = nullptr;
  std::function<graphStatus(Operator &)> verifier_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_data_slice_func_ = nullptr;
  std::string op_kernel_lib_name_;
  std::string engine_name_;
  MetaDataStore meta_data_;
  AttrStore attrs_;
};
}  // namespace ge
#endif  // GRAPH_OP_DESC_IMPL_H_
