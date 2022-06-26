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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H
#define LUOJIANET_MS_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H

#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <stack>
#include <map>
#include <tuple>
#include <set>
#include "utils/hash_map.h"
#include "backend/common/session/session_basic.h"
#include "backend/common/session/kernel_graph.h"
#include "kernel/kernel.h"
#include "backend/common/session/session_factory.h"
#include "backend/common/session/pynative_task_manager.h"

namespace luojianet_ms {
namespace session {
enum GraphType : int { COMMON_GRAPH = 0, CONDITION_GRAPH = 1, BRANCH_START = 2, BRANCH_END = 3 };

class AscendSession : public SessionBasic {
 public:
  AscendSession() { final_graph_id_ = kInvalidGraphId; }
  ~AscendSession() = default;
  void Init(uint32_t device_id) override;
  // get graph id of final graph
  GraphId GetFinalRunGraph() const override { return final_graph_id_; }
  void SyncStream() const override;

  static void BatchBuildKernel(const std::vector<std::shared_ptr<SessionTask>> &build_tasks);

 protected:
  void UnifyMindIR(const KernelGraphPtr &graph) override;
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) override;
  bool IsSupportSummary() override;
  void LoadInputData(const std::shared_ptr<KernelGraph> &kernel_graph,
                     const std::vector<tensor::TensorPtr> &inputs_const) const override;
  void PreExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                       VectorRef *const outputs) override;
  void PostExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                        VectorRef *const outputs) override;
  void ExecuteGraph(const std::shared_ptr<KernelGraph> &kernel_graph) override;
  void BuildGraphImpl(GraphId) override;

  KernelGraphPtr BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                             const std::vector<tensor::TensorPtr> &input_tensors,
                             const std::vector<int64_t> &tensors_mask) override;

  void BindAddressToTensor(const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) const;
  void RunOpImplOrigin(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                       std::vector<tensor::TensorPtr> *input_tensors, VectorRef *outputs,
                       const std::vector<int64_t> &tensors_mask) override;

  void RunOpImpl(const GraphInfo &graph_info, OpRunInfo *op_run_info, std::vector<tensor::TensorPtr> *input_tensors,
                 VectorRef *outputs, const std::vector<int64_t> &tensors_mask) override;
  void BuildOpsInGraph(const GraphId &graph_id, const std::map<AnfNodePtr, size_t> &parameter_index,
                       const std::vector<tensor::TensorPtr> &graph_inputs,
                       const std::map<KernelWithIndex, size_t> &cnode_refcount) override;
  std::string GetCommWorldGroup() override { return kHcclWorldGroup; }
  void ReportWarningMessage() override;
  void ReportErrorMessage() override;
  void SetThreadContext() override;
  void ExecuteAllTaskInQueue() override;
  void UpdateOutputTensors(const VectorRef *outputs,
                           const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                           std::map<DeviceAddressPtr, DeviceAddressPtr> *) override;
  DeviceAddressPtr AssignExtraMemForGraphOutput(const tensor::TensorPtr &tensor, const AnfNodePtr &node,
                                                size_t index) const;

 private:
  // compile child graph when session have multiple child graphs
  void CompileChildGraph(const KernelGraphPtr &child_graph);
#ifndef ENABLE_SECURITY
  void RecurseSetSummaryNodes(KernelGraph *graph, std::map<std::string, std::pair<AnfNodePtr, int>> *summary);
  void SetSummaryNodes(KernelGraph *graph) override;
#endif
  void InitRuntimeResource();
  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AssignStream(NotNull<KernelGraphPtr> kernel_graph) const;
  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  static void BuildKernel(const std::vector<CNodePtr> &kernels);
  void BuildDynamicKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void MemoryAlloc(KernelGraph *kernel_graph) const;
  void RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors, KernelGraph *kernel_graph,
                        bool is_gradient_out) const;
  void RunOpMemoryAllocNew(const std::vector<tensor::TensorPtr> &input_tensors,
                           const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                           const KernelGraph &kernel_graph) const;
  void RunOpMemoryClear(const KernelGraph *kernel_graph) const;
  void RunOpGenKernelEvent(const KernelGraph *graph) const;
  void Load(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void Execute(const std::shared_ptr<KernelGraph> &kernel_graph, bool is_task) const;
#ifndef ENABLE_SECURITY
  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;
#endif
  void LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  // below functions are used for run op
  void RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const;

  void RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph, const std::vector<KernelGraphPtr> &all_graphs);
  // merge execution order list of child graphs
  void MergeGraphExecOrder();
  // get graph order vector by graph id
  const std::vector<GraphId> &GetGraphOrder(GraphId final_graph_id) const;
  // get graph order type vector by graph id
  const std::vector<GraphType> &GetGraphOrderType(GraphId final_graph_id) const;
  // sync initial tensors' data to device
  void SyncInitialTenosrToDevice();
#ifndef ENABLE_SECURITY
  void SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph);
#endif
  // create parameter to receive data from multiple branch output
  void CreateMultiBranchOutput(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo);
  void IrFusionPass(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo);
  void HardwareOptimize(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
#ifdef ENABLE_DEBUGGER
  void LoadGraphsToDbg(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
#endif
  void AssignStaticMemory(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void UpdateRefOutputMap(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  KernelGraphPtr PreBuildOp(const OpRunInfo &op_run_info, const std::vector<tensor::TensorPtr> &input_tensors,
                            const std::vector<int64_t> &tensors_mask);
  void GetOpInputStubTensors(const CNodePtr &cnode, const std::map<AnfNodePtr, size_t> &parameter_index,
                             const std::vector<tensor::TensorPtr> &graph_inputs,
                             const std::map<KernelWithIndex, OutputTensorInfo> &node_output_info,
                             InputTensorInfo *input_tensor_info);
  void PrepareForOutputTensor(const KernelGraphPtr &graph, const std::vector<tensor::TensorPtr> &input_tensors,
                              std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                              VectorRef *outputs) const;
  std::shared_ptr<device::Bucket> CreateBucket(uint32_t bucket_id, uint32_t bucket_size) override;

  void LaunchFunc(const KernelGraphPtr &graph,
                  const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node, bool is_dynamic_shape,
                  const std::vector<tensor::TensorPtr> &input_tensors);
  KernelGraphPtr CreateKernelGraph(const GraphInfo &graph_info, OpRunInfo *op_run_info,
                                   std::vector<tensor::TensorPtr> *input_tensors,
                                   const std::vector<int64_t> &tensors_mask, bool cache_miss);
  static bool DisableLazyBuild(const OpRunInfo &op_run_info);
  void SelectKernel(const KernelGraphPtr &graph) const;
  void SetOperatorInfo(const std::vector<CNodePtr> &nodes) const;
  void RecurseSelectKernelInfo(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *memo) const;
  // key is final_graph_id,value is child graph execute order of final graph
  luojianet_ms::HashMap<GraphId, std::vector<GraphId>> graph_execute_orders_;
  // key is final_graph_id,value is the graph types of child graphs
  luojianet_ms::HashMap<GraphId, std::vector<GraphType>> graph_order_types_;
  // initial tensors, these tensor will sync data to device before run graph
  std::map<std::pair<GraphId, size_t>, tensor::TensorPtr> initial_tenosrs_;
  // final_graph_id is used in every root graph has it's own session situation
  GraphId final_graph_id_;
  // record graph ids of bp graphs that has been built in PyNative mode
  std::set<GraphId> built_graph_id_;
  // tensor with new device addr map
  std::map<tensor::TensorPtr, DeviceAddressPtr> tensor_device_addr_map_;
  // Number of operators whose precision changes after select kernel
  mutable size_t raise_precision_count_{0};
  mutable size_t reduce_precision_count_{0};
};
MS_REG_SESSION(kAscendDevice, AscendSession);
}  // namespace session
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H
