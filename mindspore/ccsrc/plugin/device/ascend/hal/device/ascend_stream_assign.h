/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_

#include <functional>
#include <unordered_map>
#include <map>
#include <set>
#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_set>
#include <utility>
#include "runtime/base.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "backend/common/session/kernel_graph.h"
#include "include/common/utils/contract.h"

namespace mindspore {
namespace device {
namespace ascend {
using std::map;
using std::queue;
using std::shared_ptr;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using CNodeKey = void *;
using GroupGraphMap = std::map<std::string, std::map<uint32_t, std::vector<CNodePtr>>>;
const uint32_t kInvalidStreamId = UINT32_MAX;
const uint32_t kInvalidEventId = UINT32_MAX;
enum StreamActiveKind { kInvalid = 0, kHead, kMiddle, kTail };
class AscendStreamAssign {
 public:
  static AscendStreamAssign &GetInstance() {
    static AscendStreamAssign instance;  // Guaranteed to be destroyed.
    return instance;
  }

  AscendStreamAssign(const AscendStreamAssign &) = delete;
  AscendStreamAssign &operator=(const AscendStreamAssign &) = delete;

  void AssignStream(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetWaitStreams(vector<uint32_t> *wait_active_stream_list);
  void GetHcomStreams(std::vector<uint32_t> *streams);
  void AssignStreamForNonTaskSink(const std::vector<CNodePtr> &kernels);
  const std::vector<std::vector<uint32_t>> &get_stream_group() const { return stream_groups_; }
  const std::map<CNodePtr, CNodePtr> &get_event_map() const { return event_map_; }

 private:
  AscendStreamAssign() = default;
  ~AscendStreamAssign() = default;

  void AssignAllNodesStream(const NotNull<KernelGraphPtr> &graph_ptr);
  std::set<uint32_t> AssignNodeStreamInOrder(const std::vector<CNodePtr> node_list);
  void ClassifyNodeByKernel(const NotNull<KernelGraphPtr> &graph_ptr, std::vector<CNodePtr> *common_list,
                            std::vector<CNodePtr> *hcom_list, std::vector<CNodePtr> *independent_list);
  void ClassifyNodeByGroupAndGraph(const std::vector<CNodePtr> hcom_list, GroupGraphMap *group_graph_map);
  void ClassifyNodeByGraph(const std::vector<CNodePtr> indepent_list,
                           std::map<uint32_t, std::vector<CNodePtr>> *graph_node_map);
  uint32_t GetNodeTaskNum(const CNodePtr &cnode);

  CNodePtr CreateSendApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id);
  CNodePtr CreateRecvApplyKernel(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t event_id, uint32_t stream_id);
  void CheckResourceAssign(const NotNull<KernelGraphPtr> &graph_ptr);
  void CheckStreamAssign(const NotNull<KernelGraphPtr> &graph_ptr);
  void CheckEventAssign(const NotNull<KernelGraphPtr> &graph_ptr);

  void UpdateAtomicAddrCleanStreamId(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActive(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForCommon(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForIndependent(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertStreamActiveForParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void ActiveRootGraphHcom(const NotNull<KernelGraphPtr> &graph_ptr, const std::set<uint32_t> &hcom_streams);
  void ActiveRootGraphIndependent(const NotNull<KernelGraphPtr> &graph_ptr,
                                  const std::set<uint32_t> &independent_streams);
  void ActiveOtherGraphParallel(const NotNull<KernelGraphPtr> &graph_ptr,
                                std::map<uint32_t, std::set<uint32_t>> other_graph);
  void InsertEventForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertCtrlForIndependentParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventForHcomParallel(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventForIndependentHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventCommonDependHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependCommon(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependCommonBak(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventHcomDependHcom(const NotNull<KernelGraphPtr> &graph_ptr);
  void InsertEventBetweenHcom(const NotNull<KernelGraphPtr> &graph_ptr,
                              const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index);
  void InsertEventHcomDependHcomAtSameGroup(const NotNull<KernelGraphPtr> &graph_ptr,
                                            std::pair<std::string, std::map<uint32_t, std::set<uint32_t>>> group_item);
  std::vector<std::pair<uint32_t, vector<size_t>>> GetStreamIDHcomMap(std::vector<CNodePtr> cnode_ptr_list,
                                                                      std::string group, size_t graph_id);

  void AdjustAtomicAddrCleanOrder(const NotNull<KernelGraphPtr> &graph_ptr);
  vector<CNodePtr> GetLastInputCnode(const NotNull<KernelGraphPtr> &graph_ptr, const CNodePtr &cur_cnode_ptr);
  bool IsSatisfiedHcom(const std::vector<std::pair<uint32_t, vector<size_t>>> &hcom_index, const CNodePtr &node_ptr,
                       size_t index);

  void GetProcessedStream(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetNeedActiveStreams(const NotNull<KernelGraphPtr> &graph_ptr);
  void ReorderIndependentOrders(const NotNull<KernelGraphPtr> &graph_ptr);

  void CheckScenario(const NotNull<KernelGraphPtr> &graph_ptr, vector<CNodePtr> *last_grad_and_status);
  CNodePtr GetCNodesNeededMoved(vector<CNodePtr> *moved_backward_cnodes, vector<CNodePtr> *moved_forward_cnodes,
                                const vector<CNodePtr> &last_grad_and_status, const NotNull<KernelGraphPtr> &graph_ptr);
  CNodePtr GetTargetOutputNode(const vector<CNodePtr> &moved_backward_cnodes, const CNodePtr first_node,
                               const NotNull<KernelGraphPtr> &graph_ptr);
  bool FinetuneSubgraphExecOrder(vector<CNodePtr> *cnodes);
  void TrailingTimeOptimizationByReorder(const NotNull<KernelGraphPtr> &graph_ptr);

  uint32_t GetMaxIndexTarget(const NotNull<KernelGraphPtr> &graph_ptr);
  uint32_t GetIndexByKey(const NotNull<KernelGraphPtr> &graph_ptr, const CNodeKey &key);
  uint32_t GetIndependentStreamSwitchStreamId(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetIndependentMaxTarget(const NotNull<KernelGraphPtr> &graph_ptr);
  bool IsNopNodeTarget(const AnfNodePtr &nop_node, const CNodePtr &target_node, const CNodePtr &cur_node,
                       bool exclude_hcom);
  bool IsTaskSink();
  bool IsHcom(const CNodePtr &cur_cnode_ptr);
  bool IsIndependentNode(const CNodePtr &node_ptr);
  vector<CNodePtr>::iterator FindTargetOp(vector<CNodePtr>::iterator begin, vector<CNodePtr>::iterator end,
                                          const CNodePtr &node, bool exclude_hcom);
  void SetLoopSink();
  void GetMaxStreamTaskNum();
  void Reset();

  // function for memory reuse
  void GetStreamRelations();
  void DFS(uint32_t start, std::vector<uint32_t> *group);
  bool IsVecExist(const std::vector<uint32_t> &group);
  void FindStreamRelations(const NotNull<KernelGraphPtr> &graph_ptr);
  void GetStreamSwitchStreamRelation(const CNodePtr &node_ptr);
  void GetStreamActiveStreamRelation(const NotNull<KernelGraphPtr> &graph_ptr, size_t index);
  StreamActiveKind GetStreamActiveKind(const NotNull<KernelGraphPtr> &graph_ptr, size_t index);
  uint32_t GetStreamByActivedStream(uint32_t actived_stream_id);
  void PrintStreamRelations();
  void PrintStreamGroups();
  void FindEventRelations(const NotNull<KernelGraphPtr> &graph_ptr);
  bool IsSatisfiedEvent(uint32_t send_stream_id, uint32_t recv_stream_id) const;
  vector<CNodePtr> GetInputKernels(const CNodePtr &cnode);

  bool ExistStreamSendAfterLastHcomNode(const NotNull<KernelGraphPtr> &graph_ptr, uint32_t graph_id);
  void GetAllGraphID(const NotNull<KernelGraphPtr> &graph_ptr, std::vector<uint32_t> *graphs_id);
  void GraphLoopSync(const NotNull<KernelGraphPtr> &root_graph, uint32_t graph_id);

  void InsertEventForMicroBatchIndependent(const NotNull<KernelGraphPtr> &graph_ptr);

  bool independent_stream_activated_{false};
  bool hcom_stream_activated_{false};
  bool loop_sink_{false};

  // key:stream id, value:task number
  std::set<uint32_t> common_stream_{};
  std::set<uint32_t> independent_stream_{};
  std::set<uint32_t> hcom_stream_{};

  std::set<uint32_t> processed_streams_{};
  std::vector<uint32_t> need_first_active_streams_{};
  std::set<CNodeKey> independent_targets_;
  std::map<std::string, uint32_t> group_stream_id_map_;

  // key:group name, value:key1:graph id, value1:stream id
  std::map<std::string, std::map<uint32_t, std::set<uint32_t>>> group_hcom_graph_map_;
  // key:graph id, value:stream set
  std::map<uint32_t, std::set<uint32_t>> independent_graph_map_;

  // attr for memory copy reuse
  std::map<uint32_t, std::vector<uint32_t>> stream_relations_{};
  std::vector<std::vector<uint32_t>> stream_groups_{};
  std::map<CNodePtr, CNodePtr> event_map_{};
  std::set<uint32_t> middle_active_streams_{};
  // new policy end
  bool IsAllOutGraphOut(const KernelGraphPtr &graph, const CNodePtr &cnode);

  uint32_t max_stream_count_ = 0;
  uint32_t max_task_count_ = 0;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_STREAM_ASSIGN_H_
