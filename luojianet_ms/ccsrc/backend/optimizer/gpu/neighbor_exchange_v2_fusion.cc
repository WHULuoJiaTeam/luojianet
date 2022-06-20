/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "backend/optimizer/gpu/neighbor_exchange_v2_fusion.h"

#include <algorithm>
#include <vector>
#include <string>

#include "backend/session/anf_runtime_algorithm.h"
#include "ir/primitive.h"
#include "utils/utils.h"
#include "runtime/device/gpu/kernel_info_setter.h"
#include "backend/optimizer/common/helper.h"
#include "backend/kernel_compiler/gpu/nccl/nccl_gpu_kernel.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kCNodePrimitiveIdx = 0;
constexpr size_t kNeighborExchangeV2InputIdx = 1;
constexpr size_t kLenTopIdx = 0;
constexpr size_t kLenBottomIdx = 1;
constexpr size_t kLenLeftIdx = 2;
constexpr size_t kLenRightIdx = 3;
constexpr size_t kHDim = 2;  // dim of H in NCHW
constexpr size_t kWDim = 3;  // dim of W in NCHW
constexpr int64_t kShapeSize = 4;
constexpr int64_t kRankIdZero = 0;
constexpr int64_t kRankIdOne = 1;
constexpr int64_t kRankIdTwo = 2;
constexpr int64_t kRankIdThree = 3;
constexpr int64_t kRankIdFour = 4;
constexpr int64_t kRankIdFive = 5;
constexpr int64_t kRankIdSix = 6;
constexpr int64_t kRankIdSeven = 7;
constexpr size_t kSizeFour = 4;
constexpr int64_t kInvalidId = -1;
constexpr int64_t kMinimumSplitNum = 2;

bool IsTop(const std::vector<int64_t> &send_rank_ids) {
  return send_rank_ids[kRankIdZero] != kInvalidId || send_rank_ids[kRankIdOne] != kInvalidId ||
         send_rank_ids[kRankIdSeven] != kInvalidId;
}

bool IsBottom(const std::vector<int64_t> &send_rank_ids) {
  return send_rank_ids[kRankIdThree] != kInvalidId || send_rank_ids[kRankIdFour] != kInvalidId ||
         send_rank_ids[kRankIdFive] != kInvalidId;
}

void AddNewOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &new_splitv, int64_t outputs_num,
                   std::vector<AnfNodePtr> *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  std::vector<AnfNodePtr> new_splitv_output;
  CreateMultipleOutputsOfAnfNode(func_graph, new_splitv, LongToSize(outputs_num), &new_splitv_output);
  inputs->insert(inputs->end(), new_splitv_output.begin(), new_splitv_output.end());
}

// cal split attrs size_splits, shapes and num_split
void CalSplitAttrs(SplitvNodeInfo *splitv_node_info) {
  MS_EXCEPTION_IF_NULL(splitv_node_info);
  splitv_node_info->size_splits.clear();
  splitv_node_info->shapes.clear();
  if (SizeToLong(splitv_node_info->base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong base_shape size: " << splitv_node_info->base_shape.size()
                      << ", it should be equal to 4.";
  }
  if (splitv_node_info->split_dim >= kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong split_dim: " << splitv_node_info->split_dim << ", it should less than 4.";
  }
  int64_t num_split = 0;
  int64_t split_middle_size = splitv_node_info->base_shape[splitv_node_info->split_dim];
  std::vector<size_t> shape_tmp(splitv_node_info->base_shape);
  // [top, bottom, left, right]
  int64_t first_size =
    splitv_node_info->split_dim == kWDim ? splitv_node_info->send_lens[kDim2] : splitv_node_info->send_lens[0];
  int64_t last_size =
    splitv_node_info->split_dim == kWDim ? splitv_node_info->send_lens[kDim3] : splitv_node_info->send_lens[1];

  if (splitv_node_info->is_first) {
    // first
    ++num_split;
    splitv_node_info->size_splits.push_back(first_size);
    split_middle_size -= first_size;
    shape_tmp[splitv_node_info->split_dim] = static_cast<size_t>(first_size);
    splitv_node_info->shapes.push_back(shape_tmp);
  }
  if (splitv_node_info->is_last) {
    // middle
    split_middle_size -= last_size;
    if (split_middle_size > 0) {
      ++num_split;
      splitv_node_info->size_splits.push_back(split_middle_size);
      shape_tmp[splitv_node_info->split_dim] = static_cast<size_t>(split_middle_size);
      splitv_node_info->shapes.push_back(shape_tmp);
    }
    // last
    ++num_split;
    splitv_node_info->size_splits.push_back(last_size);
    shape_tmp[splitv_node_info->split_dim] = static_cast<size_t>(last_size);
    splitv_node_info->shapes.push_back(shape_tmp);
  } else if (split_middle_size > 0) {
    ++num_split;
    splitv_node_info->size_splits.push_back(split_middle_size);
    shape_tmp[splitv_node_info->split_dim] = static_cast<size_t>(split_middle_size);
    splitv_node_info->shapes.push_back(shape_tmp);
  }
  splitv_node_info->num_split = num_split;
}

CNodePtr CreateSliceNode(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &slice_input,
                         const SliceNodeInfo slice_node_info, const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  if (slice_input.empty()) {
    MS_LOG(EXCEPTION) << "The input is empty, can not create slice node.";
    return nullptr;
  }

  auto slice = pass.NewCNode(slice_input, graph);
  MS_EXCEPTION_IF_NULL(slice);
  std::vector<size_t> slice_shape(slice_node_info.base_shape);
  std::vector<size_t> begins(slice_shape.size(), 0);

  slice_shape[slice_node_info.slice_dim] = slice_node_info.slice_size;
  begins[slice_node_info.slice_dim] = slice_node_info.slice_begin;

  std::vector<std::vector<size_t>> shapes = {slice_shape};
  std::vector<TypeId> dtypes(1, slice_node_info.input_dtype);
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, slice.get());
  AnfAlgo::SetNodeAttr(kAttrBegin, MakeValue(Convert2Long(begins)), slice);
  AnfAlgo::SetNodeAttr(kAttrSize, MakeValue(Convert2Long(slice_shape)), slice);
  return slice;
}

CNodePtr CreateSplitNode(const FuncGraphPtr &graph, const AnfNodePtr &split_input, SplitvNodeInfo *splitv_node_info,
                         const PatternProcessPass &pass) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(splitv_node_info);

  CalSplitAttrs(splitv_node_info);

  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};

  CNodePtr slice = nullptr;
  AbstractBasePtrList abstract_list;

  // initialize slice_node_info
  SliceNodeInfo slice_node_info;
  slice_node_info.base_shape = splitv_node_info->base_shape;
  slice_node_info.slice_dim = splitv_node_info->split_dim;
  slice_node_info.input_dtype = splitv_node_info->dtype;

  for (int64_t idx = 0; idx < splitv_node_info->num_split; ++idx) {
    slice_node_info.slice_size = splitv_node_info->size_splits[idx];
    std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)), split_input};
    slice = CreateSliceNode(graph, slice_inputs, slice_node_info, pass);
    AddNewOutputs(graph, slice, 1, &make_tuple_inputs);
    abstract_list.emplace_back(slice->abstract());
    slice_node_info.slice_begin += splitv_node_info->size_splits[idx];
  }
  if (SizeToLong(make_tuple_inputs.size()) - 1 != splitv_node_info->num_split) {
    MS_LOG(EXCEPTION) << "Wrong SplitV outputs size: " << make_tuple_inputs.size() << ", it should be equal to  "
                      << splitv_node_info->num_split << ".";
  }
  CNodePtr make_tuple = graph->NewCNode(make_tuple_inputs);
  make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  AnfAlgo::SetNodeAttr(kAttrSizeSplits, MakeValue<std::vector<int64_t>>(splitv_node_info->size_splits), make_tuple);
  return make_tuple;
}

std::vector<std::vector<size_t>> CalAllToAllvOutputShape(const std::vector<size_t> &base_shape,
                                                         const std::vector<int64_t> &recv_lens,
                                                         const std::vector<int64_t> &recv_rank_ids) {
  if (SizeToLong(base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Wrong base_shape size: " << base_shape.size() << ", it should be equal to 4.";
  }
  std::vector<std::vector<size_t>> shapes = {};
  std::vector<std::vector<size_t>> ori_shapes = {
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenTopIdx]), base_shape[kWDim]},
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenTopIdx]),
     static_cast<size_t>(recv_lens[kLenRightIdx])},
    {base_shape[0], base_shape[1], base_shape[kHDim], static_cast<size_t>(recv_lens[kLenRightIdx])},
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenBottomIdx]),
     static_cast<size_t>(recv_lens[kLenRightIdx])},
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenBottomIdx]), base_shape[kWDim]},
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenBottomIdx]),
     static_cast<size_t>(recv_lens[kLenLeftIdx])},
    {base_shape[0], base_shape[1], base_shape[kHDim], static_cast<size_t>(recv_lens[kLenLeftIdx])},
    {base_shape[0], base_shape[1], static_cast<size_t>(recv_lens[kLenTopIdx]),
     static_cast<size_t>(recv_lens[kLenLeftIdx])}};

  for (size_t idx = 0; idx < recv_rank_ids.size(); ++idx) {
    if (recv_rank_ids[idx] != kInvalidId) {
      shapes.push_back(ori_shapes[idx]);
    }
  }

  return shapes;
}

std::vector<AnfNodePtr> CreateAllToAllvInput(const std::vector<std::vector<AnfNodePtr>> &split_outputs,
                                             const std::vector<int64_t> &send_rank_ids) {
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllVOpName))};
  const std::vector<size_t> split_idx = {0, 2, 1, 3, 0, 3, 1, 2};
  const std::vector<bool> is_begin = {true, false, false, false, false, true, true, true};
  for (size_t idx = 0; idx < send_rank_ids.size(); ++idx) {
    if (send_rank_ids[idx] == kInvalidId) {
      continue;
    }
    if (is_begin[idx]) {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[split_idx[idx]].begin(),
                                split_outputs[split_idx[idx]].begin() + 1);
    } else {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[split_idx[idx]].end() - 1,
                                split_outputs[split_idx[idx]].end());
    }
  }
  return all_to_all_v_input;
}

// get center of input for grad
AnfNodePtr GetCenter(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_grad,
                     const std::vector<CNodePtr> &split_nodes, const std::vector<int64_t> &split_num,
                     const std::vector<int64_t> &send_rank_ids) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<AnfNodePtr> output;
  if (split_nodes[kRankIdTwo] == nullptr) {
    if (split_nodes[0] == nullptr) {
      return neighbor_exchange_v2_grad->input(kNeighborExchangeV2InputIdx);
    }
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>(split_num[0]), &output);
    if (output.size() < kMinimumSplitNum) {
      MS_LOG(EXCEPTION) << "Wrong split output size: " << output.size() << ", except size >= 2.";
    }
    if (send_rank_ids[kRankIdZero] == kInvalidId) {
      return output[0];
    }
    return output[1];
  }

  // split_nodes[kRankIdTwo] != nullptr
  CreateMultipleOutputsOfAnfNode(graph, split_nodes[kDim2], static_cast<size_t>(split_num[kDim2]), &output);
  if (output.size() < kMinimumSplitNum) {
    MS_LOG(EXCEPTION) << "Wrong split output size: " << output.size() << ", except size >= 2.";
  }
  if (send_rank_ids[kRankIdSix] == kInvalidId) {
    return output[0];
  }
  return output[1];
}

std::vector<AnfNodePtr> CreateAllToAllvInputForGrad(const std::vector<int64_t> &send_rank_ids,
                                                    const std::vector<std::vector<AnfNodePtr>> &split_outputs,
                                                    const std::vector<CNodePtr> &split_nodes) {
  if (send_rank_ids.size() != 8) {
    MS_LOG(EXCEPTION) << "Wrong send_rank_ids size: " << send_rank_ids.size() << ", expect size: 8.";
  }
  if (split_outputs.size() != kSizeFour) {
    MS_LOG(EXCEPTION) << "Wrong split_outputs size: " << split_outputs.size() << ", expect size: 4.";
  }
  if (split_nodes.size() != kSizeFour) {
    MS_LOG(EXCEPTION) << "Wrong split_nodes size: " << split_nodes.size() << ", expect size: 4.";
  }
  std::vector<AnfNodePtr> all_to_all_v_input = {NewValueNode(std::make_shared<Primitive>(kAllToAllVOpName))};
  // only have top-bottom split
  std::vector<size_t> side_idx = {1, 2, 3, 5, 6, 7};
  bool no_send_side = std::all_of(side_idx.begin(), side_idx.end(),
                                  [&send_rank_ids](int idx) { return send_rank_ids[idx] == kInvalidId; });
  if (no_send_side) {
    if (send_rank_ids[kRankIdZero] != kInvalidId) {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[0].begin(), split_outputs[0].begin() + 1);
    }
    if (send_rank_ids[kRankIdFour] != kInvalidId) {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[0].end() - 1, split_outputs[0].end());
    }
    return all_to_all_v_input;
  }
  // 0, 1
  if (split_nodes[1] != nullptr) {
    if (send_rank_ids[kRankIdSeven] != kInvalidId) {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin() + 1, split_outputs[1].end());
    } else {
      all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin(), split_outputs[1].end());
    }
  }
  // 2
  if (split_nodes[kIndex2] != nullptr && send_rank_ids[kRankIdTwo] != kInvalidId) {
    all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex2].end() - 1, split_outputs[kIndex2].end());
  }
  // 3, 4, 5
  if (split_nodes[kIndex3] != nullptr) {
    all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex3].rbegin(), split_outputs[kIndex3].rend());
  }
  // 6
  if (split_nodes[kIndex2] != nullptr && send_rank_ids[kRankIdSix] != kInvalidId) {
    all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[kIndex2].begin(),
                              split_outputs[kIndex2].begin() + 1);
  }
  // 7
  if (split_nodes[1] != nullptr && send_rank_ids[kRankIdSeven] != kInvalidId) {
    all_to_all_v_input.insert(all_to_all_v_input.end(), split_outputs[1].begin(), split_outputs[1].begin() + 1);
  }

  return all_to_all_v_input;
}

// alltoallv for forward & grad
CNodePtr CreateAllToAllvNode(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2_or_grad,
                             const std::vector<CNodePtr> &split_nodes, const std::vector<int64_t> &split_num,
                             bool is_grad, const PatternProcessPass &pass) {
  MS_LOG(DEBUG) << "Start to create alltoallv node.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_or_grad);
  std::vector<int64_t> send_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrSendRankIds);
  std::vector<int64_t> recv_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_or_grad, kAttrRecvLens);
  std::string group = AnfAlgo::GetNodeAttr<std::string>(neighbor_exchange_v2_or_grad, kAttrGroup);

  // get split nodes output, split_outputs: [top_bottom, left_right, top_corner, bottom_corner]
  std::vector<std::vector<AnfNodePtr>> split_outputs;
  for (size_t i = 0; i < split_nodes.size(); ++i) {
    std::vector<AnfNodePtr> output;
    if (split_nodes[i] != nullptr) {
      CreateMultipleOutputsOfAnfNode(graph, split_nodes[i], static_cast<size_t>(split_num[i]), &output);
      if (output.empty()) {
        MS_LOG(EXCEPTION) << "The node " << split_nodes[i]->DebugString()
                          << " should have at least one output, but got 0. trace: "
                          << trace::DumpSourceLines(split_nodes[i]);
      }
    }
    split_outputs.emplace_back(output);
  }

  // all_to_all_v input
  std::vector<AnfNodePtr> all_to_all_v_input;
  AnfNodePtr base_node = nullptr;
  if (is_grad) {
    all_to_all_v_input = CreateAllToAllvInputForGrad(send_rank_ids, split_outputs, split_nodes);
    base_node = GetCenter(graph, neighbor_exchange_v2_or_grad, split_nodes, split_num, send_rank_ids);
  } else {
    all_to_all_v_input = CreateAllToAllvInput(split_outputs, send_rank_ids);
    base_node = neighbor_exchange_v2_or_grad->input(kNeighborExchangeV2InputIdx);
  }

  // output shapes and dtypes
  auto base_dtype = AnfAlgo::GetOutputInferDataType(base_node, 0);
  auto base_shape = AnfAlgo::GetOutputInferShape(base_node, 0);
  if (SizeToLong(base_shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Invalid shape size " << base_shape.size() << ", only support NCHW input now!";
  }
  std::vector<std::vector<size_t>> shapes = CalAllToAllvOutputShape(base_shape, recv_lens, recv_rank_ids);

  // erase -1 in send_rank_ids
  std::vector<int64_t> real_send_rank_ids(send_rank_ids.size());
  std::vector<int64_t> real_recv_rank_ids(recv_rank_ids.size());
  auto iter1 = std::copy_if(send_rank_ids.begin(), send_rank_ids.end(), real_send_rank_ids.begin(),
                            [](const int64_t item) { return item != kInvalidId; });
  auto iter2 = std::copy_if(recv_rank_ids.begin(), recv_rank_ids.end(), real_recv_rank_ids.begin(),
                            [](const int64_t item) { return item != kInvalidId; });
  real_send_rank_ids.resize(std::distance(real_send_rank_ids.begin(), iter1));
  real_recv_rank_ids.resize(std::distance(real_recv_rank_ids.begin(), iter2));

  std::vector<TypeId> dtypes(real_recv_rank_ids.size(), base_dtype);

  // create alltoallv node
  auto all_to_all_v = pass.NewCNode(all_to_all_v_input, graph);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, all_to_all_v.get());

  AnfAlgo::SetNodeAttr(kAttrSendRankIds, MakeValue<std::vector<int64_t>>(real_send_rank_ids), all_to_all_v);
  AnfAlgo::SetNodeAttr(kAttrRecvRankIds, MakeValue<std::vector<int64_t>>(real_recv_rank_ids), all_to_all_v);
  AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(group), all_to_all_v);
  MS_LOG(INFO) << "Create AllToAllv success, send rank size " << send_rank_ids.size() << ", recv rank size "
               << recv_rank_ids.size();
  return all_to_all_v;
}

int64_t AllToAllRealIds(int64_t ids, const std::vector<int64_t> &recv_rank_ids) {
  int64_t real_ids = 0;
  for (auto i = 0; i < ids; ++i) {
    if (recv_rank_ids[i] != kInvalidId) {
      ++real_ids;
    }
  }
  return real_ids;
}
}  // namespace

// returns {top_bottom, left_right, top_corner, bottom_corner}, if no split, set it nullptr
std::vector<CNodePtr> NeighborExchangeV2Fusion::CreateSplitNodes(const FuncGraphPtr &graph,
                                                                 const CNodePtr &neighbor_exchange_v2,
                                                                 std::vector<int64_t> *split_num) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(split_num);
  std::vector<int64_t> send_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrSendRankIds);
  std::vector<int64_t> send_lens = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrSendLens);

  if (neighbor_exchange_v2->size() <= kNeighborExchangeV2InputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << neighbor_exchange_v2->DebugString() << " input size "
                      << neighbor_exchange_v2->size() << ", should be " << kNeighborExchangeV2InputIdx
                      << trace::DumpSourceLines(neighbor_exchange_v2);
  }
  std::vector<CNodePtr> split_nodes = {};

  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  auto dtype = AnfAlgo::GetOutputInferDataType(neighbor_exchange_v2_input, 0);
  auto shape = AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0);

  // Splitv for top & bottom
  SplitvNodeInfo splitv_node_info;
  splitv_node_info.is_first = IsTop(send_rank_ids);
  splitv_node_info.is_last = IsBottom(send_rank_ids);
  splitv_node_info.split_dim = kHDim;
  splitv_node_info.send_lens = send_lens;
  splitv_node_info.dtype = dtype;
  splitv_node_info.base_shape = shape;
  if (SizeToLong(splitv_node_info.base_shape.size()) != kShapeSize) {  // only support NCHW now
    MS_LOG(EXCEPTION) << "Invalid shape size " << splitv_node_info.base_shape.size() << ", only support NCHW input now!"
                      << trace::DumpSourceLines(neighbor_exchange_v2);
  }
  CNodePtr split_v_top_bottom = nullptr;
  if (splitv_node_info.is_first || splitv_node_info.is_last) {
    split_v_top_bottom = CreateSplitNode(graph, neighbor_exchange_v2_input, &splitv_node_info, *this);
  }
  split_nodes.emplace_back(split_v_top_bottom);
  split_num->push_back(splitv_node_info.num_split);

  // Splitv for left & right
  splitv_node_info.is_first = (send_rank_ids[kRankIdSix] != kInvalidId);
  splitv_node_info.is_last = (send_rank_ids[kRankIdTwo] != kInvalidId);
  splitv_node_info.split_dim = kWDim;
  CNodePtr split_v_left_right = nullptr;
  if (splitv_node_info.is_first || splitv_node_info.is_last) {
    split_v_left_right = CreateSplitNode(graph, neighbor_exchange_v2_input, &splitv_node_info, *this);
  }
  split_nodes.emplace_back(split_v_left_right);
  split_num->push_back(splitv_node_info.num_split);

  // splitv for corner
  if ((send_rank_ids[kRankIdOne] != kInvalidId) || (send_rank_ids[kRankIdSeven] != kInvalidId) ||
      (send_rank_ids[kRankIdThree] != kInvalidId) || (send_rank_ids[kRankIdFive] != kInvalidId)) {
    // top_bottom_split outputs
    std::vector<AnfNodePtr> split_outputs_top_bottom;
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>((*split_num)[0]),
                                   &split_outputs_top_bottom);
    if (split_outputs_top_bottom.empty()) {
      MS_LOG(EXCEPTION) << "The node " << split_nodes[0]->DebugString() << " should have at least one output, but got 0"
                        << trace::DumpSourceLines(split_nodes[0]);
    }

    // for top corner
    if ((send_rank_ids[kRankIdOne] != kInvalidId) || (send_rank_ids[kRankIdSeven] != kInvalidId)) {
      auto shape_tmp(shape);
      shape_tmp[kHDim] = send_lens[0];
      splitv_node_info.is_first = (send_rank_ids[kRankIdSeven] != kInvalidId);
      splitv_node_info.is_last = (send_rank_ids[kRankIdOne] != kInvalidId);
      splitv_node_info.base_shape = shape_tmp;
      CNodePtr split_v_corner_top =
        CreateSplitNode(graph, *(split_outputs_top_bottom.begin()), &splitv_node_info, *this);
      split_nodes.emplace_back(split_v_corner_top);
      split_num->push_back(splitv_node_info.num_split);
    } else {
      split_nodes.emplace_back(nullptr);
      split_num->push_back(0);
    }
    // for bottom corner
    if ((send_rank_ids[kRankIdThree] != kInvalidId) || (send_rank_ids[kRankIdFive] != kInvalidId)) {
      auto shape_tmp(shape);
      shape_tmp[kHDim] = send_lens[1];
      splitv_node_info.is_first = (send_rank_ids[kRankIdFive] != kInvalidId);
      splitv_node_info.is_last = (send_rank_ids[kRankIdThree] != kInvalidId);
      splitv_node_info.base_shape = shape_tmp;
      CNodePtr split_v_corner_bottom =
        CreateSplitNode(graph, *(split_outputs_top_bottom.end() - 1), &splitv_node_info, *this);
      split_nodes.emplace_back(split_v_corner_bottom);
      split_num->push_back(splitv_node_info.num_split);
    } else {
      split_nodes.emplace_back(nullptr);
      split_num->push_back(0);
    }
  } else {
    split_nodes.emplace_back(nullptr);
    split_num->push_back(0);
    split_nodes.emplace_back(nullptr);
    split_num->push_back(0);
  }
  return split_nodes;
}

CNodePtr NeighborExchangeV2Fusion::CreateConcatNode(const FuncGraphPtr &graph,
                                                    const std::vector<AnfNodePtr> &concat_input,
                                                    const std::vector<std::vector<size_t>> &output_shape,
                                                    const std::vector<TypeId> &output_dtype, int64_t axis,
                                                    int64_t input_nums) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto concat = NewCNode(concat_input, graph);
  MS_EXCEPTION_IF_NULL(concat);
  AnfAlgo::SetOutputInferTypeAndShape(output_dtype, output_shape, concat.get());
  AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue<int64_t>(axis), concat);
  AnfAlgo::SetNodeAttr(kAttrInputNums, MakeValue(input_nums), concat);
  std::vector<int64_t> dyn_input_size_empty{input_nums};
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue(dyn_input_size_empty), concat);
  return concat;
}

CNodePtr NeighborExchangeV2Fusion::CreateLeftRightConcat(const FuncGraphPtr &graph,
                                                         const std::vector<AnfNodePtr> &all_to_all_v_outputs,
                                                         const std::vector<int64_t> &recv_rank_ids,
                                                         const std::vector<int64_t> &recv_lens, bool is_left) const {
  MS_EXCEPTION_IF_NULL(graph);

  std::vector<AnfNodePtr> concat_input = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  int64_t input_num = 1;
  size_t first_ids = is_left ? 7 : 1;
  size_t middle_ids = is_left ? 6 : 2;
  size_t last_ids = is_left ? 5 : 3;

  auto single_shape = AnfAlgo::GetOutputInferShape(all_to_all_v_outputs[AllToAllRealIds(middle_ids, recv_rank_ids)], 0);

  if (recv_rank_ids[first_ids] != kInvalidId) {
    ++input_num;
    single_shape[kDim2] += static_cast<size_t>(recv_lens[0]);  // H in NCHW
  }
  if (recv_rank_ids[last_ids] != kInvalidId) {
    ++input_num;
    single_shape[kDim2] += static_cast<size_t>(recv_lens[1]);  // H in NCHW
  }
  if (is_left) {
    concat_input.insert(concat_input.end(), all_to_all_v_outputs.rbegin(), all_to_all_v_outputs.rbegin() + input_num);
  } else {
    concat_input.insert(concat_input.end(), all_to_all_v_outputs.begin() + AllToAllRealIds(1, recv_rank_ids),
                        all_to_all_v_outputs.begin() + input_num + AllToAllRealIds(1, recv_rank_ids));
  }

  std::vector<TypeId> concat_output_dtype = {
    AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[AllToAllRealIds(middle_ids, recv_rank_ids)], 0)};
  auto concat = CreateConcatNode(graph, concat_input, {single_shape}, concat_output_dtype, kHDim, input_num);

  return concat;
}

CNodePtr NeighborExchangeV2Fusion::CreateMiddleConcat(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                                                      const std::vector<AnfNodePtr> &all_to_all_v_outputs,
                                                      const std::vector<int64_t> &recv_rank_ids,
                                                      const std::vector<int64_t> &recv_lens, int64_t concat_dim) const {
  std::vector<AnfNodePtr> concat_input_all = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  int64_t input_num_all = 0;
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  auto single_shape = AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0);
  size_t first_idx = concat_dim == kWDim ? 6 : 0;
  size_t last_idx = concat_dim == kWDim ? 2 : 4;
  size_t first_len = concat_dim == kWDim ? static_cast<size_t>(recv_lens[kDim2]) : static_cast<size_t>(recv_lens[0]);
  size_t last_len = concat_dim == kWDim ? static_cast<size_t>(recv_lens[kDim3]) : static_cast<size_t>(recv_lens[1]);

  // left
  if (recv_rank_ids[first_idx] != kInvalidId) {
    if (concat_dim == kWDim) {
      concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.end() - 1, all_to_all_v_outputs.end());
    } else {
      concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin(), all_to_all_v_outputs.begin() + 1);
    }

    ++input_num_all;
    single_shape[concat_dim] += first_len;
  }

  concat_input_all.push_back(neighbor_exchange_v2_input);
  ++input_num_all;
  // right
  if (recv_rank_ids[last_idx] != kInvalidId) {
    if (concat_dim == kWDim) {
      concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin(), all_to_all_v_outputs.begin() + 1);
    } else {
      int64_t bottom_num = AllToAllRealIds(4, recv_rank_ids);
      concat_input_all.insert(concat_input_all.end(), all_to_all_v_outputs.begin() + bottom_num,
                              all_to_all_v_outputs.begin() + bottom_num + 1);
    }

    ++input_num_all;
    single_shape[concat_dim] += last_len;
  }

  std::vector<TypeId> concat_output_dtype = {AnfAlgo::GetOutputInferDataType(all_to_all_v_outputs[0], 0)};
  auto concat_all =
    CreateConcatNode(graph, concat_input_all, {single_shape}, concat_output_dtype, concat_dim, input_num_all);
  return concat_all;
}

CNodePtr NeighborExchangeV2Fusion::AllToAllvRecvEmpty(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                                                      const CNodePtr &all_to_all_v) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  // add depend for input & alltoallv
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                          neighbor_exchange_v2_input, all_to_all_v};
  auto depend = graph->NewCNode(depend_input);
  MS_EXCEPTION_IF_NULL(depend);
  depend->set_abstract(neighbor_exchange_v2_input->abstract());
  return depend;
}

CNodePtr NeighborExchangeV2Fusion::CreateConcatNodes(const FuncGraphPtr &graph, const CNodePtr &neighbor_exchange_v2,
                                                     const CNodePtr &all_to_all_v) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  MS_EXCEPTION_IF_NULL(all_to_all_v);
  std::vector<int64_t> recv_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2, kAttrRecvLens);

  int64_t all_to_all_output_num =
    std::count_if(recv_rank_ids.begin(), recv_rank_ids.end(), [](int64_t ids) { return ids != kInvalidId; });
  if (all_to_all_output_num == 0) {
    return AllToAllvRecvEmpty(graph, neighbor_exchange_v2, all_to_all_v);
  }

  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(all_to_all_output_num),
                                 &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(all_to_all_v);
  }

  if (recv_rank_ids[kRankIdZero] == kInvalidId && recv_rank_ids[kRankIdFour] == kInvalidId) {
    return CreateMiddleConcat(graph, neighbor_exchange_v2, all_to_all_v_outputs, recv_rank_ids, recv_lens, kWDim);
  }

  // top or bottom
  // middle concat
  auto concat_middle =
    CreateMiddleConcat(graph, neighbor_exchange_v2, all_to_all_v_outputs, recv_rank_ids, recv_lens, kHDim);

  bool is_left = recv_rank_ids[kRankIdSix] != kInvalidId || recv_rank_ids[kRankIdFive] != kInvalidId ||
                 recv_rank_ids[kRankIdSeven] != kInvalidId;
  bool is_right = recv_rank_ids[kRankIdOne] != kInvalidId || recv_rank_ids[kRankIdTwo] != kInvalidId ||
                  recv_rank_ids[kRankIdThree] != kInvalidId;
  if (!is_left && !is_right) {
    return concat_middle;
  }

  std::vector<AnfNodePtr> concat_input_all = {NewValueNode(std::make_shared<Primitive>(kConcatOpName))};
  auto neighbor_exchange_v2_input = neighbor_exchange_v2->input(kNeighborExchangeV2InputIdx);
  std::vector<size_t> shape_all = AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_input, 0);
  shape_all[kDim2] =
    recv_rank_ids[kRankIdZero] != kInvalidId ? shape_all[kDim2] + static_cast<size_t>(recv_lens[0]) : shape_all[kDim2];
  shape_all[kDim2] =
    recv_rank_ids[kRankIdFour] != kInvalidId ? shape_all[kDim2] + static_cast<size_t>(recv_lens[1]) : shape_all[kDim2];
  int64_t input_nums_all = 0;
  // left concat
  if (is_left) {
    auto concat_left = CreateLeftRightConcat(graph, all_to_all_v_outputs, recv_rank_ids, recv_lens, true);

    // connect to concat_all
    std::vector<AnfNodePtr> concat_left_outputs;
    CreateMultipleOutputsOfAnfNode(graph, concat_left, 1, &concat_left_outputs);
    if (concat_left_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << concat_left->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(concat_left);
    }
    concat_input_all.insert(concat_input_all.end(), concat_left_outputs.begin(), concat_left_outputs.end());
    ++input_nums_all;
    shape_all[kDim3] += recv_lens[kDim2];
  }

  // middle concat connect to concat_all
  std::vector<AnfNodePtr> concat_middle_outputs;
  CreateMultipleOutputsOfAnfNode(graph, concat_middle, 1, &concat_middle_outputs);
  if (concat_middle_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << concat_middle->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(concat_middle);
  }
  concat_input_all.insert(concat_input_all.end(), concat_middle_outputs.begin(), concat_middle_outputs.end());
  ++input_nums_all;

  if (is_right) {
    auto concat_right = CreateLeftRightConcat(graph, all_to_all_v_outputs, recv_rank_ids, recv_lens, false);

    // connect to concat_all
    std::vector<AnfNodePtr> concat_right_outputs;
    CreateMultipleOutputsOfAnfNode(graph, concat_right, 1, &concat_right_outputs);
    if (concat_right_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << concat_right->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(concat_right);
    }
    concat_input_all.insert(concat_input_all.end(), concat_right_outputs.begin(), concat_right_outputs.end());
    ++input_nums_all;
    shape_all[kDim3] += recv_lens[kDim3];
  }

  std::vector<TypeId> concat_right_output_dtype = {AnfAlgo::GetOutputInferDataType(concat_input_all[1], 0)};
  auto concat_all =
    CreateConcatNode(graph, concat_input_all, {shape_all}, concat_right_output_dtype, kWDim, input_nums_all);
  return concat_all;
}

// grad
// returns {top_bottom, left_right, top_corner, bottom_corner}, if no split, set it nullptr
std::vector<CNodePtr> NeighborExchangeV2GradFusion::CreateSplitNodesForGrad(const FuncGraphPtr &graph,
                                                                            const CNodePtr &neighbor_exchange_v2_grad,
                                                                            std::vector<int64_t> *split_num) const {
  MS_LOG(DEBUG) << "Start create splitv nodes.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  MS_EXCEPTION_IF_NULL(split_num);
  std::vector<int64_t> send_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendRankIds);
  std::vector<int64_t> send_lens = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendLens);

  if (neighbor_exchange_v2_grad->size() <= kNeighborExchangeV2InputIdx) {
    MS_LOG(EXCEPTION) << "Invalid cnode " << neighbor_exchange_v2_grad->DebugString() << " input size "
                      << neighbor_exchange_v2_grad->size() << ", should be " << kNeighborExchangeV2InputIdx
                      << trace::DumpSourceLines(neighbor_exchange_v2_grad);
  }

  auto neighbor_exchange_v2_grad_input = neighbor_exchange_v2_grad->input(kNeighborExchangeV2InputIdx);
  auto dtype = AnfAlgo::GetOutputInferDataType(neighbor_exchange_v2_grad_input, 0);
  auto shape = AnfAlgo::GetOutputInferShape(neighbor_exchange_v2_grad_input, 0);
  if (SizeToLong(shape.size()) != kShapeSize) {
    MS_LOG(EXCEPTION) << "Invalid shape size " << shape.size() << ", only support NCHW input now!"
                      << trace::DumpSourceLines(neighbor_exchange_v2_grad);
  }

  std::vector<CNodePtr> split_nodes = {};
  // splitv for top & bottom

  SplitvNodeInfo splitv_h_node_info;
  splitv_h_node_info.is_first = IsTop(send_rank_ids);
  splitv_h_node_info.is_last = IsBottom(send_rank_ids);
  splitv_h_node_info.split_dim = kHDim;
  splitv_h_node_info.send_lens = send_lens;
  splitv_h_node_info.dtype = dtype;
  splitv_h_node_info.base_shape = shape;
  CNodePtr split_v_top_bottom = nullptr;

  if (splitv_h_node_info.is_first || splitv_h_node_info.is_last) {
    split_v_top_bottom = CreateSplitNode(graph, neighbor_exchange_v2_grad_input, &splitv_h_node_info, *this);
  }
  split_nodes.emplace_back(split_v_top_bottom);
  split_num->push_back(splitv_h_node_info.num_split);

  // splitvs for left & right
  // inputs
  std::vector<AnfNodePtr> split_outputs_top_bottom;
  std::vector<int64_t> size_split_h;
  if (split_nodes[0] != nullptr) {
    CreateMultipleOutputsOfAnfNode(graph, split_nodes[0], static_cast<size_t>(splitv_h_node_info.num_split),
                                   &split_outputs_top_bottom);
    if (split_outputs_top_bottom.empty()) {
      MS_LOG(EXCEPTION) << "The node " << split_nodes[0]->DebugString()
                        << " should have at least one output, but got 0." << trace::DumpSourceLines(split_nodes[0]);
    }
    size_split_h = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(split_nodes[0], kAttrSizeSplits);
  } else {
    // just middle
    split_outputs_top_bottom.push_back(neighbor_exchange_v2_grad_input);
    size_split_h.push_back(shape[kHDim]);
  }

  // left_right splitv nodes from top to bottom
  SplitvNodeInfo splitv_w_node_info;
  splitv_w_node_info.is_first = (send_rank_ids[kRankIdFive] != kInvalidId) ||
                                (send_rank_ids[kRankIdSix] != kInvalidId) ||
                                (send_rank_ids[kRankIdSeven] != kInvalidId);
  splitv_w_node_info.is_last = (send_rank_ids[kRankIdOne] != kInvalidId) || (send_rank_ids[kRankIdTwo] != kInvalidId) ||
                               (send_rank_ids[kRankIdThree] != kInvalidId);
  splitv_w_node_info.split_dim = kWDim;
  splitv_w_node_info.send_lens = send_lens;
  splitv_w_node_info.dtype = dtype;

  if (splitv_w_node_info.is_first || splitv_w_node_info.is_last) {
    // if is not top
    if (!splitv_h_node_info.is_first) {
      split_nodes.push_back(nullptr);
      split_num->push_back(0);
    }
    for (size_t i = 0; i < split_outputs_top_bottom.size(); ++i) {
      std::vector<size_t> base_shape(shape);
      base_shape[kHDim] = size_split_h[i];
      splitv_w_node_info.base_shape = base_shape;
      auto split_v_left_right = CreateSplitNode(graph, split_outputs_top_bottom[i], &splitv_w_node_info, *this);
      split_nodes.emplace_back(split_v_left_right);
      split_num->push_back(splitv_w_node_info.num_split);
    }
    // if is not bottom
    if (!splitv_h_node_info.is_last) {
      split_nodes.push_back(nullptr);
      split_num->push_back(0);
    }
  } else {
    split_nodes.push_back(nullptr);
    split_num->push_back(0);
    split_nodes.push_back(nullptr);
    split_num->push_back(0);
    split_nodes.push_back(nullptr);
    split_num->push_back(0);
  }
  MS_LOG(DEBUG) << "Create splitv nodes success.";
  return split_nodes;
}

CNodePtr NeighborExchangeV2GradFusion::CreatePadNode(const FuncGraphPtr &graph, const AnfNodePtr &input,
                                                     const std::vector<int64_t> &begin,
                                                     const std::vector<int64_t> &size, const std::vector<size_t> &shape,
                                                     TypeId dtype) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  std::vector<AnfNodePtr> pad_inputs = {NewValueNode(std::make_shared<Primitive>(kPadOpName)), input};
  auto pad = NewCNode(pad_inputs, graph);
  std::vector<std::vector<int64_t>> paddings;
  for (size_t i = 0; i < shape.size(); ++i) {
    paddings.emplace_back(std::vector<int64_t>{begin[i], static_cast<int64_t>(shape[i]) - begin[i] - size[i]});
  }
  AnfAlgo::SetOutputInferTypeAndShape({dtype}, {shape}, pad.get());
  AnfAlgo::SetNodeAttr(kAttrPaddings, MakeValue(paddings), pad);
  AnfAlgo::SetNodeAttr(kAttrInputNames, MakeValue(std::vector<std::string>{"x"}), pad);
  return pad;
}

CNodePtr NeighborExchangeV2GradFusion::CreateSplitGradNodes(const FuncGraphPtr &graph,
                                                            const CNodePtr &neighbor_exchange_v2_grad,
                                                            const CNodePtr &all_to_all_v,
                                                            const std::vector<CNodePtr> &split_nodes,
                                                            const std::vector<int64_t> &split_num) const {
  MS_LOG(DEBUG) << "Start create splitvs grad nodes.";
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<int64_t> send_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrSendRankIds);
  std::vector<int64_t> recv_rank_ids =
    AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrRecvRankIds);
  std::vector<int64_t> recv_lens = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(neighbor_exchange_v2_grad, kAttrRecvLens);

  auto centerx = GetCenter(graph, neighbor_exchange_v2_grad, split_nodes, split_num, send_rank_ids);
  auto centerx_dtype = AnfAlgo::GetOutputInferDataType(centerx, 0);
  auto centerx_shape = AnfAlgo::GetOutputInferShape(centerx, 0);
  // empty
  int64_t all_to_all_output_num =
    std::count_if(recv_rank_ids.begin(), recv_rank_ids.end(), [](int64_t ids) { return ids != kInvalidId; });
  if (all_to_all_output_num == 0) {
    // add depend(alltoallv, centerx)
    std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())),
                                            centerx, all_to_all_v};
    auto depend = graph->NewCNode(depend_input);
    MS_EXCEPTION_IF_NULL(depend);
    depend->set_abstract(centerx->abstract());
    return depend;
  }
  // get alltoallv outputs
  std::vector<AnfNodePtr> all_to_all_v_outputs;
  CreateMultipleOutputsOfAnfNode(graph, all_to_all_v, static_cast<size_t>(all_to_all_output_num),
                                 &all_to_all_v_outputs);
  if (all_to_all_v_outputs.empty()) {
    MS_LOG(EXCEPTION) << "The node " << all_to_all_v->DebugString() << " should have at least one output, but got 0."
                      << trace::DumpSourceLines(all_to_all_v);
  }
  // create pad nodes
  // slice begin & size
  std::vector<std::vector<int64_t>> begins = {{0, 0, 0, 0},
                                              {0, 0, 0, static_cast<int64_t>(centerx_shape[kDim3]) - recv_lens[kDim3]},
                                              {0, 0, 0, static_cast<int64_t>(centerx_shape[kDim3]) - recv_lens[kDim3]},
                                              {0, 0, static_cast<int64_t>(centerx_shape[kDim2]) - recv_lens[kDim1],
                                               static_cast<int64_t>(centerx_shape[kDim3]) - recv_lens[kDim3]},
                                              {0, 0, static_cast<int64_t>(centerx_shape[kDim2]) - recv_lens[kDim1], 0},
                                              {0, 0, static_cast<int64_t>(centerx_shape[kDim2]) - recv_lens[kDim1], 0},
                                              {0, 0, 0, 0},
                                              {0, 0, 0, 0}};
  std::vector<std::vector<int64_t>> sizes = {
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[0],
     static_cast<int64_t>(centerx_shape[kDim3])},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[0], recv_lens[kDim3]},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]),
     static_cast<int64_t>(centerx_shape[kDim2]), recv_lens[kDim3]},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[1], recv_lens[kDim3]},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[1],
     static_cast<int64_t>(centerx_shape[kDim3])},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[1], recv_lens[kDim2]},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]),
     static_cast<int64_t>(centerx_shape[kDim2]), recv_lens[kDim2]},
    {static_cast<int64_t>(centerx_shape[0]), static_cast<int64_t>(centerx_shape[1]), recv_lens[0], recv_lens[kDim2]}};
  std::vector<CNodePtr> pad_nodes;
  size_t output_index = 0;
  for (size_t i = 0; i < recv_rank_ids.size(); ++i) {
    if (recv_rank_ids[i] != kInvalidId) {
      auto pad =
        CreatePadNode(graph, all_to_all_v_outputs[output_index], begins[i], sizes[i], centerx_shape, centerx_dtype);
      ++output_index;
      pad_nodes.emplace_back(pad);
    }
  }

  // create add node
  std::vector<AnfNodePtr> addn_inputs = {NewValueNode(std::make_shared<Primitive>(kAddNOpName)), centerx};
  int64_t pad_num = 1;
  for (auto pad : pad_nodes) {
    std::vector<AnfNodePtr> pad_outputs;
    CreateMultipleOutputsOfAnfNode(graph, pad, 1, &pad_outputs);
    if (pad_outputs.empty()) {
      MS_LOG(EXCEPTION) << "The node " << pad->DebugString() << " should have at least one output, but got 0."
                        << trace::DumpSourceLines(pad);
    }
    addn_inputs.insert(addn_inputs.end(), pad_outputs.begin(), pad_outputs.end());
    ++pad_num;
  }
  auto addn = NewCNode(addn_inputs, graph);
  MS_EXCEPTION_IF_NULL(addn);
  AnfAlgo::SetOutputInferTypeAndShape({centerx_dtype}, {centerx_shape}, addn.get());
  AnfAlgo::SetNodeAttr(kAttrDynInputSizes, MakeValue<std::vector<int64_t>>({pad_num}), addn);
  AnfAlgo::SetNodeAttr(kAttrN, MakeValue(pad_num), addn);
  MS_LOG(DEBUG) << "Create splitvs grad nodes success.";
  return addn;
}

const BaseRef NeighborExchangeV2Fusion::DefinePattern() const {
  return VectorRef({prim::kPrimNeighborExchangeV2, std::make_shared<SeqVar>()});
}

const AnfNodePtr NeighborExchangeV2Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange_v2 = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2);
  std::vector<int64_t> split_num;
  auto split_nodes = CreateSplitNodes(graph, neighbor_exchange_v2, &split_num);
  auto all_to_all_v = CreateAllToAllvNode(graph, neighbor_exchange_v2, split_nodes, split_num, false, *this);
  auto concat = CreateConcatNodes(graph, neighbor_exchange_v2, all_to_all_v);
  return concat;
}

const BaseRef NeighborExchangeV2GradFusion::DefinePattern() const {
  return VectorRef({prim::kPrimNeighborExchangeV2Grad, std::make_shared<SeqVar>()});
}
const AnfNodePtr NeighborExchangeV2GradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                       const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto neighbor_exchange_v2_grad = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(neighbor_exchange_v2_grad);
  std::vector<int64_t> split_num;
  auto split_nodes = CreateSplitNodesForGrad(graph, neighbor_exchange_v2_grad, &split_num);
  auto all_to_all_v = CreateAllToAllvNode(graph, neighbor_exchange_v2_grad, split_nodes, split_num, true, *this);
  auto add = CreateSplitGradNodes(graph, neighbor_exchange_v2_grad, all_to_all_v, split_nodes, split_num);
  return add;
}
}  // namespace opt
}  // namespace mindspore
