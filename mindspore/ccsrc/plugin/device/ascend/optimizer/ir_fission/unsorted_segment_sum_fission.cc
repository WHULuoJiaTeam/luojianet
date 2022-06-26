/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/unsorted_segment_sum_fission.h"
#include <memory>
#include <vector>
#include <algorithm>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace opt {
namespace {
bool CheckInputs(const CNodePtr &origin_node) {
  MS_EXCEPTION_IF_NULL(origin_node);
  if (common::AnfAlgo::GetInputTensorNum(origin_node) != kUnsortedSegmentSumInputTensorNum) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum has wrong inputs num, not equal " << kUnsortedSegmentSumInputTensorNum
                  << ". CNode= " << origin_node->DebugString();
    return false;
  }
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  auto y_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 1);
  if (x_shape.empty() || y_shape.empty()) {
    return false;
  }
  if (x_shape[x_shape.size() - 1] != 1) {
    MS_LOG(DEBUG) << "UnsortedSegmentSum is not need fission. The last value of input0's shape is "
                  << x_shape[x_shape.size() - 1];
    return false;
  }
  return x_shape.size() > y_shape.size();
}
}  // namespace

CNodePtr UnsortSegmentSumFission::CreatePadding(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                                const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  std::vector<AnfNodePtr> padding_inputs = {NewValueNode(std::make_shared<Primitive>(kPaddingOpName)),
                                            origin_node->input(kIndex1)};
  auto padding = NewCNode(padding_inputs, graph);
  MS_EXCEPTION_IF_NULL(padding);
  padding->set_scope(origin_node->scope());
  auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = pad_dim_size;
  if (AnfUtils::IsShapeDynamic(shape)) {
    auto min_shape = common::AnfAlgo::GetInputMinShape(origin_node, 0);
    auto max_shape = common::AnfAlgo::GetInputMaxShape(origin_node, 0);
    min_shape[shape.size() - 1] = SizeToLong(pad_dim_size);
    max_shape[shape.size() - 1] = SizeToLong(pad_dim_size);
    ShapeVector shape_tmp;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape_tmp), SizeToLong);
    BaseShapePtr base_shape = std::make_shared<abstract::Shape>(shape_tmp, min_shape, max_shape);
    common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0)},
                                                 {base_shape}, padding.get());
  } else {
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0)},
                                                {shape}, padding.get());
  }
  common::AnfAlgo::SetNodeAttr(kAttrPadDimSize, MakeValue(SizeToLong(pad_dim_size)), padding);
  return padding;
}

CNodePtr UnsortSegmentSumFission::CreateUnsortedSegmentSum(const FuncGraphPtr &graph, const CNodePtr &origin_node,
                                                           const CNodePtr &padding, const size_t &pad_dim_size) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(origin_node);
  MS_EXCEPTION_IF_NULL(padding);
  std::vector<AnfNodePtr> unsorted_segment_sum8_inputs = {
    NewValueNode(std::make_shared<Primitive>(prim::kPrimUnsortedSegmentSum->name())), padding,
    origin_node->input(kIndex2)};
  auto unsorted_segment_sum = NewCNode(unsorted_segment_sum8_inputs, graph);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum);
  unsorted_segment_sum->set_scope(origin_node->scope());
  auto shape = common::AnfAlgo::GetOutputInferShape(origin_node, 0);
  shape[shape.size() - 1] = pad_dim_size;
  if (AnfUtils::IsShapeDynamic(shape)) {
    auto min_shape = common::AnfAlgo::GetOutputMinShape(origin_node, 0);
    auto max_shape = common::AnfAlgo::GetInputMaxShape(origin_node, 0);
    min_shape[shape.size() - 1] = SizeToLong(pad_dim_size);
    max_shape[shape.size() - 1] = SizeToLong(pad_dim_size);
    ShapeVector shape_tmp;
    std::transform(shape.begin(), shape.end(), std::back_inserter(shape_tmp), SizeToLong);
    BaseShapePtr base_shape = std::make_shared<abstract::Shape>(shape_tmp, min_shape, max_shape);
    common::AnfAlgo::SetOutputTypeAndDetailShape({common::AnfAlgo::GetOutputInferDataType(origin_node, 0)},
                                                 {base_shape}, unsorted_segment_sum.get());
  } else {
    common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(origin_node, 0)}, {shape},
                                                unsorted_segment_sum.get());
  }

  common::AnfAlgo::SetNodeAttr(kAttrNumSegments, MakeValue(SizeToLong(shape[0])), unsorted_segment_sum);
  return unsorted_segment_sum;
}

CNodePtr UnsortSegmentSumFission::CreateSlice(const FuncGraphPtr &graph, const CNodePtr &unsort_segment_sum,
                                              const CNodePtr &unsorted_segment_sum8) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(unsort_segment_sum);
  MS_EXCEPTION_IF_NULL(unsorted_segment_sum8);
  std::vector<AnfNodePtr> slice_inputs = {NewValueNode(std::make_shared<Primitive>(kSliceOpName)),
                                          unsorted_segment_sum8};
  auto slice = NewCNode(slice_inputs, graph);
  MS_EXCEPTION_IF_NULL(slice);
  slice->set_scope(unsort_segment_sum->scope());
  slice->set_abstract(unsort_segment_sum->abstract());
  auto unsort_segment_sum_shape = common::AnfAlgo::GetOutputInferShape(unsort_segment_sum, 0);
  std::vector<size_t> offsets(unsort_segment_sum_shape.size(), 0);
  common::AnfAlgo::SetNodeAttr(kAttrBegin, MakeValue(Convert2Long(offsets)), slice);
  common::AnfAlgo::SetNodeAttr(kAttrSize, MakeValue(Convert2Long(unsort_segment_sum_shape)), slice);
  return slice;
}

const BaseRef UnsortSegmentSumFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  VectorRef pattern({prim::kPrimUnsortedSegmentSum, Xs});
  return pattern;
}

const AnfNodePtr UnsortSegmentSumFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto origin_node = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_node);
  if (!CheckInputs(origin_node)) {
    return nullptr;
  }
  size_t pad_dim_size;
  auto input_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(origin_node, 0);
  constexpr auto PADSIZE32 = 8;
  constexpr auto PADSIZE16 = 16;
  if (input_dtype == kNumberTypeFloat32) {
    pad_dim_size = PADSIZE32;
  } else if (input_dtype == kNumberTypeFloat16) {
    pad_dim_size = PADSIZE16;
  } else {
    MS_LOG(DEBUG) << "UnsortedSegmentSum data type not in (float32, float16), no need change";
    return nullptr;
  }

  auto padding = CreatePadding(graph, origin_node, pad_dim_size);
  auto unsorted_segment_sum8 = CreateUnsortedSegmentSum(graph, origin_node, padding, pad_dim_size);
  return CreateSlice(graph, origin_node, unsorted_segment_sum8);
}
}  // namespace opt
}  // namespace mindspore
