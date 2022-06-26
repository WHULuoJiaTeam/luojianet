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

#include <vector>
#include "transform/graph_ir/op_declare/selection_ops_declare.h"

namespace luojianet_ms::transform {
// CumsumD
INPUT_MAP(CumsumD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumsumD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumsumD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                     {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumsumD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CumsumD, kNameCumSum, ADPT_DESC(CumsumD))

// GatherV2
INPUT_MAP(GatherV2) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(axis)}};
ATTR_MAP(GatherV2) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherV2) = {{0, OUTPUT_DESC(y)}};

// CumprodD
INPUT_MAP(CumprodD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(CumprodD) = {{2, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(CumprodD) = {{"exclusive", ATTR_DESC(exclusive, AnyTraits<bool>())},
                      {"reverse", ATTR_DESC(reverse, AnyTraits<bool>())}};
OUTPUT_MAP(CumprodD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(CumprodD, kNameCumProd, ADPT_DESC(CumprodD))

INPUT_MAP(SliceD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(SliceD) = {{2, ATTR_DESC(offsets, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())},
                          {3, ATTR_DESC(size, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(SliceD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(SliceD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(SliceD, kNameSlice, ADPT_DESC(SliceD))

// TopK
INPUT_MAP(TopK) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(k)}};
ATTR_MAP(TopK) = {{"sorted", ATTR_DESC(sorted, AnyTraits<bool>())}};
OUTPUT_MAP(TopK) = {{0, OUTPUT_DESC(values)}, {1, OUTPUT_DESC(indices)}};
REG_ADPT_DESC(TopK, kNameTopK, ADPT_DESC(TopK))

// InTopK
INPUT_MAP(InTopKD) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(InTopKD) = {{"k", ATTR_DESC(k, AnyTraits<int64_t>())}};
OUTPUT_MAP(InTopKD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InTopKD, kNameInTopKD, ADPT_DESC(InTopKD))

// TileD
INPUT_MAP(TileD) = {{1, INPUT_DESC(x)}};
INPUT_ATTR_MAP(TileD) = {{2, ATTR_DESC(multiples, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(TileD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(TileD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(TileD, kNameTile, ADPT_DESC(TileD))

// OneHot
INPUT_MAP(OneHot) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(depth)}, {3, INPUT_DESC(on_value)}, {4, INPUT_DESC(off_value)}};
ATTR_MAP(OneHot) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>())}};
OUTPUT_MAP(OneHot) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(OneHot, prim::kPrimOneHot->name(), ADPT_DESC(OneHot))

// GatherV2D
INPUT_MAP(GatherV2D) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
INPUT_ATTR_MAP(GatherV2D) = {{3, ATTR_DESC(axis, AnyTraits<int64_t>())}};
ATTR_MAP(GatherV2D) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherV2D, prim::kPrimGather->name(), ADPT_DESC(GatherV2D))
REG_ADPT_DESC(Gather, kNameGather, ADPT_DESC(GatherV2D))

// ScatterNdD
INPUT_MAP(ScatterNdD) = {{1, INPUT_DESC(indices)}, {2, INPUT_DESC(x)}};
INPUT_ATTR_MAP(ScatterNdD) = {
  {3, ATTR_DESC(shape, AnyTraits<std::vector<int64_t>>(), AnyTraits<std::vector<int64_t>>())}};
ATTR_MAP(ScatterNdD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ScatterNdD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScatterNdD, kNameScatterNdD, ADPT_DESC(ScatterNdD))

// ScatterNonAliasingAdd
INPUT_MAP(ScatterNonAliasingAdd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}, {3, INPUT_DESC(updates)}};
ATTR_MAP(ScatterNonAliasingAdd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(ScatterNonAliasingAdd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ScatterNonAliasingAdd, kNameScatterNonAliasingAdd, ADPT_DESC(ScatterNonAliasingAdd))

// GatherNd
INPUT_MAP(GatherNd) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(indices)}};
ATTR_MAP(GatherNd) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherNd) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherNd, kNameGatherNd, ADPT_DESC(GatherNd))

// GatherD
INPUT_MAP(GatherD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(dim)}, {3, INPUT_DESC(index)}};
ATTR_MAP(GatherD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(GatherD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(GatherD, kNameGatherD, ADPT_DESC(GatherD))

// Range
INPUT_MAP(RangeD) = {{1, INPUT_DESC(x)}};
ATTR_MAP(RangeD) = {{"start", ATTR_DESC(start, AnyTraits<float>())},
                    {"limit", ATTR_DESC(limit, AnyTraits<float>())},
                    {"delta", ATTR_DESC(delta, AnyTraits<float>())}};
OUTPUT_MAP(RangeD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(RangeD, kNameRange, ADPT_DESC(RangeD))

// InplaceAddD
INPUT_MAP(InplaceAddD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
ATTR_MAP(InplaceAddD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(InplaceAddD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceAddD, kNameInplaceAddD, ADPT_DESC(InplaceAddD))

// InplaceSubD
INPUT_MAP(InplaceSubD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
ATTR_MAP(InplaceSubD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(InplaceSubD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceSubD, kNameInplaceSubD, ADPT_DESC(InplaceSubD))

// InplaceUpdateD
INPUT_MAP(InplaceUpdateD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(v)}};
ATTR_MAP(InplaceUpdateD) = {{"indices", ATTR_DESC(indices, AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(InplaceUpdateD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(InplaceUpdateD, kNameInplaceUpdateD, ADPT_DESC(InplaceUpdateD))

// Select
INPUT_MAP(Select) = {{1, INPUT_DESC(condition)}, {2, INPUT_DESC(x1)}, {3, INPUT_DESC(x2)}};
ATTR_MAP(Select) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Select) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Select, prim::kPrimSelect->name(), ADPT_DESC(Select))

// StridedSliceGrad
INPUT_MAP(StridedSliceGrad) = {
  {1, INPUT_DESC(dy)}, {2, INPUT_DESC(shape)}, {3, INPUT_DESC(begin)}, {4, INPUT_DESC(end)}, {5, INPUT_DESC(strides)}};
ATTR_MAP(StridedSliceGrad) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                              {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                              {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                              {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                              {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSliceGrad) = {{0, OUTPUT_DESC(output)}};
REG_ADPT_DESC(StridedSliceGrad, kNameStridedSliceGrad, ADPT_DESC(StridedSliceGrad))

// StridedSlice
INPUT_MAP(StridedSlice) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(strides)}};
ATTR_MAP(StridedSlice) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                          {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                          {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                          {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                          {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSlice) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StridedSlice, kNameStridedSlice, ADPT_DESC(StridedSlice))

// StridedSliceV2
INPUT_MAP(StridedSliceV2) = {
  {1, INPUT_DESC(x)}, {2, INPUT_DESC(begin)}, {3, INPUT_DESC(end)}, {4, INPUT_DESC(axes)}, {5, INPUT_DESC(strides)}};
ATTR_MAP(StridedSliceV2) = {{"begin_mask", ATTR_DESC(begin_mask, AnyTraits<int64_t>())},
                            {"end_mask", ATTR_DESC(end_mask, AnyTraits<int64_t>())},
                            {"ellipsis_mask", ATTR_DESC(ellipsis_mask, AnyTraits<int64_t>())},
                            {"new_axis_mask", ATTR_DESC(new_axis_mask, AnyTraits<int64_t>())},
                            {"shrink_axis_mask", ATTR_DESC(shrink_axis_mask, AnyTraits<int64_t>())}};
OUTPUT_MAP(StridedSliceV2) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(StridedSliceV2, kNameStridedSliceV2, ADPT_DESC(StridedSliceV2))

// UnsortedSegmentSum
INPUT_MAP(UnsortedSegmentSumD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
INPUT_ATTR_MAP(UnsortedSegmentSumD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};
ATTR_MAP(UnsortedSegmentSumD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentSumD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentSumD, prim::kPrimUnsortedSegmentSum->name(), ADPT_DESC(UnsortedSegmentSumD))

// UnsortedSegmentProdD
INPUT_MAP(UnsortedSegmentProdD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
INPUT_ATTR_MAP(UnsortedSegmentProdD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};
ATTR_MAP(UnsortedSegmentProdD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentProdD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentProdD, kNameUnsortedSegmentProdD, ADPT_DESC(UnsortedSegmentProdD))

// UnsortedSegmentMaxD
INPUT_MAP(UnsortedSegmentMaxD) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}};
INPUT_ATTR_MAP(UnsortedSegmentMaxD) = {{3, ATTR_DESC(num_segments, AnyTraits<int64_t>())}};
ATTR_MAP(UnsortedSegmentMaxD) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentMaxD) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentMaxD, kNameUnsortedSegmentMaxD, ADPT_DESC(UnsortedSegmentMaxD))

// UnsortedSegmentMin
INPUT_MAP(UnsortedSegmentMin) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(segment_ids)}, {3, INPUT_DESC(num_segments)}};
ATTR_MAP(UnsortedSegmentMin) = EMPTY_ATTR_MAP;
OUTPUT_MAP(UnsortedSegmentMin) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(UnsortedSegmentMin, prim::kPrimUnsortedSegmentMin->name(), ADPT_DESC(UnsortedSegmentMin))

// ReverseV2
INPUT_MAP(ReverseV2D) = {{1, INPUT_DESC(x)}};
ATTR_MAP(ReverseV2D) = {{"axis", ATTR_DESC(axis, AnyTraits<int64_t>(), AnyTraits<std::vector<int64_t>>())}};
OUTPUT_MAP(ReverseV2D) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(ReverseV2D, kNameReverseV2, ADPT_DESC(ReverseV2D))
}  // namespace luojianet_ms::transform
