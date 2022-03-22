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

#include "common/format_utils.h"
#include <set>
#include <string>
#include "ops/batch_norm.h"
#include "ops/batch_to_space.h"
#include "ops/bias_add.h"
#include "ops/depth_to_space.h"
#include "ops/fused_batch_norm.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "ops/fusion/prelu_fusion.h"
#include "ops/fusion/topk_fusion.h"
#include "ops/instance_norm.h"
#include "ops/lrn.h"
#include "ops/resize.h"
#include "ops/roi_pooling.h"
#include "ops/space_to_batch.h"
#include "ops/space_to_batch_nd.h"
#include "ops/space_to_depth.h"
#include "common/anf_util.h"

namespace luojianet_ms {
namespace dpico {
namespace {
const std::set<std::string> kAssignedFormatOpSet = {
  luojianet_ms::ops::kNameAvgPoolFusion, luojianet_ms::ops::kNameBatchNorm,
  luojianet_ms::ops::kNameBatchToSpace,  luojianet_ms::ops::kNameBiasAdd,
  luojianet_ms::ops::kNameConv2DFusion,  luojianet_ms::ops::kNameConv2dTransposeFusion,
  luojianet_ms::ops::kNameDepthToSpace,  luojianet_ms::ops::kNameFusedBatchNorm,
  luojianet_ms::ops::kNameInstanceNorm,  luojianet_ms::ops::kNameLRN,
  luojianet_ms::ops::kNameMaxPoolFusion, luojianet_ms::ops::kNamePReLUFusion,
  luojianet_ms::ops::kNameResize,        luojianet_ms::ops::kNameROIPooling,
  luojianet_ms::ops::kNameSpaceToBatch,  luojianet_ms::ops::kNameSpaceToBatchND,
  luojianet_ms::ops::kNameSpaceToDepth,  luojianet_ms::ops::kNameTopKFusion};
}  // namespace

const std::set<std::string> &GetAssignedFormatOpSet() { return kAssignedFormatOpSet; }

bool IsSpecialType(const luojianet_ms::CNodePtr &cnode) {
  return CheckPrimitiveType(cnode, luojianet_ms::prim::kPrimTupleGetItem) ||
         CheckPrimitiveType(cnode, luojianet_ms::prim::kPrimDepend) ||
         CheckPrimitiveType(cnode, luojianet_ms::prim::kPrimMakeTuple) ||
         CheckPrimitiveType(cnode, luojianet_ms::prim::kPrimReturn);
}

std::string FormatEnumToString(luojianet_ms::Format format) {
  static std::vector<std::string> names = {
    "NCHW", "NHWC", "NHWC4", "HWKC", "HWCK",   "KCHW",          "CKHW",  "KHWC", "CHWK",
    "HW",   "HW4",  "NC",    "NC4",  "NC4HW4", "NUM_OF_FORMAT", "NCDHW", "NWC",  "NCW",
  };
  if (format < luojianet_ms::NCHW || format > luojianet_ms::NCW) {
    return "";
  }
  return names[format];
}
}  // namespace dpico
}  // namespace luojianet_ms
