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

#include "minddata/dataset/engine/opt/optional/tensor_op_fusion_pass.h"

#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "minddata/dataset/kernels/image/random_crop_decode_resize_op.h"
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"

namespace mindspore {
namespace dataset {

Status TensorOpFusionPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);
  std::vector<std::shared_ptr<TensorOperation>> ops = node->operations();

  // start temporary code, to deal with pre-built TensorOperation
  std::vector<std::string> pattern = {kDecodeOp, kRandomCropAndResizeOp};
  auto itr = std::search(ops.begin(), ops.end(), pattern.begin(), pattern.end(),
                         [](auto op, const std::string &nm) { return op != nullptr ? op->Name() == nm : false; });
  if (itr != ops.end()) {
    MS_LOG(WARNING) << "Fusing pre-build Decode and RandomCropResize into one pre-build.";
    auto fused_op = dynamic_cast<RandomCropAndResizeOp *>((*(itr + 1))->Build().get());
    RETURN_UNEXPECTED_IF_NULL(fused_op);
    (*itr) = std::make_shared<transforms::PreBuiltOperation>(std::make_shared<RandomCropDecodeResizeOp>(*fused_op));
    ops.erase(itr + 1);
    node->setOperations(ops);
    *modified = true;
    return Status::OK();
  }  // end of temporary code, needs to be deleted when tensorOperation's pybind completes

  // logic below is for non-prebuilt TensorOperation
  pattern = {vision::kDecodeOperation, vision::kRandomResizedCropOperation};
  itr = std::search(ops.begin(), ops.end(), pattern.begin(), pattern.end(),
                    [](auto op, const std::string &nm) { return op != nullptr ? op->Name() == nm : false; });

  // return here if no pattern is found
  RETURN_OK_IF_TRUE(itr == ops.end());
  auto *fused_ir = dynamic_cast<vision::RandomResizedCropOperation *>((itr + 1)->get());
  RETURN_UNEXPECTED_IF_NULL(fused_ir);
  // fuse the two ops
  (*itr) = std::make_shared<vision::RandomCropDecodeResizeOperation>(*fused_ir);
  ops.erase(itr + 1);
  node->setOperations(ops);
  *modified = true;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
