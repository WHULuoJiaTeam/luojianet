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

#ifndef GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_
#define GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_

#include <map>
#include <set>
#include <utility>
#include <vector>
#include "graph/aligned_ptr.h"
#include "external/graph/types.h"
#include "inc/graph_pass.h"

namespace ge {
struct SameConstKey {
  int data_size;
  std::shared_ptr<AlignedPtr> aligned_ptr;
  DataType data_type;
  Format format;
  std::vector<int64_t> shape;

 public:
  bool operator< (const SameConstKey &key) const {
    if (data_size != key.data_size) {
      return data_size < key.data_size;
    }
    if (data_size != 0) {
      int ret = memcmp(aligned_ptr->Get(), key.aligned_ptr->Get(), data_size);
      if (ret != 0) {
        return ret < 0;
      }
    }
    if (data_type != key.data_type) {
      return data_type < key.data_type;
    }
    if (format != key.format) {
      return format < key.format;
    }
    size_t shape_size = shape.size();
    if (shape_size != key.shape.size()) {
      return shape_size < key.shape.size();
    }
    for (size_t i = 0; i < shape_size; ++i) {
      if (shape.at(i) != key.shape.at(i)) {
        return shape.at(i) < key.shape.at(i);
      }
    }
    return false;
  }
};

class ConstantFuseSamePass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  void GetFuseConstNodes(ComputeGraphPtr &graph,
      std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes);
  Status MoveOutDataEdges(NodePtr &src_node, NodePtr &dst_node);
  Status FuseConstNodes(ComputeGraphPtr &graph,
      std::map<SameConstKey, std::vector<NodePtr>> &fuse_nodes);
};
} // namespace ge
#endif // GE_GRAPH_PASSES_CONSTANT_FUSE_SAME_PASS_H_
