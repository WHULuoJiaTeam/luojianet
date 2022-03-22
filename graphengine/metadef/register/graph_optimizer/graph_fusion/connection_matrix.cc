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
#include "register/graph_optimizer/graph_fusion/connection_matrix.h"


namespace fe {
ConnectionMatrix::ConnectionMatrix() : size_(0) {};

ConnectionMatrix::~ConnectionMatrix() {}


Status ConnectionMatrix::Generate(const ge::ComputeGraph &graph) {
  int64_t max_id = 0;
  auto direct_nodes = graph.GetDirectNode();
  for (const auto &node : direct_nodes) {
    int64_t id = node->GetOpDesc()->GetId();
    if (id > max_id) {
      max_id = id;
    }
  }
  size_t total_size = static_cast<size_t>(max_id + 1);
  bit_maps.reserve(total_size);
  size_ = total_size;
  for (size_t i = 0; i < total_size; i++) {
    bit_maps.emplace_back(size_);
  }

  for (auto &node : direct_nodes) {
    auto inputs = node->GetInAllNodes();
    SetConnectivity(inputs, node);
  }

  return SUCCESS;
}

void ConnectionMatrix::Update(ge::ComputeGraph &graph, vector<ge::NodePtr> &fusion_nodes) {
  ge::LargeBitmap new_bit_vector(graph.GetDirectNode().size());
  new_bit_vector.SetValues(0);
  for (size_t i = 0; i < fusion_nodes.size(); i++) {
    new_bit_vector.Or(GetBitMap(fusion_nodes[i]));
  }
  for (auto &node : graph.GetDirectNode()) {
    bool is_connected_to_fusion = false;
    for (size_t i = 0; i < fusion_nodes.size(); i++) {
      if (GetBitMap(node).GetBit(GetIndex(fusion_nodes[i]))) {
        is_connected_to_fusion = true;
        break;
      }
    }
    if (is_connected_to_fusion) {
      GetBitMap(node).Or(new_bit_vector);
    }
  }
}

void ConnectionMatrix::SetConnectivity(const ge::Node::Vistor<ge::NodePtr> &inputs,
                                       const ge::NodePtr &node) {
  ge::LargeBitmap &bitmap = GetBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0);
  }

  bitmap.SetBit(GetIndex(node));
  for (const ge::NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetBitMap(input));
    }
  }
}

int64_t ConnectionMatrix::GetIndex(const ge::NodePtr &node) const {
  return node->GetOpDesc()->GetId();
}

bool ConnectionMatrix::IsConnected(const ge::NodePtr &a, const ge::NodePtr &b) const {
  return GetBitMap(b).GetBit(GetIndex(a));
}

const ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) const {
  return bit_maps[GetIndex(node)];
}

ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) {
  return bit_maps[GetIndex(node)];
}
}
