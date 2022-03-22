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

#include <gtest/gtest.h>

#define protected public
#define private public
#include "host_kernels/ssd_prior_box_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/passes/dimension_compute_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelSsdPriorboxKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

///  convolution   data
///        |     /
///       ssdpriorbox
///          \.
///        reshape
class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }
  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }
  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }
  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }
  ge::OpDescPtr op_desc_;
};
void make_graph_ssd(ComputeGraphPtr &graph, vector<float> temp_aspect_ratios, vector<float> max_size,
                    vector<float> min_size, vector<float> variances, bool flip) {
  NodePtr data_node = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr conv_node = NodeBuilder("Conv2D", CONV2D).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr ssd_priorbox_node = NodeBuilder("SSDPriorBox", SSDPRIORBOX)
                                  .AddInputDesc({10, 10, 10, 10}, FORMAT_NCHW, DT_FLOAT)
                                  .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                  .Build(graph);

  NodePtr reshape_node =
      NodeBuilder("reshape", RESHAPE).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), ssd_priorbox_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ssd_priorbox_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));

  auto ssdPriorbox_op = ssd_priorbox_node->GetOpDesc();
  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_OFFSET, 0.5);
  AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_ASPECT_RATIO, temp_aspect_ratios);
  AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_MAX_SIZE, max_size);
  AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_MIN_SIZE, min_size);
  AttrUtils::SetBool(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_CLIP, true);
  AttrUtils::SetBool(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_FLIP, flip);
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_IMG_H, 100);
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_IMG_W, 100);
  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_STEP_H, 0);
  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_STEP_W, 0);
  AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_VARIANCE, variances);
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_VARIANCE_NUM, 1);
}

void make_graph_ssd_for_failed(ComputeGraphPtr &graph, vector<float> temp_aspect_ratios, vector<float> max_size,
                               vector<float> min_size, vector<float> variances, bool flip, bool clip) {
  NodePtr data_node = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr conv_node = NodeBuilder("Conv2D", CONV2D).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr ssd_priorbox_node = NodeBuilder("SSDPriorBox", SSDPRIORBOX)
                                  .AddInputDesc({10, 10, 10, 10}, FORMAT_NCHW, DT_FLOAT)
                                  .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                  .Build(graph);

  NodePtr reshape_node = NodeBuilder("reshape", RESHAPE)
                             .AddInputDesc({10, 10, 10, 10}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);

  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), ssd_priorbox_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(ssd_priorbox_node->GetOutDataAnchor(0), reshape_node->GetInDataAnchor(0));

  auto ssdPriorbox_op = ssd_priorbox_node->GetOpDesc();

  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_OFFSET, 0.5);
  if (temp_aspect_ratios.size() != 0) {
    AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_ASPECT_RATIO, temp_aspect_ratios);
  }
  if (max_size.size() != 0) {
    AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_MAX_SIZE, max_size);
  }
  if (min_size.size() != 0) {
    AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_MIN_SIZE, min_size);
  }
  if (clip) {
    AttrUtils::SetBool(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_CLIP, true);
  }
  AttrUtils::SetBool(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_FLIP, flip);
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_IMG_H, 100);
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_IMG_W, 100);
  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_STEP_H, 0);
  AttrUtils::SetFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_STEP_W, 0);
  if (variances.size() != 0) {
    AttrUtils::SetListFloat(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_VARIANCE, variances);
  }
  AttrUtils::SetInt(ssdPriorbox_op, SSD_PRIOR_BOX_ATTR_VARIANCE_NUM, 1);
}
TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, ComputeFailed) {
  ComputeGraphPtr compute_graph1 = std::make_shared<ComputeGraph>("test");
  make_graph_ssd_for_failed(compute_graph1, {}, {9}, {4}, {0.1}, true, true);
  NodePtr ssd_priorbox_node1 = compute_graph1->FindNode("SSDPriorBox");
  DimensionComputePass pass;
  ge::Status ret = pass.Run(ssd_priorbox_node1);
  EXPECT_EQ(PARAM_INVALID, ret);
  ComputeGraphPtr compute_graph2 = std::make_shared<ComputeGraph>("test");
  make_graph_ssd_for_failed(compute_graph2, {1}, {}, {4}, {0.1}, true, true);
  NodePtr ssd_priorbox_node2 = compute_graph2->FindNode("SSDPriorBox");
  ret = pass.Run(ssd_priorbox_node2);
  EXPECT_EQ(PARAM_INVALID, ret);

  ComputeGraphPtr compute_graph3 = std::make_shared<ComputeGraph>("test");
  make_graph_ssd_for_failed(compute_graph3, {1}, {9}, {}, {0.1}, true, true);
  NodePtr ssd_priorbox_node3 = compute_graph3->FindNode("SSDPriorBox");
  ret = pass.Run(ssd_priorbox_node3);
  EXPECT_EQ(PARAM_INVALID, ret);

  ComputeGraphPtr compute_graph4 = std::make_shared<ComputeGraph>("test");
  make_graph_ssd_for_failed(compute_graph4, {1}, {9}, {4}, {}, true, true);
  NodePtr ssd_priorbox_node4 = compute_graph4->FindNode("SSDPriorBox");
  ret = pass.Run(ssd_priorbox_node4);
  EXPECT_EQ(PARAM_INVALID, ret);

  ComputeGraphPtr compute_graph5 = std::make_shared<ComputeGraph>("test");
  make_graph_ssd_for_failed(compute_graph5, {1}, {9}, {4}, {}, true, false);
  NodePtr ssd_priorbox_node5 = compute_graph5->FindNode("SSDPriorBox");
  ret = pass.Run(ssd_priorbox_node5);
  EXPECT_EQ(PARAM_INVALID, ret);
}
TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, ComputeSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_graph_ssd(compute_graph, {1}, {9}, {4}, {0.1}, true);

  NodePtr ssd_priorbox_node = compute_graph->FindNode("SSDPriorBox");
  DimensionComputePass pass;
  ge::Status ret = pass.Run(ssd_priorbox_node);
  EXPECT_EQ(SUCCESS, ret);
  NodePtr reshape_node = compute_graph->FindNode("reshape");
  vector<ConstGeTensorPtr> out_weights = OpDescUtils::GetWeights(reshape_node);

  const float eps = 1e-6;
  if (out_weights.size() >= 1) {
    int32_t dim_size = out_weights[0]->GetTensorDesc().GetShape().GetDim(2);
    EXPECT_EQ(10 * 10 * 2 * 4, dim_size);
    const float *top_data = (const float *)out_weights[0]->GetData().data();
    /// pick a few generated priors and compare against the expected number.
    /// first prior
    EXPECT_NEAR(top_data[0], 0.03, eps);
    EXPECT_NEAR(top_data[1], 0.03, eps);
    EXPECT_NEAR(top_data[2], 0.07, eps);
    EXPECT_NEAR(top_data[3], 0.07, eps);
    // second prior
    EXPECT_NEAR(top_data[4], 0.02, eps);
    EXPECT_NEAR(top_data[5], 0.02, eps);
    EXPECT_NEAR(top_data[6], 0.08, eps);
    EXPECT_NEAR(top_data[7], 0.08, eps);
    // prior in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, eps);

    // check variance
    top_data += dim_size;
    for (int d = 0; d < dim_size; ++d) {
      EXPECT_NEAR(top_data[d], 0.1, eps);
    }
  }
}
TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, AspectRatioNoflipSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_graph_ssd(compute_graph, {1, 2}, {9}, {4}, {0.1}, false);

  NodePtr ssd_priorbox_node = compute_graph->FindNode("SSDPriorBox");
  DimensionComputePass pass;
  ge::Status ret = pass.Run(ssd_priorbox_node);
  EXPECT_EQ(SUCCESS, ret);
  NodePtr reshape_node = compute_graph->FindNode("reshape");
  vector<ConstGeTensorPtr> out_weights = OpDescUtils::GetWeights(reshape_node);

  const float eps = 1e-6;
  if (out_weights.size() >= 1) {
    int32_t dim_size = out_weights[0]->GetTensorDesc().GetShape().GetDim(2);
    EXPECT_EQ(10 * 10 * 3 * 4, dim_size);
    const float *top_data = (const float *)out_weights[0]->GetData().data();
    /// pick a few generated priors and compare against the expected number.
    /// first prior
    EXPECT_NEAR(top_data[0], 0.03, eps);
    EXPECT_NEAR(top_data[1], 0.03, eps);
    EXPECT_NEAR(top_data[2], 0.07, eps);
    EXPECT_NEAR(top_data[3], 0.07, eps);
    // second prior
    EXPECT_NEAR(top_data[4], 0.02, eps);
    EXPECT_NEAR(top_data[5], 0.02, eps);
    EXPECT_NEAR(top_data[6], 0.08, eps);
    EXPECT_NEAR(top_data[7], 0.08, eps);
    // third prior
    EXPECT_NEAR(top_data[8], 0.05 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[9], 0.05 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[10], 0.05 + 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[11], 0.05 + 0.01 * sqrt(2.), eps);
    // prior in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 1], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 2], 0.47, eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 3], 0.47, eps);
    // prior with ratio 1:2 in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 8], 0.45 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 9], 0.45 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 10], 0.45 + 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[4 * 10 * 3 * 4 + 4 * 3 * 4 + 11], 0.45 + 0.01 * sqrt(2.), eps);
    // check variance
    top_data += dim_size;
    for (int d = 0; d < dim_size; ++d) {
      EXPECT_NEAR(top_data[d], 0.1, eps);
    }
  }
}
TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, AspectratioMultiSizeSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_graph_ssd(compute_graph, {1, 2, 0.5}, {9, 18}, {4, 8}, {0.1}, true);

  NodePtr ssd_priorbox_node = compute_graph->FindNode("SSDPriorBox");
  DimensionComputePass pass;
  ge::Status ret = pass.Run(ssd_priorbox_node);
  EXPECT_EQ(SUCCESS, ret);
  NodePtr reshape_node = compute_graph->FindNode("reshape");
  vector<ConstGeTensorPtr> out_weights = OpDescUtils::GetWeights(reshape_node);

  const float eps = 1e-6;
  if (out_weights.size() >= 1) {
    int32_t dim_size = out_weights[0]->GetTensorDesc().GetShape().GetDim(2);
    EXPECT_EQ(10 * 10 * 8 * 4, dim_size);
    const float *top_data = (const float *)out_weights[0]->GetData().data();

    /// pick a few generated priors and compare against the expected number.
    /// first prior
    EXPECT_NEAR(top_data[0], 0.03, eps);
    EXPECT_NEAR(top_data[1], 0.03, eps);
    EXPECT_NEAR(top_data[2], 0.07, eps);
    EXPECT_NEAR(top_data[3], 0.07, eps);
    // second prior
    EXPECT_NEAR(top_data[4], 0.02, eps);
    EXPECT_NEAR(top_data[5], 0.02, eps);
    EXPECT_NEAR(top_data[6], 0.08, eps);
    EXPECT_NEAR(top_data[7], 0.08, eps);
    // third prior
    EXPECT_NEAR(top_data[8], 0.05 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[9], 0.05 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[10], 0.05 + 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[11], 0.05 + 0.01 * sqrt(2.), eps);
    // forth prior
    EXPECT_NEAR(top_data[12], 0.05 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[13], 0.05 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[14], 0.05 + 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[15], 0.05 + 0.02 * sqrt(2.), eps);
    // fifth prior
    EXPECT_NEAR(top_data[16], 0.01, eps);
    EXPECT_NEAR(top_data[17], 0.01, eps);
    EXPECT_NEAR(top_data[18], 0.09, eps);
    EXPECT_NEAR(top_data[19], 0.09, eps);
    // sixth prior
    EXPECT_NEAR(top_data[20], 0.00, eps);
    EXPECT_NEAR(top_data[21], 0.00, eps);
    EXPECT_NEAR(top_data[22], 0.11, eps);
    EXPECT_NEAR(top_data[23], 0.11, eps);
    // seventh prior
    EXPECT_NEAR(top_data[24], 0.00, eps);
    EXPECT_NEAR(top_data[25], 0.05 - 0.04 / sqrt(2.), eps);
    EXPECT_NEAR(top_data[26], 0.05 + 0.04 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[27], 0.05 + 0.04 / sqrt(2.), eps);
    // forth prior
    EXPECT_NEAR(top_data[28], 0.05 - 0.04 / sqrt(2.), eps);
    EXPECT_NEAR(top_data[29], 0.00, eps);
    EXPECT_NEAR(top_data[30], 0.05 + 0.04 / sqrt(2.), eps);
    EXPECT_NEAR(top_data[31], 0.05 + 0.04 * sqrt(2.), eps);
    // prior in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4], 0.43, eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 1], 0.43, eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 2], 0.47, eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 3], 0.47, eps);
    // prior with ratio 1:2 in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 8], 0.45 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 9], 0.45 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 10], 0.45 + 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 11], 0.45 + 0.01 * sqrt(2.), eps);
    // prior with ratio 2:1 in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 12], 0.45 - 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 13], 0.45 - 0.02 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 14], 0.45 + 0.01 * sqrt(2.), eps);
    EXPECT_NEAR(top_data[8 * 10 * 4 * 4 + 8 * 4 * 4 + 15], 0.45 + 0.02 * sqrt(2.), eps);

    // check variance
    top_data += dim_size;
    for (int d = 0; d < dim_size; ++d) {
      EXPECT_NEAR(top_data[d], 0.1, eps);
    }
  }
}
TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, MultiVarianceSuccess) {
  ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_graph_ssd(compute_graph, {1}, {9}, {4}, {0.1, 0.2, 0.3, 0.4}, true);

  NodePtr ssd_priorbox_node = compute_graph->FindNode("SSDPriorBox");
  DimensionComputePass pass;
  ge::Status ret = pass.Run(ssd_priorbox_node);
  EXPECT_EQ(SUCCESS, ret);
  NodePtr reshape_node = compute_graph->FindNode("reshape");
  vector<ConstGeTensorPtr> out_weights = OpDescUtils::GetWeights(reshape_node);

  const float eps = 1e-6;
  if (out_weights.size() >= 1) {
    int32_t dim_size = out_weights[0]->GetTensorDesc().GetShape().GetDim(2);
    EXPECT_EQ(10 * 10 * 2 * 4, dim_size);
    const float *top_data = (const float *)out_weights[0]->GetData().data();
    EXPECT_NEAR(top_data[0], 0.03, eps);
    EXPECT_NEAR(top_data[1], 0.03, eps);
    EXPECT_NEAR(top_data[2], 0.07, eps);
    EXPECT_NEAR(top_data[3], 0.07, eps);
    // second prior
    EXPECT_NEAR(top_data[4], 0.02, eps);
    EXPECT_NEAR(top_data[5], 0.02, eps);
    EXPECT_NEAR(top_data[6], 0.08, eps);
    EXPECT_NEAR(top_data[7], 0.08, eps);
    // prior in the 5-th row and 5-th col
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 1], 0.43, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 2], 0.47, eps);
    EXPECT_NEAR(top_data[4 * 10 * 2 * 4 + 4 * 2 * 4 + 3], 0.47, eps);

    // check variance
    top_data += dim_size;
    for (int d = 0; d < dim_size; ++d) {
      EXPECT_NEAR(top_data[d], 0.1 * (d % 4 + 1), eps);
    }
  }
}

TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, AllSuccess) {
  int num_priors = 0;
  int dim_size = 0;
  SsdPriorboxKernel kernal;
  ge::Status ret = kernal.GetNumPriorAndDimSize(2, 2, 2, 2, 2, num_priors, dim_size);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesFoldingKernelSsdPriorboxKernel, ParamInvalid) {
  int num_priors = 0;
  int dim_size = 0;
  SsdPriorboxKernel kernal;
  ge::Status ret = kernal.GetNumPriorAndDimSize(2 * 1024 * 1024 * 1024, 2, 1, 1, 1, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);

  ret = kernal.GetNumPriorAndDimSize(4 * 1024 * 1024 * 1024 - 1, 1, 1, 1, 1, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);

  ret = kernal.GetNumPriorAndDimSize(2 * 1024 * 1024 * 1024 - 1, 1, 1, 1, 1, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);

  ret = kernal.GetNumPriorAndDimSize(1, 1, 1, 1 * 1024 * 1024 * 1024, 2, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);

  ret = kernal.GetNumPriorAndDimSize(1, 1, 1, 1024 * 1024 * 1024, 1, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);

  ret = kernal.GetNumPriorAndDimSize(1, 1, 1, 1024 * 1024 * 1024 - 1, 1, num_priors, dim_size);
  EXPECT_EQ(PARAM_INVALID, ret);
}
