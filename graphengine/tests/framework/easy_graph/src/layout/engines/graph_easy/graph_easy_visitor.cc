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

#include "easy_graph/layout/engines/graph_easy/graph_easy_visitor.h"
#include "layout/engines/graph_easy/utils/shell_executor.h"
#include "easy_graph/layout/engines/graph_easy/graph_easy_option.h"
#include "easy_graph/infra/scope_guard.h"
#include "easy_graph/graph/graph.h"
#include "easy_graph/graph/edge.h"
#include "easy_graph/graph/node.h"
#include "easy_graph/infra/log.h"

EG_NS_BEGIN

namespace {
struct SubgraphLayoutVisitor : GraphVisitor {
  SubgraphLayoutVisitor(const NodeId &id, GraphEasyLayoutContext &ctxt) : id_(id), ctxt_(ctxt) {}
  std::string layout;
  bool hasSubgraph{false};

 private:
  Status Visit(const Graph &graph) override {
    ScopeGuard guard([this, &graph]() { ctxt_.EnterGraph(graph); }, [this]() { ctxt_.ExitGraph(); });
    layout += (std::string(" -- [") + id_ + "/" + graph.GetName() + "]" +
               "{class : subgraph; label : " + graph.GetName() + ";}");
    hasSubgraph = true;
    return EG_SUCCESS;
  }

 private:
  NodeId id_;
  GraphEasyLayoutContext &ctxt_;
};

/////////////////////////////////////////////////////////////////////////
std::string GetGraphLayoutTitle(const Graph &graph, const GraphEasyLayoutContext &ctxt) {
  std::string flowDirection = (ctxt.GetOptions().dir_ == FlowDir::LR) ? "east" : "down";
  std::string graphTitle = std::string("graph { label : ") + graph.GetName() + "; flow : " + flowDirection +
      " ; } node.subgraph { border : double-dash; }";
  return graphTitle;
}
/////////////////////////////////////////////////////////////////////////
std::string GetNodeLayout(const Node &node, GraphEasyLayoutContext &ctxt) {
  const auto &id = node.GetId();
  std::string nodeBox = std::string("[") + id + "]";

  SubgraphLayoutVisitor subgraphVisitor(id, ctxt);
  node.Accept(subgraphVisitor);

  if (!subgraphVisitor.hasSubgraph || ctxt.InLinking())
    return nodeBox;

  return (std::string("( ") + id + ": " + nodeBox + subgraphVisitor.layout + ")");
}

/////////////////////////////////////////////////////////////////////////
INTERFACE(EdgeLayout) {
  EdgeLayout(GraphEasyLayoutContext & ctxt, const Edge &edge) : ctxt_(ctxt), options_(ctxt.GetOptions()), edge_(edge) {}

  std::string GetLayout() const {
    auto graph = ctxt_.GetCurrentGraph();
    if (!graph) {
      EG_FATAL("Layout context has no graph!");
      return "";
    }

    auto node_pair = graph->FindNodePair(edge_);

    if ((!node_pair.first) || (!node_pair.second)) {
      EG_FATAL("Layout context graph(%s) has not found node(%s, %s)!", graph->GetName().c_str(),
               edge_.GetSrc().getNodeId().c_str(), edge_.GetDst().getNodeId().c_str());
      return "";
    }

    std::string src_node_layout = GetNodeLayout(*node_pair.first, ctxt_);
    std::string dst_node_layout = GetNodeLayout(*node_pair.second, ctxt_);
    return src_node_layout + GetArrowLayout() + GetAttrLayout() + dst_node_layout;
  }

 private:
  ABSTRACT(std::string GetAttrLayout() const);
  ABSTRACT(std::string GetArrowLayout() const);

 protected:
  GraphEasyLayoutContext &ctxt_;
  const GraphEasyOption &options_;
  const Edge &edge_;
};

/////////////////////////////////////////////////////////////////////////
struct CtrlEdgeLayout : EdgeLayout {
  using EdgeLayout::EdgeLayout;

 private:
  std::string GetAttrLayout() const override {
    if (edge_.GetLabel() == "")
      return "";
    return std::string("{label : ") + edge_.GetLabel() + "}";
  }

  std::string GetArrowLayout() const override {
    return " ..> ";
  }
};

/////////////////////////////////////////////////////////////////////////
struct DataEdgeLayout : EdgeLayout {
  using EdgeLayout::EdgeLayout;

 private:
  std::string GetAttrLayout() const override {
    return std::string("{ ") + GetLabelAttr() + GetPortAttr() + " }";
  }

  std::string GetArrowLayout() const override {
    return " --> ";
  }

 private:
  std::string GetPortPair() const {
    return std::string("(") + std::to_string(edge_.GetSrc().getPortId()) + "," +
        std::to_string(edge_.GetDst().getPortId()) + ")";
  }

  std::string GetLabelAttr() const {
    return std::string("label :") + edge_.GetLabel() + GetPortPair() + "; ";
  }

  std::string GetPortAttr() const {
    return (options_.type_ == LayoutType::FREE) ? "" : GetOutPortAttr() + GetInPortAttr();
  }

  std::string GetOutPortAttr() const {
    return std::string(" start : ") + "front" + ", " + std::to_string(edge_.GetSrc().getPortId() * options_.scale_) +
        "; ";
  }

  std::string GetInPortAttr() const {
    return std::string(" end : ") + "back" + ", " + std::to_string(edge_.GetDst().getPortId() * options_.scale_) + "; ";
  }
};
}  // namespace

GraphEasyVisitor::GraphEasyVisitor(const GraphEasyOption &options) : ctxt_(options) {}

Status GraphEasyVisitor::Visit(const Graph &graph) {
  ctxt_.EnterGraph(graph);
  layout_ += GetGraphLayoutTitle(graph, ctxt_);
  return EG_SUCCESS;
}

Status GraphEasyVisitor::Visit(const Node &node) {
  layout_ += GetNodeLayout(node, ctxt_);
  return EG_SUCCESS;
}

Status GraphEasyVisitor::Visit(const Edge &edge) {
  ScopeGuard guard([this]() { ctxt_.LinkBegin(); }, [this]() { ctxt_.LinkEnd(); });

  auto makeEdgeLayout = [this, &edge]() -> const EdgeLayout * {
    if (edge.GetType() == EdgeType::CTRL)
      return new CtrlEdgeLayout(ctxt_, edge);
    return new DataEdgeLayout(ctxt_, edge);
  };

  std::unique_ptr<const EdgeLayout> edgeLayout(makeEdgeLayout());
  layout_ += edgeLayout->GetLayout();
  return EG_SUCCESS;
}

std::string GraphEasyVisitor::GetLayout() const {
  return layout_;
}

EG_NS_END
