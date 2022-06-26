/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/project_op.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/log_adapter.h"

namespace mindspore {
namespace dataset {
ProjectOp::ProjectOp(const std::vector<std::string> &columns_to_project)
    : PipelineOp(0), columns_to_project_(columns_to_project) {}

void ProjectOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nColumns that are projected:";
    for (size_t i = 0; i < columns_to_project_.size(); i++) {
      out << "\n" << columns_to_project_[i];
    }
    out << "\n\n";
  }
}

// Gets a row from the child operator and projects the buffer.
Status ProjectOp::GetNextRow(TensorRow *row) {
  RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
  if (!row->eoe() && !row->eof()) {
    *row = Project(*row);
  }
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
  }
  return Status::OK();
}

TensorRow ProjectOp::Project(const TensorRow &row) {
  TensorRow new_row;
  (void)std::transform(projected_column_indices_.begin(), projected_column_indices_.end(), std::back_inserter(new_row),
                       [&row](uint32_t x) { return row[x]; });
  // Now if columns changed after map, we don't know which column we should keep,
  // so temporarily we don't support print file_path after ProjectOp.
  new_row.setPath({});
  return new_row;
}

// Class functor operator () override.
// Most dataset ops operate by launching a thread (see ExecutionTree).
// However, the ProjectOp is defined as a inlined operator, so it is invalid to launch the
// functor since this op runs inlined inside another operator. The function is overloaded to
// ensure that it is not called by mistake (it will generate an error).
Status ProjectOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] ProjectOp is an inlined operator."); }

Status ProjectOp::EoeReceived(int32_t worker_id) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status ProjectOp::EofReceived(int32_t worker_id) { return Status::OK(); }

// Compute the column map and save it into our own column name map
// We cannot use the super class ComputeColMap here because we're making a modification of the
// map from the child map.
Status ProjectOp::ComputeColMap() {
  if (column_name_id_map_.empty()) {
    std::unordered_map<std::string, int32_t> child_column_name_mapping = child_[0]->column_name_id_map();
    for (size_t i = 0; i < columns_to_project_.size(); i++) {
      std::string &current_column = columns_to_project_[i];
      if (child_column_name_mapping.find(current_column) == child_column_name_mapping.end()) {
        std::string err_msg = "Invalid column, column name: " + current_column + " does not exist.";
        RETURN_STATUS_UNEXPECTED(err_msg);
      }
      // Setup the new column name mapping for ourself (base class field)
      column_name_id_map_[current_column] = i;
      projected_column_indices_.push_back(child_column_name_mapping[current_column]);
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

Status ProjectOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_IF_NOT_OK(ComputeColMap());
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(&new_row));
  (void)std::transform(projected_column_indices_.begin(), projected_column_indices_.end(), std::back_inserter(*row),
                       [&new_row](uint32_t x) { return new_row[x]; });
  // Now if columns changed after map, we don't know which column we should keep,
  // so temporarily we don't support print file_path after ProjectOp.
  new_row.setPath({});
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
