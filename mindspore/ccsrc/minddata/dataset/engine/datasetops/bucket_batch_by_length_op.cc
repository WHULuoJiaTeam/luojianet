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
#include "minddata/dataset/engine/datasetops/bucket_batch_by_length_op.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/core/tensor_shape.h"
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/util/status.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
BucketBatchByLengthOp::BucketBatchByLengthOp(const std::vector<std::string> &length_dependent_columns,
                                             const std::vector<int32_t> &bucket_boundaries,
                                             const std::vector<int32_t> &bucket_batch_sizes,
                                             std::shared_ptr<TensorOp> element_length_function, const PadInfo &pad_info,
                                             bool pad_to_bucket_boundary, bool drop_remainder,
                                             int32_t op_connector_size)
    : PipelineOp(op_connector_size),
      length_dependent_columns_(length_dependent_columns),
      bucket_boundaries_(bucket_boundaries),
      bucket_batch_sizes_(bucket_batch_sizes),
      element_length_function_(element_length_function),
      pad_info_(pad_info),
      pad_to_bucket_boundary_(pad_to_bucket_boundary),
      drop_remainder_(drop_remainder),
      batch_count_(0) {
  for (int i = 0; i < bucket_batch_sizes_.size(); i++) {
    buckets_.push_back(std::make_unique<TensorQTable>());
  }
}

Status BucketBatchByLengthOp::EoeReceived(int32_t) {
  state_ = OpState::kDeOpIdle;
  return Status::OK();
}

Status BucketBatchByLengthOp::operator()() {
  TaskManager::FindMe()->Post();

  TensorRow current_row;
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
  while (!child_iterator_->EofHandled()) {
    while (!current_row.empty()) {
      int32_t element_length;
      RETURN_IF_NOT_OK(ObtainElementLength(&element_length, current_row));

      int bucket_index = bucket_boundaries_.size() - 1;
      while (element_length < bucket_boundaries_[bucket_index]) {
        bucket_index--;
      }

      buckets_[bucket_index]->push_back(current_row);

      if (buckets_[bucket_index]->size() == bucket_batch_sizes_[bucket_index]) {
        RETURN_IF_NOT_OK(PadAndBatchBucket(bucket_index, bucket_batch_sizes_[bucket_index]));
      }

      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
    }

    // got EOE, do what we need to do with remainders in each bucket
    if (!drop_remainder_) {
      for (int i = 0; i < bucket_boundaries_.size(); i++) {
        if (!buckets_[i]->empty()) {
          RETURN_IF_NOT_OK(PadAndBatchBucket(i, buckets_[i]->size()));
        }
      }
    }

    // need to send EOE manually since we set state to idle in EoeRecieved()
    RETURN_IF_NOT_OK(out_connector_->SendEOE());

    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&current_row));
  }
  RETURN_IF_NOT_OK(out_connector_->SendEOF());

  return Status::OK();
}

Status BucketBatchByLengthOp::ObtainElementLength(int32_t *out_element_length, TensorRow element) {
  RETURN_UNEXPECTED_IF_NULL(out_element_length);
  // call pyfunc here if given pyfunc, otherwise return 0th dimension of shape of
  // the single column specified in length_dependent_columns_
  if (element_length_function_) {
    TensorRow input, output;
    size_t number_of_arguments = length_dependent_columns_.size();
    for (size_t i = 0; i < number_of_arguments; i++) {
      auto map_item = column_name_id_map_.find(length_dependent_columns_[i]);
      if (map_item == column_name_id_map_.end()) {
        RETURN_STATUS_UNEXPECTED("Invalid column, BucketBatchByLength couldn't find the specified column(" +
                                 length_dependent_columns_[i] + ") in the dataset.");
      }
      int32_t column_index = map_item->second;
      input.push_back(element[column_index]);
    }
    RETURN_IF_NOT_OK(element_length_function_->Compute(input, &output));
    RETURN_IF_NOT_OK(output.at(0)->GetItemAt(out_element_length, {0}));
    if (*out_element_length < 0) {
      RETURN_STATUS_UNEXPECTED(
        "Invalid element_length_function, element_length_function must return an integer greater than or equal to 0, "
        "but got" +
        std::to_string(*out_element_length));
    }
  } else {
    *out_element_length = element[0]->shape()[0];
  }
  return Status::OK();
}

Status BucketBatchByLengthOp::PadAndBatchBucket(int32_t bucket_index, int32_t batch_size) {
  std::unique_ptr<TensorQTable> *bucket = &buckets_[bucket_index];

  PadInfo pad_info_copy = pad_info_;
  if (pad_to_bucket_boundary_) {
    for (auto &pair : pad_info_copy) {
      std::vector<dsize_t> pad_shape = pair.second.first.AsVector();

      for (size_t i = 0; i < pad_shape.size(); i++) {
        if (pad_shape[i] == TensorShape::kDimUnknown) {
          if (bucket_index + 1 >= bucket_boundaries_.size()) {
            std::string error_message =
              "Invalid data, requested to pad to bucket boundary failed, bucket index should be less than " +
              std::to_string(bucket_boundaries_.size()) + ", but got " + std::to_string(bucket_index);
            return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, error_message);
          }

          pad_shape[i] = bucket_boundaries_[bucket_index + 1] - 1;
        }
      }

      pair.second.first = TensorShape(pad_shape);
    }
  }

  // PadColumns will change the data in bucket
  RETURN_IF_NOT_OK(BatchOp::PadColumns(bucket, pad_info_copy, column_name_id_map_));

  TensorRow batched_bucket;
  RETURN_IF_NOT_OK(BatchOp::BatchRows(bucket, &batched_bucket, batch_size));
  (*bucket)->clear();

  RETURN_IF_NOT_OK(out_connector_->Add(std::move(batched_bucket)));

  batch_count_++;

  return Status::OK();
}

// Computing the assignment of the column name map and check compute input columns.
Status BucketBatchByLengthOp::ComputeColMap() {
  RETURN_IF_NOT_OK(DatasetOp::ComputeColMap());

  for (const auto &inCol : length_dependent_columns_) {
    bool found = column_name_id_map_.find(inCol) != column_name_id_map_.end() ? true : false;
    if (!found) {
      std::string err_msg = "Invalid parameter, input column name: " + inCol + " doesn't exist in the dataset columns.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
