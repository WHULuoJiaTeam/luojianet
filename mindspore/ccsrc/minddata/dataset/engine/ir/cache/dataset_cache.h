/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_H_

#include <memory>

#include "minddata/dataset/engine/datasetops/dataset_op.h"
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "minddata/dataset/util/status.h"

namespace mindspore::dataset {
class DatasetCache {
 public:
  virtual ~DatasetCache() = default;

  virtual Status Build() = 0;

  virtual Status ValidateParams() = 0;

  virtual Status CreateCacheOp(int32_t num_workers, int32_t connector_queue_size, std::shared_ptr<SamplerObj> sampler,
                               std::shared_ptr<DatasetOp> *ds) = 0;

  virtual Status CreateCacheLookupOp(int32_t num_workers, int32_t connector_queue_size,
                                     std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetOp> *ds) = 0;

  virtual Status CreateCacheMergeOp(int32_t num_workers, int32_t connector_queue_size,
                                    std::shared_ptr<DatasetOp> *ds) = 0;

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

#ifndef ENABLE_ANDROID
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<DatasetCache> *cache);
#endif
};
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_IR_CACHE_DATASET_CACHE_H_
