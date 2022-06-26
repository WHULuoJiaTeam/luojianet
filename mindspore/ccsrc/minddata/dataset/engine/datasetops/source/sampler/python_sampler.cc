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

#include "minddata/dataset/engine/datasetops/source/sampler/python_sampler.h"

#include <string>
#include <utility>

namespace mindspore {
namespace dataset {

PythonSamplerRT::PythonSamplerRT(int64_t num_samples, py::object py_sampler_instance, int64_t samples_per_tensor)
    : SamplerRT(num_samples, samples_per_tensor),
      need_to_reset_(false),
      py_sampler_instance(std::move(py_sampler_instance)) {}

Status PythonSamplerRT::GetNextSample(TensorRow *out) {
  RETURN_UNEXPECTED_IF_NULL(out);
  if (need_to_reset_) {
    (*out) = TensorRow(TensorRow::kFlagEOE);
  } else {
    if (HasChildSampler()) {
      RETURN_IF_NOT_OK(child_[0]->GetNextSample(&child_ids_));
    }

    std::shared_ptr<Tensor> sample_ids;
    {
      py::gil_scoped_acquire gil_acquire;
      if (Py_IsInitialized() == 0) {
        return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
      }
      try {
        py::object py_ret = py_sampler_instance.attr("_get_indices")();
        py::array np_sample_ids = py_ret.cast<py::array>();
        RETURN_IF_NOT_OK(Tensor::CreateFromNpArray(np_sample_ids, &sample_ids));  // copy numpy to tensor

        if (HasChildSampler()) {
          for (auto it = sample_ids->begin<int64_t>(); it != sample_ids->end<int64_t>(); ++it) {
            int64_t associated_child_id = 0;
            RETURN_IF_NOT_OK(GetAssociatedChildId(&associated_child_id, associated_child_id));
            *it = associated_child_id;
          }
        }
      } catch (const py::error_already_set &e) {
        return Status(StatusCode::kMDPyFuncException, e.what());
      } catch (const py::cast_error &e) {
        return Status(StatusCode::kMDPyFuncException,
                      "Invalid data, Python sampler iterator should return an integer index, but error raised: " +
                        std::string(e.what()));
      }
    }
    (*out) = {sample_ids};
    need_to_reset_ = true;
  }
  return Status::OK();
}

Status PythonSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }
  CHECK_FAIL_RETURN_UNEXPECTED(
    num_rows_ > 0, "[Internal ERROR] num_rows must be greater than 0, but got " + std::to_string(num_rows_));
  // Special value of 0 for num_samples means that the user wants to sample the entire set of data.
  // If the user asked to sample more rows than exists in the dataset, adjust the num_samples accordingly.
  if (num_samples_ == 0 || num_samples_ > num_rows_) {
    num_samples_ = num_rows_;
  }
  {
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
    }
    try {
      py_sampler_instance.attr("_handshake")(num_rows_, num_samples_);
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException,
                    "[Internal ERROR] python sampler execute _handshake failed: " + std::string(e.what()));
    }
  }

  is_initialized = true;
  return Status::OK();
}

Status PythonSamplerRT::ResetSampler() {
  CHECK_FAIL_RETURN_UNEXPECTED(need_to_reset_, "[Internal ERROR] Reset() Sampler called early or late.");
  need_to_reset_ = false;
  py::gil_scoped_acquire gil_acquire;
  if (Py_IsInitialized() == 0) {
    return Status(StatusCode::kMDPythonInterpreterFailure, "[Internal ERROR] Python Interpreter is finalized");
  }
  try {
    py_sampler_instance.attr("reset")();
  } catch (const py::error_already_set &e) {
    return Status(StatusCode::kMDPyFuncException, e.what());
  }

  if (HasChildSampler()) {
    RETURN_IF_NOT_OK(child_[0]->ResetSampler());
  }

  return Status::OK();
}

void PythonSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: PythonSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}
}  // namespace dataset
}  // namespace mindspore
