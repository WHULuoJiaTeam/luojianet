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

#ifndef INC_COMMON_UTILS_AI_CORE_OP_SLICE_INFO_H
#define INC_COMMON_UTILS_AI_CORE_OP_SLICE_INFO_H

#include <vector>
#include "graph/op_desc.h"
#include "aicore_util_types.h"

namespace fe {
class InputSplitInfoImpl;
using InputSplitInfoImplPtr = std::shared_ptr<InputSplitInfoImpl>;
class InputSplitInfo;
using InputSplitInfoPtr = std::shared_ptr<InputSplitInfo>;
class OutputSplitInfoImpl;
using OutputSplitInfoImplPtr = std::shared_ptr<OutputSplitInfoImpl>;
class OutputSplitInfo;
using OutputSplitInfoPtr = std::shared_ptr<OutputSplitInfo>;
class InputReduceInfoImpl;
using InputReduceInfoImplPtr = std::shared_ptr<InputReduceInfoImpl>;
class InputReduceInfo;
using InputReduceInfoPtr = std::shared_ptr<InputReduceInfo>;
class OutputReduceInfoImpl;
using OutputReduceInfoImplPtr = std::shared_ptr<OutputReduceInfoImpl>;
class OutputReduceInfo;
using OutputReduceInfoPtr = std::shared_ptr<OutputReduceInfo>;
class AxisSplitMapImpl;
using AxisSplitMapImplPtr = std::shared_ptr<AxisSplitMapImpl>;
class AxisSplitMap;
using AxisSplitMapPtr = std::shared_ptr<AxisSplitMap>;
class AxisReduceMapImpl;
using AxisReduceMapImplPtr = std::shared_ptr<AxisReduceMapImpl>;
class AxisReduceMap;
using AxisReduceMapPtr = std::shared_ptr<AxisReduceMap>;
class OpCalcInfoImpl;
using OpCalcInfoImplPtr = std::shared_ptr<OpCalcInfoImpl>;
class OpCalcInfo;
using OpCalcInfoPtr = std::shared_ptr<OpCalcInfo>;

class InputSplitInfo {
 public:
  InputSplitInfo();
  InputSplitInfo &operator = (const InputSplitInfo &input_split_info);
  ~InputSplitInfo();
  bool Initialize();
  size_t GetIndex() const;
  std::vector<int64_t> GetAxis() const;
  std::vector<int64_t> GetHeadOverLap() const;
  std::vector<int64_t> GetTailOverLap() const;
  void SetIndex(const size_t& idx);
  void SetAxis(std::vector<int64_t>& axis);
  void SetHeadOverLap(std::vector<int64_t>& head_over_lap);
  void SetTailOverLap(std::vector<int64_t>& tail_over_lap);
  bool IsPtrNull();
 private:
  InputSplitInfoImplPtr split_impl_{nullptr};
};

class OutputSplitInfo {
 public:
  OutputSplitInfo();
  OutputSplitInfo &operator = (const OutputSplitInfo &output_split_info);
  ~OutputSplitInfo();
  bool Initialize();
  size_t GetIndex() const;
  std::vector<int64_t> GetAxis() const;
  void SetIndex(const size_t& idx);
  void SetAxis(std::vector<int64_t>& axis);
  bool IsPtrNull();
 private:
  OutputSplitInfoImplPtr split_impl_{nullptr};
};

class InputReduceInfo {
 public:
  InputReduceInfo();
  InputReduceInfo &operator = (const InputReduceInfo &input_reduce_info);
  ~InputReduceInfo();
  bool Initialize();
  size_t GetIndex() const;
  std::vector<int64_t> GetAxis() const;
  void SetIndex(const size_t& idx);
  void SetAxis(std::vector<int64_t>& axis);
  bool IsPtrNull();
 private:
  InputReduceInfoImplPtr reduce_impl_{nullptr};
};

class OutputReduceInfo {
 public:
  OutputReduceInfo();
  OutputReduceInfo &operator = (const OutputReduceInfo &output_reduce_info);
  ~OutputReduceInfo();
  bool Initialize();
  size_t GetIndex() const;
  OpReduceType GetReduceType() const;
  bool GetIsAtomic() const;
  void SetIndex(const size_t& idx);
  void SetReduceType(const OpReduceType& reduce_type);
  void SetIsAtomic(const bool& is_atomic);
  bool IsPtrNull();
 private:
  OutputReduceInfoImplPtr reduce_impl_{nullptr};
};

class AxisSplitMap {
 public:
  friend class AxisSplitMapImpl;
  AxisSplitMap();
  AxisSplitMap &operator = (const AxisSplitMap &axis_split_map);
  ~AxisSplitMap();
  bool Initialize();
  std::vector<InputSplitInfoPtr> GetInputSplitInfos() const;
  std::vector<OutputSplitInfoPtr> GetOutputSplitInfos() const;
  std::vector<InputSplitInfo> GetInputSplitInfoVec() const;
  std::vector<OutputSplitInfo> GetOutputSplitInfoVec() const;
  void AddInputSplitInfo(InputSplitInfo& input_split_info);
  void SetInputSplitInfos(std::vector<InputSplitInfo>& input_split_vec);
  void SetInputSplitInfos(std::vector<InputSplitInfoPtr>& input_split_vec);
  void AddOutputSplitInfo(OutputSplitInfo& output_split_info);
  void SetOutputSplitInfos(std::vector<OutputSplitInfo>& output_split_vec);
  void SetOutputSplitInfos(std::vector<OutputSplitInfoPtr>& output_split_vec);
  bool IsPtrNull();
 private:
  AxisSplitMapImplPtr aixs_split_impl_{nullptr};
};

class AxisReduceMap {
 public:
  AxisReduceMap();
  AxisReduceMap &operator = (const AxisReduceMap &axis_reduce_map);
  ~AxisReduceMap();
  bool Initialize();
  friend class AxisReduceMapImpl;
  std::vector<InputReduceInfoPtr> GetInputReduceInfos() const;
  std::vector<OutputReduceInfoPtr> GetOutputReduceInfos() const;
  std::vector<InputReduceInfo> GetInputReduceInfoVec() const;
  std::vector<OutputReduceInfo> GetOutputReduceInfoVec() const;
  void AddInputReduceInfo(InputReduceInfo& input_reduce_info);
  void SetInputReduceInfos(std::vector<InputReduceInfo>& input_reduce_vec);
  void SetInputReduceInfos(std::vector<InputReduceInfoPtr>& input_reduce_vec);
  void AddOutputReduceInfo(OutputReduceInfo& output_reduce_info);
  void SetOutputReduceInfos(std::vector<OutputReduceInfo>& output_reduce_vec);
  void SetOutputReduceInfos(std::vector<OutputReduceInfoPtr>& output_reduce_vec);
  bool IsPtrNull();
 private:
  AxisReduceMapImplPtr aixs_reduce_impl_{nullptr};
};

class OpCalcInfo {
 public:
  OpCalcInfo();
  ~OpCalcInfo();
  bool Initialize();
  std::vector<AxisSplitMapPtr> GetAxisSplitMaps() const;
  std::vector<AxisReduceMapPtr> GetAxisReduceMaps() const;
  std::vector<AxisSplitMap> GetAxisSplitMapVec() const;
  std::vector<AxisReduceMap> GetAxisReduceMapVec() const;
  OpL1FusionType GetL1FusionEnable() const;
  int64_t GetMinTbeL1Space() const;
  void AddAxisSplitMap(AxisSplitMap& axis_split_map);
  void SetAxisSplitMaps(std::vector<AxisSplitMap>& axis_split_vec);
  void SetAxisSplitMaps(std::vector<AxisSplitMapPtr>& axis_split_vec);
  void AddAxisReduceMap(AxisReduceMap& axis_reduce_map);
  void SetAxisReduceMaps(std::vector<AxisReduceMap>& axis_reduce_vec);
  void SetAxisReduceMaps(std::vector<AxisReduceMapPtr>& axis_reduce_vec);
  void SetL1FusionEnable(const OpL1FusionType& l1_fusion_enable);
  void SetMinTbeL1Space(const int64_t& min_tbe_l1_space);
  void DelAxisSplitMapBaseAxis(std::vector<int64_t>& axis);
  bool IsPtrNull();
 private:
  OpCalcInfoImplPtr op_calc_info_impl_{nullptr};
};
}  // namespace fe
#endif // INC_COMMON_UTILS_AI_CORE_OP_SLICE_INFO_H
