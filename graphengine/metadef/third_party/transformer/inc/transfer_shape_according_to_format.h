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

#ifndef COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
#define COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_

#include <memory.h>
#include <functional>
#include <vector>
#include "graph/types.h"
#include "axis_util.h"
#include "graph/ge_tensor.h"

namespace transformer {
using std::vector;

enum OpImplType {
  EN_IMPL_CUSTOM_CONSTANT_CCE = 0,    // custom constant op
  EN_IMPL_CUSTOM_TIK,                 // custom tik op
  EN_IMPL_CUSTOM_TBE,                 // custom tbe op
  EN_IMPL_HW_CONSTANT_CCE,            // Huawei built-in constant op
  EN_IMPL_HW_GENERAL_CCE,             // Huawei built-in cce op
  EN_IMPL_HW_TIK,                     // Huawei built-in tik op
  EN_IMPL_HW_TBE,                     // Huawei built-in tbe op
  EN_IMPL_RL,                         // RL op
  EN_IMPL_PLUGIN_TBE,                 // Huawei built-in tbe plugin op
  EN_IMPL_VECTOR_CORE_HW_TBE,         // Huawei built-in tbe op
  EN_IMPL_VECTOR_CORE_CUSTOM_TBE,     // custom tbe op
  EN_IMPL_NON_PERSISTENT_CUSTOM_TBE,  // custom tbe op
  EN_RESERVED                         // reserved value
};

const uint32_t SHAPE_NUMBER_16 = 16;
const uint32_t SHAPE_NUMBER_32 = 32;
const uint32_t SHAPE_NUMBER_64 = 64;
const uint32_t SHAPE_NUMBER_128 = 128;
const uint32_t SHAPE_NUMBER_256 = 256;
const uint32_t SHAPE_DIM_VALUE_C04 = 4;
const uint32_t NI = 16;
const uint32_t MINUS_VALUE_ONE = 1;
const uint32_t MINUS_VALUE_TWO = 2;
const uint32_t SIZE_OF_CN = 2;
const uint32_t MINIMUM_NZ_SHAPE_DIM_NUM = 2;
const uint32_t GROUPS_DEFAULT_VALUE = 1;
const uint32_t UNKNOWN_SHAPE_VALUE = -1;
const uint32_t MINIMUM_ND_TO_RNN_SHAPE_NUM = 2;

const int32_t LSTM_NI = 4;
const int32_t X0 = 16;
const std::vector<uint32_t> vector_of_dtype_and_c0 = {
    SHAPE_NUMBER_16,  // DT_FLOAT = 0,
    SHAPE_NUMBER_16,  // DT_FLOAT16 = 1,
    SHAPE_NUMBER_32,  // DT_INT8 = 2,
    SHAPE_NUMBER_16,  // DT_INT32 = 3,
    SHAPE_NUMBER_32,  // DT_UINT8 = 4,
    SHAPE_NUMBER_16,
    SHAPE_NUMBER_16,  // DT_INT16 = 6,
    SHAPE_NUMBER_16,  // DT_UINT16 = 7,
    SHAPE_NUMBER_16,  // DT_UINT32 = 8,
    SHAPE_NUMBER_16,  // DT_INT64 = 9,
    SHAPE_NUMBER_16,  // DT_UINT64 = 10,
    SHAPE_NUMBER_16,  // DT_DOUBLE = 11,
    SHAPE_NUMBER_16,  // DT_BOOL = 12,
    SHAPE_NUMBER_16,  // DT_DUAL = 13,
    SHAPE_NUMBER_16,  // DT_DUAL_SUB_INT8 = 14,
    SHAPE_NUMBER_16,  // DT_DUAL_SUB_UINT8 = 15,
    SHAPE_NUMBER_16,  // DT_COMPLEX64 = 16,
    SHAPE_NUMBER_16,  // DT_COMPLEX128 = 17,
    SHAPE_NUMBER_16,  // DT_QINT8 = 18,
    SHAPE_NUMBER_16,  // DT_QINT16 = 19,
    SHAPE_NUMBER_16,  // DT_QINT32 = 20,
    SHAPE_NUMBER_16,  // DT_QUINT8 = 21,
    SHAPE_NUMBER_16,  // DT_QUINT16 = 22,
    SHAPE_NUMBER_16,  // DT_RESOURCE = 23,
    SHAPE_NUMBER_16,  // DT_STRING_REF = 24,
    SHAPE_NUMBER_16,  // DT_DUAL = 25,
    SHAPE_NUMBER_16,  // DT_VARIANT = 26,
    SHAPE_NUMBER_16,  // DT_BF16 = 27,
    SHAPE_NUMBER_16,  // DT_UNDEFINED,
    SHAPE_NUMBER_64,  // DT_INT4 = 29,
};
/* The first parameter is axis value, second is new shape and third is
 * op implementation type. */
using GetNewShapeByAxisValueAndFormat = std::function<bool(ge::GeShape&, vector<int64_t>&)>;

using GetNewShapeByAxisValueAndFormatPtr = std::shared_ptr<GetNewShapeByAxisValueAndFormat>;

struct CalcShapeExtraAttr {
  int64_t hidden_size;
  int64_t input_size;
};

struct ShapeAndFormatInfo {
  ge::GeShape &oldShape;
  const ge::Format &oldFormat;
  const ge::Format &newFormat;
  const ge::DataType &currentDataType;
  int64_t group_count;
  CalcShapeExtraAttr extra_attr;
  ShapeAndFormatInfo(ge::GeShape &old_shape, const ge::Format &old_format, const ge::Format &new_format,
                     const ge::DataType &data_type)
                     : oldShape(old_shape), oldFormat(old_format), newFormat(new_format), currentDataType(data_type),
                       group_count(1), extra_attr({1, 1}) {}
};

using ShapeAndFormat = struct ShapeAndFormatInfo;

static const std::unordered_set<int> FE_ORIGIN_FORMAT_VECTOR = {
  ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
  ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
  ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND
};

class ShapeTransferAccordingToFormat {
 public:
  ShapeTransferAccordingToFormat();

  ~ShapeTransferAccordingToFormat(){};

  ShapeTransferAccordingToFormat(const ShapeTransferAccordingToFormat&) = delete;

  ShapeTransferAccordingToFormat &operator=(const ShapeTransferAccordingToFormat&) = delete;

  bool GetShapeAccordingToFormat(ShapeAndFormat &inputAndOutputInfo, int64_t* c = nullptr);

  /* ----------Below is the function of getting new shape---------------------- */
  static bool GetNDC1HWC0ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetNCHWShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetNHWCShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetNC1HWC0ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFzShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetHWCNShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetC1HWNCoC0ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetNzShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFz3DShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFz3DTransposeShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFzLstmShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFzC04ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetFzGShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static bool GetCHWNShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value);

  static int64_t GetAsisEnlargeValue(const int64_t& cin, const int64_t& cout, const int64_t& c0, const int64_t& group);

};
} // namespace transformer

#endif  // COMMON_UTILS_TRANSFORMER_INC_TRANSFER_SHAPE_ACCORDING_TO_FORMAT_H_
