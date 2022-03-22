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

#include "transfer_shape_according_to_format.h"
#include <algorithm>
#include "framework/common/debug/ge_log.h"

namespace transformer {
using namespace ge;

namespace {
  static std::unique_ptr<AxisUtil> axisutil_object(new(std::nothrow) AxisUtil());
  static std::map<ge::Format, GetNewShapeByAxisValueAndFormatPtr> getNewShapeFuncMap = {
    {ge::FORMAT_NCHW, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue)},
    {ge::FORMAT_NHWC, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue)},
    {ge::FORMAT_NC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_NDC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNDC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzShapeByAxisValue)},
    {ge::FORMAT_HWCN, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue)},
    {ge::FORMAT_C1HWNCoC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_NZ, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNzShapeByAxisValue)},
    {ge::FORMAT_NC1HWC0_C04, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_C04, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzC04ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_G, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzGShapeByAxisValue)},
    {ge::FORMAT_CHWN, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetCHWNShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_3D, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFz3DShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE,
        std::make_shared<GetNewShapeByAxisValueAndFormat>(
            ShapeTransferAccordingToFormat::GetFz3DTransposeShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_ZN_LSTM, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzLstmShapeByAxisValue)}};

    static std::map<ge::DataType, uint32_t> mapOfDtypeAndC0 = {
    {ge::DT_FLOAT16, SHAPE_NUMBER_16}, {ge::DT_FLOAT, SHAPE_NUMBER_16},  {ge::DT_INT8, SHAPE_NUMBER_32},
    {ge::DT_INT16, SHAPE_NUMBER_16},   {ge::DT_INT32, SHAPE_NUMBER_16},  {ge::DT_INT64, SHAPE_NUMBER_16},
    {ge::DT_UINT8, SHAPE_NUMBER_32},   {ge::DT_UINT16, SHAPE_NUMBER_16}, {ge::DT_UINT32, SHAPE_NUMBER_16},
    {ge::DT_UINT64, SHAPE_NUMBER_16},  {ge::DT_BOOL, SHAPE_NUMBER_16},   {ge::DT_UINT1, SHAPE_NUMBER_256},
    {ge::DT_INT2, SHAPE_NUMBER_128},   {ge::DT_UINT2, SHAPE_NUMBER_128}, {ge::DT_INT4, SHAPE_NUMBER_64}};
}
ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat(void) {}

bool ShapeTransferAccordingToFormat::GetNDC1HWC0ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>
        &axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  shape.SetDimNum(DIM_SIZE_SIX);
  shape.SetDim(0, axis_value[AXIS_N]);
  shape.SetDim(1, axis_value[AXIS_D]);
  shape.SetDim(2, axis_value[AXIS_C1]);
  shape.SetDim(3, axis_value[AXIS_H]);
  shape.SetDim(4, axis_value[AXIS_W]);
  shape.SetDim(5, axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, axis_value[AXIS_N]);
  shape.SetDim(1, axis_value[AXIS_C]);
  shape.SetDim(2, axis_value[AXIS_H]);
  shape.SetDim(3, axis_value[AXIS_W]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, axis_value[AXIS_N]);
  shape.SetDim(1, axis_value[AXIS_H]);
  shape.SetDim(2, axis_value[AXIS_W]);
  shape.SetDim(3, axis_value[AXIS_C]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_SIZE_FIVE);
  shape.SetDim(0, axis_value[AXIS_N]);
  shape.SetDim(1, axis_value[AXIS_C1]);
  shape.SetDim(2, axis_value[AXIS_H]);
  shape.SetDim(3, axis_value[AXIS_W]);
  shape.SetDim(4, axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  size_t size_of_original_vec = shape.GetDimNum();
  if (size_of_original_vec == SIZE_OF_CN) {
    /* size_of_original_vec - 1 mean the last value of original vec
     * size_of_original_vec - 2 mean the second last value of original vec */
    shape.SetDim((size_of_original_vec - MINUS_VALUE_ONE),
        DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE), SHAPE_NUMBER_16));
    shape.SetDim((size_of_original_vec - MINUS_VALUE_TWO),
        DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_TWO), axis_value[AXIS_C0]));
    shape.AppendDim(SHAPE_NUMBER_16);
    shape.AppendDim(axis_value[AXIS_C0]);
  } else {
    bool has_unknown_shape = axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                             axis_value[AXIS_C1] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_G] == UNKNOWN_SHAPE_VALUE;
    int64_t hwc1 = UNKNOWN_SHAPE_VALUE;
    int64_t axis_n_val = axis_value[AXIS_N];
    if (!has_unknown_shape) {
      int64_t group_val = axis_value[AXIS_G];
      int64_t axis_c1_val = axis_value[AXIS_C1];
      int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
      int64_t axis_c_val = axis_value[AXIS_C];
      if (group_val > GROUPS_DEFAULT_VALUE && axis_n_val >= group_val) {
        int64_t enlarge_value =
                GetAsisEnlargeValue(axis_c_val, axis_n_val / group_val, axis_value[AXIS_C0], group_val);
        axis_g_val = DivisionCeiling(group_val, enlarge_value);
        INT64_MULCHECK(axis_c_val, enlarge_value);
        axis_c_val *= enlarge_value;
        INT64_MULCHECK(axis_n_val / group_val, enlarge_value);
        axis_n_val = (axis_n_val / group_val) * enlarge_value;
        axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
      }
      INT64_MULCHECK(axis_g_val, axis_c1_val);
      int64_t g_c1_val = axis_g_val * axis_c1_val;
      INT64_MULCHECK(g_c1_val, axis_value[AXIS_H]);
      g_c1_val *= axis_value[AXIS_H];
      INT64_MULCHECK(g_c1_val, axis_value[AXIS_W]);
      hwc1 = g_c1_val * axis_value[AXIS_W];
    }
    shape.SetDimNum(DIM_DEFAULT_SIZE);
    shape.SetDim(0, hwc1);
    shape.SetDim(1, DivisionCeiling(axis_n_val, NI));
    shape.SetDim(2, NI);
    shape.SetDim(3, axis_value[AXIS_C0]);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, axis_value[AXIS_H]);
  shape.SetDim(1, axis_value[AXIS_W]);
  shape.SetDim(2, axis_value[AXIS_C]);
  shape.SetDim(3, axis_value[AXIS_N]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue(ge::GeShape &shape,
                                                                  const vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_SIZE_SIX);
  shape.SetDim(0, axis_value[AXIS_C1]);
  shape.SetDim(1, axis_value[AXIS_H]);
  shape.SetDim(2, axis_value[AXIS_W]);
  shape.SetDim(3, axis_value[AXIS_N]);
  shape.SetDim(4, axis_value[AXIS_Co]);
  shape.SetDim(5, axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNzShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>& axis_value) {

  CHECK(shape.IsScalar(), GELOGD("Origin shape is empty!"), return true);
  CHECK(axis_value.empty() || axis_value.size() <= AXIS_C0,
        GELOGD("AxisValue is empty or its size %zu <= AXIS_C0[%u]", axis_value.size(), AXIS_C0), return true);
  size_t size_of_original_vec = shape.GetDimNum();
  if (size_of_original_vec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    GELOGD("nd_value's dim num is less than 2!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  int64_t dim_back_two = shape.GetDim(size_of_original_vec - MINUS_VALUE_TWO);
  int64_t dim_back_one = shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE);
  shape.SetDim((size_of_original_vec - MINUS_VALUE_ONE), DivisionCeiling(dim_back_two, (int64_t)SHAPE_NUMBER_16));

  shape.SetDim((size_of_original_vec - MINUS_VALUE_TWO), DivisionCeiling(dim_back_one, axis_value[AXIS_C0]));
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool CheckInputParam(const ShapeAndFormat& shapeAndFormatInfo, ge::Format primary_new_format) {
  bool invalid_format =
      (shapeAndFormatInfo.oldFormat == ge::FORMAT_RESERVED || shapeAndFormatInfo.oldFormat >= ge::FORMAT_END) ||
      (primary_new_format == ge::FORMAT_RESERVED || primary_new_format >= ge::FORMAT_END);
  if (invalid_format) {
    GELOGE(GRAPH_FAILED, "Old format %u or new format %u is invalid!", shapeAndFormatInfo.oldFormat,
           primary_new_format);
    return false;
  }

  if (shapeAndFormatInfo.currentDataType == ge::DT_UNDEFINED ||
      shapeAndFormatInfo.currentDataType >= ge::DT_MAX) {
    GELOGE(GRAPH_FAILED, "currentDataType %u is invalid!", shapeAndFormatInfo.currentDataType);
    return false;
  }
  if (axisutil_object == nullptr) {
    return false;
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat& shapeAndFormatInfo, int64_t* c) {
  if (shapeAndFormatInfo.oldFormat == ge::FORMAT_ND &&
      FE_ORIGIN_FORMAT_VECTOR.count(static_cast<int>(shapeAndFormatInfo.newFormat)) > 0) {
    GELOGD("Do not need to do shape transformation from ND to original format.");
    return SUCCESS;
  }

  ge::Format primary_new_format = static_cast<Format>(GetPrimaryFormat(shapeAndFormatInfo.newFormat));
  if (!CheckInputParam(shapeAndFormatInfo, primary_new_format)) {
    return false;
  }

  if (!axisutil_object->HasAxisValueFunc(shapeAndFormatInfo.oldFormat)) {
    return true;
  }

  auto iterGetNewShapeFunc = getNewShapeFuncMap.find(primary_new_format);
  if (iterGetNewShapeFunc == getNewShapeFuncMap.end()) {
    GELOGD("Can not get new shape of new format %u!", primary_new_format);
    return true;
  }
  GELOGD("Original format is %u, new format %u", shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
  GetNewShapeByAxisValueAndFormatPtr getNewShapeFunc = iterGetNewShapeFunc->second;

  vector<int64_t> axis_value(static_cast<size_t>(AXIS_BOTTOM), 1);

  int64_t group = static_cast<int64_t>(ge::GetSubFormat(shapeAndFormatInfo.newFormat));
  if (group > GROUPS_DEFAULT_VALUE) {
    axis_value[AXIS_G] = group;
  }

  uint32_t c0;
  if (mapOfDtypeAndC0.empty()) {
    c0 = SHAPE_NUMBER_16;
  } else {
    auto iterGetC0 = mapOfDtypeAndC0.find(shapeAndFormatInfo.currentDataType);
    if (iterGetC0 == mapOfDtypeAndC0.end()) {
      GELOGE(GRAPH_FAILED, "Dtype is not support.");
      return true;
    }
    c0 = iterGetC0->second;
  }

  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (shapeAndFormatInfo.newFormat == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }
  if (axisutil_object == nullptr) {
    return false;
  }
  bool ret = axisutil_object->GetAxisValueByOriginFormat(
      shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.oldShape, c0, axis_value);
  if (ret != true && shapeAndFormatInfo.newFormat != ge::FORMAT_FRACTAL_NZ) {
    return true;
  }

  (*getNewShapeFunc)(shapeAndFormatInfo.oldShape, axis_value);
  if (c != nullptr) {
    *c = axis_value[AXIS_C];
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetCHWNShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, axis_value[AXIS_C]);
  shape.SetDim(1, axis_value[AXIS_H]);
  shape.SetDim(2, axis_value[AXIS_W]);
  shape.SetDim(3, axis_value[AXIS_N]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFz3DShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }

  /* axis_value is initialized as a size 6 vector. */
  bool has_unknown_shape = axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C1] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_G] == UNKNOWN_SHAPE_VALUE;

  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  int64_t group_val = axis_value[AXIS_G];
  int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
  int64_t axis_n_val = axis_value[AXIS_N];
  int64_t axis_c_val = axis_value[AXIS_C];
  int64_t axis_c1_val = axis_value[AXIS_C1];
  if (!has_unknown_shape) {
    if (group_val > GROUPS_DEFAULT_VALUE && axis_n_val >= group_val) {
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val / group_val,
                                                  axis_value[AXIS_C0], group_val);
      axis_g_val = DivisionCeiling(group_val, enlarge_value);
      INT64_MULCHECK(axis_c_val, enlarge_value);
      axis_c_val *= enlarge_value;
      INT64_MULCHECK(axis_n_val / group_val, enlarge_value);
      axis_n_val = (axis_n_val / group_val) * enlarge_value;
      axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
    }
    INT64_MULCHECK(axis_g_val, axis_c1_val);
    int64_t g_c1_val = axis_g_val * axis_c1_val;
    INT64_MULCHECK(g_c1_val, axis_value[AXIS_D]);
    g_c1_val *= axis_value[AXIS_D];
    INT64_MULCHECK(g_c1_val, axis_value[AXIS_H]);
    g_c1_val *= axis_value[AXIS_H];
    INT64_MULCHECK(g_c1_val, axis_value[AXIS_W]);
    gdhwc1 = g_c1_val * axis_value[AXIS_W];
  }
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, gdhwc1);
  shape.SetDim(1, DivisionCeiling(axis_n_val, NI));
  shape.SetDim(2, NI);
  shape.SetDim(3, axis_value[AXIS_C0]);

  return true;
}

bool ShapeTransferAccordingToFormat::GetFz3DTransposeShapeByAxisValue(ge::GeShape &shape, const vector<int64_t>
        &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  for (auto ele : axis_value) {
    GELOGI("value is %ld", ele);
  }
  int64_t n1 = DivisionCeiling(axis_value[AXIS_N], NI);
  int64_t dhwn1 = n1 * axis_value[AXIS_H] * axis_value[AXIS_W] * axis_value[AXIS_D];
  if (n1 == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
      axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE) {
    dhwn1 = UNKNOWN_SHAPE_VALUE;
  }

  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, dhwn1);
  if (axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    shape.SetDim(1, UNKNOWN_SHAPE_VALUE);
  } else {
    shape.SetDim(1, axis_value[AXIS_C1]);
  }
  shape.SetDim(2, NI);
  shape.SetDim(3, axis_value[AXIS_C0]);

  return true;
}

bool ShapeTransferAccordingToFormat::GetFzLstmShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  int64_t h = axis_value[AXIS_N] / LSTM_NI;
  int64_t i = axis_value[AXIS_C] - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);

  int64_t second_element_of_fz_lstm = LSTM_NI * DivisionCeiling(h, NI);
  if (axis_value[AXIS_N] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, first_element_of_fz_lstm);
  shape.SetDim(1, second_element_of_fz_lstm);
  shape.SetDim(2, NI);
  shape.SetDim(3, NI);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzC04ShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  int64_t x = SHAPE_DIM_VALUE_C04 * axis_value[AXIS_H] * axis_value[AXIS_W];
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, DivisionCeiling(x, X0));
  shape.SetDim(1, DivisionCeiling(axis_value[AXIS_N], NI));
  shape.SetDim(2, NI);
  shape.SetDim(3, X0);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzGShapeByAxisValue(ge::GeShape &shape, const vector<int64_t> &axis_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  int64_t new_c = axis_value[AXIS_C] * axis_value[AXIS_G];
  int64_t new_c1 = DivisionCeiling(new_c, axis_value[AXIS_C0]);
  int64_t hwc1 = new_c1 * axis_value[AXIS_H] * axis_value[AXIS_W];
  shape.SetDimNum(DIM_DEFAULT_SIZE);
  shape.SetDim(0, hwc1);
  shape.SetDim(1, DivisionCeiling(axis_value[AXIS_N], NI));
  shape.SetDim(2, NI);
  shape.SetDim(3, axis_value[AXIS_C0]);
  return true;
}

int64_t GetGreatestCommonDivisor(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  return GetGreatestCommonDivisor(y, x % y);
}

int64_t GetLeastCommonMultiple(int64_t x, int64_t y) {
  if (x == 0 || y == 0) {
    return 0;
  }
  return (x * y) / GetGreatestCommonDivisor(x, y);
}

int64_t ShapeTransferAccordingToFormat::GetAsisEnlargeValue(const int64_t& cin, const int64_t& cout, const int64_t& c0,
                                                            const int64_t& group) {
  if (cin == 0 || cout == 0) {
    return 0;
  }
  int64_t tmp = GetLeastCommonMultiple(GetLeastCommonMultiple(cin, c0) / cin, GetLeastCommonMultiple(cout, NI) / cout);
  return std::min(tmp, group);
}
} // namespace transformer
