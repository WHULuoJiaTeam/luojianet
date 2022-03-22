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

#include "axis_util.h"
#include "graph/types.h"
#include "framework/common/debug/ge_log.h"

namespace transformer {
using std::vector;
using namespace ge;
namespace {
static std::map<ge::Format, GetAxisValueInfoByFormatPtr> getAxisValueFuncMap = {
    {FORMAT_NCHW, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByNCHW)},
    {FORMAT_NHWC, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByNHWC)},
    {FORMAT_NC1HWC0, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByNC1HWC0)},
    {FORMAT_HWCN, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByHWCN)},
    {FORMAT_ND, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByND)},
    {FORMAT_C1HWNCoC0, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByC1HWNCoC0)},
    {FORMAT_NDHWC, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByNDHWC)},
    {FORMAT_NCDHW, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByNCDHW)},
    {FORMAT_DHWCN, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByDHWCN)},
    {FORMAT_DHWNC, std::make_shared<GetAxisValueInfoByFormat>(AxisUtil::GetAxisValueByDHWNC)}};
}


AxisUtil::AxisUtil() {}

bool AxisUtil::GetAxisValueByOriginFormat(const Format &format, const ge::GeShape &shape, const uint32_t &c0,
                                          vector<int64_t> &axisValue) {
  auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
  if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
    GELOGI("Can not get axis value of old format %u!", format);
    return false;
  }
  GetAxisValueInfoByFormatPtr getAxisFunc = iterGetAxisFunc->second;
  CHECK_NOTNULL(getAxisFunc);
  return (*getAxisFunc)(shape, c0, axisValue);
}

bool AxisUtil::HasAxisValueFunc(const Format &format) {
  auto iterGetAxisFunc = getAxisValueFuncMap.find(format);
  if (iterGetAxisFunc == getAxisValueFuncMap.end()) {
    GELOGI("Can not get axis value of format %u!", format);
    return false;
  }
  return true;
}

bool AxisUtil::CheckParams(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  if (shape.GetDimNum() < DIM_DEFAULT_SIZE) {
    /* Before this funcion, we should call function PadDimensionTo4. */
    GELOGI("Dimension size %zu is invalid.", shape.GetDimNum());
    return false;
  }
  if (c0 == 0) {
    GELOGE(GRAPH_FAILED, "[ERROR]c0 is zero!");
    return false;
  }

  return true;
}

bool AxisUtil::GetAxisValueByND(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  /* To differentiate the input datatype of int8 and others */
  axisValue[AXIS_C0] = c0;
  if (shape.GetDimNum() == NCHW_DIMENSION_NUM) {
    axisValue[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
    axisValue[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
    axisValue[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
    axisValue[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
    axisValue[AXIS_C1] = DivisionCeiling(shape.GetDim(AXIS_NCHW_DIM_C), (int64_t)c0);
    axisValue[AXIS_Co] = c0;
  }
  return true;
}

bool AxisUtil::GetAxisValueByNCHW(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level.If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(shape, c0, axisValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
  axisValue[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
  axisValue[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
  axisValue[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
  axisValue[AXIS_C1] = DivisionCeiling(shape.GetDim(AXIS_NCHW_DIM_C), (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNHWC(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level.If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(shape, c0, axisValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = shape.GetDim(AXIS_NHWC_DIM_N);
  axisValue[AXIS_C] = shape.GetDim(AXIS_NHWC_DIM_C);
  axisValue[AXIS_H] = shape.GetDim(AXIS_NHWC_DIM_H);
  axisValue[AXIS_W] = shape.GetDim(AXIS_NHWC_DIM_W);
  axisValue[AXIS_C1] = DivisionCeiling(shape.GetDim(AXIS_NHWC_DIM_C), (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByNC1HWC0(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  CHECK(CheckParams(shape, c0, axisValue) != true, GELOGE(GRAPH_FAILED,"[ERROR]Parameter is invalid!"),
        return false);

  if (shape.GetDimNum() == DIM_SIZE_FIVE) {
    axisValue[AXIS_C1] = shape.GetDim(AXIS_NC1HWC0_DIM_C1);
    axisValue[AXIS_C0] = shape.GetDim(AXIS_NC1HWC0_DIM_C0);
    axisValue[AXIS_C] = axisValue[AXIS_C1] * axisValue[AXIS_C0];
  } else {
    axisValue[AXIS_C1] = DivisionCeiling(shape.GetDim(AXIS_NCHW_DIM_C), (int64_t)c0);
    axisValue[AXIS_C0] = c0;
    axisValue[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
  }

  axisValue[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
  axisValue[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
  axisValue[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
  return true;
}

bool AxisUtil::GetAxisValueByHWCN(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  // TODO: temporarily modified to warning level. If modified normally, it needs complementary dimension for origin shape
  CHECK(CheckParams(shape, c0, axisValue) != true, GELOGW("[WARNING]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = shape.GetDim(AXIS_HWCN_DIM_N);
  axisValue[AXIS_C] = shape.GetDim(AXIS_HWCN_DIM_C);
  axisValue[AXIS_H] = shape.GetDim(AXIS_HWCN_DIM_H);
  axisValue[AXIS_W] = shape.GetDim(AXIS_HWCN_DIM_W);
  axisValue[AXIS_C1] = DivisionCeiling(shape.GetDim(AXIS_HWCN_DIM_C), (int64_t)c0);
  axisValue[AXIS_Co] = c0;
  return true;
}

bool AxisUtil::GetAxisValueByC1HWNCoC0(const ge::GeShape &shape, const uint32_t &c0, vector<int64_t> &axisValue) {
  CHECK(axisValue.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axisValue[AXIS_C0] = c0;
  CHECK(CheckParams(shape, c0, axisValue) != true, GELOGE(GRAPH_FAILED, "[ERROR]Parameter is invalid!"),
        return false);

  axisValue[AXIS_N] = shape.GetDim(AXIS_C1HWNCoC0_DIM_N);
  axisValue[AXIS_C] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1) * c0;
  axisValue[AXIS_H] = shape.GetDim(AXIS_C1HWNCoC0_DIM_H);
  axisValue[AXIS_W] = shape.GetDim(AXIS_C1HWNCoC0_DIM_W);
  axisValue[AXIS_C1] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1);
  axisValue[AXIS_Co] = shape.GetDim(AXIS_C1HWNCoC0_DIM_Co);
  return true;
}

bool AxisUtil::GetAxisValueByNDHWC(const ge::GeShape &shape, const uint32_t& c0, vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_N] = shape.GetDim(NDHWC_DIM_N);
  int64_t axis_c_val = shape.GetDim(NDHWC_DIM_C);

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = shape.GetDim(NDHWC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NDHWC_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = shape.GetDim(NDHWC_DIM_D);
  return true;
}

bool AxisUtil::GetAxisValueByNCDHW(const ge::GeShape &shape, const uint32_t& c0, vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_N] = shape.GetDim(NCDHW_DIM_N);
  int64_t axis_c_val = shape.GetDim(NCDHW_DIM_C);

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = shape.GetDim(NCDHW_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NCDHW_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = shape.GetDim(NCDHW_DIM_D);
  return true;
}

bool AxisUtil::GetAxisValueByDHWCN(const ge::GeShape &shape, const uint32_t& c0, vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_N] = shape.GetDim(DHWCN_DIM_N);
  int64_t axis_c_val = shape.GetDim(DHWCN_DIM_C);

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = shape.GetDim(DHWCN_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWCN_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = shape.GetDim(DHWCN_DIM_D);
  return true;
}

bool AxisUtil::GetAxisValueByDHWNC(const ge::GeShape &shape, const uint32_t& c0, vector<int64_t>& axis_value) {
  CHECK(axis_value.empty(), GELOGI("AxisValue is empty!"), return true);
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);

  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_N] = shape.GetDim(DHWNC_DIM_N);
  int64_t axis_c_val = shape.GetDim(DHWNC_DIM_C);

  axis_value[AXIS_C] = axis_c_val;
  axis_value[AXIS_H] = shape.GetDim(DHWNC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWNC_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_c_val, c0);
  axis_value[AXIS_C0] = c0;
  axis_value[AXIS_Co] = c0;
  axis_value[AXIS_D] = shape.GetDim(DHWNC_DIM_D);
  return true;
}
} // namespace transformer
