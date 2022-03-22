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

#ifndef GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_
#define GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_

namespace ge {
namespace formats {
static const int kCubeSize = 16;
static const int kNiSize = 16;
static const int64_t kShapeItemNumMAX = 1024UL * 1024UL * 1024UL * 1024UL;


enum NchwDimIndex {
  kNchwN,
  kNchwC,
  kNchwH,
  kNchwW,
  kNchwDimsNum
};

enum NhwcDimIndex {
  kNhwcN,
  kNhwcH,
  kNhwcW,
  kNhwcC,
  kNhwcDimsNum
};

enum HwcnDimIndex {
  kHwcnH,
  kHwcnW,
  kHwcnC,
  kHwcnN,
  kHwcnDimsNum
};

enum ChwnDimIndex {
  kChwnC,
  kChwnH,
  kChwnW,
  kChwnN,
  kChwnDimsNum
};

enum Nc1hwc0DimIndex {
  kNc1hwc0N,
  kNc1hwc0C1,
  kNc1hwc0H,
  kNc1hwc0W,
  kNc1hwc0C0,
  kNc1hwc0DimsNum
};

enum C1hwncoc0DimIndex {
  kC1hwncoc0C1,
  kC1hwncoc0H,
  kC1hwncoc0W,
  kC1hwncoc0N,
  kC1hwncoc0Co,
  kC1hwncoc0C0,
  kC1hwncoc0DimsNum
};

enum FracZDimIndex {
  kFracZHWC1,
  kFracZN0,
  kFracZNi,
  kFracZC0,
  kFracZDimsNum
};

enum DhwcnDimIndex {
  kDhwcnD,
  kDhwcnH,
  kDhwcnW,
  kDhwcnC,
  kDhwcnN,
  kDhwcnDimsNum
};

enum DhwncDimIndex {
  kDhwncD,
  kDhwncH,
  kDhwncW,
  kDhwncN,
  kDhwncC,
  kDhwncDimsNum
};
}  // namespace formats
}  // namespace ge
#endif  // GE_COMMON_FORMATS_UTILS_FORMATS_DEFINITIONS_H_
