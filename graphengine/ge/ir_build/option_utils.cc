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
#include "ir_build/option_utils.h"
#include "common/util/error_manager/error_manager.h"
#include "external/ge/ge_api_types.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "graph/compute_graph.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"

using std::pair;
using std::string;
using std::vector;

namespace ge {
namespace {
const int64_t kDynamicInputDim = -1;
const int64_t kDynamicImageSizeNum = 2;
const size_t kMaxDynamicDimNum = 100;
const size_t kMaxNDDimNum = 4;
const size_t kMinNDDimNum = 1;
const size_t kSquareBracketsSize = 2;
const size_t kRangePairSize = 2;
const size_t kShapeRangeSize = 2;
const size_t kShapeRangeStrIndex = 2;
const size_t kShapeRangeStrSize = 1;
// datatype/formats from user to GE, Unified to util interface file later
const std::map<std::string, ge::DataType> kOutputTypeSupportDatatype = {
    {"FP32", ge::DT_FLOAT}, {"FP16", ge::DT_FLOAT16}, {"UINT8", ge::DT_UINT8}};
const char *const kOutputTypeSupport = "only support FP32, FP16, UINT8";
const std::set<std::string> kBufferOptimizeSupportOption = {"l1_optimize", "l2_optimize", "off_optimize",
                                                            "l1_and_l2_optimize"};
// The function is incomplete. Currently, only l2_optimize, off_optimize is supported.
const char *const kBufferOptimizeSupport = "only support l2_optimize, off_optimize";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT = "high_performance";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_PRECISON = "high_precision";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PRECISION_FOR_ALL = "high_precision_for_all";
const char *const IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PERFORMANCE_FOR_ALL = "high_performance_for_all";
const char *const kInputShapeSample1 = "\"input_name1:n1,c1,h1,w1\"";
const char *const kInputShapeSample2 = "\"input_name1:1,3,224,224\"";
const char *const kSplitError1 = "size not equal to 2 split by \":\"";
const char *const kEmptyError = "can not be empty";
const char *const kFloatNumError = "exist float number";
const char *const kDigitError = "is not digit";
const char *const kCompressWeightError = "it must be appointed when appoint parameter[--optypelist_for_implmode]";
const char *const kSelectImplmodeError = "only support high_performance, high_precision, "
                                         "high_precision_for_all, high_performance_for_all";
const char *const kDynamicBatchSizeError = "It can only contains digit, \",\", \" \"";
const char *const kDynamicImageSizeError = "It can only contains digit, \",\", \" \" and \";\"";
const char *const kKeepDtypeError = "file not found";
const char *const kInputShapeRangeInvalid = "format of shape range is invalid";
const char *const kInputShapeRangeSizeInvalid = " shape range size less than 2 is invalid";
const char *const kShapeRangeValueConvertError = "transfer from string to int64 error";
const char *const kInputShapeRangeSample1 = "\"input_name1:[n1~n2,c1,h1,w1]\"";
const char *const kInputShapeRangeSample2 = "\"[1~20]\"";
const char *const kInputShapeRangeSample3 = "\"[1~20,3,3~6,-1]\"";
const char *const kInputShapeRangeSample4 = "\"[1~20,3,3~6,-1],[1~20,3,3~6,-1]\"";

vector<string> SplitInputShape(const std::string &input_shape) {
  vector<string> shape_pair_vec;
  size_t pos = input_shape.rfind(":");
  if (pos != std::string::npos) {
    shape_pair_vec.emplace_back(input_shape.substr(0, pos));
    shape_pair_vec.emplace_back(input_shape.substr(pos + 1, input_shape.size() - pos));
  }
  return shape_pair_vec;
}

static bool StringToLongNoThrow(const string &str, long &val) {
  try {
    val = std::stol(str);
    return true;
  } catch (const std::invalid_argument) {
    REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                       std::vector<string>({str, kShapeRangeValueConvertError, kInputShapeRangeSample3}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter] str:%s to long failed, reason: %s, correct sample is %s.",
           str.c_str(), kShapeRangeValueConvertError, kInputShapeRangeSample3);
  } catch (const std::out_of_range) {
    REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                       std::vector<string>({str, kShapeRangeValueConvertError, kInputShapeRangeSample3}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter] str:%s to long failed, reason: %s, correct sample is %s.",
           str.c_str(), kShapeRangeValueConvertError, kInputShapeRangeSample3);
  }
  return false;
}

static bool ParseShapeRangePair(const string &shape_range,
                                const vector<string> &range_pair_set,
                                std::pair<int64_t, int64_t> &range_pair) {
  if (range_pair_set.size() == 1) {
    long range_value = 0;
    if (!StringToLongNoThrow(range_pair_set.at(0), range_value)) {
      return false;
    }
    if (range_value < 0) {
      range_pair = std::make_pair(1, range_value);
    } else {
      range_pair = std::make_pair(range_value, range_value);
    }
  } else if (range_pair_set.size() == kRangePairSize) {
    // unknown dim, should get range.
    long range_left = 0;
    if (!StringToLongNoThrow(range_pair_set.at(0), range_left)) {
      return false;
    }
    long range_right = 0;
    if (!StringToLongNoThrow(range_pair_set.at(1), range_right)) {
      return false;
    }
    if ((range_left < 0) || (range_right < 0)) {
      REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                         std::vector<string>({shape_range, kInputShapeRangeInvalid, kInputShapeRangeSample3}));
      GELOGE(PARAM_INVALID,
             "[Parse][InputParameter] [--input_shape_range]'s shape range[%s] failed,"
             "reason: %s, correct sample is %s.",
             shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3);
      return false;
    }
    range_pair = std::make_pair(range_left, range_right);
  } else {
    REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                       std::vector<string>({shape_range, kInputShapeRangeInvalid, kInputShapeRangeSample3}));
    GELOGE(PARAM_INVALID, "[Parse][Parameter]shape_range:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample3);
    return false;
  }
  return true;
}
}  // namespace

Status CheckInputFormat(const string &input_format) {
  if (input_format.empty()) {
    return ge::SUCCESS;
  }
  if (!ge::TypeUtils::IsFormatValid(input_format.c_str())) {
    ErrorManager::GetInstance().ATCReportErrMessage(
      "E10001", {"parameter", "value", "reason"}, {"--input_format", input_format, "input format is invalid!"});
    GELOGE(ge::PARAM_INVALID, "[Check][InputFormat] --input_format[%s] is invalid!", input_format.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

bool CheckDynamicBatchSizeInputShapeValid(map<string, vector<int64_t>> shape_map,
                                          std::string &dynamic_batch_size) {
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    if (shape.empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10012");
      GELOGE(ge::PARAM_INVALID,
          "[Check][DynamicBatchSizeInputShape] shape size can not be less than 1 when set --dynamic_batch_size.");
      return false;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }

    bool ret = multibatch::CheckDynamicBatchShape(shape, iter->first);
    if (ret) {
      size++;
    }
  }

  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10031");
    GELOGE(ge::PARAM_INVALID,
        "[Check][DynamicBatchSizeInputShape]At least one batch n must be equal to -1 when set dynamic_batch_size.");
    return false;
  }

  for (char c : dynamic_batch_size) {
    if (!isdigit(c) && (c != ',') && (c != ' ')) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10003", {"parameter", "value", "reason"},
          {"dynamic_batch_size", dynamic_batch_size, kDynamicBatchSizeError});
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicBatchSizeInputShape] --dynamic_batch_size:%s is invalid. reason: %s",
          dynamic_batch_size.c_str(), kDynamicBatchSizeError);
      return false;
    }
  }
  if (dynamic_batch_size.back() == ',') {
    dynamic_batch_size.erase(dynamic_batch_size.end() - 1);
  }
  return true;
}

bool CheckDynamicImagesizeInputShapeValid(map<string, vector<int64_t>> shape_map,
                                          const std::string input_format, std::string &dynamic_image_size) {
  if (!input_format.empty() && !ge::TypeUtils::IsFormatValid(input_format.c_str())) {
    GELOGE(ge::PARAM_INVALID,
        "[Check][DynamicImagesizeInputShape] input_format [%s] invalid, can not support now.", input_format.c_str());
    REPORT_INPUT_ERROR("E10003", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({"input_format", input_format, "this format is not support"}));
    return false;
  }
  int32_t size = 0;
  for (auto iter = shape_map.begin(); iter != shape_map.end(); ++iter) {
    vector<int64_t> shape = iter->second;
    // only support four dim
    if (shape.size() != DIM_DEFAULT_SIZE) {
      if (std::count(shape.begin(), shape.end(), kDynamicInputDim) > 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10019");
        GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape] --input_shape invalid,"
            " only height and width can be -1 when set --dynamic_image_size.");
        return false;
      }
      continue;
    }

    if (std::count(shape.begin(), shape.end(), kDynamicInputDim) == 0) {
      continue;
    }
    auto ret = multibatch::CheckDynamicImageSizeShape(shape, iter->first, input_format);
    if (ret) {
      size++;
    } else {
      return ret;
    }
  }
  if (size == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10019");
    GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape]--input shape invalid, "
        "only height and width can be -1 when set --dynamic_image_size.");
    return false;
  }

  EraseEndSemicolon(dynamic_image_size);
  for (char c : dynamic_image_size) {
    bool is_char_valid = isdigit(c) || (c == ',') || (c == ' ') || (c == ';');
    if (!is_char_valid) {
      ErrorManager::GetInstance().ATCReportErrMessage(
              "E10003", {"parameter", "value", "reason"},
              {"dynamic_image_size", dynamic_image_size, kDynamicImageSizeError});
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicImageSizeInputShape] --dynamic_image_size:%s is invalid. reason: %s",
             dynamic_image_size.c_str(), kDynamicImageSizeError);
      return false;
    }
  }
  // Different parameter sets are split string by ';'
  std::vector<std::string> split_set = StringUtils::Split(dynamic_image_size, ';');
  // Different dimensions are split by ','
  std::vector<std::string> split_dim;
  for (auto str : split_set) {
    split_dim = StringUtils::Split(str, ',');
    if (split_dim.size() != static_cast<size_t>(kDynamicImageSizeNum)) {
      REPORT_INPUT_ERROR("E10020", std::vector<std::string>({"dynamic_image_size"}),
                         std::vector<std::string>({dynamic_image_size}));
      GELOGE(ge::PARAM_INVALID,
          "[Check][DynamicImagesizeInputShape] invalid value:%s number of dimensions of each group must be %ld.",
          dynamic_image_size.c_str(), kDynamicImageSizeNum);
      return false;
    }
  }

  return true;
}

bool CheckDynamicDimsInputShapeValid(const map<string, vector<int64_t>> &shape_map,
                                     string input_format, string &dynamic_dims) {
  if (input_format != "ND") {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--input_format", input_format.c_str(), "input_format must be ND when set dynamic_dims"});
    GELOGE(ge::PARAM_INVALID, "[Check][DynamicDimsInputShape]--input_format must be ND when set dynamic_dims.");
    return false;
  }

  int32_t dynamic_dim = 0;
  for (auto &info_shapes : shape_map) {
    auto &shapes = info_shapes.second;
    if (shapes.size() > kMaxNDDimNum || shapes.size() < kMinNDDimNum) {
      ErrorManager::GetInstance().ATCReportErrMessage(
          "E10001", {"parameter", "value", "reason"},
          {"--input_shape's dim", std::to_string(shapes.size()), "Dim num must within [1, 4] when set dynamic_dims"});
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicDimsInputShape]Dim num must within [%zu, %zu] when set dynamic_dims.",
             kMinNDDimNum, kMaxNDDimNum);
      return false;
    }
    dynamic_dim += std::count(shapes.begin(), shapes.end(), kDynamicInputDim);
  }
  if (dynamic_dim == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--input_shape's dynamic dim num", "0", "at least one dim should be -1 when set dynamic_dims"});
    GELOGE(ge::PARAM_INVALID,
           "[Check][DynamicDimsInputShape]--input_shape invalid,"
           "at least one dim should be -1 when set dynamic_dims.");
    return false;
  }

  if (!CheckAndParseDynamicDims(dynamic_dim, dynamic_dims)) {
    GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims]failed, %s invalid.", dynamic_dims.c_str());
    return false;
  }

  return true;
}

bool CheckAndParseDynamicDims(int32_t dynamic_dim_num, std::string &dynamic_dims) {
  EraseEndSemicolon(dynamic_dims);
  if (dynamic_dims.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage(
        "E10001", {"parameter", "value", "reason"},
        {"--dynamic_dims", dynamic_dims.c_str(), "dynamic_dims can not be empty"});
    GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims]--dynamic_dims can not be empty.");
    return false;
  }
  // Different parameter sets are split by ';'
  vector<string> split_set = StringUtils::Split(dynamic_dims, ';');
  if (split_set.size() > kMaxDynamicDimNum) {
    REPORT_INPUT_ERROR(
        "E10036", std::vector<std::string>({"shapesize", "maxshapesize"}),
        std::vector<std::string>({std::to_string(split_set.size()), std::to_string(kMaxDynamicDimNum + 1)}));
    GELOGE(ge::PARAM_INVALID,
        "[CheckAndParse][DynamicDims]dynamic_dims's num of parameter set can not exceed %zu.", kMaxDynamicDimNum);
    return false;
  }
  for (auto split_dim : split_set) {
    vector<string> one_set = StringUtils::Split(split_dim, ',');
    if (one_set.size() != static_cast<size_t>(dynamic_dim_num)) {
      REPORT_INPUT_ERROR(
          "E10003", std::vector<std::string>({"parameter", "value", "reason"}),
          std::vector<std::string>({"dynamic_dims", dynamic_dims,
            "Each gear setting needs to be consistent with the number of -1 in the inputshape"}));
      GELOGE(ge::PARAM_INVALID, "[CheckAndParse][DynamicDims] --dynamic_dims:%s invalid. "
          "reason: Each gear setting needs to be consistent with the number of -1 in the inputshape.",
          dynamic_dims.c_str());
      return false;
    }
    for (auto dim : one_set) {
      for (auto c : dim) {
        if (!isdigit(c)) {
          ErrorManager::GetInstance().ATCReportErrMessage(
              "E10001", {"parameter", "value", "reason"},
              {"--dynamic_dims's parameter", dim.c_str(), "must be positive integer"});
          GELOGE(ge::PARAM_INVALID,
              "[CheckAndParse][DynamicDims]--dynamic_dims:%s parameter must be positive integer.",
              dynamic_dims.c_str());
          return false;
        }
      }
    }
  }
  return true;
}

bool ParseSingleShapeRange(std::string &shape_range, vector<pair<int64_t, int64_t>> &shape_range_vec) {
  vector<char> square_brackets;
  for (auto ch : shape_range) {
    if (ch == '[' || ch == ']') {
      square_brackets.push_back(ch);
    }
  }

  bool is_square_brackets = (square_brackets.size() == kSquareBracketsSize) &&
                            (square_brackets[0] == '[') && (square_brackets[1] == ']');
  if (!is_square_brackets) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10048", {"shape_range", "reason", "sample"},
                                                    {shape_range, kInputShapeRangeInvalid, kInputShapeRangeSample2});
    GELOGE(PARAM_INVALID, "[Parse][Parameter] shape_range:%s invalid, reason: %s, correct sample is %s.",
        shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample2);
    return false;
  }
  // trim start bytes, after that, single input should be "1~20,3,3~6,-1"
  if (ge::StringUtils::StartWith(shape_range, "[")) {
    shape_range = shape_range.substr(1, shape_range.size() - 1);
  }
  // parse shape_range of single input. eg. "1~20,3,3~6,-1"
  vector<string> dim_range_set = ge::StringUtils::Split(shape_range, ',');
  for (const auto &range_pair_str : dim_range_set) {
    vector<string> range_pair_set = ge::StringUtils::Split(range_pair_str, '~');
    pair<int64_t, int64_t> range_pair;
    if (!ParseShapeRangePair(shape_range, range_pair_set, range_pair)) {
      GELOGE(PARAM_INVALID, "[Parse][RangePair] parse range pair failed.");
      return false;
    }
    shape_range_vec.emplace_back(range_pair);
  }
  return true;
}

/**
 * Parser shape_range from string to map
 * shape_range from option normally is "input1:[1~20,3,3~6,-1];input2:[1~20,3,3~6,-1]"
 * @param shape_range
 */
Status ParseInputShapeRange(const std::string &shape_range,
                            std::map<string, std::vector<std::pair<int64_t, int64_t>>> &shape_range_map) {
  GELOGD("Input shape range %s", shape_range.c_str());

  vector<string> shape_range_vec = StringUtils::Split(shape_range, ';');
  const int DEFAULT_SHAPE_RANGE_PAIR_SIZE = 2;
  for (const auto &shape_range_item : shape_range_vec) {
    vector<string> shape_range_pair_vec = SplitInputShape(shape_range_item);
    if (shape_range_pair_vec.size() != DEFAULT_SHAPE_RANGE_PAIR_SIZE) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10048", {"shape_range", "reason", "sample"},
                                                      {shape_range, kSplitError1, kInputShapeRangeSample1});
      GELOGE(PARAM_INVALID, "[Parse][Parameter]--input shape_range:%s invalid, reason: %s, correct sample is %s.",
          shape_range.c_str(), kSplitError1, kInputShapeRangeSample1);
      return PARAM_INVALID;
    }
    if (shape_range_pair_vec[1].empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10048", {"shape", "reason", "sample"},
                                                      {shape_range, kEmptyError, kInputShapeRangeSample1});
      GELOGE(PARAM_INVALID, "[Parse][Parameter]shape_range:%s invalid,reason: %s, correct sample is %s.",
          shape_range.c_str(), kEmptyError, kInputShapeRangeSample1);
      return PARAM_INVALID;
    }

    string shape_range_str = shape_range_pair_vec[1];
    vector<pair<int64_t, int64_t>> shape_range_val;
    if (!ParseSingleShapeRange(shape_range_str, shape_range_val)) {
      GELOGE(PARAM_INVALID, "[Parse][Parameter] shape_range_str: %s invalid.", shape_range_str.c_str());
      return PARAM_INVALID;
    }
    shape_range_map.emplace(make_pair(StringUtils::Trim(shape_range_pair_vec[0]), shape_range_val));
  }
  return SUCCESS;
}

/**
 * Parser shape_range from string to vector
 * shape_range from option normally is "[1~20,3,3~6,-1],[1~20,3,3~6,-1]"
 * @param shape_range
 */
Status ParseInputShapeRange(const std::string &shape_range,
                            std::vector<std::vector<std::pair<int64_t, int64_t>>> &range) {
  GELOGD("Input shape range %s", shape_range.c_str());

  if (shape_range.size() < kShapeRangeSize) {
    REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                       std::vector<std::string>({shape_range, kInputShapeRangeSizeInvalid, kInputShapeRangeSample4}));
    GELOGE(PARAM_INVALID, "[Parse][ShapeRange] str:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeSizeInvalid, kInputShapeRangeSample4);
    return PARAM_INVALID;
  }
  // different shape_range of single input are split by ']'
  vector<string> shape_range_set = ge::StringUtils::Split(shape_range, ']');
  if (shape_range_set.empty()) {
    REPORT_INPUT_ERROR("E10048", std::vector<std::string>({"shape_range", "reason", "sample"}),
                       std::vector<string>({shape_range, kInputShapeRangeInvalid, kInputShapeRangeSample4}));
    GELOGE(PARAM_INVALID, "[Parse][ShapeRange] str:%s invalid, reason: %s, correct sample is %s.",
           shape_range.c_str(), kInputShapeRangeInvalid, kInputShapeRangeSample4);
    return PARAM_INVALID;
  }
  for (auto &shape_range_str : shape_range_set) {
    if (shape_range_str.size() < kShapeRangeStrSize) {
      // shape_range_str should be "[2~3,1"
      // or ",[2~3,1". because we should trim '[' or ',['.
      // For scaler input, shape range should be "[]"
      // so shape_range_str.size() < 1 is invalid
      continue;
    }
    // trim start bytes, after that, single input should be "1~20,3,3~6,-1"
    if (ge::StringUtils::StartWith(shape_range_str, "[")) {
      shape_range_str = shape_range_str.substr(1, shape_range_str.size());
    }
    if (ge::StringUtils::StartWith(shape_range_str, ",")) {
      shape_range_str = shape_range_str.substr(kShapeRangeStrIndex, shape_range_str.size());
    }

    // parse shape_range of single input. eg. "1~20,3,3~6,-1"
    std::vector<std::pair<int64_t, int64_t>> range_of_single_input;
    vector<string> dim_range_set = ge::StringUtils::Split(shape_range_str, ',');
    for (const auto &range_pair_str : dim_range_set) {
      if (range_pair_str.empty()) {
        // for scaler input ,range is empty. use [0,0] as scaler range.
        range_of_single_input.emplace_back(std::make_pair(0, 0));
        continue;
      }
      vector<string> range_pair_set = ge::StringUtils::Split(range_pair_str, '~');
      pair<int64_t, int64_t> range_pair;
      if (!ParseShapeRangePair(shape_range_str, range_pair_set, range_pair)) {
        GELOGE(PARAM_INVALID, "[Parse][RangePair] Parse range pair failed.");
        return PARAM_INVALID;
      }
      range_of_single_input.emplace_back(range_pair);
    }
    range.emplace_back(range_of_single_input);
  }
  return SUCCESS;
}

Status CheckDynamicInputParamValid(string &dynamic_batch_size, string &dynamic_image_size, string &dynamic_dims,
    const string input_shape, const string input_shape_range, const string input_format, bool &is_dynamic_input) {
  int32_t param_size = static_cast<int32_t>(!dynamic_batch_size.empty()) +
      static_cast<int32_t>(!dynamic_image_size.empty()) + static_cast<int32_t>(!dynamic_dims.empty());
  if (param_size > 1) {
    REPORT_INPUT_ERROR("E10009", std::vector<std::string>(), std::vector<std::string>());
    GELOGE(ge::PARAM_INVALID,
           "[Parse][Parameter]dynamic_batch_size, dynamic_image_size and dynamic_dims can only be set one");
    return ge::PARAM_INVALID;
  }

  if (param_size == 0) {
    if (input_shape_range.find(":") != string::npos) {
      if (!input_shape_range.empty()) {
        std::map<string, std::vector<std::pair<int64_t, int64_t>>> shape_range_map;
        if (ParseInputShapeRange(input_shape_range, shape_range_map) != SUCCESS) {
          GELOGE(ge::PARAM_INVALID, "[Parse][InputShapeRange] failed, range: %s", input_shape_range.c_str());
          return ge::PARAM_INVALID;
        }
      }
    }
    return ge::SUCCESS;
  }

  map<string, vector<int64_t>> shape_map;
  vector<pair<string, vector<int64_t>>> user_shape_map;
  is_dynamic_input = true;
  if (input_shape.empty()) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10004", {"parameter"}, {"input_shape"});
    GELOGE(ge::PARAM_INVALID,
           "[Check][Parameter:input_shape]The input_shape can not be empty in dynamic input size scenario.");
    return ge::PARAM_INVALID;
  }

  if (!ParseInputShape(input_shape, shape_map, user_shape_map, is_dynamic_input)) {
    GELOGE(ge::PARAM_INVALID, "[Parse][InputShape]input_shape: %s invalid.", input_shape.c_str());
    return ge::PARAM_INVALID;
  }

  if (!dynamic_batch_size.empty()) {
    if (!CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicBatchSizeInputShape] input_shape: %s invalid.", input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }

  if (!dynamic_image_size.empty()) {
    if (!CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicImagesizeInputShape] %s invalid. dynamic_image_size:%s ",
             input_shape.c_str(), dynamic_image_size.c_str());
      return ge::PARAM_INVALID;
    }
  }

  if (!dynamic_dims.empty()) {
    if (!CheckDynamicDimsInputShapeValid(shape_map, input_format, dynamic_dims)) {
      GELOGE(ge::PARAM_INVALID, "[Check][DynamicDimsInputShape]: %s of input shape: %s failed.", dynamic_dims.c_str(),
             input_shape.c_str());
      return ge::PARAM_INVALID;
    }
  }
  return ge::SUCCESS;
}

bool ParseInputShape(const string &input_shape, map<string, vector<int64_t>> &shape_map,
                     vector<pair<string, vector<int64_t>>> &user_shape_map, bool is_dynamic_input) {
  vector<string> shape_vec = StringUtils::Split(input_shape, ';');
  const int DEFAULT_SHAPE_PAIR_SIZE = 2;
  for (const auto &shape : shape_vec) {
    vector<string> shape_pair_vec = SplitInputShape(shape);
    if (shape_pair_vec.size() != DEFAULT_SHAPE_PAIR_SIZE) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                      {shape, kSplitError1, kInputShapeSample1});
      GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
             shape.c_str(), kSplitError1, kInputShapeSample1);
      return false;
    }
    if (shape_pair_vec[1].empty()) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                      {shape, kEmptyError, kInputShapeSample1});
      GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
             shape.c_str(), kEmptyError, kInputShapeSample1);
      return false;
    }

    vector<string> shape_value_strs = StringUtils::Split(shape_pair_vec[1], ',');
    vector<int64_t> shape_values;
    for (auto &shape_value_str : shape_value_strs) {
      // stoul: The method may throw an exception: invalid_argument/out_of_range
      if (std::string::npos != shape_value_str.find('.')) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                        {shape, kFloatNumError, kInputShapeSample2});
        GELOGW("Parse input parameter [--input_shape]'s shape[%s] failed, reason: %s, correct sample is %s.",
               shape.c_str(), kFloatNumError, kInputShapeSample2);
        return false;
      }

      long left_result = 0;
      try {
        left_result = stol(StringUtils::Trim(shape_value_str));
        if (!shape_value_str.empty() && (shape_value_str.front() == '-')) {
          // The value maybe dynamic shape [-1], need substr it and verify isdigit.
          shape_value_str = shape_value_str.substr(1);
        }
        for (char c : shape_value_str) {
          if (!isdigit(c)) {
            ErrorManager::GetInstance().ATCReportErrMessage("E10002", {"shape", "reason", "sample"},
                                                            {shape, kDigitError, kInputShapeSample2});
            GELOGE(PARAM_INVALID, "[Check][Param]--input_shape's shape value[%s] is not digit",
                   shape_value_str.c_str());
            return false;
          }
        }
      } catch (const std::out_of_range &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10013", {"parameter", "value"},
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]'s value[%s] cause out of range execption!", shape_value_str.c_str());
        return false;
      } catch (const std::invalid_argument &) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10014", {"parameter", "value"},
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]'s value[%s] cause invalid argument!", shape_value_str.c_str());
        return false;
      } catch (...) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10015", {"parameter", "value"},
                                                        {"--input_shape", shape_value_str});
        GELOGW("Input parameter[--input_shape]'s value[%s] cause unkown execption!", shape_value_str.c_str());
        return false;
      }
      int64_t result = left_result;
      // - 1 is not currently supported
      if (!is_dynamic_input && result <= 0) {
        ErrorManager::GetInstance().ATCReportErrMessage("E10011", {"shape", "result"},
            {shape, std::to_string(result)});
        GELOGW(
            "Input parameter[--input_shape]'s shape value[%s] is invalid, "
            "expect positive integer, but value is %ld.",
            shape.c_str(), result);
        return false;
      }
      shape_values.push_back(result);
    }

    shape_map.emplace(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
    user_shape_map.push_back(make_pair(StringUtils::Trim(shape_pair_vec[0]), shape_values));
  }

  return true;
}

Status CheckOutputTypeParamValid(const std::string output_type) {
  if ((!output_type.empty()) && (kOutputTypeSupportDatatype.find(output_type) == kOutputTypeSupportDatatype.end())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"--output_type", output_type, kOutputTypeSupport});
    GELOGE(ge::PARAM_INVALID,
           "[Check][Param]Invalid value for --output_type[%s], %s.", output_type.c_str(), kOutputTypeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckBufferOptimizeParamValid(const std::string buffer_optimize) {
  if ((!buffer_optimize.empty()) &&
      (kBufferOptimizeSupportOption.find(buffer_optimize) == kBufferOptimizeSupportOption.end())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"--buffer_optimize", buffer_optimize, kBufferOptimizeSupport});
    GELOGE(ge::PARAM_INVALID,
           "[Check][BufferOptimize]Invalid value for [%s], %s.", buffer_optimize.c_str(), kBufferOptimizeSupport);
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckCompressWeightParamValid(const std::string enable_compress_weight,
    const std::string compress_weight_conf) {
  if ((!compress_weight_conf.empty()) &&
      (!CheckInputPathValid(compress_weight_conf, "--compress_weight_conf"))) {
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath]compress weight config file not found, file_name:%s",
           compress_weight_conf.c_str());
    return ge::PARAM_INVALID;
  }
  if ((enable_compress_weight != "") && (enable_compress_weight != "true") && (enable_compress_weight != "false")) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10005", {"parameter", "value"},
                                                    {"enable_compress_weight", enable_compress_weight});
    GELOGE(ge::PARAM_INVALID, "[Check][Param:enable_compress_weight]"
           "Input parameter[--enable_compress_weight]'s value:%s must be true or false.",
           enable_compress_weight.c_str());
    return ge::PARAM_INVALID;
  }

  if ((enable_compress_weight == "true") && (!compress_weight_conf.empty())) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10047", {"parameter0", "parameter1"},
                                                    {"enable_compress_weight", "compress_weight_conf"});
    GELOGE(ge::PARAM_INVALID,
           "[Check][CompressWeight]enable_compress_weight and compress_weight_conf can not both exist!!");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckKeepTypeParamValid(const std::string &keep_dtype) {
  if ((!keep_dtype.empty()) && (!CheckInputPathValid(keep_dtype, "--keep_dtype"))) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"--keep_dtype", keep_dtype, kKeepDtypeError});
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath::--keep_dtype] file not found, file_name:%s", keep_dtype.c_str());
    return ge::PARAM_INVALID;
  }

  return ge::SUCCESS;
}

int CheckLogParamValidAndSetLogLevel(const std::string log) {
  int ret = -1;
  char *npu_collect_path = std::getenv("NPU_COLLECT_PATH");
  if (npu_collect_path != nullptr && log == "null") {
    return 0;
  }

  if (log == "default") {
    ret = 0;
  } else if (log == "null") {
    ret = dlog_setlevel(-1, DLOG_NULL, 0);
  } else if (log == "debug") {
    ret = dlog_setlevel(-1, DLOG_DEBUG, 1);
  } else if (log == "info") {
    ret = dlog_setlevel(-1, DLOG_INFO, 1);
  } else if (log == "warning") {
    ret = dlog_setlevel(-1, DLOG_WARN, 1);
  } else if (log == "error") {
    ret = dlog_setlevel(-1, DLOG_ERROR, 1);
  } else {
    GELOGE(ge::PARAM_INVALID,
           "[Check][LogParam]log:%s invalid, only support debug, info, warning, error, null", log.c_str());
    return ret;
  }
  if (ret != 0) {
    GELOGE(ge::PARAM_INVALID, "[Set][LogLevel] fail, level:%s.", log.c_str());
  }
  return ret;
}

Status CheckInsertOpConfParamValid(const std::string insert_op_conf) {
  if ((!insert_op_conf.empty()) &&
      (!CheckInputPathValid(insert_op_conf, "--insert_op_conf"))) {
    GELOGE(ge::PARAM_INVALID, "[Check][InputPath]file not found: %s", insert_op_conf.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckDisableReuseMemoryParamValid(const std::string disable_reuse_memory) {
  if ((disable_reuse_memory != "") && (disable_reuse_memory != "0") && (disable_reuse_memory != "1")) {
    REPORT_INPUT_ERROR("E10006", std::vector<std::string>({"parameter", "value"}),
                       std::vector<std::string>({"disable_reuse_memory", disable_reuse_memory}));
    GELOGE(ge::PARAM_INVALID, "[Check][DisableReuseMemory]disable_reuse_memory must be 1 or 0.");
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckEnableSingleStreamParamValid(const std::string enable_single_stream) {
  if ((enable_single_stream != "") && (enable_single_stream != "true") && (enable_single_stream != "false")) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10005", {"parameter", "value"},
                                                    {"enable_single_stream", enable_single_stream});
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--enable_single_stream] value:%s must be true or false.",
           enable_single_stream.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckImplmodeParamValid(const std::string &optypelist_for_implmode, std::string &op_select_implmode) {
  // only appointed op_select_implmode, can user appoint optypelist_for_implmode
  if (optypelist_for_implmode != "" && op_select_implmode == "") {
    ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                    {"--op_select_implmode", op_select_implmode.c_str(),
                                                     kCompressWeightError});
    GELOGE(ge::PARAM_INVALID, "[Check][Param:--op_select_implmode]value:%s invalid, %s.",
           op_select_implmode.c_str(), kCompressWeightError);
    return ge::PARAM_INVALID;
  }
  // op_select_implmode default value is high_performance
  if (op_select_implmode == "") {
    op_select_implmode = IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT;
  } else {
    if (op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_DEFAULT &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_PRECISON &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PRECISION_FOR_ALL &&
      op_select_implmode != IR_OPTION_OP_SELECT_IMPLMODE_HIGH_PERFORMANCE_FOR_ALL) {
      ErrorManager::GetInstance().ATCReportErrMessage("E10001", {"parameter", "value", "reason"},
                                                      {"--op_select_implmode", op_select_implmode.c_str(),
                                                       kSelectImplmodeError});
      GELOGE(ge::PARAM_INVALID, "[Check][Implmode]Invalid value for --op_select_implmode[%s], %s.",
             op_select_implmode.c_str(), kSelectImplmodeError);
      return ge::PARAM_INVALID;
    }
  }

  return ge::SUCCESS;
}

Status CheckModifyMixlistParamValid(const std::map<std::string, std::string> &options) {
  std::string precision_mode;
  auto it = options.find(ge::PRECISION_MODE);
  if (it != options.end()) {
    precision_mode = it->second;
  }
  it = options.find(ge::MODIFY_MIXLIST);
  if (it != options.end() && CheckModifyMixlistParamValid(precision_mode, it->second) != ge::SUCCESS) {
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

Status CheckModifyMixlistParamValid(const std::string &precision_mode, const std::string &modify_mixlist) {
  if (!modify_mixlist.empty() && precision_mode != "allow_mix_precision") {
    REPORT_INPUT_ERROR("E10001", std::vector<std::string>({"parameter", "value", "reason"}),
                       std::vector<std::string>({ge::MODIFY_MIXLIST, modify_mixlist, kModifyMixlistError}));
    GELOGE(ge::PARAM_INVALID, "[Check][ModifyMixlist] Failed, %s", kModifyMixlistError);
    return ge::PARAM_INVALID;
  }
  GELOGI("Option set successfully, option_key=%s, option_value=%s", ge::MODIFY_MIXLIST.c_str(), modify_mixlist.c_str());

  return ge::SUCCESS;
}

void PrintOptionMap(std::map<std::string, std::string> &options, std::string tips) {
  for (auto iter = options.begin(); iter != options.end(); iter++) {
    std::string key = iter->first;
    std::string option_name = iter->second;
    GELOGD("%s set successfully, option_key=%s, option_value=%s", tips.c_str(), key.c_str(), option_name.c_str());
  }
}

void EraseEndSemicolon(string &param) {
  if (param.empty()) {
    return;
  }
  if (param.back() == ';') {
    param.erase(param.end() - 1);
  }
}

Status UpdateDataOpShape(const OpDescPtr &op, map<string, vector<int64_t>> &shape_map) {
  GE_CHECK_NOTNULL(op);
  if (shape_map.empty()) {
    GELOGI("Shape map of data op [%s] is empty, no need to update.", op->GetName().c_str());
    return SUCCESS;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  string data_op_name = op->GetName();
  auto iter = shape_map.find(data_op_name);
  if (iter != shape_map.end()) {
    tensor_input->SetShape(ge::GeShape(iter->second));
    tensor_output->SetShape(ge::GeShape(iter->second));
    GELOGI("Update input [%s] shape info", data_op_name.c_str());
  } else {
    GELOGI("No need update input [%s] attr because not found from input_shape.", data_op_name.c_str());
  }

  return SUCCESS;
}

Status UpdateDataOpShapeRange(const OpDescPtr &op,
                              const map<string, vector<pair<int64_t, int64_t>>> &name_shape_range_map) {
  GE_CHECK_NOTNULL(op);
  if (name_shape_range_map.empty()) {
    GELOGI("Shape range name map of data op [%s] is empty.", op->GetName().c_str());
    return SUCCESS;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  string data_op_name = op->GetName();
  auto origin_shape = tensor_input->GetShape();
  auto iter = name_shape_range_map.find(data_op_name);
  if (iter != name_shape_range_map.end()) {
    auto cur_shape_range = iter->second;
    if (TensorUtils::CheckShapeByShapeRange(origin_shape, cur_shape_range) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Check][OpDescPtr] Check shape by shape range failed for op:%s.", data_op_name.c_str());
      return PARAM_INVALID;
    }
    std::vector<int64_t> dims;
    for (size_t idx = 0; idx < cur_shape_range.size(); ++idx) {
      auto left_range = cur_shape_range[idx].first;
      auto right_range = cur_shape_range[idx].second;
      if (left_range != right_range) {
        dims.push_back(UNKNOWN_DIM);
      } else {
        dims.push_back(left_range);
      }
    }
    origin_shape = GeShape(dims);
    tensor_input->SetShape(origin_shape);
    tensor_input->SetShapeRange(cur_shape_range);
    tensor_output->SetShape(origin_shape);
    tensor_output->SetShapeRange(cur_shape_range);
    GELOGI("Update input [%s] shape range and shape [%s] info success.",
           data_op_name.c_str(), origin_shape.ToString().c_str());
  } else {
    GELOGI("No need to update input [%s] attr because not found from input_shape_range.", data_op_name.c_str());
  }

  return SUCCESS;
}

Status UpdateDataOpShapeRange(const OpDescPtr &op,
                              const vector<vector<pair<int64_t, int64_t>>> &index_shape_range_map) {
  GE_CHECK_NOTNULL(op);
  if (index_shape_range_map.empty()) {
    GELOGI("Shape range index map of data op [%s] is empty.", op->GetName().c_str());
    return SUCCESS;
  }

  GeAttrValue::INT index = 0;
  if (!AttrUtils::GetInt(op, ATTR_NAME_INDEX, index)) {
    GELOGW("[%s] Get index from data attr failed.", op->GetName().c_str());
    return SUCCESS;
  }

  if ((index < 0) || (static_cast<size_t>(index) >= index_shape_range_map.size())) {
    std::string situation = "data op index[" + std::to_string(index) + "]";
    std::string reason = "it must less than user_input size[" + std::to_string(index_shape_range_map.size()) + "]";
    REPORT_INPUT_ERROR("E19025", std::vector<std::string>({"situation", "reason"}),
                       std::vector<std::string>({situation, reason}));
    GELOGE(PARAM_INVALID, "user_input size = %zu, graph data op index = %ld.", index_shape_range_map.size(), index);
    return FAILED;
  }

  auto tensor_input = op->MutableInputDesc(0);
  auto tensor_output = op->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_input);
  GE_CHECK_NOTNULL(tensor_output);
  string data_op_name = op->GetName();
  auto origin_shape = tensor_input->GetShape();
  auto cur_shape_range = index_shape_range_map[index];
  if (TensorUtils::CheckShapeByShapeRange(origin_shape, cur_shape_range) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][OpDescPtr] Check shape by shape range failed for op:%s.", data_op_name.c_str());
    return PARAM_INVALID;
  }
  std::vector<int64_t> dims;
  for (size_t idx = 0; idx < cur_shape_range.size(); ++idx) {
    auto left_range = cur_shape_range[idx].first;
    auto right_range = cur_shape_range[idx].second;
    if (left_range != right_range) {
      dims.push_back(UNKNOWN_DIM);
    } else {
      dims.push_back(left_range);
    }
  }
  origin_shape = GeShape(dims);
  tensor_input->SetShape(origin_shape);
  tensor_input->SetShapeRange(cur_shape_range);
  tensor_output->SetShape(origin_shape);
  tensor_output->SetShapeRange(cur_shape_range);
  GELOGI("Update input [%s] shape range and shape [%s] info success.",
         data_op_name.c_str(), origin_shape.ToString().c_str());

  return SUCCESS;
}

static Status CheckInputShapeRangeNode(const ComputeGraphPtr &compute_graph,
                                       const map<string, vector<pair<int64_t, int64_t>>> &shape_range_map) {
  for (const auto &it : shape_range_map) {
    std::string node_name = it.first;
    ge::NodePtr node = compute_graph->FindNode(node_name);
    if (node == nullptr) {
      REPORT_INPUT_ERROR("E10016", std::vector<std::string>({"parameter", "opname"}),
                         std::vector<std::string>({"input_shape_range", node_name}));
      GELOGE(PARAM_INVALID, "[Check][InputNode]Input parameter[--input_shape_range]'s opname[%s] is not exist in model",
             node_name.c_str());
      return PARAM_INVALID;
    }
    if (node->GetType() != DATA) {
      REPORT_INPUT_ERROR("E10017", std::vector<std::string>({"parameter", "opname"}),
                         std::vector<std::string>({"input_shape_range", node_name}));
      GELOGE(PARAM_INVALID, "[Check][InputNode]Input parameter[--input_shape_range]'s opname[%s] is not a input opname",
             node_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status UpdateDynamicInputShapeRange(const ge::ComputeGraphPtr &compute_graph, const string &input_shape_range) {
  if (input_shape_range.empty()) {
    return SUCCESS;
  }
  GE_CHECK_NOTNULL(compute_graph);

  map<string, vector<pair<int64_t, int64_t>>> shape_range_map;
  if (ParseInputShapeRange(input_shape_range, shape_range_map) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parse][InputShapeRange] input_shape_range:%s invalid.", input_shape_range.c_str());
    return PARAM_INVALID;
  }

  if (CheckInputShapeRangeNode(compute_graph, shape_range_map) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][InputShapeRange]check input shape range:%s failed.", input_shape_range.c_str());
    return PARAM_INVALID;
  }

  for (NodePtr &input_node : compute_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(input_node);
    OpDescPtr op = input_node->GetOpDesc();
    GE_CHECK_NOTNULL(op);
    if (op->GetType() == DATA) {
      if (UpdateDataOpShapeRange(op, shape_range_map) != SUCCESS) {
        GELOGE(FAILED, "[Update][InputShapeRange] fail for op:%s.", op->GetName().c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}
}  // namespace ge
