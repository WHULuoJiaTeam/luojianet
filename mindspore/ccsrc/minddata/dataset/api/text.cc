/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/include/dataset/text.h"

#include <unistd.h>
#include <fstream>
#include <regex>

#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/text/ir/kernels/text_ir.h"
#include "mindspore/core/ir/dtype/type_id.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
// Transform operations for text.
namespace text {
constexpr size_t size_two = 2;
constexpr size_t size_three = 3;
constexpr int64_t value_one = 1;
constexpr int64_t value_two = 2;
constexpr size_t kMaxLoggedRows = 10;

// FUNCTIONS TO CREATE TEXT OPERATIONS
// (In alphabetical order)

#ifndef _WIN32
// BasicTokenizer
struct BasicTokenizer::Data {
  Data(bool lower_case, bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
       bool with_offsets)
      : lower_case_(lower_case),
        keep_whitespace_(keep_whitespace),
        normalize_form_(normalize_form),
        preserve_unused_token_(preserve_unused_token),
        with_offsets_(with_offsets) {}
  bool lower_case_;
  bool keep_whitespace_;
  NormalizeForm normalize_form_;
  bool preserve_unused_token_;
  bool with_offsets_;
};

BasicTokenizer::BasicTokenizer(bool lower_case, bool keep_whitespace, const NormalizeForm normalize_form,
                               bool preserve_unused_token, bool with_offsets)
    : data_(std::make_shared<Data>(lower_case, keep_whitespace, normalize_form, preserve_unused_token, with_offsets)) {}

std::shared_ptr<TensorOperation> BasicTokenizer::Parse() {
  return std::make_shared<BasicTokenizerOperation>(data_->lower_case_, data_->keep_whitespace_, data_->normalize_form_,
                                                   data_->preserve_unused_token_, data_->with_offsets_);
}

// BertTokenizer
struct BertTokenizer::Data {
  Data(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator, int32_t max_bytes_per_token,
       const std::vector<char> &unknown_token, bool lower_case, bool keep_whitespace,
       const NormalizeForm normalize_form, bool preserve_unused_token, bool with_offsets)
      : vocab_(vocab),
        suffix_indicator_(CharToString(suffix_indicator)),
        max_bytes_per_token_(max_bytes_per_token),
        unknown_token_(CharToString(unknown_token)),
        lower_case_(lower_case),
        keep_whitespace_(keep_whitespace),
        normalize_form_(normalize_form),
        preserve_unused_token_(preserve_unused_token),
        with_offsets_(with_offsets) {}
  std::shared_ptr<Vocab> vocab_;
  std::string suffix_indicator_;
  int32_t max_bytes_per_token_;
  std::string unknown_token_;
  bool lower_case_;
  bool keep_whitespace_;
  NormalizeForm normalize_form_;
  bool preserve_unused_token_;
  bool with_offsets_;
};

BertTokenizer::BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                             int32_t max_bytes_per_token, const std::vector<char> &unknown_token, bool lower_case,
                             bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                             bool with_offsets)
    : data_(std::make_shared<Data>(vocab, suffix_indicator, max_bytes_per_token, unknown_token, lower_case,
                                   keep_whitespace, normalize_form, preserve_unused_token, with_offsets)) {}

std::shared_ptr<TensorOperation> BertTokenizer::Parse() {
  return std::make_shared<BertTokenizerOperation>(
    data_->vocab_, data_->suffix_indicator_, data_->max_bytes_per_token_, data_->unknown_token_, data_->lower_case_,
    data_->keep_whitespace_, data_->normalize_form_, data_->preserve_unused_token_, data_->with_offsets_);
}

// CaseFold
CaseFold::CaseFold() = default;

std::shared_ptr<TensorOperation> CaseFold::Parse() { return std::make_shared<CaseFoldOperation>(); }

// FilterWikipediaXML
FilterWikipediaXML::FilterWikipediaXML() {}

std::shared_ptr<TensorOperation> FilterWikipediaXML::Parse() { return std::make_shared<FilterWikipediaXMLOperation>(); }
#endif

// JiebaTokenizer
struct JiebaTokenizer::Data {
  Data(const std::vector<char> &hmm_path, const std::vector<char> &mp_path, const JiebaMode &mode, bool with_offsets)
      : hmm_path_(CharToString(hmm_path)),
        mp_path_(CharToString(mp_path)),
        mode_(mode),
        with_offsets_(with_offsets),
        words_list_({}) {}
  std::string hmm_path_;
  std::string mp_path_;
  JiebaMode mode_;
  bool with_offsets_;
  std::vector<std::pair<std::string, int64_t>> words_list_;
};

JiebaTokenizer::JiebaTokenizer(const std::vector<char> &hmm_path, const std::vector<char> &mp_path,
                               const JiebaMode &mode, bool with_offsets)
    : data_(std::make_shared<Data>(hmm_path, mp_path, mode, with_offsets)) {}

std::shared_ptr<TensorOperation> JiebaTokenizer::Parse() {
  std::shared_ptr<JiebaTokenizerOperation> jieba_tokenizer =
    std::make_shared<JiebaTokenizerOperation>(data_->hmm_path_, data_->mp_path_, data_->mode_, data_->with_offsets_);
  for (auto &word : data_->words_list_) {
    Status rc = jieba_tokenizer->AddWord(word.first, word.second);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      return {};
    }
  }
  return jieba_tokenizer;
}

Status JiebaTokenizer::AddWordChar(const std::vector<char> &word, int64_t freq) {
  if (word.empty()) {
    std::string err_msg = "JiebaTokenizer : The parameter word is empty or not provided.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (freq < 0) {
    std::string err_msg = "JiebaTokenizer : The parameter freq must be greater than or equal to 0.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  (void)data_->words_list_.emplace_back(CharToString(word), freq);
  return Status::OK();
}

Status JiebaTokenizer::AddDictChar(const std::vector<std::pair<std::vector<char>, int64_t>> &user_dict) {
  for (auto &word_freq_pair : user_dict) {
    RETURN_IF_NOT_OK(AddWordChar(word_freq_pair.first, word_freq_pair.second));
  }
  return Status::OK();
}

Status JiebaTokenizer::AddDictChar(const std::vector<char> &file_path) {
  std::vector<std::pair<std::string, int64_t>> user_dict;
  RETURN_IF_NOT_OK(ParserFile(CharToString(file_path), &user_dict));
  RETURN_IF_NOT_OK(AddDictChar(PairStringInt64ToPairCharInt64(user_dict)));
  return Status::OK();
}

Status JiebaTokenizer::ParserFile(const std::string &file_path,
                                  std::vector<std::pair<std::string, int64_t>> *const user_dict) {
  RETURN_UNEXPECTED_IF_NULL(user_dict);
  auto realpath = FileUtils::GetRealPath(file_path.c_str());
  if (!realpath.has_value()) {
    std::string err_msg = "Get real path failed, path: " + file_path;
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::ifstream ifs(realpath.value());
  if (!ifs) {
    std::string err_msg = "JiebaTokenizer : Fail to load dictionary from the input file, check the file path.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) {
      continue;
    }
    std::regex regex("^\\s*([^\\s*]+?)\\s*([0-9]+)?\\s*$");
    std::smatch tokens;
    if (std::regex_match(line, tokens, regex)) {
      if (tokens.size() == size_two) {
        (void)user_dict->emplace_back(tokens.str(value_one), 0);
      } else if (tokens.size() == size_three) {
        (void)user_dict->emplace_back(tokens.str(value_one), strtoll(tokens.str(value_two).c_str(), nullptr, 0));
      } else {
        continue;
      }
    } else {
      continue;
    }
  }
  ifs.close();
  MS_LOG(INFO) << "JiebaTokenizer::AddDict: The size of user input dictionary is: " << user_dict->size();
  MS_LOG(INFO) << "Valid rows in input dictionary (Maximum of first 10 rows are shown.):";
  for (std::size_t i = 0; i != user_dict->size(); ++i) {
    if (i >= kMaxLoggedRows) break;
    MS_LOG(INFO) << user_dict->at(i).first << " " << user_dict->at(i).second;
  }
  return Status::OK();
}

// Lookup
struct Lookup::Data {
  Data(const std::shared_ptr<Vocab> &vocab, const std::optional<std::vector<char>> &unknown_token,
       mindspore::DataType data_type)
      : vocab_(vocab), data_type_(dataset::MSTypeToDEType(static_cast<TypeId>(data_type))) {
    if (unknown_token == std::nullopt) {
      unknown_token_ = std::nullopt;
    } else {
      unknown_token_ = std::string(unknown_token->begin(), unknown_token->end());
    }
  }
  std::shared_ptr<Vocab> vocab_;
  std::optional<std::string> unknown_token_;
  dataset::DataType data_type_;
};

Lookup::Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::vector<char>> &unknown_token,
               mindspore::DataType data_type)
    : data_(std::make_shared<Data>(vocab, unknown_token, data_type)) {
  data_->data_type_ = dataset::MSTypeToDEType(static_cast<TypeId>(data_type));
}

std::shared_ptr<TensorOperation> Lookup::Parse() {
  return std::make_shared<LookupOperation>(data_->vocab_, data_->unknown_token_, data_->data_type_);
}

// Ngram
struct Ngram::Data {
  Data(const std::vector<int32_t> &ngrams, const std::pair<std::vector<char>, int32_t> &left_pad,
       const std::pair<std::vector<char>, int32_t> &right_pad, const std::vector<char> &separator)
      : ngrams_(ngrams),
        left_pad_(PairCharToString(left_pad)),
        right_pad_(PairCharToString(right_pad)),
        separator_(CharToString(separator)) {}
  std::vector<int32_t> ngrams_;
  std::pair<std::string, int32_t> left_pad_;
  std::pair<std::string, int32_t> right_pad_;
  std::string separator_;
};

Ngram::Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::vector<char>, int32_t> &left_pad,
             const std::pair<std::vector<char>, int32_t> &right_pad, const std::vector<char> &separator)
    : data_(std::make_shared<Data>(ngrams, left_pad, right_pad, separator)) {}

std::shared_ptr<TensorOperation> Ngram::Parse() {
  return std::make_shared<NgramOperation>(data_->ngrams_, data_->left_pad_, data_->right_pad_, data_->separator_);
}

#ifndef _WIN32
// NormalizeUTF8
struct NormalizeUTF8::Data {
  explicit Data(NormalizeForm normalize_form) : normalize_form_(normalize_form) {}
  NormalizeForm normalize_form_;
};

NormalizeUTF8::NormalizeUTF8(NormalizeForm normalize_form) : data_(std::make_shared<Data>(normalize_form)) {}

std::shared_ptr<TensorOperation> NormalizeUTF8::Parse() {
  return std::make_shared<NormalizeUTF8Operation>(data_->normalize_form_);
}

// RegexReplace
struct RegexReplace::Data {
  Data(const std::vector<char> &pattern, const std::vector<char> &replace, bool replace_all)
      : pattern_(CharToString(pattern)), replace_(CharToString(replace)), replace_all_(replace_all) {}
  std::string pattern_;
  std::string replace_;
  bool replace_all_;
};

RegexReplace::RegexReplace(const std::vector<char> &pattern, const std::vector<char> &replace, bool replace_all)
    : data_(std::make_shared<Data>(pattern, replace, replace_all)) {}

std::shared_ptr<TensorOperation> RegexReplace::Parse() {
  return std::make_shared<RegexReplaceOperation>(data_->pattern_, data_->replace_, data_->replace_all_);
}

// RegexTokenizer
struct RegexTokenizer::Data {
  Data(const std::vector<char> &delim_pattern, const std::vector<char> &keep_delim_pattern, bool with_offsets)
      : delim_pattern_(CharToString(delim_pattern)),
        keep_delim_pattern_(CharToString(keep_delim_pattern)),
        with_offsets_(with_offsets) {}
  std::string delim_pattern_;
  std::string keep_delim_pattern_;
  bool with_offsets_;
};

RegexTokenizer::RegexTokenizer(const std::vector<char> &delim_pattern, const std::vector<char> &keep_delim_pattern,
                               bool with_offsets)
    : data_(std::make_shared<Data>(delim_pattern, keep_delim_pattern, with_offsets)) {}

std::shared_ptr<TensorOperation> RegexTokenizer::Parse() {
  return std::make_shared<RegexTokenizerOperation>(data_->delim_pattern_, data_->keep_delim_pattern_,
                                                   data_->with_offsets_);
}
#endif

// SentencePieceTokenizer
struct SentencePieceTokenizer::Data {
  Data(const std::shared_ptr<SentencePieceVocab> &vocab, SPieceTokenizerOutType out_type)
      : vocab_(vocab), vocab_path_(""), out_type_(out_type) {}
  Data(const std::vector<char> &vocab_path, SPieceTokenizerOutType out_type)
      : vocab_(nullptr), vocab_path_(CharToString(vocab_path)), out_type_(out_type) {}
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::string vocab_path_;
  SPieceTokenizerOutType out_type_;
};

SentencePieceTokenizer::SentencePieceTokenizer(const std::shared_ptr<SentencePieceVocab> &vocab,
                                               SPieceTokenizerOutType out_type)
    : data_(std::make_shared<Data>(vocab, out_type)) {}

SentencePieceTokenizer::SentencePieceTokenizer(const std::vector<char> &vocab_path, SPieceTokenizerOutType out_type)
    : data_(std::make_shared<Data>(vocab_path, out_type)) {}

std::shared_ptr<TensorOperation> SentencePieceTokenizer::Parse() {
  if (data_->vocab_ != nullptr) {
    return std::make_shared<SentencePieceTokenizerOperation>(data_->vocab_, data_->out_type_);
  } else {
    return std::make_shared<SentencePieceTokenizerOperation>(data_->vocab_path_, data_->out_type_);
  }
}

// SlidingWindow
struct SlidingWindow::Data {
  Data(const int32_t width, const int32_t axis) : width_(width), axis_(axis) {}
  int32_t width_;
  int32_t axis_;
};

SlidingWindow::SlidingWindow(const int32_t width, const int32_t axis) : data_(std::make_shared<Data>(width, axis)) {}

std::shared_ptr<TensorOperation> SlidingWindow::Parse() {
  return std::make_shared<SlidingWindowOperation>(data_->width_, data_->axis_);
}

// ToNumber
struct ToNumber::Data {
  dataset::DataType data_type_;
};

ToNumber::ToNumber(mindspore::DataType data_type) : data_(std::make_shared<Data>()) {
  data_->data_type_ = dataset::MSTypeToDEType(static_cast<TypeId>(data_type));
}

std::shared_ptr<TensorOperation> ToNumber::Parse() { return std::make_shared<ToNumberOperation>(data_->data_type_); }

// ToVectors
struct ToVectors::Data {
  Data(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init, bool lower_case_backup)
      : vectors_(vectors), unk_init_(unk_init), lower_case_backup_(lower_case_backup) {}
  std::shared_ptr<Vectors> vectors_;
  std::vector<float> unk_init_;
  bool lower_case_backup_;
};

ToVectors::ToVectors(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init,
                     bool lower_case_backup)
    : data_(std::make_shared<Data>(vectors, unk_init, lower_case_backup)) {}

std::shared_ptr<TensorOperation> ToVectors::Parse() {
  return std::make_shared<ToVectorsOperation>(data_->vectors_, data_->unk_init_, data_->lower_case_backup_);
}

// TruncateSequencePair
struct TruncateSequencePair::Data {
  explicit Data(int32_t max_length) : max_length_(max_length) {}
  int32_t max_length_;
};

TruncateSequencePair::TruncateSequencePair(int32_t max_length) : data_(std::make_shared<Data>(max_length)) {}

std::shared_ptr<TensorOperation> TruncateSequencePair::Parse() {
  return std::make_shared<TruncateSequencePairOperation>(data_->max_length_);
}

// UnicodeCharTokenizer
struct UnicodeCharTokenizer::Data {
  explicit Data(bool with_offsets) : with_offsets_(with_offsets) {}
  bool with_offsets_;
};

UnicodeCharTokenizer::UnicodeCharTokenizer(bool with_offsets) : data_(std::make_shared<Data>(with_offsets)) {}

std::shared_ptr<TensorOperation> UnicodeCharTokenizer::Parse() {
  return std::make_shared<UnicodeCharTokenizerOperation>(data_->with_offsets_);
}

// WordpieceTokenizer
struct WordpieceTokenizer::Data {
  Data(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator, int32_t max_bytes_per_token,
       const std::vector<char> &unknown_token, bool with_offsets)
      : vocab_(vocab),
        suffix_indicator_(CharToString(suffix_indicator)),
        max_bytes_per_token_(max_bytes_per_token),
        unknown_token_(CharToString(unknown_token)),
        with_offsets_(with_offsets) {}
  std::shared_ptr<Vocab> vocab_;
  std::string suffix_indicator_;
  int32_t max_bytes_per_token_;
  std::string unknown_token_;
  bool with_offsets_;
};

WordpieceTokenizer::WordpieceTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                                       int32_t max_bytes_per_token, const std::vector<char> &unknown_token,
                                       bool with_offsets)
    : data_(std::make_shared<Data>(vocab, suffix_indicator, max_bytes_per_token, unknown_token, with_offsets)) {}

std::shared_ptr<TensorOperation> WordpieceTokenizer::Parse() {
  return std::make_shared<WordpieceTokenizerOperation>(
    data_->vocab_, data_->suffix_indicator_, data_->max_bytes_per_token_, data_->unknown_token_, data_->with_offsets_);
}

#ifndef _WIN32
// UnicodeScriptTokenizer
struct UnicodeScriptTokenizer::Data {
  Data(bool keep_whitespace, bool with_offsets) : keep_whitespace_(keep_whitespace), with_offsets_(with_offsets) {}
  bool keep_whitespace_;
  bool with_offsets_;
};

UnicodeScriptTokenizer::UnicodeScriptTokenizer(bool keep_whitespace, bool with_offsets)
    : data_(std::make_shared<Data>(keep_whitespace, with_offsets)) {}

std::shared_ptr<TensorOperation> UnicodeScriptTokenizer::Parse() {
  return std::make_shared<UnicodeScriptTokenizerOperation>(data_->keep_whitespace_, data_->with_offsets_);
}

// WhitespaceTokenizer
struct WhitespaceTokenizer::Data {
  explicit Data(bool with_offsets) : with_offsets_(with_offsets) {}
  bool with_offsets_;
};

WhitespaceTokenizer::WhitespaceTokenizer(bool with_offsets) : data_(std::make_shared<Data>(with_offsets)) {}

std::shared_ptr<TensorOperation> WhitespaceTokenizer::Parse() {
  return std::make_shared<WhitespaceTokenizerOperation>(data_->with_offsets_);
}
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
