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
#include <memory>
#include <string>
#include <string_view>

#include "common/common.h"
#include "minddata/dataset/text/kernels/basic_tokenizer_op.h"
#include "minddata/dataset/text/kernels/case_fold_op.h"
#include "minddata/dataset/text/kernels/normalize_utf8_op.h"
#include "minddata/dataset/text/kernels/regex_replace_op.h"
#include "minddata/dataset/text/kernels/regex_tokenizer_op.h"
#include "minddata/dataset/text/kernels/unicode_char_tokenizer_op.h"
#include "minddata/dataset/text/kernels/unicode_script_tokenizer_op.h"
#include "minddata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "gtest/gtest.h"
#include "utils/log_adapter.h"

using namespace mindspore::dataset;

class MindDataTestTokenizerOp : public UT::Common {
 public:
  void CheckEqual(const std::shared_ptr<Tensor> &o,
                  const std::vector<dsize_t> &index,
                  const std::string &expect) {
    std::string_view str;
    Status s = o->GetItemAt(&str, index);
    EXPECT_TRUE(s.IsOk());
    EXPECT_EQ(str, expect);
  }
};

TEST_F(MindDataTestTokenizerOp, TestUnicodeCharTokenizerOp) {
  MS_LOG(INFO) << "Doing TestUnicodeCharTokenizerOp.";
  std::unique_ptr<UnicodeCharTokenizerOp> op(new UnicodeCharTokenizerOp(true));
std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Hello World!", &input);  TensorRow output;
  Status s = op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 12);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "H");
  CheckEqual(output[0], {1}, "e");
  CheckEqual(output[0], {2}, "l");
  CheckEqual(output[0], {3}, "l");
  CheckEqual(output[0], {4}, "o");
  CheckEqual(output[0], {5}, " ");
  CheckEqual(output[0], {6}, "W");
  CheckEqual(output[0], {7}, "o");
  CheckEqual(output[0], {8}, "r");
  CheckEqual(output[0], {9}, "l");
  CheckEqual(output[0], {10}, "d");
  CheckEqual(output[0], {11}, "!");

  Tensor::CreateScalar<std::string>("中国 你好!", &input);
  output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 6);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "中");
  CheckEqual(output[0], {1}, "国");
  CheckEqual(output[0], {2}, " ");
  CheckEqual(output[0], {3}, "你");
  CheckEqual(output[0], {4}, "好");
  CheckEqual(output[0], {5}, "!");

  Tensor::CreateScalar<std::string>("中", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "中");

  Tensor::CreateScalar<std::string>("H", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "H");

  Tensor::CreateScalar<std::string>("  ", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 2);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output[0]->ToString();
  CheckEqual(output[0], {0}, " ");
  CheckEqual(output[0], {1}, " ");

  Tensor::CreateScalar<std::string>("", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor6: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "");
}

TEST_F(MindDataTestTokenizerOp, TestWhitespaceTokenizerOp) {
  MS_LOG(INFO) << "Doing TestWhitespaceTokenizerOp.";
  std::unique_ptr<WhitespaceTokenizerOp> op(new WhitespaceTokenizerOp(true));
std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China.", &input);  TensorRow output;
  Status s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 3);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Welcome");
  CheckEqual(output[0], {1}, "to");
  CheckEqual(output[0], {2}, "China.");

  Tensor::CreateScalar<std::string>("  hello", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "hello");

  Tensor::CreateScalar<std::string>("hello", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "hello");

  Tensor::CreateScalar<std::string>("hello  ", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "hello");

  Tensor::CreateScalar<std::string>("  ", &input);
output.clear();
  s = op->Compute(TensorRow(0, {input}), &output);
    EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "");
}

TEST_F(MindDataTestTokenizerOp, TestUnicodeScriptTokenizer) {
  MS_LOG(INFO) << "Doing TestUnicodeScriptTokenizer.";
  std::unique_ptr<UnicodeScriptTokenizerOp> keep_whitespace_op(new UnicodeScriptTokenizerOp(true, true));
  std::unique_ptr<UnicodeScriptTokenizerOp> skip_whitespace_op(new UnicodeScriptTokenizerOp(false, true));

 std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China. \n 中国\t北京", &input);
 TensorRow output;
  Status s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 10);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor1: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Welcome");
  CheckEqual(output[0], {1}, " ");
  CheckEqual(output[0], {2}, "to");
  CheckEqual(output[0], {3}, " ");
  CheckEqual(output[0], {4}, "China");
  CheckEqual(output[0], {5}, ".");
  CheckEqual(output[0], {6}, " \n ");
  CheckEqual(output[0], {7}, "中国");
  CheckEqual(output[0], {8}, "\t");
  CheckEqual(output[0], {9}, "北京");
  output.clear();
  s = skip_whitespace_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 6);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor2: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Welcome");
  CheckEqual(output[0], {1}, "to");
  CheckEqual(output[0], {2}, "China");
  CheckEqual(output[0], {3}, ".");
  CheckEqual(output[0], {4}, "中国");
  CheckEqual(output[0], {5}, "北京");

  Tensor::CreateScalar<std::string>("  Welcome to 中国.  ", &input);
 output.clear();
  s = skip_whitespace_op->Compute(TensorRow(0, {input}), &output);  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 4);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor3: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Welcome");
  CheckEqual(output[0], {1}, "to");
  CheckEqual(output[0], {2}, "中国");
  CheckEqual(output[0], {3}, ".");
  output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 8);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor4: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "  ");
  CheckEqual(output[0], {1}, "Welcome");
  CheckEqual(output[0], {2}, " ");
  CheckEqual(output[0], {3}, "to");
  CheckEqual(output[0], {4}, " ");
  CheckEqual(output[0], {5}, "中国");
  CheckEqual(output[0], {6}, ".");
  CheckEqual(output[0], {7}, "  ");

  Tensor::CreateScalar<std::string>("Hello", &input);
output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor5: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Hello");

  Tensor::CreateScalar<std::string>("H", &input);
output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor6: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "H");

  Tensor::CreateScalar<std::string>("", &input);
  output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor7: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "");

  Tensor::CreateScalar<std::string>("Hello中国Hello世界", &input);
 output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output); EXPECT_TRUE(s.IsOk());  EXPECT_EQ(output[0]->Size(), 4);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor8: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "Hello");
  CheckEqual(output[0], {1}, "中国");
  CheckEqual(output[0], {2}, "Hello");
  CheckEqual(output[0], {3}, "世界");

  Tensor::CreateScalar<std::string>("   ", &input);
 output.clear();
  s = keep_whitespace_op->Compute(TensorRow(0, {input}), &output);
   EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor10: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "   ");
  Tensor::CreateScalar<std::string>("   ", &input);
  output.clear();
  s = skip_whitespace_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output[0]->Size(), 1);
  EXPECT_EQ(output[0]->Rank(), 1);
  MS_LOG(INFO) << "Out tensor11: " << output[0]->ToString();
  CheckEqual(output[0], {0}, "");
}

TEST_F(MindDataTestTokenizerOp, TestCaseFold) {
  MS_LOG(INFO) << "Doing TestCaseFold.";
  std::unique_ptr<CaseFoldOp> case_fold_op(new CaseFoldOp());
  std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China. \n 中国\t北京", &input);

  std::shared_ptr<Tensor> output;
  Status s = case_fold_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 0);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {}, "welcome to china. \n 中国\t北京");
}

TEST_F(MindDataTestTokenizerOp, TestNormalize) {
  MS_LOG(INFO) << "Doing TestNormalize.";
  std::unique_ptr<NormalizeUTF8Op> nfc_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfc));
  std::unique_ptr<NormalizeUTF8Op> nfkc_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfkc));
  std::unique_ptr<NormalizeUTF8Op> nfd_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfd));
  std::unique_ptr<NormalizeUTF8Op> nfkd_normalize_op(new NormalizeUTF8Op(NormalizeForm::kNfkd));
  std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("ṩ", &input);
  std::shared_ptr<Tensor> output;
  Status s = nfc_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFC str:" << output->ToString();

  nfkc_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFKC str:" << output->ToString();

  nfd_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFD str:" << output->ToString();

  nfkd_normalize_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  MS_LOG(INFO) << "NFKD str:" << output->ToString();
}

TEST_F(MindDataTestTokenizerOp, TestRegexReplace) {
  MS_LOG(INFO) << "Doing TestRegexReplace.";
  std::unique_ptr<RegexReplaceOp> regex_replace_op(new RegexReplaceOp("\\s+", "_", true));
  std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China. \n 中国\t北京", &input);
  std::shared_ptr<Tensor> output;
  Status s = regex_replace_op->Compute(input, &output);
  EXPECT_TRUE(s.IsOk());
  EXPECT_EQ(output->Size(), 1);
  EXPECT_EQ(output->Rank(), 0);
  MS_LOG(INFO) << "Out tensor1: " << output->ToString();
  CheckEqual(output, {}, "Welcome_to_China._中国_北京");
}

TEST_F(MindDataTestTokenizerOp, TestRegexTokenizer) {
  MS_LOG(INFO) << "Doing TestRegexTokenizerOp.";
  std::unique_ptr<RegexTokenizerOp> regex_tokenizer_op(new RegexTokenizerOp("\\p{Cc}|\\p{Cf}|\\s+", "", true));
std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China. \n 中国\t北京", &input);
     TensorRow output;
  Status s = regex_tokenizer_op->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
}

TEST_F(MindDataTestTokenizerOp, TestBasicTokenizer) {
  MS_LOG(INFO) << "Doing TestBasicTokenizer.";
  // bool lower_case, bool keep_whitespace,
  // NormalizeForm  normalization_form, bool preserve_unused_token
  std::unique_ptr<BasicTokenizerOp> basic_tokenizer(new BasicTokenizerOp(true, true, NormalizeForm::kNone, false,true));
std::shared_ptr<Tensor> input;
  Tensor::CreateScalar<std::string>("Welcome to China. 中国\t北京", &input);
  TensorRow output;
  Status s = basic_tokenizer->Compute(TensorRow(0, {input}), &output);
  EXPECT_TRUE(s.IsOk());
}