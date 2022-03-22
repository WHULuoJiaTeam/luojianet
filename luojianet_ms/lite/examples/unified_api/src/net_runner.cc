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

#include "src/net_runner.h"
#include <math.h>
#include <getopt.h>
#include <stdio.h>
#include <malloc.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <fstream>
#include <utility>
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/serialization.h"
#include "include/api/callback/loss_monitor.h"
#include "include/api/metrics/accuracy.h"
#include "include/api/callback/ckpt_saver.h"
#include "include/api/callback/train_accuracy.h"
#include "include/api/callback/lr_scheduler.h"
#include "src/utils.h"
#include "include/dataset/datasets.h"
#include "include/dataset/vision_lite.h"
#include "include/dataset/transforms.h"

using luojianet_ms::AccuracyMetrics;
using luojianet_ms::Model;
using luojianet_ms::TrainAccuracy;
using luojianet_ms::TrainCallBack;
using luojianet_ms::TrainCallBackData;
using luojianet_ms::dataset::Dataset;
using luojianet_ms::dataset::Mnist;
using luojianet_ms::dataset::SequentialSampler;
using luojianet_ms::dataset::TensorOperation;
using luojianet_ms::dataset::transforms::TypeCast;
using luojianet_ms::dataset::vision::Normalize;
using luojianet_ms::dataset::vision::Resize;

constexpr int kPrintNum = 10;
constexpr float kScalePoint = 255.0f;
constexpr int kBatchSize = 2;
constexpr int kNCHWDims = 4;
constexpr int kNCHWCDim = 2;
constexpr int kPrintTimes = 100;
constexpr int kSaveEpochs = 3;
constexpr float kGammaFactor = 0.7f;
constexpr static int kElem2Print = 10;

class Rescaler : public luojianet_ms::TrainCallBack {
 public:
  explicit Rescaler(float scale) : scale_(scale) {
    if (scale_ == 0) {
      scale_ = 1.0;
    }
  }
  ~Rescaler() override = default;
  void StepBegin(const luojianet_ms::TrainCallBackData &cb_data) override {
    auto inputs = cb_data.model_->GetInputs();
    auto *input_data = reinterpret_cast<float *>(inputs.at(0).MutableData());
    for (int k = 0; k < inputs.at(0).ElementNum(); k++) input_data[k] /= scale_;
  }

 private:
  float scale_ = 1.0;
};

/* This is an example of a user defined Callback to measure memory and latency of execution */
class Measurement : public luojianet_ms::TrainCallBack {
 public:
  explicit Measurement(unsigned int epochs)
      : epochs_(epochs), time_avg_(std::chrono::duration<double, std::milli>(0)) {}
  ~Measurement() override = default;
  void EpochBegin(const luojianet_ms::TrainCallBackData &cb_data) override {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  luojianet_ms::CallbackRetValue EpochEnd(const luojianet_ms::TrainCallBackData &cb_data) override {
    end_time_ = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration<double, std::milli>(end_time_ - start_time_);
    time_avg_ += time;
    return luojianet_ms::kContinue;
  }
  void End(const luojianet_ms::TrainCallBackData &cb_data) override {
    if (epochs_ > 0) {
      std::cout << "AvgRunTime: " << time_avg_.count() / epochs_ << " ms" << std::endl;
    }

    struct mallinfo info = mallinfo();
    std::cout << "Total allocation: " << info.arena + info.hblkhd << std::endl;
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time_;
  std::chrono::duration<double, std::milli> time_avg_;
  unsigned int epochs_;
};

NetRunner::~NetRunner() {
  if (model_ != nullptr) {
    delete model_;
  }
  if (graph_ != nullptr) {
    delete graph_;
  }
}

void NetRunner::InitAndFigureInputs() {
  auto context = std::make_shared<luojianet_ms::Context>();
  auto cpu_context = std::make_shared<luojianet_ms::CPUDeviceInfo>();
  cpu_context->SetEnableFP16(enable_fp16_);
  context->MutableDeviceInfo().push_back(cpu_context);

  graph_ = new luojianet_ms::Graph();
  MS_ASSERT(graph_ != nullptr);

  auto status = luojianet_ms::Serialization::Load(ms_file_, luojianet_ms::kMindIR, graph_);
  if (status != luojianet_ms::kSuccess) {
    std::cout << "Error " << status << " during serialization of graph " << ms_file_;
    MS_ASSERT(status != luojianet_ms::kSuccess);
  }

  auto cfg = std::make_shared<luojianet_ms::TrainCfg>();
  if (enable_fp16_) {
    cfg.get()->optimization_level_ = luojianet_ms::kO2;
  }

  model_ = new luojianet_ms::Model();
  MS_ASSERT(model_ != nullptr);

  status = model_->Build(luojianet_ms::GraphCell(*graph_), context, cfg);
  if (status != luojianet_ms::kSuccess) {
    std::cout << "Error " << status << " during build of model " << ms_file_;
    MS_ASSERT(status != luojianet_ms::kSuccess);
  }

  acc_metrics_ = std::shared_ptr<AccuracyMetrics>(new AccuracyMetrics);
  MS_ASSERT(acc_metrics_ != nullptr);
  model_->InitMetrics({acc_metrics_.get()});

  auto inputs = model_->GetInputs();
  MS_ASSERT(inputs.size() >= 1);
  auto nhwc_input_dims = inputs.at(0).Shape();

  batch_size_ = nhwc_input_dims.at(0);
  h_ = nhwc_input_dims.at(1);
  w_ = nhwc_input_dims.at(kNCHWCDim);
}

float NetRunner::CalculateAccuracy(int max_tests) {
  test_ds_ = Mnist(data_dir_ + "/test", "all");
  TypeCast typecast_f(luojianet_ms::DataType::kNumberTypeFloat32);
  Resize resize({h_, w_});
  test_ds_ = test_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast(luojianet_ms::DataType::kNumberTypeInt32);
  test_ds_ = test_ds_->Map({&typecast}, {"label"});
  test_ds_ = test_ds_->Batch(batch_size_, true);

  model_->Evaluate(test_ds_, {});
  std::cout << "Accuracy is " << acc_metrics_->Eval() << std::endl;

  return 0.0;
}

int NetRunner::InitDB() {
  train_ds_ = Mnist(data_dir_ + "/train", "all", std::make_shared<SequentialSampler>(0, 0));

  TypeCast typecast_f(luojianet_ms::DataType::kNumberTypeFloat32);
  Resize resize({h_, w_});
  train_ds_ = train_ds_->Map({&resize, &typecast_f}, {"image"});

  TypeCast typecast(luojianet_ms::DataType::kNumberTypeInt32);
  train_ds_ = train_ds_->Map({&typecast}, {"label"});

  train_ds_ = train_ds_->Batch(batch_size_, true);

  if (verbose_) {
    std::cout << "DatasetSize is " << train_ds_->GetDatasetSize() << std::endl;
  }
  if (train_ds_->GetDatasetSize() == 0) {
    std::cout << "No relevant data was found in " << data_dir_ << std::endl;
    MS_ASSERT(train_ds_->GetDatasetSize() != 0);
  }
  return 0;
}

int NetRunner::TrainLoop() {
  luojianet_ms::LossMonitor lm(kPrintTimes);
  luojianet_ms::TrainAccuracy am(1);

  luojianet_ms::CkptSaver cs(kSaveEpochs, std::string("lenet"));
  Rescaler rescale(kScalePoint);
  Measurement measure(epochs_);

  if (virtual_batch_ > 0) {
    model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &measure});
  } else {
    struct luojianet_ms::StepLRLambda step_lr_lambda(1, kGammaFactor);
    luojianet_ms::LRScheduler step_lr_sched(luojianet_ms::StepLRLambda, static_cast<void *>(&step_lr_lambda), 1);
    model_->Train(epochs_, train_ds_, {&rescale, &lm, &cs, &am, &step_lr_sched, &measure});
  }

  return 0;
}

int NetRunner::Main() {
  InitAndFigureInputs();

  InitDB();

  TrainLoop();

  CalculateAccuracy();

  if (epochs_ > 0) {
    auto trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_trained.ms";
    luojianet_ms::Serialization::ExportModel(*model_, luojianet_ms::kMindIR, trained_fn, luojianet_ms::kNoQuant, false);
    trained_fn = ms_file_.substr(0, ms_file_.find_last_of('.')) + "_infer.ms";
    luojianet_ms::Serialization::ExportModel(*model_, luojianet_ms::kMindIR, trained_fn, luojianet_ms::kNoQuant, true);
  }
  return 0;
}

void NetRunner::Usage() {
  std::cout << "Usage: net_runner -f <.ms model file> -d <data_dir> [-e <num of training epochs>] "
            << "[-v (verbose mode)] [-s <save checkpoint every X iterations>]" << std::endl;
}

bool NetRunner::ReadArgs(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "f:e:d:s:ihc:vob:")) != -1) {
    switch (opt) {
      case 'f':
        ms_file_ = std::string(optarg);
        break;
      case 'e':
        epochs_ = atoi(optarg);
        break;
      case 'd':
        data_dir_ = std::string(optarg);
        break;
      case 'v':
        verbose_ = true;
        break;
      case 's':
        save_checkpoint_ = atoi(optarg);
        break;
      case 'o':
        enable_fp16_ = true;
        break;
      case 'b':
        virtual_batch_ = atoi(optarg);
        break;
      case 'h':
      default:
        Usage();
        return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  NetRunner nr;

  if (nr.ReadArgs(argc, argv)) {
    nr.Main();
  } else {
    return -1;
  }
  return 0;
}
