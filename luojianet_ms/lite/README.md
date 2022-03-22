[查看中文](./README_CN.md)

## What Is LuoJiaNet Lite

LuoJiaNet lite is a high-performance, lightweight open source reasoning framework that can be used to meet the needs of AI applications on mobile devices. LuoJiaNet Lite focuses on how to deploy AI technology more effectively on devices. It has been integrated into HMS (Huawei Mobile Services) to provide inferences for applications such as image classification, object detection and OCR. LuoJiaNet Lite will promote the development and enrichment of the AI software/hardware application ecosystem.

<img src="../../docs/LuoJiaNet-Lite-architecture.png" alt="LuoJiaNet Lite Architecture" width="600"/>

For more details please check out our [LuoJiaNet Lite Architecture Guide](https://www.luojianet_ms.cn/lite/docs/en/master/architecture_lite.html).

### LuoJiaNet Lite features

1. Cooperative work with LuoJiaNet training
   - Provides training, optimization, and deployment.
   - The unified IR realizes the device-cloud AI application integration.

2. Lightweight
   - Provides model compress, which could help to improve performance as well.
   - Provides the ultra-lightweight reasoning solution LuoJiaNet Micro to meet the deployment requirements in extreme environments such as smart watches and headphones.

3. High-performance
   - The built-in high-performance kernel computing library NNACL supports multiple convolution optimization algorithms such as Slide window, im2col+gemm, winograde, etc.
   - Assembly code to improve performance of kernel operators. Supports CPU, GPU, and NPU.
4. Versatility
   - Supports IOS, Android.
   - Supports Lite OS.
   - Supports mobile device, smart screen, pad, and IOT devices.
   - Supports third party models such as TFLite, CAFFE and ONNX.

## LuoJiaNet Lite AI deployment procedure

1. Model selection and personalized training

   Select a new model or use an existing model for incremental training using labeled data. When designing a model for mobile device, it is necessary to consider the model size, accuracy and calculation amount.

   The LuoJiaNet team provides a series of pre-training models used for image classification, object detection. You can use these pre-trained models in your application.

   The pre-trained model provided by LuoJiaNet: [Image Classification](https://download.luojianet_ms.cn/model_zoo/official/lite/). More models will be provided in the feature.

   LuoJiaNet allows you to retrain pre-trained models to perform other tasks.

2. Model converter and optimization

   If you use LuoJiaNet or a third-party model, you need to use [LuoJiaNet Lite Model Converter Tool](https://www.luojianet_ms.cn/lite/docs/en/master/use/converter_tool.html) to convert the model into LuoJiaNet Lite model. The LuoJiaNet Lite model converter tool provides the converter of TensorFlow Lite, Caffe, ONNX to LuoJiaNet Lite model, fusion and quantization could be introduced during convert procedure.

   LuoJiaNet also provides a tool to convert models running on IoT devices .

3. Model deployment

   This stage mainly realizes model deployment, including model management, deployment, operation and maintenance monitoring, etc.

4. Inference

   Load the model and perform inference. [Inference](https://www.luojianet_ms.cn/lite/docs/en/master/use/runtime.html) is the process of running input data through the model to get output.

   LuoJiaNet provides pre-trained model that can be deployed on mobile device [example](https://www.luojianet_ms.cn/lite/examples/en).

## LuoJiaNet Lite benchmark test result

We test a couple of networks on HUAWEI Mate40 (Hisilicon Kirin9000e) mobile phone, and get the test results below for your reference.

| NetWork             | Thread Number | Average Run Time(ms) |
| ------------------- | ------------- | -------------------- |
| basic_squeezenet    | 4             | 6.415                |
| inception_v3        | 4             | 36.767               |
| mobilenet_v1_10_224 | 4             | 4.936                |
| mobilenet_v2_10_224 | 4             | 3.644                |
| resnet_v2_50        | 4             | 25.071               |
