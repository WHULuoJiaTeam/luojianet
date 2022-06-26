/*
 * Copyright 2022 Huawei Technologies Co., Ltd
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

package com.mindspore;

import com.mindspore.config.*;
import com.mindspore.lite.NativeLibrary;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(JUnit4.class)
public class ModelTest {

    @Test
    public void testBuildByGraphSuccess() {
        try {
            NativeLibrary.load();
            System.err.println("System: NativeLibrary load success.");
        } catch (Exception e) {
            System.err.println("Failed to load MindSporLite native library.");
            e.printStackTrace();
        }
        System.out.println(Version.version());
        Graph g = new Graph();
        assertTrue(g.load("../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms"));
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        TrainCfg cfg = new TrainCfg();
        cfg.init();
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(g, context, cfg);
        assertTrue(isSuccess);
        isSuccess = liteModel.setLearningRate(1.0f);
        assertTrue(isSuccess);
        isSuccess = liteModel.setupVirtualBatch(2,1.0f,0.5f);
        assertTrue(isSuccess);
        liteModel.free();
    }

    @Test
    public void testBuildByGraphFailed() {
        Graph g = new Graph();
        assertTrue(g.load("../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms"));
        MSContext context = new MSContext();
        TrainCfg cfg = new TrainCfg();
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(g, context, cfg);
        assertFalse(isSuccess);
        liteModel.free();
    }

    @Test
    public void testBuildByInferGraphSuccess() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        Graph g = new Graph();
        assertTrue(g.load(modelFile));
        MSContext context = new MSContext();
        context.init();
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(g, context, null);
        assertTrue(isSuccess);
        liteModel.free();
    }

    @Test
    public void testBuildByFileSuccess() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        assertTrue(isSuccess);
        liteModel.free();
    }

    @Test
    public void testBuildByBufferSuccess() {
        String fileName = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        FileChannel fc = null;
        MappedByteBuffer byteBuffer = null;
        try {
            fc = new RandomAccessFile(fileName, "r").getChannel();
            byteBuffer = fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size()).load();
        } catch (IOException e) {
            e.printStackTrace();
        }
        assertNotNull(byteBuffer);
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(byteBuffer, 0, context);
        assertTrue(isSuccess);
        liteModel.free();
    }

    @Test
    public void testBuildByFileFailed() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        assertFalse(isSuccess);
        liteModel.free();
    }

    @Test
    public void testPredictFailed() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        assertTrue(isSuccess);
        isSuccess = liteModel.predict();
        assertFalse(isSuccess);
        liteModel.free();
    }

    @Test
    public void testPredictSuccess() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        assertTrue(isSuccess);
        List<MSTensor> msTensorList = liteModel.getInputs();
        assertEquals(1, msTensorList.size());
        for (MSTensor msTensor : msTensorList) {
            byte[] temp = new byte[msTensor.elementsNum()];
            msTensor.setData(temp);
        }
        isSuccess = liteModel.predict();
        assertFalse(isSuccess);
        List<MSTensor> outputs = liteModel.getOutputs();
        for (MSTensor output : outputs) {
            System.out.println("output-------" + output.tensorName());
        }
        String outputTensorName = "Default/network-WithLossCell/_backbone-LeNet5/fc3-Dense/BiasAdd-op121";
        MSTensor output = liteModel.getOutputByTensorName(outputTensorName);
        assertEquals(80, output.size());
        output = liteModel.getOutputByTensorName("Default/network-WithLossCell/_loss_fn-L1Loss/ReduceMean-op112");
        assertEquals(0, output.size());
        List<MSTensor> inputs = liteModel.getInputs();
        for (MSTensor input : inputs) {
            System.out.println(input.tensorName());
        }
        for (String name : liteModel.getOutputTensorNames()) {
            System.out.println("output tensor name:" + name);
        }
        List<MSTensor> outputTensors = liteModel.getOutputsByNodeName("Default/network-WithLossCell/_backbone-LeNet5" +
                "/fc3-Dense/MatMul-op118");
        assertEquals(1, outputTensors.size());
        assertEquals(outputTensorName, outputTensors.get(0).tensorName());
        liteModel.free();
    }

    @Test
    public void testResize() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        List<MSTensor> inputs = liteModel.getInputs();
        int[][] newShape = {{2, 32, 32, 1}};
        System.out.println();
        isSuccess = liteModel.resize(inputs, newShape);
        assertTrue(isSuccess);
        newShape[0][3] = 3;
        isSuccess = liteModel.resize(inputs, newShape);
        assertFalse(isSuccess);
        liteModel.free();
    }

    @Test
    public void testExport() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_tod_infer.ms";
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(modelFile, 0, context);
        assertTrue(isSuccess);
        isSuccess = liteModel.export(null, 0, true, null);
        assertFalse(isSuccess);
        String outputName= "Default/network-WithLossCell/_backbone-LeNet5/conv2-Conv2d/Conv2D-op98";
        List<String> outputTensorNames = new ArrayList<>();
        outputTensorNames.add(outputName);
        isSuccess = liteModel.export("./test.ms", 0, false, outputTensorNames);
        assertFalse(isSuccess);
        liteModel.free();
    }


    @Test
    public void testFeatureMap() {
        String modelFile = "../test/ut/src/runtime/kernel/arm/test_data/nets/lenet_train.ms";
        Graph g = new Graph();
        assertTrue(g.load(modelFile));
        MSContext context = new MSContext();
        context.init(1, 0);
        context.addDeviceInfo(DeviceType.DT_CPU, false, 0);
        TrainCfg cfg = new TrainCfg();
        cfg.init();
        Model liteModel = new Model();
        boolean isSuccess = liteModel.build(g, context, cfg);
        assertTrue(isSuccess);
        int[] tensorShape = {6, 5, 5, 1};
        float[] tensorData = new float[6 * 5 * 5];
        Arrays.fill(tensorData, 0);
        ByteBuffer byteBuf = ByteBuffer.allocateDirect(6 * 5 * 5 * 4);
        FloatBuffer floatBuf = byteBuf.asFloatBuffer();
        floatBuf.put(tensorData);
        MSTensor newTensor = MSTensor.createTensor("conv1.weight", DataType.kNumberTypeFloat32, tensorShape, byteBuf);
        List<MSTensor> msTensors = new ArrayList<>();
        msTensors.add(newTensor);
        isSuccess = liteModel.updateFeatureMaps(msTensors);
        assertTrue(isSuccess);
        List<MSTensor> weights = liteModel.getFeatureMaps();
        for (MSTensor weight : weights) {
            if (weight.tensorName().equals("conv1.weight")) {
                float[] weightData = weight.getFloatData();
                assertEquals(0L, weightData[0], 0.0);
                break;
            }
        }
        newTensor.free();
        liteModel.free();
    }
}
