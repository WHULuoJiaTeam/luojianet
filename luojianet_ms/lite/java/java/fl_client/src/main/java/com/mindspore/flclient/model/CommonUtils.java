/*
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

package com.luojianet_ms.flclient.model;

import com.luojianet_ms.flclient.Common;
import com.luojianet_ms.lite.MSTensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class CommonUtils {
    private static final Logger logger = Logger.getLogger(CommonUtils.class.toString());

    /**
     * Get max value index.
     *
     * @param scores score array.
     * @param start  start index.
     * @param end    end index.
     * @return max score index.
     */
    public static int getMaxScoreIndex(float[] scores, int start, int end) {
        if (scores == null || scores.length == 0) {
            logger.severe(Common.addTag("scores cannot be empty"));
            return -1;
        }
        if (start >= scores.length || start < 0 || end > scores.length || end < 0) {
            logger.severe(Common.addTag("start,end cannot out of scores length"));
            return -1;
        }
        float maxScore = scores[start];
        int maxIdx = start;
        for (int i = start; i < end; i++) {
            if (scores[i] > maxScore) {
                maxIdx = i;
                maxScore = scores[i];
            }
        }
        return maxIdx - start;
    }

    /**
     * convert tensor to feature map
     *
     * @param tensors input tensors
     * @return feature map
     */
    public static Map<String, float[]> convertTensorToFeatures(List<MSTensor> tensors) {
        if (tensors == null) {
            logger.severe(Common.addTag("tensors cannot be null"));
            return new HashMap<>();
        }
        Map<String, float[]> features = new HashMap<>(tensors.size());
        for (MSTensor mstensor : tensors) {
            if (mstensor == null) {
                logger.severe(Common.addTag("tensors cannot be null"));
                return new HashMap<>();
            }
            features.put(mstensor.tensorName(), mstensor.getFloatData());
        }
        return features;
    }

}
