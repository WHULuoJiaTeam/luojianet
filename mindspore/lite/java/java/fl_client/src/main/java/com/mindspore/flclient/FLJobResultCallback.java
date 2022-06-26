/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

package com.mindspore.flclient;

import java.util.logging.Logger;

/**
 * Define job result callback function.
 *
 * @since 2021-06-30
 */
public class FLJobResultCallback implements IFLJobResultCallback {
    private static final Logger LOGGER = Logger.getLogger(FLJobResultCallback.class.toString());

    /**
     * Called at the end of an iteration for Fl job
     *
     * @param modelName    the name of model
     * @param iterationSeq Iteration number
     * @param resultCode   Status Code
     */
    @Override
    public void onFlJobIterationFinished(String modelName, int iterationSeq, int resultCode) {
        LOGGER.info(Common.addTag("[onFlJobIterationFinished] modelName: " + modelName + " iterationSeq: " +
                iterationSeq + " resultCode: " + resultCode));
    }

    /**
     * Called on completion for Fl job
     *
     * @param modelName      the name of model
     * @param iterationCount total Iteration numbers
     * @param resultCode     Status Code
     */
    @Override
    public void onFlJobFinished(String modelName, int iterationCount, int resultCode) {
        LOGGER.info(Common.addTag("[onFlJobFinished] modelName: " + modelName + " iterationCount: " +
                iterationCount + " resultCode: " + resultCode));
    }
}