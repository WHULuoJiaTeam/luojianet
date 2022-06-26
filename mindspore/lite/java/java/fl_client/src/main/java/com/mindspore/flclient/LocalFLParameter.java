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

import com.mindspore.lite.config.MSConfig;

import org.bouncycastle.math.ec.rfc7748.X25519;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Defines global parameters used internally during federated learning.
 *
 * @since 2021-06-30
 */
public class LocalFLParameter {
    private static final Logger LOGGER = Logger.getLogger(LocalFLParameter.class.toString());

    /**
     * Seed length used to generate random perturbations
     */
    public static final int SEED_SIZE = 32;

    /**
     * The length of IV value
     */
    public static final int I_VEC_LEN = 16;

    /**
     * The length of salt value
     */
    public static final int SALT_SIZE = 32;

    /**
     * the key length
     */
    public static final int KEY_LEN = X25519.SCALAR_SIZE;

    /**
     * The model name supported by federated learning tasks: "lenet".
     */
    public static final String LENET = "lenet";

    /**
     * The model name supported by federated learning tasks: "albert".
     */
    public static final String ALBERT = "albert";

    /**
     * The deployment environment supported by federated learning tasks: "android".
     */
    public static final String ANDROID = "android";

    /**
     * The deployment environment supported by federated learning tasks: "x86".
     */
    public static final String X86 = "x86";
    private static volatile LocalFLParameter localFLParameter;

    private List<String> classifierWeightName = new ArrayList<>();
    private List<String> albertWeightName = new ArrayList<>();
    private String flID;
    private String encryptLevel = EncryptLevel.NOT_ENCRYPT.toString();
    private String earlyStopMod = EarlyStopMod.NOT_EARLY_STOP.toString();
    private String serverMod = ServerMod.HYBRID_TRAINING.toString();
    private boolean stopJobFlag = false;
    private MSConfig msConfig = new MSConfig();
    private boolean useSSL = true;
    private float lr = 0.1f;
    private Map<String, float[]> oldFeatureMap;
    private byte uploadCompressType = 0;
    private int seed = 0;
    private float uploadSparseRatio = 0.08f;


    private LocalFLParameter() {
        // set classifierWeightName albertWeightName
        Common.setClassifierWeightName(classifierWeightName);
        Common.setAlbertWeightName(albertWeightName);
    }

    /**
     * Get the singleton object of the class LocalFLParameter.
     *
     * @return the singleton object of the class LocalFLParameter.
     */
    public static LocalFLParameter getInstance() {
        LocalFLParameter localRef = localFLParameter;
        if (localRef == null) {
            synchronized (LocalFLParameter.class) {
                localRef = localFLParameter;
                if (localRef == null) {
                    localFLParameter = localRef = new LocalFLParameter();
                }
            }
        }
        return localRef;
    }

    public List<String> getClassifierWeightName() {
        if (classifierWeightName.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <classifierWeightName> is null, please " +
                    "set it before use"));
            throw new IllegalArgumentException();
        }
        return classifierWeightName;
    }

    public void setClassifierWeightName(List<String> classifierWeightName) {
        this.classifierWeightName = classifierWeightName;
    }

    public List<String> getAlbertWeightName() {
        if (albertWeightName.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <classifierWeightName> is null, please " +
                    "set it before use"));
            throw new IllegalArgumentException();
        }
        return albertWeightName;
    }

    public void setAlbertWeightName(List<String> albertWeightName) {
        this.albertWeightName = albertWeightName;
    }

    public String getFlID() {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <flID> is null, please set it before " +
                    "use"));
            throw new IllegalArgumentException();
        }
        return flID;
    }

    public void setFlID(String flID) {
        if (flID == null || flID.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <flID> is null, please check it before " +
                    "set"));
            throw new IllegalArgumentException();
        }
        this.flID = flID;
    }

    public EncryptLevel getEncryptLevel() {
        return EncryptLevel.valueOf(encryptLevel);
    }

    public void setEncryptLevel(String encryptLevel) {
        if (encryptLevel == null || encryptLevel.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <encryptLevel> is null, please check it " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
        if ((!EncryptLevel.DP_ENCRYPT.toString().equals(encryptLevel)) &&
                (!EncryptLevel.NOT_ENCRYPT.toString().equals(encryptLevel)) &&
                (!EncryptLevel.SIGNDS.toString().equals(encryptLevel)) &&
                (!EncryptLevel.PW_ENCRYPT.toString().equals(encryptLevel))) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <encryptLevel> is " + encryptLevel + " ," +
                    " it must be DP_ENCRYPT or NOT_ENCRYPT or PW_ENCRYPT or SIGNDS, please check it before setting"));
            throw new IllegalArgumentException();
        }
        this.encryptLevel = encryptLevel;
    }

    public EarlyStopMod getEarlyStopMod() {
        return EarlyStopMod.valueOf(earlyStopMod);
    }

    public void setEarlyStopMod(String earlyStopMod) {
        if (earlyStopMod == null || earlyStopMod.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <earlyStopMod> is null, please check it " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
        if ((!EarlyStopMod.NOT_EARLY_STOP.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.LOSS_ABS.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.LOSS_DIFF.toString().equals(earlyStopMod)) &&
                (!EarlyStopMod.WEIGHT_DIFF.toString().equals(earlyStopMod))) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <earlyStopMod> is " + earlyStopMod + " ," +
                    " it must be NOT_EARLY_STOP or LOSS_ABS or LOSS_DIFF or WEIGHT_DIFF, please check it before " +
                    "setting"));
            throw new IllegalArgumentException();
        }
        this.earlyStopMod = earlyStopMod;
    }

    public String getServerMod() {
        return serverMod;
    }

    public void setServerMod(String serverMod) {
        if (serverMod == null || serverMod.isEmpty()) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <serverMod> is null, please check it " +
                    "before setting"));
            throw new IllegalArgumentException();
        }
        if ((!ServerMod.HYBRID_TRAINING.toString().equals(serverMod)) &&
                (!ServerMod.FEDERATED_LEARNING.toString().equals(serverMod))) {
            LOGGER.severe(Common.addTag("[localFLParameter] the parameter of <serverMod> is " + serverMod + " , it " +
                    "must be HYBRID_TRAINING or FEDERATED_LEARNING, please check it before setting"));
            throw new IllegalArgumentException();
        }
        this.serverMod = serverMod;
    }

    public boolean isStopJobFlag() {
        return stopJobFlag;
    }

    public void setStopJobFlag(boolean stopJobFlag) {
        this.stopJobFlag = stopJobFlag;
    }

    public MSConfig getMsConfig() {
        return msConfig;
    }

    public void setMsConfig(int DeviceType, int threadNum, int cpuBindMode, boolean enable_fp16) {
        // arg 0: DeviceType:DT_CPU -> 0
        // arg 1: ThreadNum -> 2
        // arg 2: cpuBindMode:NO_BIND ->  0
        // arg 3: enable_fp16 -> false
        msConfig.init(DeviceType, threadNum, cpuBindMode, enable_fp16);
    }

    public boolean isUseSSL() {
        return useSSL;
    }

    public void setUseSSL(boolean useSSL) {
        this.useSSL = useSSL;
    }

    public float getLr() {
        return lr;
    }

    public void setLr(float lr) {
        this.lr = lr;
    }

    public Map<String, float[]> getOldFeatureMap() {
        return oldFeatureMap;
    }

    public void setOldFeatureMap(Map<String, float[]> oldFeatureMap) {
        this.oldFeatureMap = oldFeatureMap;
    }

    public byte getUploadCompressType() {
        return uploadCompressType;
    }

    public void setUploadCompressType(byte uploadCompressType) {
        this.uploadCompressType = uploadCompressType;
    }

    public int getSeed() {
        return seed;
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public float getUploadSparseRatio() {
        return uploadSparseRatio;
    }

    public void setUploadSparseRatio(float uploadSparseRatio) {
        this.uploadSparseRatio = uploadSparseRatio;
    }
}
