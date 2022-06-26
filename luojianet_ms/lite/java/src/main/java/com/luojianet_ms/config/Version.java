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

package com.luojianet_ms.config;

import com.luojianet_ms.lite.NativeLibrary;

/**
 * Define luojianet_ms version info.
 *
 * @since v1.0
 */
public class Version {
    static {
        try {
            NativeLibrary.loadLibs();
        } catch (Exception e) {
            System.err.println("Failed to load LuoJiaNETLite native library.");
            e.printStackTrace();
            throw e;
        }
    }

    /**
     * Get LUOJIANET_MS Lite version info.
     *
     * @return LUOJIANET_MS Lite version info.
     */
    public static native String version();
}
