/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.mindspore.flclient.cipher;

/**
 * Consts used for certification signature
 *
 * @since 2021-8-27
 */
public class CipherConsts {
    /**
     * provider name
     */
    public static final String PROVIDER_NAME = "HwUniversalKeyStoreProvider";

    /**
     * keyStore type
     */
    public static final String KEYSTORE_TYPE = "HwKeyStore";

    /**
     * sign algorithm
     */
    public static final String SIGN_ALGORITHM = "SHA256withRSA/PSS";
}
