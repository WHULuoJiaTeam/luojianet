/*
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 */

package com.luojianet_ms.flclient.pki;

import java.security.cert.Certificate;

/**
 * PkiBean entity
 *
 * @since 2021-08-25
 */
public class PkiBean {
    private byte[] signData;

    private Certificate[] certificates;

    public PkiBean(byte[] signData, Certificate[] certificates) {
        this.signData = signData;
        this.certificates = certificates;
    }

    public byte[] getSignData() {
        return signData;
    }

    public void setSignData(byte[] signData) {
        this.signData = signData;
    }

    public Certificate[] getCertificates() {
        return certificates;
    }

    public void setCertificates(Certificate[] certificates) {
        this.certificates = certificates;
    }
}
