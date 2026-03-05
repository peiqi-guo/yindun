/*
 * 作用：定义进程内检测状态载体（detecting/risk/mean/pFake）。
 * 思路：最小字段满足UI刷新与调试观察。
 */
package com.example.deepfakerisk.service;

import com.example.deepfakerisk.model.RiskState;

public class DetectionState {
    public final boolean detecting;
    public final RiskState risk;
    public final float mean;
    public final float pFake;

    public DetectionState(boolean detecting, RiskState risk, float mean, float pFake) {
        this.detecting = detecting;
        this.risk = risk;
        this.mean = mean;
        this.pFake = pFake;
    }

    public static DetectionState idle() {
        return new DetectionState(false, RiskState.SAFE, 0f, 0f);
    }
}
