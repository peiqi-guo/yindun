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
