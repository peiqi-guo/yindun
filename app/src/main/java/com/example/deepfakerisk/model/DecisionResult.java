package com.example.deepfakerisk.model;

public class DecisionResult {
    public final RiskState riskState;
    public final float mean;
    public final float pFake;

    public DecisionResult(RiskState riskState, float mean, float pFake) {
        this.riskState = riskState;
        this.mean = mean;
        this.pFake = pFake;
    }
}
