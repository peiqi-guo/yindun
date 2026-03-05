package com.example.deepfakerisk.infer;

import com.example.deepfakerisk.model.RiskState;

public interface InferenceListener {
    void onResult(RiskState risk, float pWindow, float pFrame, float logit);
}
