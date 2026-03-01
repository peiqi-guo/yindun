package com.example.deepfakerisk.model;

import android.graphics.Bitmap;

public class StubModelRunner implements ModelRunner {
    @Override
    public float infer(Bitmap input224) {
        // TODO: Replace with real deepfake model inference (TFLite/ONNX/etc.)
        double t = System.currentTimeMillis() / 1000.0;
        float value = (float) ((Math.sin(t) + 1.0) / 2.0);
        return Math.max(0f, Math.min(1f, value));
    }
}
