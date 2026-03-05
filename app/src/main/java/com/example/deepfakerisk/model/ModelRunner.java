package com.example.deepfakerisk.model;

import android.graphics.Bitmap;

public interface ModelRunner {
    float infer(Bitmap input224);
}
