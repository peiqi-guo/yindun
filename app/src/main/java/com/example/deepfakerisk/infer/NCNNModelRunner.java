package com.example.deepfakerisk.infer;

import android.content.Context;

import com.tencent.ncnn.Mat;
import com.tencent.ncnn.Net;

public class NCNNModelRunner {
    private final Context context;
    private final Net net = new Net();
    private boolean loaded;

    public NCNNModelRunner(Context context) {
        this.context = context.getApplicationContext();
    }

    public void init() {
        if (loaded) {
            return;
        }

        int pm = net.loadParam(context.getAssets(), "ncnn/mobilenetv3_small.param");
        if (pm != 0) {
            throw new IllegalStateException("loadParam failed: " + pm);
        }
        int bm = net.loadModel(context.getAssets(), "ncnn/mobilenetv3_small.bin");
        if (bm != 0) {
            throw new IllegalStateException("loadModel failed: " + bm);
        }
        loaded = true;
    }

    public float infer(float[] chw) {
        if (!loaded) {
            throw new IllegalStateException("NCNNModelRunner not initialized");
        }
        if (chw.length != 3 * 224 * 224) {
            throw new IllegalArgumentException("Expected CHW size " + (3 * 224 * 224) + ", got " + chw.length);
        }

        Mat input = new Mat(224, 224, 3);
        input.clone_from_pixels(chw, 224, 224);

        Net.Extractor ex = net.createExtractor();
        ex.input("input", input);

        Mat out = new Mat();
        ex.extract("logits", out);
        return out.get(0);
    }
}
