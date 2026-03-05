package com.example.deepfakerisk.infer;

public class PerfSnapshot {
    public final long inferCount;
    public final float avgInferMs;
    public final long droppedNoFrame;

    public PerfSnapshot(long inferCount, float avgInferMs, long droppedNoFrame) {
        this.inferCount = inferCount;
        this.avgInferMs = avgInferMs;
        this.droppedNoFrame = droppedNoFrame;
    }
}
