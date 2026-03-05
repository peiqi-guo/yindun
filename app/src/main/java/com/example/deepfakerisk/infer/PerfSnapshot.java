/*
 * 作用：承载推理周期统计快照（次数、平均耗时、无新帧跳过数）。
 * 思路：把统计值对象化，便于日志输出与后续扩展指标。
 */
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
