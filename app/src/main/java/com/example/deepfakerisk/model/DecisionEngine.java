/*
 * 作用：按时间窗口平滑p_fake并产出风险等级。
 * 思路：队列均值+连续触发规则，抑制单帧抖动带来的误报。
 */
package com.example.deepfakerisk.model;

import java.util.ArrayDeque;

public class DecisionEngine {
    private final int windowSize;
    private final float tLow;
    private final float tHigh;

    private final ArrayDeque<Float> queue;
    private int highConsecutive;
    private int lowConsecutive;

    public DecisionEngine() {
        this(10, 0.5f, 0.7f);
    }

    public DecisionEngine(int windowSize, float tLow, float tHigh) {
        this.windowSize = windowSize;
        this.tLow = tLow;
        this.tHigh = tHigh;
        this.queue = new ArrayDeque<>(windowSize);
    }

    public synchronized DecisionResult add(float pFake) {
        if (queue.size() == windowSize) {
            queue.removeFirst();
        }
        queue.addLast(pFake);

        float sum = 0f;
        for (Float f : queue) {
            sum += f;
        }
        float mean = queue.isEmpty() ? 0f : sum / queue.size();

        RiskState risk;
        if (mean > tHigh) {
            highConsecutive += 1;
            lowConsecutive = 0;
            risk = highConsecutive >= 3 ? RiskState.DANGEROUS : RiskState.SAFE;
        } else if (mean > tLow) {
            lowConsecutive += 1;
            highConsecutive = 0;
            risk = lowConsecutive >= 5 ? RiskState.SUSPICIOUS : RiskState.SAFE;
        } else {
            lowConsecutive = 0;
            highConsecutive = 0;
            risk = RiskState.SAFE;
        }

        return new DecisionResult(risk, mean, pFake);
    }
}
