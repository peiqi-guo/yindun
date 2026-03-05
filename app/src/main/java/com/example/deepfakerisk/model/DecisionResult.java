/*
 * 作用：封装决策引擎输出（风险状态、窗口均值、当前帧概率）。
 * 思路：将算法输出结构化，避免多返回值带来的可读性问题。
 */
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
