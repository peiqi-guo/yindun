/*
 * 作用：定义推理结果回调接口，解耦推理线程与上层状态发布。
 * 思路：用简单接口承载risk/p_window/p_frame/logit，便于替换上层消费方。
 */
package com.example.deepfakerisk.infer;

import com.example.deepfakerisk.model.RiskState;

public interface InferenceListener {
    void onResult(RiskState risk, float pWindow, float pFrame, float logit);
}
