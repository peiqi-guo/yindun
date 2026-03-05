/*
 * 作用：抽象模型推理接口，便于后续切换不同后端。
 * 思路：保持接口最小化（输入Bitmap或张量、输出概率/得分）。
 */
package com.example.deepfakerisk.model;

import android.graphics.Bitmap;

public interface ModelRunner {
    float infer(Bitmap input224);
}
