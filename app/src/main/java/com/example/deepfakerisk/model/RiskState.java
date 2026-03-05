/*
 * 作用：定义风险等级枚举（SAFE/SUSPICIOUS/DANGEROUS）。
 * 思路：用统一枚举贯穿推理、状态总线与UI显示。
 */
package com.example.deepfakerisk.model;

public enum RiskState {
    SAFE,
    SUSPICIOUS,
    DANGEROUS
}
