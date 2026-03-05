/*
 * 作用：表示ROI矩形（left/top/right/bottom）。
 * 思路：保持轻量不可变数据结构，供裁剪与边界检查复用。
 */
package com.example.deepfakerisk.model;

public class FaceBox {
    public final int left;
    public final int top;
    public final int right;
    public final int bottom;

    public FaceBox(int left, int top, int right, int bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }
}
