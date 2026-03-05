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
