package com.example.deepfakerisk.infer;

import android.graphics.Bitmap;

public class Preprocessor {
    private final int width;
    private final int height;
    private final int[] pixelBuffer;
    private final float[] chwBuffer;

    private final float[] mean = new float[]{0.485f, 0.456f, 0.406f};
    private final float[] std = new float[]{0.229f, 0.224f, 0.225f};

    public Preprocessor() {
        this(224, 224);
    }

    public Preprocessor(int width, int height) {
        this.width = width;
        this.height = height;
        this.pixelBuffer = new int[width * height];
        this.chwBuffer = new float[3 * width * height];
    }

    public float[] toCHW(Bitmap bitmap224) {
        if (bitmap224.getWidth() != width || bitmap224.getHeight() != height) {
            throw new IllegalArgumentException("Expected " + width + "x" + height
                    + ", got " + bitmap224.getWidth() + "x" + bitmap224.getHeight());
        }

        bitmap224.getPixels(pixelBuffer, 0, width, 0, 0, width, height);

        int hw = width * height;
        for (int i = 0; i < pixelBuffer.length; i++) {
            int c = pixelBuffer[i];
            float r = ((c >> 16) & 0xFF) / 255f;
            float g = ((c >> 8) & 0xFF) / 255f;
            float b = (c & 0xFF) / 255f;

            chwBuffer[i] = (r - mean[0]) / std[0];
            chwBuffer[hw + i] = (g - mean[1]) / std[1];
            chwBuffer[2 * hw + i] = (b - mean[2]) / std[2];
        }
        return chwBuffer;
    }
}
