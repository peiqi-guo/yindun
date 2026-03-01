package com.example.deepfakerisk.service;

import android.content.Intent;

public final class ProjectionGrantStore {
    public static volatile Integer resultCode;
    public static volatile Intent resultData;

    private ProjectionGrantStore() {
    }

    public static void clear() {
        resultCode = null;
        resultData = null;
    }
}
