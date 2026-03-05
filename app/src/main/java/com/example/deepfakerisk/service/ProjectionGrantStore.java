/*
 * 作用：临时缓存MediaProjection授权结果（resultCode/data）。
 * 思路：进程内轻量缓存，减少重复弹授权。
 */
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
