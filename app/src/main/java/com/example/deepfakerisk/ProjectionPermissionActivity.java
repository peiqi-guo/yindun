/*
 * 作用：仅用于弹出MediaProjection系统授权，并把授权结果转交给前台服务。
 * 思路：Activity尽量保持“短生命周期中转站”，避免业务状态堆积。
 */
package com.example.deepfakerisk;

import android.app.Activity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.deepfakerisk.service.ProjectionGrantStore;
import com.example.deepfakerisk.service.ScreenCaptureService;

public class ProjectionPermissionActivity extends AppCompatActivity {
    private static final int REQ_CAPTURE = 9001;
    private MediaProjectionManager projectionManager;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        projectionManager = getSystemService(MediaProjectionManager.class);
        startActivityForResult(projectionManager.createScreenCaptureIntent(), REQ_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_CAPTURE && resultCode == Activity.RESULT_OK && data != null) {
            ProjectionGrantStore.resultCode = resultCode;
            ProjectionGrantStore.resultData = data;

            Intent startIntent = new Intent(this, ScreenCaptureService.class);
            startIntent.setAction(ScreenCaptureService.ACTION_START);
            startIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
            startIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_DATA, data);
            ContextCompat.startForegroundService(this, startIntent);
        }
        finish();
    }
}
