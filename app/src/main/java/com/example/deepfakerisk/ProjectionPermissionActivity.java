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

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        MediaProjectionManager projectionManager = getSystemService(MediaProjectionManager.class);
        startActivityForResult(projectionManager.createScreenCaptureIntent(), REQ_CAPTURE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_CAPTURE && resultCode == Activity.RESULT_OK && data != null) {
            ProjectionGrantStore.resultCode = resultCode;
            ProjectionGrantStore.resultData = data;

            Intent serviceIntent = new Intent(this, ScreenCaptureService.class);
            serviceIntent.setAction(ScreenCaptureService.ACTION_START);
            serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, resultCode);
            serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_DATA, data);
            ContextCompat.startForegroundService(this, serviceIntent);
        }
        finish();
    }
}
