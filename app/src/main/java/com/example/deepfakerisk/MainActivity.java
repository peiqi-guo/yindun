package com.example.deepfakerisk;

import android.Manifest;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import com.example.deepfakerisk.overlay.FloatingOverlayManager;
import com.example.deepfakerisk.service.ScreenCaptureService;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        TextView tvStatus = findViewById(R.id.tvStatus);
        Button btnOverlayPermission = findViewById(R.id.btnOverlayPermission);
        Button btnToggleBubble = findViewById(R.id.btnToggleBubble);
        Switch switchAnti = findViewById(R.id.switchAntiMistouch);

        switchAnti.setChecked(SettingsStore.isAntiMistouchEnabled(this));
        switchAnti.setOnCheckedChangeListener((buttonView, isChecked) ->
                SettingsStore.setAntiMistouchEnabled(this, isChecked));

        btnOverlayPermission.setOnClickListener(v -> {
            if (!Settings.canDrawOverlays(this)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName()));
                startActivity(intent);
            }
        });

        btnToggleBubble.setOnClickListener(v -> {
            if (!Settings.canDrawOverlays(this)) {
                tvStatus.setText("请先授予悬浮窗权限");
                return;
            }
            if (FloatingOverlayManager.isShowing()) {
                FloatingOverlayManager.hide(this);
                tvStatus.setText("悬浮球已隐藏");
                btnToggleBubble.setText("显示悬浮球");
            } else {
                FloatingOverlayManager.show(this);
                tvStatus.setText("悬浮球已显示");
                btnToggleBubble.setText("隐藏悬浮球");
            }
        });

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, 1001);
        }

        Intent initIntent = new Intent(this, ScreenCaptureService.class);
        initIntent.setAction(ScreenCaptureService.ACTION_INIT);
        ContextCompat.startForegroundService(this, initIntent);
    }

    @Override
    protected void onResume() {
        super.onResume();
        TextView tvStatus = findViewById(R.id.tvStatus);
        tvStatus.setText(Settings.canDrawOverlays(this) ? "悬浮窗权限已授予" : "请先授予悬浮窗权限");
    }
}
