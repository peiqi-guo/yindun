/*
 * 作用：应用主入口，负责权限引导、悬浮球开关、诊断开关等用户侧操作。
 * 思路：把一次性权限申请与持久化设置放在Activity；重逻辑交给Service/Worker。
 */
package com.example.deepfakerisk;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.Switch;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.example.deepfakerisk.overlay.FloatingOverlayManager;

public class MainActivity extends AppCompatActivity {

    private TextView tvStatus;
    private Button btnToggleBubble;

    private final ActivityResultLauncher<String> notificationPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (!granted) {
                    tvStatus.setText("通知权限被拒绝：检测仍可运行，但通知展示可能受限");
                }
            });

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        tvStatus = findViewById(R.id.tvStatus);
        Button btnOverlayPermission = findViewById(R.id.btnOverlayPermission);
        btnToggleBubble = findViewById(R.id.btnToggleBubble);
        Switch switchAnti = findViewById(R.id.switchAntiMistouch);
        Switch switchDiagnostic = findViewById(R.id.switchPerformanceDiagnostic);

        switchAnti.setChecked(SettingsStore.isAntiMistouchEnabled(this));
        switchAnti.setOnCheckedChangeListener((buttonView, checked) ->
                SettingsStore.setAntiMistouchEnabled(this, checked));

        switchDiagnostic.setChecked(SettingsStore.isPerformanceDiagnosticEnabled(this));
        switchDiagnostic.setOnCheckedChangeListener((buttonView, checked) ->
                SettingsStore.setPerformanceDiagnosticEnabled(this, checked));

        btnOverlayPermission.setOnClickListener(v -> {
            if (!Settings.canDrawOverlays(this)) {
                startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                        Uri.parse("package:" + getPackageName())));
            }
        });

        btnToggleBubble.setOnClickListener(v -> {
            if (!Settings.canDrawOverlays(this)) {
                tvStatus.setText("请先授予悬浮窗权限");
                return;
            }

            if (FloatingOverlayManager.isShowing()) {
                FloatingOverlayManager.hide(this);
                SettingsStore.setBubbleEnabled(this, false);
                tvStatus.setText("悬浮球已隐藏");
                btnToggleBubble.setText("显示悬浮球");
            } else {
                FloatingOverlayManager.show(this);
                SettingsStore.setBubbleEnabled(this, true);
                tvStatus.setText("悬浮球已显示");
                btnToggleBubble.setText("隐藏悬浮球");
            }
        });

        requestNotificationPermissionIfNeeded();
    }

    @Override
    protected void onResume() {
        super.onResume();
        tvStatus.setText(Settings.canDrawOverlays(this) ? "悬浮窗权限已授予" : "请先授予悬浮窗权限");
        btnToggleBubble.setText(FloatingOverlayManager.isShowing() ? "隐藏悬浮球" : "显示悬浮球");
    }

    private void requestNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            return;
        }
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            return;
        }
        notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS);
    }
}
