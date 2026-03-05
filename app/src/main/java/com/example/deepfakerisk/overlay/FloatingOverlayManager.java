package com.example.deepfakerisk.overlay;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.provider.Settings;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.ContextCompat;

import com.example.deepfakerisk.MainActivity;
import com.example.deepfakerisk.ProjectionPermissionActivity;
import com.example.deepfakerisk.SettingsStore;
import com.example.deepfakerisk.model.RiskState;
import com.example.deepfakerisk.service.DetectionState;
import com.example.deepfakerisk.service.DetectionStateBus;
import com.example.deepfakerisk.service.ProjectionGrantStore;
import com.example.deepfakerisk.service.ScreenCaptureService;

public final class FloatingOverlayManager {

    private static View bubbleView;
    private static View panelView;
    private static WindowManager windowManager;
    private static WindowManager.LayoutParams bubbleLp;
    private static long lastClickTs;

    private static final Handler MAIN = new Handler(Looper.getMainLooper());
    private static DetectionStateBus.Listener listener;

    private FloatingOverlayManager() {
    }

    public static boolean isShowing() {
        return bubbleView != null;
    }

    public static void show(Context context) {
        if (bubbleView != null || !Settings.canDrawOverlays(context)) {
            return;
        }

        windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        bubbleView = buildBubble(context.getApplicationContext());
        bubbleLp = new WindowManager.LayoutParams(
                dp(context, 56),
                dp(context, 56),
                overlayType(),
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT
        );
        bubbleLp.gravity = Gravity.TOP | Gravity.START;
        bubbleLp.x = 0;
        bubbleLp.y = dp(context, 120);
        windowManager.addView(bubbleView, bubbleLp);
        SettingsStore.setBubbleEnabled(context, true);
    }

    public static void hide(Context context) {
        unregisterPanelListener();

        if (panelView != null && windowManager != null) {
            windowManager.removeView(panelView);
        }
        panelView = null;

        if (bubbleView != null && windowManager != null) {
            windowManager.removeView(bubbleView);
        }
        bubbleView = null;
        bubbleLp = null;
        SettingsStore.setBubbleEnabled(context, false);

        ProjectionGrantStore.clear();

        Intent stopIntent = new Intent(context, ScreenCaptureService.class);
        stopIntent.setAction(ScreenCaptureService.ACTION_STOP);
        ContextCompat.startForegroundService(context, stopIntent);
    }

    private static View buildBubble(Context context) {
        TextView tv = new TextView(context);
        tv.setText("DF");
        tv.setTextColor(Color.WHITE);
        tv.setGravity(Gravity.CENTER);
        tv.setTextSize(16f);

        GradientDrawable drawable = new GradientDrawable();
        drawable.setShape(GradientDrawable.OVAL);
        drawable.setColor(Color.parseColor("#AA6200EE"));
        tv.setBackground(drawable);

        final float[] downX = new float[1];
        final float[] downY = new float[1];
        final float[] downRawX = new float[1];
        final float[] downRawY = new float[1];
        final boolean[] moved = new boolean[1];

        tv.setOnTouchListener((v, event) -> {
            if (bubbleLp == null) {
                return false;
            }

            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    downX[0] = bubbleLp.x;
                    downY[0] = bubbleLp.y;
                    downRawX[0] = event.getRawX();
                    downRawY[0] = event.getRawY();
                    moved[0] = false;
                    return true;
                case MotionEvent.ACTION_MOVE:
                    float dx = event.getRawX() - downRawX[0];
                    float dy = event.getRawY() - downRawY[0];
                    if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
                        moved[0] = true;
                    }
                    bubbleLp.x = (int) (downX[0] + dx);
                    bubbleLp.y = (int) (downY[0] + dy);
                    windowManager.updateViewLayout(tv, bubbleLp);
                    return true;
                case MotionEvent.ACTION_UP:
                    if (moved[0]) {
                        snapToEdge(context, bubbleLp);
                    } else {
                        handleClick(context);
                    }
                    return true;
                default:
                    return false;
            }
        });

        tv.setOnLongClickListener(v -> {
            togglePanel(context);
            return true;
        });

        return tv;
    }

    private static void handleClick(Context context) {
        boolean antiEnabled = SettingsStore.isAntiMistouchEnabled(context);
        if (antiEnabled) {
            long now = SystemClock.elapsedRealtime();
            if (now - lastClickTs > 600L) {
                lastClickTs = now;
                Toast.makeText(context, "防误触：请双击确认", Toast.LENGTH_SHORT).show();
                return;
            }
        }

        DetectionState state = DetectionStateBus.getState();
        if (state.detecting) {
            stopDetection(context);
        } else {
            startDetection(context);
        }
        lastClickTs = 0L;
    }

    private static void togglePanel(Context context) {
        if (panelView == null) {
            showPanel(context);
        } else {
            hidePanel();
        }
    }

    private static void showPanel(Context context) {
        if (panelView != null) return;

        LinearLayout layout = new LinearLayout(context);
        layout.setOrientation(LinearLayout.VERTICAL);
        int padding = dp(context, 12);
        layout.setPadding(padding, padding, padding, padding);

        GradientDrawable bg = new GradientDrawable();
        bg.setCornerRadius(dp(context, 10));
        bg.setColor(Color.parseColor("#CC222222"));
        layout.setBackground(bg);

        TextView statusTv = new TextView(context);
        statusTv.setTextColor(Color.WHITE);
        TextView valueTv = new TextView(context);
        valueTv.setTextColor(Color.WHITE);

        Button btnToggle = new Button(context);
        Button btnExitBubble = new Button(context);
        btnExitBubble.setText("退出悬浮球");
        Button btnSettings = new Button(context);
        btnSettings.setText("进入设置");

        layout.addView(statusTv);
        layout.addView(valueTv);
        layout.addView(btnToggle);
        layout.addView(btnSettings);
        layout.addView(btnExitBubble);

        btnToggle.setOnClickListener(v -> {
            if (DetectionStateBus.getState().detecting) {
                stopDetection(context);
            } else {
                startDetection(context);
            }
        });
        btnExitBubble.setOnClickListener(v -> hide(context));
        btnSettings.setOnClickListener(v -> {
            Intent intent = new Intent(context, MainActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
        });

        panelView = layout;
        WindowManager.LayoutParams panelLp = new WindowManager.LayoutParams(
                dp(context, 220),
                WindowManager.LayoutParams.WRAP_CONTENT,
                overlayType(),
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE,
                PixelFormat.TRANSLUCENT
        );
        panelLp.gravity = Gravity.TOP | Gravity.START;
        panelLp.x = dp(context, 16);
        panelLp.y = dp(context, 200);
        windowManager.addView(panelView, panelLp);

        listener = state -> MAIN.post(() -> {
            statusTv.setText("状态: " + state.risk.name());
            valueTv.setText(String.format("s=%.3f, p_fake=%.3f", state.mean, state.pFake));
            btnToggle.setText(state.detecting ? "Stop" : "Start");
            if (state.risk == RiskState.DANGEROUS) {
                statusTv.setTextColor(Color.RED);
            } else if (state.risk == RiskState.SUSPICIOUS) {
                statusTv.setTextColor(Color.YELLOW);
            } else {
                statusTv.setTextColor(Color.GREEN);
            }
        });
        DetectionStateBus.register(listener);
    }

    private static void hidePanel() {
        unregisterPanelListener();
        if (panelView != null && windowManager != null) {
            windowManager.removeView(panelView);
        }
        panelView = null;
    }

    private static void unregisterPanelListener() {
        if (listener != null) {
            DetectionStateBus.unregister(listener);
            listener = null;
        }
    }

    private static void startDetection(Context context) {
        if (!Settings.canDrawOverlays(context)) {
            Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + context.getPackageName()));
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
            return;
        }

        if (ProjectionGrantStore.resultCode == null || ProjectionGrantStore.resultData == null) {
            Intent intent = new Intent(context, ProjectionPermissionActivity.class);
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            context.startActivity(intent);
            return;
        }

        Intent serviceIntent = new Intent(context, ScreenCaptureService.class);
        serviceIntent.setAction(ScreenCaptureService.ACTION_START);
        serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_CODE, ProjectionGrantStore.resultCode);
        serviceIntent.putExtra(ScreenCaptureService.EXTRA_RESULT_DATA, ProjectionGrantStore.resultData);
        ContextCompat.startForegroundService(context, serviceIntent);
    }

    private static void stopDetection(Context context) {
        Intent stopIntent = new Intent(context, ScreenCaptureService.class);
        stopIntent.setAction(ScreenCaptureService.ACTION_STOP);
        ContextCompat.startForegroundService(context, stopIntent);
    }

    private static void snapToEdge(Context context, WindowManager.LayoutParams lp) {
        int screenWidth = context.getResources().getDisplayMetrics().widthPixels;
        lp.x = (lp.x + dp(context, 28) > screenWidth / 2) ? screenWidth - dp(context, 56) : 0;
        if (windowManager != null && bubbleView != null) {
            windowManager.updateViewLayout(bubbleView, lp);
        }
    }

    private static int dp(Context context, int value) {
        return (int) TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                value,
                context.getResources().getDisplayMetrics()
        );
    }

    private static int overlayType() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            return WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY;
        }
        return WindowManager.LayoutParams.TYPE_PHONE;
    }
}
