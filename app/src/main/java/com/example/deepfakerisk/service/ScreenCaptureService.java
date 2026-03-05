package com.example.deepfakerisk.service;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.view.WindowManager;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import com.example.deepfakerisk.MainActivity;
import com.example.deepfakerisk.R;
import com.example.deepfakerisk.model.DecisionEngine;
import com.example.deepfakerisk.model.DecisionResult;
import com.example.deepfakerisk.model.FaceBox;
import com.example.deepfakerisk.model.RiskState;
import com.example.deepfakerisk.model.StubModelRunner;

import java.nio.ByteBuffer;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

public class ScreenCaptureService extends Service {

    public static final String ACTION_INIT = "com.example.deepfakerisk.action.INIT";
    public static final String ACTION_START = "com.example.deepfakerisk.action.START";
    public static final String ACTION_STOP = "com.example.deepfakerisk.action.STOP";

    public static final String EXTRA_RESULT_CODE = "extra_result_code";
    public static final String EXTRA_RESULT_DATA = "extra_result_data";

    private static final String CHANNEL_ID = "capture_channel";
    private static final int NOTIFICATION_ID = 101;

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    private final AtomicReference<Bitmap> latestFrame = new AtomicReference<>(null);

    private final java.util.concurrent.ExecutorService captureExecutor = Executors.newSingleThreadExecutor();
    private ScheduledExecutorService inferExecutor;

    private final StubModelRunner modelRunner = new StubModelRunner();
    private final DecisionEngine decisionEngine = new DecisionEngine();

    private int width;
    private int height;
    private int densityDpi;

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createChannel();
        startForeground(NOTIFICATION_ID, buildNotification(false));
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null || intent.getAction() == null) {
            return START_STICKY;
        }

        String action = intent.getAction();
        if (ACTION_INIT.equals(action)) {
            // no-op
        } else if (ACTION_START.equals(action)) {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
            Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
            if (resultCode != -1 && resultData != null) {
                startDetection(resultCode, resultData);
            }
        } else if (ACTION_STOP.equals(action)) {
            stopDetection();
        }

        return START_STICKY;
    }

    private void startDetection(int resultCode, Intent resultData) {
        if (mediaProjection != null) {
            return;
        }

        MediaProjectionManager manager = getSystemService(MediaProjectionManager.class);
        mediaProjection = manager.getMediaProjection(resultCode, resultData);

        DisplayMetrics metrics = new DisplayMetrics();
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (getDisplay() != null) {
                getDisplay().getRealMetrics(metrics);
            }
        } else {
            //noinspection deprecation
            ((WindowManager) getSystemService(WINDOW_SERVICE)).getDefaultDisplay().getRealMetrics(metrics);
        }

        width = metrics.widthPixels;
        height = metrics.heightPixels;
        densityDpi = metrics.densityDpi;

        rebuildVirtualDisplay();
        startInferenceLoop();
        DetectionStateBus.update(new DetectionState(true, RiskState.SAFE, 0f, 0f));
        updateNotification(true);
    }

    private void rebuildVirtualDisplay() {
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }

        imageReader = ImageReader.newInstance(width, height, ImageFormat.RGBA_8888, 2);
        imageReader.setOnImageAvailableListener(reader -> captureExecutor.execute(() -> {
            Image image = reader.acquireLatestImage();
            if (image == null) {
                return;
            }
            try {
                Bitmap bitmap = imageToBitmap(image);
                Bitmap old = latestFrame.getAndSet(bitmap);
                if (old != null && !old.isRecycled()) {
                    old.recycle();
                }
            } finally {
                // 必须及时 close，避免 ImageReader 堆积泄漏
                image.close();
            }
        }), null);

        virtualDisplay = mediaProjection.createVirtualDisplay(
                "DeepfakeRiskDisplay",
                width,
                height,
                densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(),
                null,
                null
        );
    }

    private void startInferenceLoop() {
        if (inferExecutor != null) {
            inferExecutor.shutdownNow();
        }
        inferExecutor = Executors.newSingleThreadScheduledExecutor();
        inferExecutor.scheduleAtFixedRate(() -> {
            try {
                maybeRebuildForSizeChange();
                Bitmap frame = latestFrame.get();
                if (frame == null) {
                    return;
                }
                FaceBox faceBox = buildCenterFaceBox(frame.getWidth(), frame.getHeight());
                Bitmap roi224 = cropAndResize(frame, faceBox);
                float pFake = modelRunner.infer(roi224);
                DecisionResult result = decisionEngine.add(pFake);
                DetectionStateBus.update(new DetectionState(true, result.riskState, result.mean, result.pFake));
                roi224.recycle();
            } catch (Throwable ignored) {
            }
        }, 0, 200, TimeUnit.MILLISECONDS);
    }

    private void maybeRebuildForSizeChange() {
        DisplayMetrics metrics = getResources().getDisplayMetrics();
        if (metrics.widthPixels != width || metrics.heightPixels != height || metrics.densityDpi != densityDpi) {
            width = metrics.widthPixels;
            height = metrics.heightPixels;
            densityDpi = metrics.densityDpi;
            rebuildVirtualDisplay();
        }
    }

    private FaceBox buildCenterFaceBox(int w, int h) {
        // TODO: Replace with real face detector.
        int boxW = (int) (w * 0.35f);
        int boxH = (int) (h * 0.35f);

        int cx = w / 2;
        int cy = h / 2;

        int left = cx - boxW / 2;
        int top = cy - boxH / 2;
        int right = cx + boxW / 2;
        int bottom = cy + boxH / 2;

        int expandX = (int) (boxW * 0.2f);
        int expandY = (int) (boxH * 0.2f);

        return new FaceBox(
                Math.max(0, left - expandX),
                Math.max(0, top - expandY),
                Math.min(w, right + expandX),
                Math.min(h, bottom + expandY)
        );
    }

    private Bitmap cropAndResize(Bitmap frame, FaceBox box) {
        int safeW = Math.max(1, box.right - box.left);
        int safeH = Math.max(1, box.bottom - box.top);
        Bitmap roi = Bitmap.createBitmap(frame, box.left, box.top, safeW, safeH);
        Bitmap scaled = Bitmap.createScaledBitmap(roi, 224, 224, true);
        roi.recycle();
        return scaled;
    }

    private Bitmap imageToBitmap(Image image) {
        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int rowPadding = rowStride - pixelStride * image.getWidth();

        Bitmap bitmap = Bitmap.createBitmap(
                image.getWidth() + rowPadding / pixelStride,
                image.getHeight(),
                Bitmap.Config.ARGB_8888
        );
        bitmap.copyPixelsFromBuffer(buffer);

        Bitmap cropped = Bitmap.createBitmap(bitmap, 0, 0, image.getWidth(), image.getHeight());
        bitmap.recycle();
        return cropped;
    }

    private void stopDetection() {
        if (inferExecutor != null) {
            inferExecutor.shutdownNow();
            inferExecutor = null;
        }

        Bitmap old = latestFrame.getAndSet(null);
        if (old != null && !old.isRecycled()) {
            old.recycle();
        }

        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
        if (mediaProjection != null) {
            mediaProjection.stop();
            mediaProjection = null;
        }

        DetectionStateBus.update(DetectionState.idle());
        updateNotification(false);
    }

    @Override
    public void onDestroy() {
        stopDetection();
        captureExecutor.shutdownNow();
        super.onDestroy();
    }

    private void createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel(
                    CHANNEL_ID,
                    "Screen Capture",
                    NotificationManager.IMPORTANCE_LOW
            );
            getSystemService(NotificationManager.class).createNotificationChannel(channel);
        }
    }

    private void updateNotification(boolean detecting) {
        getSystemService(NotificationManager.class).notify(NOTIFICATION_ID, buildNotification(detecting));
    }

    private Notification buildNotification(boolean detecting) {
        PendingIntent contentIntent = PendingIntent.getActivity(
                this,
                0,
                new Intent(this, MainActivity.class),
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        Intent stop = new Intent(this, ScreenCaptureService.class);
        stop.setAction(ACTION_STOP);
        PendingIntent stopIntent = PendingIntent.getService(
                this,
                100,
                stop,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.mipmap.ic_launcher)
                .setContentTitle("Deepfake Risk 预警")
                .setContentText(detecting ? "检测进行中" : "检测已停止")
                .setOngoing(true)
                .setContentIntent(contentIntent)
                .addAction(0, "Stop", stopIntent)
                .build();
    }
}
