/*
 * 作用：前台服务，负责录屏采集、最新帧缓存、推理调度、状态发布与资源释放。
 * 思路：采集与推理解耦，采用最新帧策略+周期诊断日志，保障长时间稳定运行。
 */
package com.example.deepfakerisk.service;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Debug;
import android.os.IBinder;
import android.provider.Settings;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;

import com.example.deepfakerisk.MainActivity;
import com.example.deepfakerisk.R;
import com.example.deepfakerisk.SettingsStore;
import com.example.deepfakerisk.infer.InferenceListener;
import com.example.deepfakerisk.infer.InferenceWorker;
import com.example.deepfakerisk.infer.NCNNModelRunner;
import com.example.deepfakerisk.infer.PerfSnapshot;
import com.example.deepfakerisk.infer.Preprocessor;
import com.example.deepfakerisk.model.DecisionEngine;
import com.example.deepfakerisk.model.RiskState;
import com.example.deepfakerisk.overlay.FloatingOverlayManager;

import java.nio.ByteBuffer;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class ScreenCaptureService extends Service {

    private static final String TAG = "ScreenCaptureService";
    private static final String CHANNEL_ID = "capture_channel";
    private static final int NOTIFICATION_ID = 101;

    public static final String ACTION_START = "com.example.deepfakerisk.action.START";
    public static final String ACTION_STOP = "com.example.deepfakerisk.action.STOP";
    public static final String EXTRA_RESULT_CODE = "extra_result_code";
    public static final String EXTRA_RESULT_DATA = "extra_result_data";

    private MediaProjection mediaProjection;
    private VirtualDisplay virtualDisplay;
    private ImageReader imageReader;

    // 只保留最新帧，防积压（防泄露点 #1）
    private final AtomicReference<Bitmap> latestFrame = new AtomicReference<>(null);
    private final AtomicLong latestFrameToken = new AtomicLong(0L);
    private final ExecutorService captureExecutor = Executors.newSingleThreadExecutor();

    private NCNNModelRunner ncnnModelRunner;
    private DecisionEngine decisionEngine;
    private InferenceWorker inferenceWorker;

    private ScheduledExecutorService perfScheduler;

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

        // Service 重启恢复：至少恢复悬浮球与通知
        if (SettingsStore.isBubbleEnabled(this) && Settings.canDrawOverlays(this)) {
            FloatingOverlayManager.show(getApplicationContext());
        }

        ncnnModelRunner = new NCNNModelRunner(getApplicationContext());
        decisionEngine = new DecisionEngine();

        try {
            ncnnModelRunner.init();
            selfCheck();
        } catch (Throwable t) {
            Log.e(TAG, "NCNN init/self-check failed", t);
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent != null ? intent.getAction() : null;
        if (ACTION_START.equals(action)) {
            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1);
            Intent resultData = intent.getParcelableExtra(EXTRA_RESULT_DATA);
            if (resultCode != -1 && resultData != null) {
                startDetection(resultCode, resultData);
            } else {
                Log.w(TAG, "ACTION_START ignored: missing MediaProjection grant data");
            }
        } else if (ACTION_STOP.equals(action)) {
            stopDetection();
        } else {
            // START_STICKY 场景：确保通知存在
            updateNotification(DetectionStateBus.getState().detecting);
        }
        return START_STICKY;
    }

    private void selfCheck() {
        try {
            Bitmap bmp = BitmapFactory.decodeStream(getAssets().open("test_face.jpg"));
            if (bmp == null) {
                Log.w(TAG, "self-check skipped: assets/test_face.jpg decode failed");
                return;
            }
            Bitmap resized = Bitmap.createScaledBitmap(bmp, 224, 224, true);
            Preprocessor preprocessor = new Preprocessor(224, 224);
            float[] chw = preprocessor.toCHW(resized);
            float logit = ncnnModelRunner.infer(chw);
            float pFake = (float) (1.0 / (1.0 + Math.exp(-logit)));
            Log.i(TAG, "self-check done, logit=" + logit + ", p_fake=" + pFake);
            bmp.recycle();
            resized.recycle();
        } catch (Throwable t) {
            Log.w(TAG, "self-check skipped: test image missing or infer failed", t);
        }
    }

    private void startDetection(int resultCode, Intent resultData) {
        if (mediaProjection != null) return;

        MediaProjectionManager projectionManager = getSystemService(MediaProjectionManager.class);
        mediaProjection = projectionManager.getMediaProjection(resultCode, resultData);

        updateDisplayMetrics();
        rebuildVirtualDisplay();
        startInferenceWorker();
        startPerfLogger();

        DetectionStateBus.update(new DetectionState(true, RiskState.SAFE, 0f, 0f));
        updateNotification(true);
    }

    private void updateDisplayMetrics() {
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
    }

    private void startInferenceWorker() {
        if (inferenceWorker != null) {
            inferenceWorker.stop();
        }

        inferenceWorker = new InferenceWorker(
                latestFrame,
                latestFrameToken,
                ncnnModelRunner,
                decisionEngine,
                new InferenceListener() {
                    @Override
                    public void onResult(RiskState risk, float pWindow, float pFrame, float logit) {
                        DetectionStateBus.update(new DetectionState(true, risk, pWindow, pFrame));
                    }
                }
        );
        inferenceWorker.start();
    }

    private void startPerfLogger() {
        if (perfScheduler != null) {
            perfScheduler.shutdownNow();
        }
        perfScheduler = Executors.newSingleThreadScheduledExecutor();
        perfScheduler.scheduleAtFixedRate(() -> {
            maybeRebuildForSizeChange();
            if (!SettingsStore.isPerformanceDiagnosticEnabled(this)) {
                return;
            }

            PerfSnapshot snapshot = inferenceWorker != null
                    ? inferenceWorker.drainAndResetPerfStats()
                    : new PerfSnapshot(0L, 0f, 0L);
            float fps = snapshot.inferCount / 5f;
            float pssMb = Debug.getPss() / 1024f;
            float javaMb = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / 1024f / 1024f;

            Log.d(TAG, String.format(Locale.US,
                    "diag: fps=%.2f, infer=%.2fms, dropNoFrame=%d, pss=%.2fMB, java=%.2fMB",
                    fps, snapshot.avgInferMs, snapshot.droppedNoFrame, pssMb, javaMb));
        }, 5, 5, TimeUnit.SECONDS);
    }

    private void stopPerfLogger() {
        if (perfScheduler != null) {
            perfScheduler.shutdownNow();
            perfScheduler = null;
        }
    }

    private void maybeRebuildForSizeChange() {
        int oldW = width;
        int oldH = height;
        int oldDpi = densityDpi;

        updateDisplayMetrics();
        if (oldW != width || oldH != height || oldDpi != densityDpi) {
            Log.i(TAG, "display changed: " + oldW + "x" + oldH + " -> " + width + "x" + height + ", rebuild VirtualDisplay");
            rebuildVirtualDisplay();
        }
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
                // 防泄露点 #2：严格 finally close image
                Bitmap bitmap = imageToBitmap(image);
                Bitmap old = latestFrame.getAndSet(bitmap);
                if (old != null && !old.isRecycled()) {
                    old.recycle();
                }
                latestFrameToken.incrementAndGet();
            } finally {
                image.close();
            }
        }), null);

        virtualDisplay = mediaProjection != null
                ? mediaProjection.createVirtualDisplay(
                "DeepfakeRiskDisplay",
                width,
                height,
                densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(),
                null,
                null)
                : null;
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
        stopPerfLogger();

        if (inferenceWorker != null) {
            inferenceWorker.stop();
            inferenceWorker = null;
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
        // 防泄露点 #3：销毁路径统一释放所有图像与投屏资源
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
