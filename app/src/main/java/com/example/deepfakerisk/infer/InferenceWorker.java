package com.example.deepfakerisk.infer;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

import com.example.deepfakerisk.model.DecisionEngine;
import com.example.deepfakerisk.model.DecisionResult;
import com.example.deepfakerisk.model.FaceBox;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

public class InferenceWorker {
    private final AtomicReference<Bitmap> latestFrameRef;
    private final AtomicLong frameTokenRef;
    private final NCNNModelRunner modelRunner;
    private final DecisionEngine decisionEngine;
    private final InferenceListener callback;

    private final Preprocessor preprocessor = new Preprocessor(224, 224);

    // 复用对象，减少GC抖动
    private final Bitmap reusableInput = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888);
    private final Canvas reusableCanvas = new Canvas(reusableInput);
    private final Rect srcRect = new Rect();
    private final Rect dstRect = new Rect(0, 0, 224, 224);
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);

    private ScheduledExecutorService scheduler;
    private long lastFrameToken = -1L;

    private long inferCount;
    private long inferCostNs;
    private long droppedNoFrame;

    public InferenceWorker(
            AtomicReference<Bitmap> latestFrameRef,
            AtomicLong frameTokenRef,
            NCNNModelRunner modelRunner,
            DecisionEngine decisionEngine,
            InferenceListener callback
    ) {
        this.latestFrameRef = latestFrameRef;
        this.frameTokenRef = frameTokenRef;
        this.modelRunner = modelRunner;
        this.decisionEngine = decisionEngine;
        this.callback = callback;
    }

    public void start() {
        if (scheduler != null) {
            return;
        }
        scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(() -> {
            try {
                Bitmap frame = latestFrameRef.get();
                if (frame == null) {
                    droppedNoFrame++;
                    return;
                }

                long token = frameTokenRef.get();
                if (token == lastFrameToken) {
                    droppedNoFrame++;
                    return;
                }
                lastFrameToken = token;

                long startNs = System.nanoTime();

                FaceBox box = buildCenterFaceBox(frame.getWidth(), frame.getHeight());
                renderRoiToReusable224(frame, box);
                float[] chw = preprocessor.toCHW(reusableInput);
                float logit = modelRunner.infer(chw);
                float pFrame = sigmoid(logit);
                DecisionResult result = decisionEngine.add(pFrame);
                callback.onResult(result.riskState, result.mean, pFrame, logit);

                inferCount++;
                inferCostNs += (System.nanoTime() - startNs);
            } catch (Throwable ignored) {
            }
        }, 0, 200, TimeUnit.MILLISECONDS);
    }

    public void stop() {
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler = null;
        }
        if (!reusableInput.isRecycled()) {
            reusableInput.recycle();
        }
    }

    public PerfSnapshot drainAndResetPerfStats() {
        long count = inferCount;
        long total = inferCostNs;
        long dropped = droppedNoFrame;
        inferCount = 0;
        inferCostNs = 0;
        droppedNoFrame = 0;

        float avgMs = count == 0 ? 0f : (total / 1_000_000f / count);
        return new PerfSnapshot(count, avgMs, dropped);
    }

    private FaceBox buildCenterFaceBox(int w, int h) {
        int boxW = (int) (w * 0.35f);
        int boxH = (int) (h * 0.35f);
        int cx = w / 2;
        int cy = h / 2;

        int left = cx - boxW / 2;
        int top = cy - boxH / 2;
        int right = cx + boxW / 2;
        int bottom = cy + boxH / 2;

        // 扩框20% + 边界检查
        int expandX = (int) (boxW * 0.2f);
        int expandY = (int) (boxH * 0.2f);

        return new FaceBox(
                Math.max(0, left - expandX),
                Math.max(0, top - expandY),
                Math.min(w, right + expandX),
                Math.min(h, bottom + expandY)
        );
    }

    private void renderRoiToReusable224(Bitmap frame, FaceBox box) {
        int right = Math.max(box.left + 1, box.right);
        int bottom = Math.max(box.top + 1, box.bottom);
        srcRect.set(box.left, box.top, right, bottom);
        reusableCanvas.drawBitmap(frame, srcRect, dstRect, paint);
    }

    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }
}
