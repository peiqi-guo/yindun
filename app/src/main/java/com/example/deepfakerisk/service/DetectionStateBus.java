/*
 * 作用：进程内状态总线，向悬浮窗等订阅者广播检测状态。
 * 思路：CopyOnWrite监听集合换取线程安全和简单实现。
 */
package com.example.deepfakerisk.service;

import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

public final class DetectionStateBus {

    public interface Listener {
        void onStateChanged(DetectionState state);
    }

    private static final Set<Listener> LISTENERS = new CopyOnWriteArraySet<>();
    private static volatile DetectionState state = DetectionState.idle();

    private DetectionStateBus() {
    }

    public static void register(Listener listener) {
        LISTENERS.add(listener);
        listener.onStateChanged(state);
    }

    public static void unregister(Listener listener) {
        LISTENERS.remove(listener);
    }

    public static DetectionState getState() {
        return state;
    }

    public static void update(DetectionState newState) {
        state = newState;
        for (Listener listener : LISTENERS) {
            listener.onStateChanged(newState);
        }
    }
}
