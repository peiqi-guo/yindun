package com.example.deepfakerisk;

import android.content.Context;
import android.content.SharedPreferences;

public final class SettingsStore {
    private static final String PREF = "deepfake_demo_pref";
    private static final String KEY_ANTI = "anti_mistouch";
    private static final String KEY_DIAG = "perf_diagnostic";
    private static final String KEY_BUBBLE_ENABLED = "bubble_enabled";

    private SettingsStore() {
    }

    public static boolean isAntiMistouchEnabled(Context context) {
        SharedPreferences sp = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        return sp.getBoolean(KEY_ANTI, true);
    }

    public static void setAntiMistouchEnabled(Context context, boolean enabled) {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_ANTI, enabled)
                .apply();
    }

    public static boolean isPerformanceDiagnosticEnabled(Context context) {
        SharedPreferences sp = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        return sp.getBoolean(KEY_DIAG, false);
    }

    public static void setPerformanceDiagnosticEnabled(Context context, boolean enabled) {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_DIAG, enabled)
                .apply();
    }

    public static boolean isBubbleEnabled(Context context) {
        SharedPreferences sp = context.getSharedPreferences(PREF, Context.MODE_PRIVATE);
        return sp.getBoolean(KEY_BUBBLE_ENABLED, false);
    }

    public static void setBubbleEnabled(Context context, boolean enabled) {
        context.getSharedPreferences(PREF, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_BUBBLE_ENABLED, enabled)
                .apply();
    }
}
