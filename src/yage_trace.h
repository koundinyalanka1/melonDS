/*
    yage_trace.h — debug instrumentation for NDS save / firmware diagnosis.

    Traces the cartridge backup-memory (.sav) SPI conversation and the DS
    firmware SPI reads. The goal is to make the exact command/response a game
    rejects with "A communication error has occurred." visible in logcat, so the
    divergence between the open (DraStic / generated) BIOS+firmware and real
    hardware can be pinpointed and fixed in software — no copyrighted files.

    Capture on device with:
        adb logcat -s YAGE-NDSTRACE:I
    (or `adb logcat | grep NDSTRACE`), reproduce the error, then share the log.

    IMPORTANT: set YAGE_NDS_TRACE to 0 for release builds. It is intentionally
    verbose and only meant for debugging.
*/

#ifndef YAGE_TRACE_H
#define YAGE_TRACE_H

// Flip to 1 to trace NDS save/firmware SPI when diagnosing a game; keep 0 for
// normal/release builds (the macros below compile to nothing when 0).
#define YAGE_NDS_TRACE 0

#if YAGE_NDS_TRACE
  #ifdef __ANDROID__
    #include <android/log.h>
    #define NDSTRACE(...) __android_log_print(ANDROID_LOG_INFO, "YAGE-NDSTRACE", __VA_ARGS__)
  #else
    #include <stdio.h>
    #define NDSTRACE(...) do { printf("[YAGE-NDSTRACE] " __VA_ARGS__); printf("\n"); } while (0)
  #endif
#else
  #define NDSTRACE(...) do {} while (0)
#endif

#endif // YAGE_TRACE_H
