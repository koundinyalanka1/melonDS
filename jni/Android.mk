LOCAL_PATH := $(call my-dir)

ROOT_DIR     := $(LOCAL_PATH)/..
MELON_DIR    := $(ROOT_DIR)/src
CORE_DIR     := $(MELON_DIR)/libretro
JIT_ARCH     :=
HAVE_THREADS := 1
# YAGE: OpenGL renderer enabled for the Android/GLES3 build.
HAVE_OPENGL := 1

ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  JIT_ARCH := armv7
else ifeq ($(TARGET_ARCH_ABI),arm64-v8a)
  JIT_ARCH := aarch64
else ifeq ($(TARGET_ARCH_ABI),x86_64)
  JIT_ARCH := x64
endif

include $(ROOT_DIR)/Makefile.common

CORE_FLAGS := -D__LIBRETRO__ $(INCFLAGS) $(DEFINES)

include $(CLEAR_VARS)
LOCAL_MODULE    := retro
LOCAL_SRC_FILES := $(SOURCES_C) $(SOURCES_CXX) $(SOURCES_S)
LOCAL_CFLAGS    := $(CORE_FLAGS)
LOCAL_CPPFLAGS  := -std=c++17 -fexceptions -Wno-invalid-offsetof -Wno-macro-redefined $(CORE_FLAGS)
LOCAL_LDFLAGS   := -Wl,-version-script=$(CORE_DIR)/link.T

# ── YAGE: GLES3 / Android OpenGL ─────────────────────────────────────────
# Force-include the GLES3 compatibility shim (gles3_compat.h) which stubs the
# desktop-only GL entry points referenced inside HAVE_OPENGL blocks, and link
# the GLES3 / EGL / log libraries.
LOCAL_CFLAGS  += -DHAVE_OPENGLES3 -DHAVE_OPENGLES -include $(LOCAL_PATH)/gles3_compat.h
LOCAL_LDLIBS  += -lGLESv3 -lEGL -llog

# ── YAGE: cross-TU inlining + tight symbol visibility (all ABIs) ──────────
# ThinLTO lets -O3 inline melonDS's per-frame bus read/write + render helpers
# into the interpreter/JIT hot paths. Hidden visibility is safe because the
# version script (link.T) is what actually controls the exported libretro
# entry points, so nothing the frontend needs gets stripped.
LOCAL_CFLAGS   += -flto=thin -fvisibility=hidden -fno-semantic-interposition
LOCAL_CPPFLAGS += -flto=thin -fvisibility=hidden -fvisibility-inlines-hidden -fno-semantic-interposition
LOCAL_LDFLAGS  += -flto=thin

# ── YAGE: fleet-safe 32-bit tuning for A53/A55-class Android-TV SoCs ───────
# Use -mtune (NOT -mcpu): it schedules for the dominant in-order Cortex-A53/A55
# WITHOUT raising the ISA above the armv7-a baseline, so the binary still runs
# on genuine pre-ARMv8 (A7/A9) 32-bit devices. The first AArch32 JIT backend
# is helper-call based, so these builds still benefit from interpreter-oriented
# scheduling until native opcode coverage lands.
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  LOCAL_CFLAGS   += -march=armv7-a -mtune=cortex-a53 -mfpu=neon
  LOCAL_CPPFLAGS += -march=armv7-a -mtune=cortex-a53 -mfpu=neon
endif

# ── YAGE: A32 JIT instruction-kind profiler (armeabi-v7a only) ───────────────
# Set to 1 to enable: counts native-emitted vs interpreter-fallback per
# instruction kind, IO bucket hit rates, and direct-region chain hit rates.
# Logs via LogA32JitProfile every ~1M events (melonDS-JIT tag).
# Keep at 0 for shipping builds — adds ~5-10% overhead from atomic counters.
# To enable: set A32JIT_PROFILE := 1 below, rebuild armeabi-v7a, play a 3D
# scene for 2+ minutes, then grep logcat for "melonDS-JIT.*profile".
A32JIT_PROFILE := 0
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  LOCAL_CPPFLAGS += -DA32JIT_PROFILE=$(A32JIT_PROFILE)
endif

include $(BUILD_SHARED_LIBRARY)
