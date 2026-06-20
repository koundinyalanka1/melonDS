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

# ── YAGE: CPU-neutral 32-bit ARM baseline ─────────────────────────────────
# Keep armeabi-v7a codegen generic: no -mcpu and no -mtune for a specific
# Cortex family.  NEON remains enabled because this core has explicit NEON
# fast paths, but scheduling/ISA selection stays at the portable ARMv7-A level.
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  LOCAL_CFLAGS   += -march=armv7-a -mfpu=neon
  LOCAL_CPPFLAGS += -march=armv7-a -mfpu=neon
endif

# ── YAGE: A32 JIT instruction-kind profiler (armeabi-v7a only) ───────────────
# Set to 1 to enable: counts native-emitted vs interpreter-fallback per
# instruction kind, IO bucket hit rates, and direct-region chain hit rates.
# Logs via LogA32JitProfile every ~1M events (melonDS-JIT tag).
# Keep at 0 for shipping builds — adds ~5-10% overhead from atomic counters.
# To enable for a local profiling build:
#   A32JIT_PROFILE=1 FORCE_REBUILD=1 CORE_FILTER=melonds ABIS=armeabi-v7a \
#     ./scripts/build_libretro_cores.sh
# Then play a 3D scene for 2+ minutes and grep logcat for
# "melonDS-JIT.*profile".
A32JIT_PROFILE ?= 0
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a)
  LOCAL_CPPFLAGS += -DA32JIT_PROFILE=$(A32JIT_PROFILE)
endif

include $(BUILD_SHARED_LIBRARY)
