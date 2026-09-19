APP_ABI := all
APP_STL := c++_static
APP_OPTIM := release

# YAGE: pin the minimum API level.
#
# ndk-build otherwise defaults APP_PLATFORM to the NDK's own minimum (android-21
# on r28), and bionic only exposes the GNU `char* strerror_r()` from API 23 —
# below that it declares the POSIX `int` variant and src/dolphin/CommonFuncs.cpp
# fails to compile. android-24 also matches the ASharedMemory_create() path the
# JIT fastmem allocator prefers (API 26, with a /dev/ashmem fallback below it).
APP_PLATFORM := android-24

# YAGE: -O3 without -ffast-math.
#
# -ffast-math was previously applied to every TU, including GPU3D/GPU3D_Soft/SPU.
# It permits value-changing reassociation and enables FTZ/DAZ, so the Android
# build could diverge from every other platform on a core where determinism is
# load-bearing (savestates, netplay, TAS). -fno-math-errno keeps the part that
# actually helps codegen (libm calls treated as pure) without altering results.
APP_CFLAGS := -O3 -DNDEBUG -fno-math-errno
APP_CPPFLAGS := -O3 -DNDEBUG -fno-math-errno
