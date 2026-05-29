/*
    GPU2D_OpenGL — GPU-accelerated DS 2D renderer (libretro / GLES3).
    See GPU2D_OpenGL.h. Phase C1: skeleton that delegates to SoftRenderer.

    This file is part of melonDS (GPLv3); see the project license.
*/

#include "GPU2D_OpenGL.h"

#ifdef __ANDROID__
#include <android/log.h>
#define MELONDS_2D_LOG(...) __android_log_print(ANDROID_LOG_INFO, "melonDS-GLES", __VA_ARGS__)
#else
#include <cstdio>
#define MELONDS_2D_LOG(...) do { fprintf(stderr, "melonDS-GLES: " __VA_ARGS__); fprintf(stderr, "\n"); } while (0)
#endif

namespace GPU2D
{

GLRenderer2D::GLRenderer2D()
{
}

GLRenderer2D::~GLRenderer2D()
{
}

bool GLRenderer2D::Init()
{
    // C1: no GL resources are created yet. The renderer is functional because
    // it forwards every call to the software backend, so the libretro GL
    // display path (which uploads GPU::Framebuffer) keeps producing correct
    // frames. GPU passes are introduced in C2+ and will progressively replace
    // these passthrough calls.
    MELONDS_2D_LOG("GPU2D: GL 2D renderer selected (C1 passthrough to software)");
    return true;
}

void GLRenderer2D::SetFramebuffer(u32* unitA, u32* unitB)
{
    Framebuffer[0] = unitA;
    Framebuffer[1] = unitB;
    // Keep the passthrough backend pointed at the same output buffers.
    Soft.SetFramebuffer(unitA, unitB);
}

void GLRenderer2D::DrawScanline(u32 line, Unit* unit)
{
    Soft.DrawScanline(line, unit);
}

void GLRenderer2D::DrawSprites(u32 line, Unit* unit)
{
    Soft.DrawSprites(line, unit);
}

void GLRenderer2D::VBlankEnd(Unit* unitA, Unit* unitB)
{
    Soft.VBlankEnd(unitA, unitB);
}

}
