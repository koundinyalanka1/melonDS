/*
    GPU2D_OpenGL — GPU-accelerated DS 2D renderer (libretro / GLES3).

    Reimplements the design of upstream melonDS's GLRenderer2D (GPU2D_OpenGL)
    against this fork's (pre-refactor) global GPU2D API, targeting GLES 3.x so
    BG/OBJ rasterisation and the 2D-over-3D composite run on the GPU instead of
    the CPU SoftRenderer.

    Phase C1: skeleton. Subclasses Renderer2D, is selectable via the
    `melonds_2d_renderer` core option + GLES gating, and currently DELEGATES to
    an internal SoftRenderer (correct output) while the GPU passes are built out
    in later phases (C2: BG layers, C3: sprites, C4: compositor + 3D + capture).

    This file is part of melonDS (GPLv3); see the project license.
*/

#pragma once

#include "GPU2D.h"
#include "GPU2D_Soft.h"

namespace GPU2D
{

class GLRenderer2D : public Renderer2D
{
public:
    GLRenderer2D();
    ~GLRenderer2D() override;

    // Sets up GL state for the 2D renderer. Returns false if it cannot be
    // initialised (insufficient GLES caps, allocation failure, ...), in which
    // case the caller must fall back to the SoftRenderer.
    bool Init();

    void SetFramebuffer(u32* unitA, u32* unitB) override;

    void DrawScanline(u32 line, Unit* unit) override;
    void DrawSprites(u32 line, Unit* unit) override;
    void VBlankEnd(Unit* unitA, Unit* unitB) override;

private:
    // C1 passthrough backend. Replaced incrementally by GPU passes; kept as a
    // correctness fallback for unimplemented paths during the C2–C4 build-out.
    SoftRenderer Soft;
};

}
