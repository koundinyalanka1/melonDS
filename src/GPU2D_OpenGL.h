/*
    GPU2D_OpenGL — GPU-accelerated DS 2D renderer (libretro / GLES3).

    Reimplements the design of upstream melonDS's GLRenderer2D (GPU2D_OpenGL)
    against this fork's (pre-refactor) global GPU2D API, targeting GLES 3.x so
    BG/OBJ rasterisation and the 2D-over-3D composite run on the GPU instead of
    the CPU SoftRenderer.

    Structural note: upstream uses one renderer instance PER screen (constructed
    with a specific GPU2D&). This fork uses ONE Renderer2D instance serving both
    units (methods take Unit*). So per-unit GL resources are held as [2] arrays
    and indexed by unit->Num.

    Build-out phases:
      C1  skeleton + selection + gating (delegates to SoftRenderer)        [done]
      C2  background layers: shaders (C2.0), GL resources + InitShaders     [C2.1 here]
          (C2.2 config capture, C2.3 prerender)
      C3  sprites      C4  compositor + 3D + capture      C5  display/upscale

    Until the compositor (C4) is wired, DrawScanline/DrawSprites/VBlankEnd still
    delegate to SoftRenderer so output stays correct.

    This file is part of melonDS (GPLv3); see the project license.
*/

#pragma once

#include "OpenGLSupport.h"
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
    // initialised (shader compile failure, etc.); the caller then falls back
    // to the SoftRenderer.
    bool Init();

    void SetFramebuffer(u32* unitA, u32* unitB) override;

    void DrawScanline(u32 line, Unit* unit) override;
    void DrawSprites(u32 line, Unit* unit) override;
    void VBlankEnd(Unit* unitA, Unit* unitB) override;

private:
    // ── GL setup (C2.1) ───────────────────────────────────────────────────
    bool InitShaders();
    bool InitResources();   // textures / FBOs / UBOs for both units
    void DeInitGL();

    bool GLReady = false;

    // LayerPre program (shared between units). The fork's
    // OpenGL::BuildShaderProgram fills a GLuint[3] = {vs, fs, program}.
    GLuint LayerPreShader[3] = {0, 0, 0};
    GLint  LayerPreCurBGULoc = -1;

    // ── std140 config structs (BG subset for C2) ─────────────────────────
    struct sBGConfig
    {
        u32 Size[2];
        u32 Type;
        u32 PalOffset;
        u32 TileOffset;
        u32 MapOffset;
        u32 Clamp;
        u32 __pad0[1];
    };
    struct sLayerConfig
    {
        u32 uVRAMMask;
        u32 __pad0[3];
        sBGConfig uBGConfig[4];
    } LayerConfig[2];

    struct sScanline
    {
        s32 BGOffset[4][4];     // really [4][2], padded to vec4
        s32 BGRotscale[2][4];
        u32 BackColor;
        u32 WinRegs;
        u32 WinMask;
        u32 __pad0[1];
        s32 WinPos[4];
        u32 BGMosaicEnable[4];
        s32 MosaicSize[4];
    };
    struct sScanlineConfig
    {
        sScanline uScanline[192];
    } ScanlineConfig[2];

    // ── Per-unit GL resources (indexed by unit->Num) ──────────────────────
    GLuint VRAMTex_BG[2]        = {0, 0};
    GLuint PalTex_BG[2]         = {0, 0};
    GLuint LayerConfigUBO[2]    = {0, 0};
    GLuint ScanlineConfigUBO[2] = {0, 0};

    // pre-rendered BG layer textures: all possible sizes (22), per unit.
    GLuint AllBGLayerTex[2][22] = {};
    GLuint AllBGLayerFB[2][22]  = {};

    // C1/C2 passthrough backend (correctness fallback until C4 closes the loop).
    SoftRenderer Soft;
};

}
