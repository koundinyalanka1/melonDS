/*
    GPU2D_OpenGL — GPU-accelerated DS 2D renderer (libretro / GLES3).
    See GPU2D_OpenGL.h.

    C2.1: real GL init — compiles the ported LayerPre shaders on the GPU and
    creates per-unit VRAM/palette textures + Layer/Scanline UBOs. Rendering
    still delegates to SoftRenderer (passthrough) until the compositor lands
    (C4); if GL init fails, Init() returns false and the caller falls back to
    the SoftRenderer.

    This file is part of melonDS (GPLv3); see the project license.
*/

#include "GPU2D_OpenGL.h"
#include "GPU2D_OpenGL_shaders.h"
#include "GPU.h"
#include "GPU3D.h"

#ifdef __ANDROID__
#include <android/log.h>
#define MELONDS_2D_LOG(...) __android_log_print(ANDROID_LOG_INFO, "melonDS-GLES", __VA_ARGS__)
#else
#include <cstdio>
#define MELONDS_2D_LOG(...) do { fprintf(stderr, "melonDS-GLES: " __VA_ARGS__); fprintf(stderr, "\n"); } while (0)
#endif

namespace GPU2D
{

// Equivalent of upstream OpenGLSupport's glDefaultTexParams (absent in this
// fork): nearest filtering, clamp-to-edge — what the DS 2D textures want.
static void texParamsDefault(GLenum target)
{
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

GLRenderer2D::GLRenderer2D()
{
}

GLRenderer2D::~GLRenderer2D()
{
    DeInitGL();
}

bool GLRenderer2D::Init()
{
    if (!InitShaders())
    {
        MELONDS_2D_LOG("GPU2D: GL 2D shader compile FAILED — falling back to software");
        DeInitGL();
        return false;
    }
    if (!InitResources())
    {
        MELONDS_2D_LOG("GPU2D: GL 2D resource alloc FAILED — falling back to software");
        DeInitGL();
        return false;
    }
    GLReady = true;
    MELONDS_2D_LOG("GPU2D: GL 2D renderer initialised (C2.1: resources ready, "
                   "rendering still via software passthrough until C4)");
    return true;
}

bool GLRenderer2D::InitShaders()
{
    // Compile + link the background layer pre-pass program. This is the first
    // on-device validation of the GLES3-ported LayerPre shaders (C2.0).
    if (!OpenGL::BuildShaderProgram(k2DLayerPreVS, k2DLayerPreFS, LayerPreShader, "2DLayerPreShader"))
        return false;

    glBindAttribLocation(LayerPreShader[2], 0, "vPosition");
    glBindFragDataLocation(LayerPreShader[2], 0, "oColor");

    if (!OpenGL::LinkShaderProgram(LayerPreShader))
        return false;

    glUseProgram(LayerPreShader[2]);

    GLint loc;
    loc = glGetUniformLocation(LayerPreShader[2], "VRAMTex");
    if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(LayerPreShader[2], "PalTex");
    if (loc >= 0) glUniform1i(loc, 1);

    GLuint blk = glGetUniformBlockIndex(LayerPreShader[2], "ubBGConfig");
    if (blk != GL_INVALID_INDEX)
        glUniformBlockBinding(LayerPreShader[2], blk, 20);

    LayerPreCurBGULoc = glGetUniformLocation(LayerPreShader[2], "uCurBG");

    return true;
}

bool GLRenderer2D::InitResources()
{
    for (int u = 0; u < 2; u++)
    {
        // Raw BG VRAM as an integer texture (1 byte/texel). Engine A (unit 0)
        // can map up to 512 KB of BG VRAM; engine B up to 128 KB.
        int bgheight = (u == 0) ? 512 : 128;

        glGenTextures(1, &VRAMTex_BG[u]);
        glBindTexture(GL_TEXTURE_2D, VRAMTex_BG[u]);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1024, bgheight, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);

        // BG palette: 256 entries wide; standard pal row + 4 BGs * 16 ext-pal
        // slots tall. RGB5_A1 matches the NDS 15-bit palette layout. GLES3 has
        // no GL_UNSIGNED_SHORT_1_5_5_5_REV (desktop-only), so we use
        // GL_UNSIGNED_SHORT_5_5_5_1 and swizzle NDS BGR555 entries at upload
        // time (C2.2) — same approach as the GPU3D_OpenGL palette path.
        glGenTextures(1, &PalTex_BG[u]);
        glBindTexture(GL_TEXTURE_2D, PalTex_BG[u]);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB5_A1, 256, 1 + (4 * 16), 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, nullptr);

        glGenBuffers(1, &LayerConfigUBO[u]);
        glBindBuffer(GL_UNIFORM_BUFFER, LayerConfigUBO[u]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(sLayerConfig), nullptr, GL_STREAM_DRAW);

        glGenBuffers(1, &ScanlineConfigUBO[u]);
        glBindBuffer(GL_UNIFORM_BUFFER, ScanlineConfigUBO[u]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(sScanlineConfig), nullptr, GL_STREAM_DRAW);
    }

    // UBO structs must be std140 16-byte aligned.
    static_assert((sizeof(sLayerConfig) & 15) == 0, "sLayerConfig not std140-aligned");
    static_assert((sizeof(sScanlineConfig) & 15) == 0, "sScanlineConfig not std140-aligned");

    return true;
}

void GLRenderer2D::DeInitGL()
{
    if (LayerPreShader[2])
    {
        glDeleteProgram(LayerPreShader[2]);
        LayerPreShader[0] = LayerPreShader[1] = LayerPreShader[2] = 0;
    }
    for (int u = 0; u < 2; u++)
    {
        if (VRAMTex_BG[u])        { glDeleteTextures(1, &VRAMTex_BG[u]);        VRAMTex_BG[u] = 0; }
        if (PalTex_BG[u])         { glDeleteTextures(1, &PalTex_BG[u]);         PalTex_BG[u] = 0; }
        if (LayerConfigUBO[u])    { glDeleteBuffers(1, &LayerConfigUBO[u]);     LayerConfigUBO[u] = 0; }
        if (ScanlineConfigUBO[u]) { glDeleteBuffers(1, &ScanlineConfigUBO[u]);  ScanlineConfigUBO[u] = 0; }
    }
    GLReady = false;
}

// ── Register capture (C2.2) ───────────────────────────────────────────────
// Ported from upstream GLRenderer2D::UpdateScanlineConfig, adapted to this
// fork's per-Unit API:
//   GPU2D.x              -> unit->x
//   GPU.GPU3D.GetRenderXPos() -> GPU3D::RenderXPos
//   GPU2D.BGMosaicLine   -> (line - unit->BGMosaicY)   [fork SoftRenderer does
//                            yoff = BGYPos + line; yoff -= BGMosaicY]
//   GPU.Palette          -> GPU::Palette
// Not called during software passthrough (mutates window-tracking state).
void GLRenderer2D::UpdateScanlineConfig(int line, Unit* unit)
{
    auto& cfg = ScanlineConfig[unit->Num].uScanline[line];

    u32 bgmode = unit->DispCnt & 0x7;
    bool xmosaic = (unit->BGMosaicSize[0] > 0);
    int mosaicLine = line - (int)unit->BGMosaicY;

    if (unit->DispCnt & (1<<3))
    {
        // 3D layer
        int xpos = GPU3D::RenderXPos & 0x1FF;
        cfg.BGOffset[0][0] = xpos - ((xpos & 0x100) << 1);
        cfg.BGOffset[0][1] = line;
        cfg.BGMosaicEnable[0] = false;
    }
    else
    {
        // text layer
        cfg.BGOffset[0][0] = unit->BGXPos[0];
        if (unit->BGCnt[0] & (1<<6))
        {
            cfg.BGOffset[0][1] = unit->BGYPos[0] + mosaicLine;
            cfg.BGMosaicEnable[0] = xmosaic;
        }
        else
        {
            cfg.BGOffset[0][1] = unit->BGYPos[0] + line;
            cfg.BGMosaicEnable[0] = false;
        }
    }

    // BG1 — always a text layer
    cfg.BGOffset[1][0] = unit->BGXPos[1];
    if (unit->BGCnt[1] & (1<<6))
    {
        cfg.BGOffset[1][1] = unit->BGYPos[1] + mosaicLine;
        cfg.BGMosaicEnable[1] = xmosaic;
    }
    else
    {
        cfg.BGOffset[1][1] = unit->BGYPos[1] + line;
        cfg.BGMosaicEnable[1] = false;
    }

    if ((bgmode == 2) || (bgmode >= 4 && bgmode <= 6))
    {
        // BG2 rotscale layer
        cfg.BGOffset[2][0] = unit->BGXRefInternal[0];
        cfg.BGOffset[2][1] = unit->BGYRefInternal[0];
        cfg.BGRotscale[0][0] = unit->BGRotA[0];
        cfg.BGRotscale[0][1] = unit->BGRotB[0];
        cfg.BGRotscale[0][2] = unit->BGRotC[0];
        cfg.BGRotscale[0][3] = unit->BGRotD[0];
    }
    else
    {
        // BG2 text layer
        cfg.BGOffset[2][0] = unit->BGXPos[2];
        if (unit->BGCnt[2] & (1<<6))
            cfg.BGOffset[2][1] = unit->BGYPos[2] + mosaicLine;
        else
            cfg.BGOffset[2][1] = unit->BGYPos[2] + line;
    }
    cfg.BGMosaicEnable[2] = (unit->BGCnt[2] & (1<<6)) ? xmosaic : false;

    if (bgmode >= 1 && bgmode <= 5)
    {
        // BG3 rotscale layer
        cfg.BGOffset[3][0] = unit->BGXRefInternal[1];
        cfg.BGOffset[3][1] = unit->BGYRefInternal[1];
        cfg.BGRotscale[1][0] = unit->BGRotA[1];
        cfg.BGRotscale[1][1] = unit->BGRotB[1];
        cfg.BGRotscale[1][2] = unit->BGRotC[1];
        cfg.BGRotscale[1][3] = unit->BGRotD[1];
    }
    else
    {
        // BG3 text layer
        cfg.BGOffset[3][0] = unit->BGXPos[3];
        if (unit->BGCnt[3] & (1<<6))
            cfg.BGOffset[3][1] = unit->BGYPos[3] + mosaicLine;
        else
            cfg.BGOffset[3][1] = unit->BGYPos[3] + line;
    }
    cfg.BGMosaicEnable[3] = (unit->BGCnt[3] & (1<<6)) ? xmosaic : false;

    u16* pal = (u16*)&GPU::Palette[unit->Num ? 0x400 : 0];
    cfg.BackColor = pal[0];

    // mosaic sizes
    cfg.MosaicSize[0] = unit->BGMosaicSize[0];
    cfg.MosaicSize[1] = unit->BGMosaicSize[1];
    cfg.MosaicSize[2] = unit->OBJMosaicSize[0];
    cfg.MosaicSize[3] = unit->OBJMosaicSize[1];

    // windows
    if (unit->DispCnt & 0xE000) cfg.WinRegs = unit->WinCnt[2];
    else                        cfg.WinRegs = 0xFF;
    if (unit->DispCnt & (1<<15)) cfg.WinRegs |= (unit->WinCnt[3] << 8);
    else                         cfg.WinRegs |= 0xFF00;
    if (unit->DispCnt & (1<<14)) cfg.WinRegs |= (unit->WinCnt[1] << 16);
    else                         cfg.WinRegs |= 0xFF0000;
    if (unit->DispCnt & (1<<13)) cfg.WinRegs |= (unit->WinCnt[0] << 24);
    else                         cfg.WinRegs |= 0xFF000000;

    cfg.WinMask = 0;

    if ((unit->DispCnt & (1<<13)) && (unit->Win0Active & 0x1))
    {
        int x0 = unit->Win0Coords[0];
        int x1 = unit->Win0Coords[1];
        if (x0 <= x1)
        {
            cfg.WinPos[0] = x0; cfg.WinPos[1] = x1;
            if (unit->Win0Active == 0x3) cfg.WinMask |= (1<<0);
            cfg.WinMask |= (1<<1);
            unit->Win0Active &= ~0x2;
        }
        else
        {
            cfg.WinPos[0] = x1; cfg.WinPos[1] = x0;
            if (unit->Win0Active == 0x3) cfg.WinMask |= (1<<0);
            cfg.WinMask |= (1<<2);
            unit->Win0Active |= 0x2;
        }
    }
    else
    {
        cfg.WinPos[0] = 256; cfg.WinPos[1] = 256;
    }

    if ((unit->DispCnt & (1<<14)) && (unit->Win1Active & 0x1))
    {
        int x0 = unit->Win1Coords[0];
        int x1 = unit->Win1Coords[1];
        if (x0 <= x1)
        {
            cfg.WinPos[2] = x0; cfg.WinPos[3] = x1;
            if (unit->Win1Active == 0x3) cfg.WinMask |= (1<<3);
            cfg.WinMask |= (1<<4);
            unit->Win1Active &= ~0x2;
        }
        else
        {
            cfg.WinPos[2] = x1; cfg.WinPos[3] = x0;
            if (unit->Win1Active == 0x3) cfg.WinMask |= (1<<3);
            cfg.WinMask |= (1<<5);
            unit->Win1Active |= 0x2;
        }
    }
    else
    {
        cfg.WinPos[2] = 256; cfg.WinPos[3] = 256;
    }
}

// ── Rendering: software passthrough until the GPU pipeline closes (C4) ─────

void GLRenderer2D::SetFramebuffer(u32* unitA, u32* unitB)
{
    Framebuffer[0] = unitA;
    Framebuffer[1] = unitB;
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
