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

#include <cstring>   // memcpy / memset

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

        // The 22 intermediate BG layer textures (+ FBOs) per unit, sized for
        // every possible BG layer size. Order matches upstream so BGBaseIndex
        // selects the right slot. RGBA8 colour-renderable on GLES3.
        static const u16 bgsizes[8][3] = {
            { 128,  128, 2},
            { 256,  256, 4},
            { 256,  512, 4},
            { 512,  256, 4},
            { 512,  512, 4},
            { 512, 1024, 1},
            {1024,  512, 1},
            {1024, 1024, 2},
        };

        glGenTextures(22, AllBGLayerTex[u]);
        glGenFramebuffers(22, AllBGLayerFB[u]);

        int l = 0;
        for (int j = 0; j < 8; j++)
        {
            const u16* sz = bgsizes[j];
            for (int k = 0; k < sz[2]; k++)
            {
                glBindTexture(GL_TEXTURE_2D, AllBGLayerTex[u][l]);
                texParamsDefault(GL_TEXTURE_2D);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sz[0], sz[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

                glBindFramebuffer(GL_FRAMEBUFFER, AllBGLayerFB[u][l]);
                glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, AllBGLayerTex[u][l], 0);
                glDrawBuffer(GL_COLOR_ATTACHMENT0);

                l++;
            }
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Fullscreen [0,1] rect (two triangles) that the LayerPre VS expands to the
    // current layer's clip-space quad and layer-sized texcoords.
    static const float rectVtx[12] = {
        0.0f, 0.0f,  1.0f, 0.0f,  1.0f, 1.0f,
        0.0f, 0.0f,  1.0f, 1.0f,  0.0f, 1.0f,
    };
    glGenVertexArrays(1, &RectVtxArray);
    glGenBuffers(1, &RectVtxBuffer);
    glBindVertexArray(RectVtxArray);
    glBindBuffer(GL_ARRAY_BUFFER, RectVtxBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(rectVtx), rectVtx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);   // vPosition (bound to location 0 in InitShaders)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

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
    if (RectVtxArray)  { glDeleteVertexArrays(1, &RectVtxArray);  RectVtxArray = 0; }
    if (RectVtxBuffer) { glDeleteBuffers(1, &RectVtxBuffer);      RectVtxBuffer = 0; }
    for (int u = 0; u < 2; u++)
    {
        if (VRAMTex_BG[u])        { glDeleteTextures(1, &VRAMTex_BG[u]);        VRAMTex_BG[u] = 0; }
        if (PalTex_BG[u])         { glDeleteTextures(1, &PalTex_BG[u]);         PalTex_BG[u] = 0; }
        if (LayerConfigUBO[u])    { glDeleteBuffers(1, &LayerConfigUBO[u]);     LayerConfigUBO[u] = 0; }
        if (ScanlineConfigUBO[u]) { glDeleteBuffers(1, &ScanlineConfigUBO[u]);  ScanlineConfigUBO[u] = 0; }
        if (AllBGLayerTex[u][0])  { glDeleteTextures(22, AllBGLayerTex[u]);     memset(AllBGLayerTex[u], 0, sizeof(AllBGLayerTex[u])); }
        if (AllBGLayerFB[u][0])   { glDeleteFramebuffers(22, AllBGLayerFB[u]);  memset(AllBGLayerFB[u], 0, sizeof(AllBGLayerFB[u])); }
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

// base index for a BG layer within the AllBGLayer texture arrays, indexed by
// [BG type][BG size] (from upstream GLRenderer2D::BGBaseIndex).
static const u8 BGBaseIndex[4][4] = {
    {2, 10, 6, 14},     // text mode
    {0, 4, 16, 20},     // rotscale
    {0, 4, 12, 16},     // bitmap
    {18, 19, 12, 16},   // large bitmap
};

// Ported from upstream GLRenderer2D::UpdateLayerConfig, adapted to the fork's
// per-Unit API. Display-capture BG types (7/8) are deferred — treated as a
// plain direct-colour bitmap (Type 5) — so we don't depend on the modern
// GetCaptureInfo_BG(); revisit when wiring capture in C4.
void GLRenderer2D::UpdateLayerConfig(Unit* unit)
{
    const int num = unit->Num;

    // recomputed below: which layers get a prerendered BG texture this frame
    BGLayerActive[num] = 0;

    u32 tilebase, mapbase;
    if (!num)
    {
        tilebase = ((unit->DispCnt >> 24) & 0x7) << 16;
        mapbase  = ((unit->DispCnt >> 27) & 0x7) << 16;
    }
    else
    {
        tilebase = 0;
        mapbase  = 0;
    }

    int layertype[4] = {1, 1, 0, 0};
    switch (unit->DispCnt & 0x7)
    {
        case 0: layertype[2] = 1; layertype[3] = 1; break;
        case 1: layertype[2] = 1; layertype[3] = 2; break;
        case 2: layertype[2] = 2; layertype[3] = 2; break;
        case 3: layertype[2] = 1; layertype[3] = 3; break;
        case 4: layertype[2] = 2; layertype[3] = 3; break;
        case 5: layertype[2] = 3; layertype[3] = 3; break;
        case 6: layertype[0] = 0; layertype[1] = 0;
                layertype[2] = 4; layertype[3] = 0; break;
        case 7: layertype[2] = 0; layertype[3] = 0; break;
    }

    for (int layer = 0; layer < 4; layer++)
    {
        int type = layertype[layer];
        if (!type) continue;

        // active layer — gets a prerendered texture unless it turns out to be
        // the 3D layer (cleared in that branch below).
        BGLayerActive[num] |= (1u << layer);

        u16 bgcnt = unit->BGCnt[layer];
        auto& cfg = LayerConfig[num].uBGConfig[layer];

        cfg.TileOffset = tilebase + (((bgcnt >> 2) & 0xF) << 14);
        cfg.MapOffset  = mapbase  + (((bgcnt >> 8) & 0x1F) << 11);
        cfg.PalOffset  = 0;

        BGVRAMRange[num][layer][0] = cfg.TileOffset;
        BGVRAMRange[num][layer][2] = cfg.MapOffset;

        if ((layer == 0) && (unit->DispCnt & (1<<3)))
        {
            // 3D layer — composited from GPU3D's output (C4), not prerendered.
            BGLayerActive[num] &= ~(1u << layer);
            cfg.Size[0] = 256; cfg.Size[1] = 192;
            cfg.Type = 6;
            cfg.Clamp = 1;
            BGVRAMRange[num][layer][0] = 0xFFFFFFFF;
            BGVRAMRange[num][layer][1] = 0xFFFFFFFF;
            BGVRAMRange[num][layer][2] = 0xFFFFFFFF;
            BGVRAMRange[num][layer][3] = 0xFFFFFFFF;
        }
        else if (type == 1)
        {
            // text layer
            u32 tilesz = 0, mapsz = 0;
            switch (bgcnt >> 14)
            {
                case 0: cfg.Size[0] = 256; cfg.Size[1] = 256; mapsz = 0x800;  break;
                case 1: cfg.Size[0] = 512; cfg.Size[1] = 256; mapsz = 0x1000; break;
                case 2: cfg.Size[0] = 256; cfg.Size[1] = 512; mapsz = 0x1000; break;
                case 3: cfg.Size[0] = 512; cfg.Size[1] = 512; mapsz = 0x2000; break;
            }

            if (bgcnt & (1<<7))
            {
                // 256-color
                cfg.Type = 1;
                if (unit->DispCnt & (1<<30))
                {
                    int paloff = layer;
                    if ((layer < 2) && (bgcnt & (1<<13))) paloff += 2;
                    cfg.PalOffset = 1 + (16 * paloff);
                }
                tilesz = 0x10000;
            }
            else
            {
                // 16-color
                cfg.Type = 0;
                tilesz = 0x8000;
            }
            cfg.Clamp = 0;

            int n = BGBaseIndex[0][bgcnt >> 14] + layer;
            BGLayerTex[num][layer] = AllBGLayerTex[num][n];
            BGLayerFB[num][layer]  = AllBGLayerFB[num][n];

            BGVRAMRange[num][layer][1] = tilesz;
            BGVRAMRange[num][layer][3] = mapsz;
        }
        else if (type == 2)
        {
            // affine layer
            u32 mapsz = 0;
            switch (bgcnt >> 14)
            {
                case 0: cfg.Size[0] = 128;  cfg.Size[1] = 128;  mapsz = 0x100;  break;
                case 1: cfg.Size[0] = 256;  cfg.Size[1] = 256;  mapsz = 0x400;  break;
                case 2: cfg.Size[0] = 512;  cfg.Size[1] = 512;  mapsz = 0x1000; break;
                case 3: cfg.Size[0] = 1024; cfg.Size[1] = 1024; mapsz = 0x4000; break;
            }
            cfg.Type = 2;
            cfg.Clamp = !(bgcnt & (1<<13));

            int n = BGBaseIndex[1][bgcnt >> 14] + layer - 2;
            BGLayerTex[num][layer] = AllBGLayerTex[num][n];
            BGLayerFB[num][layer]  = AllBGLayerFB[num][n];

            BGVRAMRange[num][layer][1] = 0x4000;
            BGVRAMRange[num][layer][3] = mapsz;
        }
        else if (type == 3)
        {
            // extended layer
            if (bgcnt & (1<<7))
            {
                // bitmap modes
                u32 mapsz = 0;
                switch (bgcnt >> 14)
                {
                    case 0: cfg.Size[0] = 128; cfg.Size[1] = 128; mapsz = 0x4000;  break;
                    case 1: cfg.Size[0] = 256; cfg.Size[1] = 256; mapsz = 0x10000; break;
                    case 2: cfg.Size[0] = 512; cfg.Size[1] = 256; mapsz = 0x20000; break;
                    case 3: cfg.Size[0] = 512; cfg.Size[1] = 512; mapsz = 0x40000; break;
                }

                u32 tileoffset = 0;
                u32 mapoffset = ((bgcnt >> 8) & 0x1F) << 14;

                BGVRAMRange[num][layer][0] = 0xFFFFFFFF;
                BGVRAMRange[num][layer][1] = 0xFFFFFFFF;
                BGVRAMRange[num][layer][2] = mapoffset;
                BGVRAMRange[num][layer][3] = mapsz;

                if (bgcnt & (1<<2))
                {
                    // direct colour: 2 bytes/pixel. Display-capture detection
                    // (Type 7/8) deferred to C4 — treat as plain direct bitmap.
                    mapsz <<= 1;
                    BGVRAMRange[num][layer][3] = mapsz;
                    cfg.Type = 5;
                }
                else
                    cfg.Type = 4;

                cfg.TileOffset = tileoffset;
                cfg.MapOffset  = mapoffset;

                int n = BGBaseIndex[2][bgcnt >> 14] + layer - 2;
                BGLayerTex[num][layer] = AllBGLayerTex[num][n];
                BGLayerFB[num][layer]  = AllBGLayerFB[num][n];
            }
            else
            {
                // rotscale w/ tiles (always 256-color)
                u32 mapsz = 0;
                switch (bgcnt >> 14)
                {
                    case 0: cfg.Size[0] = 128;  cfg.Size[1] = 128;  mapsz = 0x200;  break;
                    case 1: cfg.Size[0] = 256;  cfg.Size[1] = 256;  mapsz = 0x800;  break;
                    case 2: cfg.Size[0] = 512;  cfg.Size[1] = 512;  mapsz = 0x2000; break;
                    case 3: cfg.Size[0] = 1024; cfg.Size[1] = 1024; mapsz = 0x8000; break;
                }

                cfg.Type = 3;
                if (unit->DispCnt & (1<<30))
                {
                    int paloff = layer;
                    if ((layer < 2) && (bgcnt & (1<<13))) paloff += 2;
                    cfg.PalOffset = 1 + (16 * paloff);
                }

                int n = BGBaseIndex[1][bgcnt >> 14] + layer - 2;
                BGLayerTex[num][layer] = AllBGLayerTex[num][n];
                BGLayerFB[num][layer]  = AllBGLayerFB[num][n];

                BGVRAMRange[num][layer][1] = 0x10000;
                BGVRAMRange[num][layer][3] = mapsz;
            }

            cfg.Clamp = !(bgcnt & (1<<13));
        }
        else // type == 4 — large bitmap layer
        {
            u32 mapsz = 0;
            switch (bgcnt >> 14)
            {
                case 0: cfg.Size[0] = 512;  cfg.Size[1] = 1024; mapsz = 0x80000; break;
                case 1: cfg.Size[0] = 1024; cfg.Size[1] = 512;  mapsz = 0x80000; break;
                case 2: cfg.Size[0] = 512;  cfg.Size[1] = 256;  mapsz = 0x20000; break;
                case 3: cfg.Size[0] = 512;  cfg.Size[1] = 512;  mapsz = 0x40000; break;
            }

            cfg.Type = 4;
            cfg.TileOffset = 0;
            cfg.MapOffset  = 0;
            cfg.Clamp = !(bgcnt & (1<<13));

            int n = BGBaseIndex[3][bgcnt >> 14];
            BGLayerTex[num][layer] = AllBGLayerTex[num][n];
            BGLayerFB[num][layer]  = AllBGLayerFB[num][n];

            BGVRAMRange[num][layer][0] = 0xFFFFFFFF;
            BGVRAMRange[num][layer][1] = 0xFFFFFFFF;
            BGVRAMRange[num][layer][3] = mapsz;
        }
    }

    glBindBuffer(GL_UNIFORM_BUFFER, LayerConfigUBO[num]);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(sLayerConfig), &LayerConfig[num]);
}

// ── Background prerender (C2.3) ────────────────────────────────────────────
// Ported from upstream GLRenderer2D's per-frame BG pre-pass (the bgDirty/
// LayerPre block in RenderScreen + PrerenderLayer), adapted to the fork's
// per-Unit API. Upstream tracks dirty VRAM ranges via a bitfield; we instead
// upload only the tile/map ranges the active layers reference (BGVRAMRange,
// filled by UpdateLayerConfig). Driven by C4; not yet wired here.

// Upload one [start, start+size) VRAM byte range into the bound R8UI texture.
// VRAM is laid out 1024 bytes/row, so a byte range maps to whole texture rows.
// start == 0xFFFFFFFF marks "no range" (e.g. bitmap layers have no tile range).
static void uploadVRAMRange(const u8* vram, u32 start, u32 size, int bgheight)
{
    if (start == 0xFFFFFFFF || size == 0) return;

    u32 limit = (u32)bgheight * 1024;
    if (start >= limit) return;
    u32 end = start + size;
    if (end > limit) end = limit;

    int startRow = (int)(start >> 10);
    int endRow   = (int)((end + 1023) >> 10);
    if (endRow > bgheight) endRow = bgheight;
    if (endRow <= startRow) return;

    glTexSubImage2D(GL_TEXTURE_2D, 0,
                    0, startRow,
                    1024, endRow - startRow,
                    GL_RED_INTEGER, GL_UNSIGNED_BYTE,
                    &vram[(size_t)startRow * 1024]);
}

void GLRenderer2D::UploadBGVRAM(Unit* unit)
{
    const int num = unit->Num;

    u8* vram; u32 vrammask;
    unit->GetBGVRAM(vram, vrammask);
    int bgheight = (int)((vrammask + 1) >> 10);   // VRAM rows = texture height

    glBindTexture(GL_TEXTURE_2D, VRAMTex_BG[num]);

    for (int layer = 0; layer < 4; layer++)
    {
        if (!(BGLayerActive[num] & (1u << layer))) continue;
        // [0]=tile start, [1]=tile size; [2]=map/bitmap start, [3]=map/bitmap size
        uploadVRAMRange(vram, BGVRAMRange[num][layer][0], BGVRAMRange[num][layer][1], bgheight);
        uploadVRAMRange(vram, BGVRAMRange[num][layer][2], BGVRAMRange[num][layer][3], bgheight);
    }
}

void GLRenderer2D::UploadBGPalette(Unit* unit)
{
    const int num = unit->Num;

    // Standard BG palette (256 entries) then the 4 ext-pal slots × 16 palettes,
    // matching the PalTex_BG layout (rows: 0 = standard, 1.. = ext-pal).
    static u16 temp[256 * (1 + 4 * 16)];
    memcpy(&temp[0], &GPU::Palette[num ? 0x400 : 0], 256 * 2);
    for (int s = 0; s < 4; s++)
        for (int p = 0; p < 16; p++)
        {
            u16* pal = unit->GetBGExtPal(s, p);
            memcpy(&temp[(1 + (s * 16) + p) * 256], pal, 256 * 2);
        }

    // GLES3 has no GL_UNSIGNED_SHORT_1_5_5_5_REV, so swizzle NDS BGR555 entries
    // into 5_5_5_1 layout at upload time (same as the GPU3D_OpenGL palette path):
    //   R(0x001F)<<11 | G(0x03E0)<<1 | B(0x7C00)>>9 | (top bit)->A
    static u16 swiz[256 * (1 + 4 * 16)];
    const int n = 256 * (1 + 4 * 16);
    for (int i = 0; i < n; i++)
    {
        u16 s = temp[i];
        swiz[i] = ((s & 0x001F) << 11) | ((s & 0x03E0) << 1) | ((s & 0x7C00) >> 9) | ((s >> 15) & 0x1);
    }

    glBindTexture(GL_TEXTURE_2D, PalTex_BG[num]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1 + (4 * 16),
                    GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, swiz);
}

void GLRenderer2D::PrerenderLayer(int layer, Unit* unit)
{
    const int num = unit->Num;
    auto& cfg = LayerConfig[num].uBGConfig[layer];

    if (cfg.Type >= 6) return;   // 3D layer — composited from GPU3D output at C4

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, BGLayerFB[num][layer]);

    glUniform1i(LayerPreCurBGULoc, layer);
    glViewport(0, 0, cfg.Size[0], cfg.Size[1]);

    glBindBuffer(GL_ARRAY_BUFFER, RectVtxBuffer);
    glBindVertexArray(RectVtxArray);
    glDrawArrays(GL_TRIANGLES, 0, 2 * 3);
}

void GLRenderer2D::PrerenderBGLayers(Unit* unit)
{
    const int num = unit->Num;

    UpdateLayerConfig(unit);   // fills LayerConfig + BGVRAMRange + BGLayerActive

    UploadBGVRAM(unit);
    UploadBGPalette(unit);

    if (!BGLayerActive[num]) return;

    glUseProgram(LayerPreShader[2]);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);

    glBindBufferBase(GL_UNIFORM_BUFFER, 20, LayerConfigUBO[num]);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, VRAMTex_BG[num]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, PalTex_BG[num]);

    for (int layer = 0; layer < 4; layer++)
    {
        if (!(BGLayerActive[num] & (1u << layer))) continue;
        PrerenderLayer(layer, unit);
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
