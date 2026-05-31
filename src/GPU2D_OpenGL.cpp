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
#include "GPU3D_OpenGL.h"

#include <cstring>   // memcpy / memset
#include <cstddef>   // offsetof

#ifndef YAGE_MELONDS_GL_DIAG
#define YAGE_MELONDS_GL_DIAG 0
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define MELONDS_2D_LOG(...) __android_log_print(ANDROID_LOG_INFO, "melonDS-GLES", __VA_ARGS__)
#else
#include <cstdio>
#define MELONDS_2D_LOG(...) do { fprintf(stderr, "melonDS-GLES: " __VA_ARGS__); fprintf(stderr, "\n"); } while (0)
#endif

namespace GPU2D
{

// ── TEMP diagnostic instrumentation (C4.2 black-screen / 0x0502 hunt) ─────────
// This renderer had NO internal glGetError checks, so a GL_INVALID_OPERATION
// raised in the 2D frame path only surfaced downstream as the misleading
// "0x0502 at glBindFramebuffer" from libretro/opengl.cpp's present-path CHECK_GL.
// GL2D_CHK drains + tags the error queue per stage so the next on-device log
// names the real failing call; GL2D_FBO logs incompleteness of the 2D FBOs
// (which, unlike the 3D ones, were never status-checked on the real driver).
// Logged at INFO via MELONDS_2D_LOG so it is platform-agnostic. Remove once the
// pipeline is validated.
#if YAGE_MELONDS_GL_DIAG
#define GL2D_CHK(tag) do { GLenum e_; while ((e_ = glGetError()) != GL_NO_ERROR) \
    MELONDS_2D_LOG("2D-GL 0x%04x @ %s", e_, (tag)); } while (0)
static inline void GL2D_FBO(const char* tag)
{
    GLenum s = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (s != GL_FRAMEBUFFER_COMPLETE)
        MELONDS_2D_LOG("2D-FBO %s INCOMPLETE 0x%04x", tag, s);
}
#else
#define GL2D_CHK(tag) do { (void)sizeof(tag); } while (0)
static inline void GL2D_FBO(const char*) {}
#endif

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
    MELONDS_2D_LOG("GPU2D: GL 2D renderer ACTIVE (C5: BG + OBJ + compositor "
                   "drive frames; OutputTex is presented directly; "
                   "3D-layer hook active; upscaling pending)");
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

    // ── Sprite pre-pass program (OBJ → 1024×512 atlas) ─────────────────────
    // Integer vertex attribs (glVertexAttribIPointer): vPosition (ivec2 quad
    // corner) at 0, vSpriteIndex (int) at 1; single colour output at 0.
    if (!OpenGL::BuildShaderProgram(k2DSpritePreVS, k2DSpritePreFS, SpritePreShader, "2DSpritePreShader"))
        return false;

    glBindAttribLocation(SpritePreShader[2], 0, "vPosition");
    glBindAttribLocation(SpritePreShader[2], 1, "vSpriteIndex");
    glBindFragDataLocation(SpritePreShader[2], 0, "oColor");

    if (!OpenGL::LinkShaderProgram(SpritePreShader))
        return false;

    glUseProgram(SpritePreShader[2]);
    loc = glGetUniformLocation(SpritePreShader[2], "VRAMTex"); if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(SpritePreShader[2], "PalTex");  if (loc >= 0) glUniform1i(loc, 1);
    blk = glGetUniformBlockIndex(SpritePreShader[2], "ubSpriteConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(SpritePreShader[2], blk, 21);

    // ── Sprite composite program (atlas → OBJ layer, MRT colour+flags) ─────
    // vPosition at 0, vTexcoord at 1, vSpriteIndex at 2; outputs oColor@0,
    // oFlags@1 (pinned via layout() in the FS; the bind calls are no-ops).
    if (!OpenGL::BuildShaderProgram(k2DSpriteVS, k2DSpriteFS, SpriteShader, "2DSpriteShader"))
        return false;

    glBindAttribLocation(SpriteShader[2], 0, "vPosition");
    glBindAttribLocation(SpriteShader[2], 1, "vTexcoord");
    glBindAttribLocation(SpriteShader[2], 2, "vSpriteIndex");
    glBindFragDataLocation(SpriteShader[2], 0, "oColor");
    glBindFragDataLocation(SpriteShader[2], 1, "oFlags");

    if (!OpenGL::LinkShaderProgram(SpriteShader))
        return false;

    glUseProgram(SpriteShader[2]);
    loc = glGetUniformLocation(SpriteShader[2], "SpriteTex");     if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(SpriteShader[2], "Capture128Tex"); if (loc >= 0) glUniform1i(loc, 1);
    loc = glGetUniformLocation(SpriteShader[2], "Capture256Tex"); if (loc >= 0) glUniform1i(loc, 2);
    blk = glGetUniformBlockIndex(SpriteShader[2], "ubSpriteConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(SpriteShader[2], blk, 21);
    blk = glGetUniformBlockIndex(SpriteShader[2], "ubSpriteScanlineConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(SpriteShader[2], blk, 24);
    SpriteRenderTransULoc = glGetUniformLocation(SpriteShader[2], "uRenderTransparent");

    // ── Mosaic lookup texture (16 mosaic sizes × 256 x-positions, R8I) ─────
    // mosaic_tex[m][x] = the x-coord the pixel at x snaps to under mosaic m
    // (the compositor / sprite path use it for horizontal mosaic).
    {
        u8* mosaic_tex = new u8[256 * 16];
        for (int m = 0; m < 16; m++)
        {
            int mosx = 0;
            for (int x = 0; x < 256; x++)
            {
                mosaic_tex[(m * 256) + x] = (u8)mosx;
                if (mosx == m) mosx = 0;
                else           mosx++;
            }
        }
        glGenTextures(1, &MosaicTex);
        glBindTexture(GL_TEXTURE_2D, MosaicTex);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8I, 256, 16, 0, GL_RED_INTEGER, GL_BYTE, mosaic_tex);
        delete[] mosaic_tex;
    }

    // ── Compositor program (BG+OBJ → OutputTex) ────────────────────────────
    if (!OpenGL::BuildShaderProgram(k2DCompositorVS, k2DCompositorFS, CompositorShader, "2DCompositorShader"))
        return false;

    glBindAttribLocation(CompositorShader[2], 0, "vPosition");
    glBindFragDataLocation(CompositorShader[2], 0, "oColor");

    if (!OpenGL::LinkShaderProgram(CompositorShader))
        return false;

    glUseProgram(CompositorShader[2]);
    loc = glGetUniformLocation(CompositorShader[2], "BGLayerTex[0]"); if (loc >= 0) glUniform1i(loc, 0);
    loc = glGetUniformLocation(CompositorShader[2], "BGLayerTex[1]"); if (loc >= 0) glUniform1i(loc, 1);
    loc = glGetUniformLocation(CompositorShader[2], "BGLayerTex[2]"); if (loc >= 0) glUniform1i(loc, 2);
    loc = glGetUniformLocation(CompositorShader[2], "BGLayerTex[3]"); if (loc >= 0) glUniform1i(loc, 3);
    loc = glGetUniformLocation(CompositorShader[2], "OBJLayerTex");   if (loc >= 0) glUniform1i(loc, 4);
    loc = glGetUniformLocation(CompositorShader[2], "Capture128Tex"); if (loc >= 0) glUniform1i(loc, 5);
    loc = glGetUniformLocation(CompositorShader[2], "Capture256Tex"); if (loc >= 0) glUniform1i(loc, 6);
    loc = glGetUniformLocation(CompositorShader[2], "MosaicTex");     if (loc >= 0) glUniform1i(loc, 7);

    blk = glGetUniformBlockIndex(CompositorShader[2], "ubBGConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(CompositorShader[2], blk, 20);
    blk = glGetUniformBlockIndex(CompositorShader[2], "ubScanlineConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(CompositorShader[2], blk, 22);
    blk = glGetUniformBlockIndex(CompositorShader[2], "ubCompositorConfig");
    if (blk != GL_INVALID_INDEX) glUniformBlockBinding(CompositorShader[2], blk, 23);

    CompositorScaleULoc = glGetUniformLocation(CompositorShader[2], "uScaleFactor");

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

        // BG VRAM mask in texture rows (bgheight-1), for the shader's
        // (addr >> 10) & uVRAMMask wraparound — mirrors SpriteConfig.uVRAMMask.
        LayerConfig[u].uVRAMMask = (u32)(bgheight - 1);

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

    // ── OBJ (sprite) resources, per unit ───────────────────────────────────
    for (int u = 0; u < 2; u++)
    {
        // raw OBJ VRAM as an integer texture: engine A maps up to 256 KB of OBJ
        // VRAM, engine B up to 128 KB.
        int objheight = (u == 0) ? 256 : 128;

        glGenTextures(1, &VRAMTex_OBJ[u]);
        glBindTexture(GL_TEXTURE_2D, VRAMTex_OBJ[u]);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8UI, 1024, objheight, 0, GL_RED_INTEGER, GL_UNSIGNED_BYTE, nullptr);

        // OBJ palette: standard row + 16 ext-pal slots. Same BGR555→5_5_5_1
        // swizzle-at-upload as the BG palette (no GL_UNSIGNED_SHORT_1_5_5_5_REV
        // on GLES3).
        glGenTextures(1, &PalTex_OBJ[u]);
        glBindTexture(GL_TEXTURE_2D, PalTex_OBJ[u]);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB5_A1, 256, 1 + 16, 0, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, nullptr);

        // 1024×512 sprite atlas (16×8 grid of 64×64 cells) + its FBO.
        glGenTextures(1, &SpriteTex[u]);
        glBindTexture(GL_TEXTURE_2D, SpriteTex[u]);
        texParamsDefault(GL_TEXTURE_2D);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 512, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glGenFramebuffers(1, &SpriteFB[u]);
        glBindFramebuffer(GL_FRAMEBUFFER, SpriteFB[u]);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, SpriteTex[u], 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);

        // upscaled OBJ layer (2D-array: layer 0 = colour, layer 1 = flags) +
        // depth. Just allocate the names here; SetScaleFactor sizes them.
        glGenTextures(1, &OBJLayerTex[u]);
        glBindTexture(GL_TEXTURE_2D_ARRAY, OBJLayerTex[u]);
        texParamsDefault(GL_TEXTURE_2D_ARRAY);

        glGenTextures(1, &OBJDepthTex[u]);
        glBindTexture(GL_TEXTURE_2D, OBJDepthTex[u]);
        texParamsDefault(GL_TEXTURE_2D);

        glGenFramebuffers(1, &OBJLayerFB[u]);

        glGenBuffers(1, &SpriteConfigUBO[u]);
        glBindBuffer(GL_UNIFORM_BUFFER, SpriteConfigUBO[u]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(sSpriteConfig), nullptr, GL_STREAM_DRAW);

        glGenBuffers(1, &SpriteScanlineConfigUBO[u]);
        glBindBuffer(GL_UNIFORM_BUFFER, SpriteScanlineConfigUBO[u]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(sSpriteScanlineConfig), nullptr, GL_STREAM_DRAW);

        // OBJ VRAM mask in texture rows (objheight-1), for the shader's
        // (addr >> 10) & uVRAMMask wraparound.
        SpriteConfig[u].uVRAMMask = (u32)(objheight - 1);

        // compositor: output texture (sized by SetScaleFactor) + config UBO.
        glGenTextures(1, &OutputTex[u]);
        glBindTexture(GL_TEXTURE_2D, OutputTex[u]);
        texParamsDefault(GL_TEXTURE_2D);

        glGenFramebuffers(1, &OutputFB[u]);

        glGenBuffers(1, &CompositorConfigUBO[u]);
        glBindBuffer(GL_UNIFORM_BUFFER, CompositorConfigUBO[u]);
        glBufferData(GL_UNIFORM_BUFFER, sizeof(sCompositorConfig), nullptr, GL_STREAM_DRAW);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // sprite vertex streams (shared scratch, reused per unit). Pre-pass: 2× pos
    // + index per vertex, 6 verts × 128 OBJ. Composite: 2× pos + 2× texcoord +
    // index, 6 verts × up to 256 quads (each OBJ may wrap → up to 2 quads). All
    // attribs are SHORT (positions can be negative for off-screen sprites).
    {
        int sprPreSize = (3 * 6) * 128;
        SpritePreVtxData = new u16[sprPreSize];
        glGenBuffers(1, &SpritePreVtxBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, SpritePreVtxBuffer);
        glBufferData(GL_ARRAY_BUFFER, sprPreSize * sizeof(u16), nullptr, GL_STREAM_DRAW);
        glGenVertexArrays(1, &SpritePreVtxArray);
        glBindVertexArray(SpritePreVtxArray);
        glEnableVertexAttribArray(0);   // vPosition (ivec2)
        glVertexAttribIPointer(0, 2, GL_SHORT, 3 * sizeof(u16), (void*)0);
        glEnableVertexAttribArray(1);   // vSpriteIndex (int)
        glVertexAttribIPointer(1, 1, GL_SHORT, 3 * sizeof(u16), (void*)(2 * sizeof(u16)));

        int sprSize = (5 * 6) * 256;
        SpriteVtxData = new u16[sprSize];
        glGenBuffers(1, &SpriteVtxBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, SpriteVtxBuffer);
        glBufferData(GL_ARRAY_BUFFER, sprSize * sizeof(u16), nullptr, GL_STREAM_DRAW);
        glGenVertexArrays(1, &SpriteVtxArray);
        glBindVertexArray(SpriteVtxArray);
        glEnableVertexAttribArray(0);   // vPosition (ivec2)
        glVertexAttribIPointer(0, 2, GL_SHORT, 5 * sizeof(u16), (void*)0);
        glEnableVertexAttribArray(1);   // vTexcoord (ivec2)
        glVertexAttribIPointer(1, 2, GL_SHORT, 5 * sizeof(u16), (void*)(2 * sizeof(u16)));
        glEnableVertexAttribArray(2);   // vSpriteIndex (int)
        glVertexAttribIPointer(2, 1, GL_SHORT, 5 * sizeof(u16), (void*)(4 * sizeof(u16)));
        glBindVertexArray(0);
    }

    // 1×1 dummy 2D-array bound to the Sprite FS's deferred capture samplers.
    {
        const u8 zero[4] = {0, 0, 0, 0};
        glGenTextures(1, &DummyTexArray);
        glBindTexture(GL_TEXTURE_2D_ARRAY, DummyTexArray);
        texParamsDefault(GL_TEXTURE_2D_ARRAY);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, 1, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, zero);
    }

    // size the OBJ render target to the current scale (1× until C5 wires the
    // melonds_opengl_resolution option).
    SetScaleFactor(ScaleFactor);

    // UBO structs must be std140 16-byte aligned.
    static_assert((sizeof(sLayerConfig) & 15) == 0, "sLayerConfig not std140-aligned");
    static_assert((sizeof(sScanlineConfig) & 15) == 0, "sScanlineConfig not std140-aligned");
    static_assert((sizeof(sSpriteConfig) & 15) == 0, "sSpriteConfig not std140-aligned");
    static_assert((sizeof(sSpriteScanlineConfig) & 15) == 0, "sSpriteScanlineConfig not std140-aligned");
    static_assert((sizeof(sSpriteConfig::sOAM) == 64), "sOAM must be 64 bytes (std140 array stride)");
    static_assert((sizeof(sCompositorConfig) & 15) == 0, "sCompositorConfig not std140-aligned");

    return true;
}

void GLRenderer2D::SetScaleFactor(int scale)
{
    ScaleFactor = scale;
    ScreenW = 256 * scale;
    ScreenH = 192 * scale;

    if (CompositorShader[2])
    {
        glUseProgram(CompositorShader[2]);
        glUniform1i(CompositorScaleULoc, ScaleFactor);
    }

    const GLenum fbassign2[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

    for (int u = 0; u < 2; u++)
    {
        // OBJ layer: 2 array layers (colour @0, flags @1) at screen scale.
        glBindTexture(GL_TEXTURE_2D_ARRAY, OBJLayerTex[u]);
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA, ScreenW, ScreenH, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindTexture(GL_TEXTURE_2D, OBJDepthTex[u]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT16, ScreenW, ScreenH, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, OBJLayerFB[u]);
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, OBJLayerTex[u], 0, 0);
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, OBJLayerTex[u], 0, 1);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, OBJDepthTex[u], 0);
        glDrawBuffers(2, fbassign2);
        GL2D_FBO("OBJLayerFB");

        // compositor output: single RGBA8 colour target at screen scale.
        glBindTexture(GL_TEXTURE_2D, OutputTex[u]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ScreenW, ScreenH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

        glBindFramebuffer(GL_FRAMEBUFFER, OutputFB[u]);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, OutputTex[u], 0);
        glDrawBuffer(GL_COLOR_ATTACHMENT0);
        GL2D_FBO("OutputFB");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL2D_CHK("SetScaleFactor");
}

void GLRenderer2D::DeInitGL()
{
    if (LayerPreShader[2])
    {
        glDeleteProgram(LayerPreShader[2]);
        LayerPreShader[0] = LayerPreShader[1] = LayerPreShader[2] = 0;
    }
    if (SpritePreShader[2])
    {
        glDeleteProgram(SpritePreShader[2]);
        SpritePreShader[0] = SpritePreShader[1] = SpritePreShader[2] = 0;
    }
    if (SpriteShader[2])
    {
        glDeleteProgram(SpriteShader[2]);
        SpriteShader[0] = SpriteShader[1] = SpriteShader[2] = 0;
    }
    if (CompositorShader[2])
    {
        glDeleteProgram(CompositorShader[2]);
        CompositorShader[0] = CompositorShader[1] = CompositorShader[2] = 0;
    }
    if (RectVtxArray)  { glDeleteVertexArrays(1, &RectVtxArray);  RectVtxArray = 0; }
    if (RectVtxBuffer) { glDeleteBuffers(1, &RectVtxBuffer);      RectVtxBuffer = 0; }

    if (SpritePreVtxArray)  { glDeleteVertexArrays(1, &SpritePreVtxArray);  SpritePreVtxArray = 0; }
    if (SpritePreVtxBuffer) { glDeleteBuffers(1, &SpritePreVtxBuffer);      SpritePreVtxBuffer = 0; }
    if (SpriteVtxArray)     { glDeleteVertexArrays(1, &SpriteVtxArray);     SpriteVtxArray = 0; }
    if (SpriteVtxBuffer)    { glDeleteBuffers(1, &SpriteVtxBuffer);         SpriteVtxBuffer = 0; }
    if (MosaicTex)          { glDeleteTextures(1, &MosaicTex);              MosaicTex = 0; }
    if (DummyTexArray)      { glDeleteTextures(1, &DummyTexArray);          DummyTexArray = 0; }

    delete[] SpritePreVtxData; SpritePreVtxData = nullptr;
    delete[] SpriteVtxData;    SpriteVtxData = nullptr;

    for (int u = 0; u < 2; u++)
    {
        if (VRAMTex_BG[u])        { glDeleteTextures(1, &VRAMTex_BG[u]);        VRAMTex_BG[u] = 0; }
        if (PalTex_BG[u])         { glDeleteTextures(1, &PalTex_BG[u]);         PalTex_BG[u] = 0; }
        if (LayerConfigUBO[u])    { glDeleteBuffers(1, &LayerConfigUBO[u]);     LayerConfigUBO[u] = 0; }
        if (ScanlineConfigUBO[u]) { glDeleteBuffers(1, &ScanlineConfigUBO[u]);  ScanlineConfigUBO[u] = 0; }
        if (AllBGLayerTex[u][0])  { glDeleteTextures(22, AllBGLayerTex[u]);     memset(AllBGLayerTex[u], 0, sizeof(AllBGLayerTex[u])); }
        if (AllBGLayerFB[u][0])   { glDeleteFramebuffers(22, AllBGLayerFB[u]);  memset(AllBGLayerFB[u], 0, sizeof(AllBGLayerFB[u])); }

        if (VRAMTex_OBJ[u])            { glDeleteTextures(1, &VRAMTex_OBJ[u]);            VRAMTex_OBJ[u] = 0; }
        if (PalTex_OBJ[u])             { glDeleteTextures(1, &PalTex_OBJ[u]);             PalTex_OBJ[u] = 0; }
        if (SpriteTex[u])              { glDeleteTextures(1, &SpriteTex[u]);              SpriteTex[u] = 0; }
        if (SpriteFB[u])               { glDeleteFramebuffers(1, &SpriteFB[u]);           SpriteFB[u] = 0; }
        if (OBJLayerTex[u])            { glDeleteTextures(1, &OBJLayerTex[u]);            OBJLayerTex[u] = 0; }
        if (OBJDepthTex[u])            { glDeleteTextures(1, &OBJDepthTex[u]);            OBJDepthTex[u] = 0; }
        if (OBJLayerFB[u])             { glDeleteFramebuffers(1, &OBJLayerFB[u]);         OBJLayerFB[u] = 0; }
        if (SpriteConfigUBO[u])        { glDeleteBuffers(1, &SpriteConfigUBO[u]);         SpriteConfigUBO[u] = 0; }
        if (SpriteScanlineConfigUBO[u]){ glDeleteBuffers(1, &SpriteScanlineConfigUBO[u]); SpriteScanlineConfigUBO[u] = 0; }
        if (OutputTex[u])              { glDeleteTextures(1, &OutputTex[u]);              OutputTex[u] = 0; }
        if (OutputFB[u])               { glDeleteFramebuffers(1, &OutputFB[u]);           OutputFB[u] = 0; }
        if (CompositorConfigUBO[u])    { glDeleteBuffers(1, &CompositorConfigUBO[u]);     CompositorConfigUBO[u] = 0; }
    }
    GLReady = false;
    GPU::GL2DActive = false;
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

    // The software renderer advances affine/extended BG reference points after
    // drawing each scanline. GL captures all scanlines up front, so it must do
    // the same bookkeeping here or mode-5 BG2/BG3 sample the same reference row
    // for the whole frame, dropping scenery such as NSMB's castle/ground.
    if ((unit->DispCnt & 0x0400) && ((bgmode == 2) || (bgmode >= 4 && bgmode <= 6)))
    {
        unit->BGXRefInternal[0] += unit->BGRotB[0];
        unit->BGYRefInternal[0] += unit->BGRotD[0];
    }
    if ((unit->DispCnt & 0x0800) && (bgmode >= 1 && bgmode <= 5))
    {
        unit->BGXRefInternal[1] += unit->BGRotB[1];
        unit->BGYRefInternal[1] += unit->BGRotD[1];
    }

    u16* pal = (u16*)&GPU::Palette[unit->Num ? 0x400 : 0];
    cfg.BackColor = pal[0];

    // Master brightness (reg 0x6C): final whole-screen brightness fade, applied
    // per-scanline in the compositor FS. Matches SoftRenderer's ColorBrightness
    // Up/Down (no rounding, factor clamped to 16). Used by virtually every game's
    // scene fade-in/out transitions.
    cfg.MasterBright = unit->MasterBrightness;

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

    // Sync the lazy-coherency flat VRAM buffers from the underlying bank data.
    // The soft renderer does this per DrawScanline; we must do it per frame here.
    if (num == 0)
    {
        auto bgDirty = GPU::VRAMDirty_ABG.DeriveState(GPU::VRAMMap_ABG);
        GPU::MakeVRAMFlat_ABGCoherent(bgDirty);
    }
    else
    {
        auto bgDirty = GPU::VRAMDirty_BBG.DeriveState(GPU::VRAMMap_BBG);
        GPU::MakeVRAMFlat_BBGCoherent(bgDirty);
    }

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

    // Sync extended palette flat buffers.
    //
    // The GL renderer coheres only ONCE per frame, whereas the SoftRenderer
    // derives every scanline. melonDS's dirty tracking is incremental:
    // DeriveState() copies dirty regions then CLEARS the per-bank dirty bits.
    // Games like NSMB write their BG extended palette ONCE (static), so a single
    // coarse once-per-frame consume can drop that lone dirty region — leaving the
    // ext-pal flat buffer all-zero and the mode-5 hills washed/wrong (the std
    // palette, which lives in dedicated RAM, was fine; only the VRAM-backed
    // ext-pal broke). Force a full re-copy every frame by invalidating the
    // tracker's recorded mapping first: Reset() sets Mapping[]=0x8000, so the
    // next DeriveState() sees a mismatch on every slot and does a full SetRange
    // copy of each mapped slot. ~32 KB/frame — correctness over deltas.
    if (num == 0)
    {
        GPU::VRAMDirty_ABGExtPal.Reset();
        auto epDirty = GPU::VRAMDirty_ABGExtPal.DeriveState(GPU::VRAMMap_ABGExtPal);
        GPU::MakeVRAMFlat_ABGExtPalCoherent(epDirty);
    }
    else
    {
        GPU::VRAMDirty_BBGExtPal.Reset();
        auto epDirty = GPU::VRAMDirty_BBGExtPal.DeriveState(GPU::VRAMMap_BBGExtPal);
        GPU::MakeVRAMFlat_BBGExtPalCoherent(epDirty);
    }

    // Standard BG palette (256 entries) then the 4 ext-pal slots × 16 palettes,
    // matching the PalTex_BG layout (rows: 0 = standard, 1.. = ext-pal).
    static u16 temp[kBGPalEntries];
    memcpy(&temp[0], &GPU::Palette[num ? 0x400 : 0], 256 * 2);
    for (int s = 0; s < 4; s++)
        for (int p = 0; p < 16; p++)
        {
            u16* pal = unit->GetBGExtPal(s, p);
            memcpy(&temp[(1 + (s * 16) + p) * 256], pal, 256 * 2);
        }

    // ── BGPAL-PROBE (one-shot, engine A, mode-5/ext-pal): dump the ACTUAL BG
    // palette being uploaded, at the moment it is read. Ground truth for whether
    // the hills SHOULD be vibrant. Scans for the first nonzero colours in the
    // standard region (rows 0) and the ext-pal region (rows 1..). Remove later.
    {
        static bool bgpalProbe[2] = { false, false };
        if (YAGE_MELONDS_GL_DIAG && !bgpalProbe[num] && num == 0 &&
            (unit->DispCnt & (1u << 30)))
        {
            bgpalProbe[num] = true;
            MELONDS_2D_LOG("BGPAL-PROBE u%d DispCnt=%08X bgExtPalEn=%d extPalMap=[%u,%u,%u,%u]",
                           num, unit->DispCnt, !!(unit->DispCnt & (1u << 30)),
                           GPU::VRAMMap_ABGExtPal[0], GPU::VRAMMap_ABGExtPal[1],
                           GPU::VRAMMap_ABGExtPal[2], GPU::VRAMMap_ABGExtPal[3]);
            // standard region (temp[0..255])
            int shown = 0;
            for (int i = 0; i < 256 && shown < 6; i++)
            {
                u16 c = temp[i];
                if (!c) continue;
                MELONDS_2D_LOG("BGPAL-PROBE std[%d]=0x%04X rgb=(%d,%d,%d)", i, c,
                               (c & 0x1F) << 3, ((c >> 5) & 0x1F) << 3, ((c >> 10) & 0x1F) << 3);
                shown++;
            }
            if (shown == 0) MELONDS_2D_LOG("BGPAL-PROBE std region ALL ZERO");
            // ext-pal region (temp[256..])
            shown = 0;
            for (int i = 256; i < kBGPalEntries && shown < 10; i++)
            {
                u16 c = temp[i];
                if (!c) continue;
                int row = i / 256, ent = i % 256;
                MELONDS_2D_LOG("BGPAL-PROBE ext row%d(slot%d pal%d) e%d=0x%04X rgb=(%d,%d,%d)",
                               row, (row - 1) / 16, (row - 1) % 16, ent, c,
                               (c & 0x1F) << 3, ((c >> 5) & 0x1F) << 3, ((c >> 10) & 0x1F) << 3);
                shown++;
            }
            if (shown == 0) MELONDS_2D_LOG("BGPAL-PROBE ext region ALL ZERO");
        }
    }

    // Skip swizzle + GPU upload when the palette data hasn't changed since the
    // last upload. bionic's memcmp is NEON-vectorised (~7 µs for 33 KB) vs the
    // swizzle loop + glTexSubImage2D (~350 µs). Most NDS games only change
    // palettes on level transitions, so this saves work on the vast majority of
    // rendered frames. The Reset+DeriveState+MakeFlat above must still run every
    // frame (it keeps the flat VRAM buffer coherent with the VRAM bank writes).
    if (PalBGValid[num] &&
        memcmp(temp, PrevPalBG[num], sizeof(u16) * kBGPalEntries) == 0)
        return;

    memcpy(PrevPalBG[num], temp, sizeof(u16) * kBGPalEntries);
    PalBGValid[num] = true;

    // GLES3 has no GL_UNSIGNED_SHORT_1_5_5_5_REV, so swizzle NDS BGR555 entries
    // into 5_5_5_1 layout at upload time (same as the GPU3D_OpenGL palette path):
    //   R(0x001F)<<11 | G(0x03E0)<<1 | B(0x7C00)>>9 | (top bit)->A
    static u16 swiz[kBGPalEntries];
    for (int i = 0; i < kBGPalEntries; i++)
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


    // ── One-shot debug dump: first frame where at least one BG layer is enabled ──
    static bool bgDumped[2]   = { false, false };
    static bool pxDumped[2]   = { false, false };
    bool doPixelReadback = false;
    if (YAGE_MELONDS_GL_DIAG && !bgDumped[num] &&
        ((unit->DispCnt >> 8) & 0x0F) != 0)
    {
        bgDumped[num]   = true;
        doPixelReadback = true;
        u8* vram; u32 vrammask;
        unit->GetBGVRAM(vram, vrammask);
        u32 scanLen = vrammask + 1;
        int nonzero = 0;
        if (vram) { for (u32 i = 0; i < scanLen; i++) if (vram[i]) nonzero++; }
        MELONDS_2D_LOG("BG-DBG u%d (BG-enabled): DispCnt=%08X uVRAMMask=%u BGAct=%X vramNZ=%d/%u",
                       num, unit->DispCnt, LayerConfig[num].uVRAMMask, BGLayerActive[num],
                       nonzero, scanLen);
        for (int l = 0; l < 4; l++)
        {
            if (!(BGLayerActive[num] & (1u << l))) continue;
            auto& cfg = LayerConfig[num].uBGConfig[l];
            u32 ts = BGVRAMRange[num][l][0], tsz = BGVRAMRange[num][l][1];
            u32 ms = BGVRAMRange[num][l][2], msz = BGVRAMRange[num][l][3];
            int tnz = 0, mnz = 0;
            if (vram && ts != 0xFFFFFFFF && tsz > 0)
                for (u32 i = ts; i < ts+tsz && i <= vrammask; i++) if (vram[i]) tnz++;
            if (vram && ms != 0xFFFFFFFF && msz > 0)
                for (u32 i = ms; i < ms+msz && i <= vrammask; i++) if (vram[i]) mnz++;
            u8 tb0 = (ts != 0xFFFFFFFF && vram && ts <= vrammask) ? vram[ts] : 0xFF;
            u8 mb0 = (ms != 0xFFFFFFFF && vram && ms <= vrammask) ? vram[ms] : 0xFF;
            MELONDS_2D_LOG("  L%d type=%u size=%dx%d tOff=%u[%u] tNZ=%d t[0]=0x%02x mOff=%u[%u] mNZ=%d m[0]=0x%02x",
                           l, cfg.Type, cfg.Size[0], cfg.Size[1], ts, tsz, tnz, tb0, ms, msz, mnz, mb0);
        }
    }

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

        // One-shot pixel readback on the first BG-enabled frame only.
        if (doPixelReadback && !pxDumped[num])
        {
            glFinish();
            u8 px[4] = {0, 0, 0, 0};
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, BGLayerFB[num][layer]);
            glReadBuffer(GL_COLOR_ATTACHMENT0);
            glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            GL2D_CHK("readback");
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
            MELONDS_2D_LOG("  L%d pixel(4,4): r=%u g=%u b=%u a=%u", layer, px[0], px[1], px[2], px[3]);
            if (layer == 3) pxDumped[num] = true;
        }
    }
}

// ── Sprites (C3.1) ─────────────────────────────────────────────────────────
// Ported from upstream GLRenderer2D::{UpdateOAM,PrerenderSprites,DoRenderSprites,
// RenderSprites}, adapted to the fork's per-Unit API:
//   GPU2D.x          -> unit->x
//   GPU2D.OBJMosaicLine -> unit->OBJMosaicY (the fork's mosaic-snapped line)
//   local OAM copy   -> read directly from GPU::OAM[unit base]
//   GetCaptureInfo_OBJ -> absent → OBJ display-capture deferred (bitmap sprites
//                         stay Type 2), mirroring the BG capture deferral.
// Not called during software passthrough — go live at C4.

// Scan OAM for `unit` over scanlines [ystart, yend) and build the per-OBJ config.
// Resets NumSprites/SpriteUseMosaic, so call once per frame for the whole range.
void GLRenderer2D::UpdateOAM(int ystart, int yend, Unit* unit)
{
    const int num = unit->Num;
    auto& cfg = SpriteConfig[num];
    u16* oam = (u16*)&GPU::OAM[num ? 0x400 : 0];

    NumSprites[num] = 0;
    SpriteUseMosaic[num] = false;

    // rotscale parameters: 32 groups of 4 s16 interleaved through OAM.
    for (int i = 0; i < 32; i++)
    {
        s16* rotscale = (s16*)&oam[(i * 16) + 3];
        auto& rotdst = cfg.uRotscale[i];
        rotdst[0] = rotscale[0];
        rotdst[1] = rotscale[4];
        rotdst[2] = rotscale[8];
        rotdst[3] = rotscale[12];
    }

    static const u8 spritewidth[16] =
    {
        8, 16, 8, 8,  16, 32, 8, 8,  32, 32, 16, 8,  64, 64, 32, 8
    };
    static const u8 spriteheight[16] =
    {
        8, 8, 16, 8,  16, 8, 32, 8,  32, 16, 32, 8,  64, 32, 64, 8
    };

    for (int sprnum = 0; sprnum < 128; sprnum++)
    {
        u16* attrib = &oam[sprnum * 4];

        u32 sprtype = (attrib[0] >> 8) & 0x3;
        if (sprtype == 2) // sprite disabled
            continue;

        // X > 255 → negative (-256..-1); Y > 127 → both positive and negative.
        s32 xpos = (s32)(attrib[1] << 23) >> 23;
        s32 ypos = (s32)(attrib[0] << 24) >> 24;

        u32 sizeparam = (attrib[0] >> 14) | ((attrib[1] & 0xC000) >> 12);
        s32 width = spritewidth[sizeparam];
        s32 height = spriteheight[sizeparam];
        s32 boundwidth = width;
        s32 boundheight = height;

        if (sprtype == 3)
        {
            // double-size rotscale sprite
            boundwidth <<= 1;
            boundheight <<= 1;
        }

        if (xpos <= -boundwidth)
            continue;

        bool yc0 = ((ypos + boundheight) > ystart) && (ypos < yend);
        bool yc1 = (((ypos & 0xFF) + boundheight) > ystart) && ((ypos & 0xFF) < yend);
        if (!(yc0 || yc1))
            continue;

        u32 sprmode = (attrib[0] >> 10) & 0x3;
        if (sprmode == 3)
        {
            if ((unit->DispCnt & 0x60) == 0x60)
                continue;
            if ((attrib[2] >> 12) == 0)
                continue;
        }

        if (NumSprites[num] >= 128)
            break;

        auto& sprcfg = cfg.uOAM[NumSprites[num]];

        sprcfg.Position[0] = (u32)xpos;
        sprcfg.Position[1] = (u32)ypos;
        sprcfg.Size[0] = width;
        sprcfg.Size[1] = height;
        sprcfg.BoundSize[0] = boundwidth;
        sprcfg.BoundSize[1] = boundheight;

        if (sprtype & 1)
        {
            sprcfg.Flip[0] = 0;
            sprcfg.Flip[1] = 0;
            sprcfg.Rotscale = (attrib[1] >> 9) & 0x1F;
        }
        else
        {
            sprcfg.Flip[0] = !!(attrib[1] & (1 << 12));
            sprcfg.Flip[1] = !!(attrib[1] & (1 << 13));
            sprcfg.Rotscale = (u32)-1;
        }

        sprcfg.OBJMode = sprmode;
        sprcfg.Mosaic = (!!(attrib[0] & (1 << 12))) && (sprmode != 2);
        sprcfg.BGPrio = (attrib[2] >> 10) & 0x3;

        u32 tilenum = attrib[2] & 0x3FF;

        if (sprmode == 3)
        {
            // bitmap sprite
            sprcfg.Type = 2;

            if (unit->DispCnt & (1 << 6))
            {
                // 1D mapping
                sprcfg.TileOffset = tilenum << (7 + ((unit->DispCnt >> 22) & 0x1));
                sprcfg.TileStride = width * 2;
            }
            else
            {
                bool is256 = !!(unit->DispCnt & (1 << 5));
                if (is256)
                {
                    // 2D mapping, 256 pixels
                    sprcfg.TileOffset = ((tilenum & 0x01F) << 4) + ((tilenum & 0x3E0) << 7);
                    sprcfg.TileStride = 256 * 2;
                }
                else
                {
                    // 2D mapping, 128 pixels
                    sprcfg.TileOffset = ((tilenum & 0x00F) << 4) + ((tilenum & 0x3F0) << 7);
                    sprcfg.TileStride = 128 * 2;
                }
                // OBJ display-capture detection (sprite Type 3/4) deferred — the
                // fork has no GetCaptureInfo_OBJ. Bitmap sprites stay Type 2
                // (direct VRAM read). Revisit when capture lands at C4.
            }

            sprcfg.PalOffset = 1 + (attrib[2] >> 12); // alpha
        }
        else
        {
            if (unit->DispCnt & (1 << 4))
            {
                // 1D mapping
                sprcfg.TileOffset = tilenum << (5 + ((unit->DispCnt >> 20) & 0x3));
                sprcfg.TileStride = (width >> 3) * 32;
                if (attrib[0] & (1 << 13))
                    sprcfg.TileStride <<= 1;
            }
            else
            {
                // 2D mapping
                sprcfg.TileOffset = tilenum << 5;
                sprcfg.TileStride = 32 * 32;
            }

            if (attrib[0] & (1 << 13))
            {
                // 256-color sprite
                sprcfg.Type = 1;
                if (unit->DispCnt & (1u << 31))
                    sprcfg.PalOffset = 1 + (attrib[2] >> 12);
                else
                    sprcfg.PalOffset = 0;
            }
            else
            {
                // 16-color sprite
                sprcfg.Type = 0;
                sprcfg.PalOffset = (attrib[2] >> 12) << 4;
            }
        }

        NumSprites[num]++;

        if (sprcfg.Mosaic && (unit->OBJMosaicSize[0] > 0))
            SpriteUseMosaic[num] = true;
    }

    glBindBuffer(GL_UNIFORM_BUFFER, SpriteConfigUBO[num]);
    glBufferSubData(GL_UNIFORM_BUFFER,
                    0,
                    offsetof(sSpriteConfig, uOAM) + (NumSprites[num] * sizeof(cfg.uOAM[0])),
                    &cfg);
}

// Upload the whole OBJ VRAM region into the unit's R8UI texture. The fork has
// no dirty tracking (cf. UploadBGVRAM uploading referenced ranges); sprite tile
// reads are scattered, so we upload all of it each frame — correctness first;
// delta uploads are a Phase-D perf optimisation. Call before PrerenderSprites.
void GLRenderer2D::UploadOBJVRAM(Unit* unit)
{
    const int num = unit->Num;

    if (num == 0)
    {
        auto objDirty = GPU::VRAMDirty_AOBJ.DeriveState(GPU::VRAMMap_AOBJ);
        GPU::MakeVRAMFlat_AOBJCoherent(objDirty);
    }
    else
    {
        auto objDirty = GPU::VRAMDirty_BOBJ.DeriveState(GPU::VRAMMap_BOBJ);
        GPU::MakeVRAMFlat_BOBJCoherent(objDirty);
    }

    u8* vram; u32 vrammask;
    unit->GetOBJVRAM(vram, vrammask);
    int objheight = (int)((vrammask + 1) >> 10);   // VRAM rows = texture height

    glBindTexture(GL_TEXTURE_2D, VRAMTex_OBJ[num]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 1024, objheight,
                    GL_RED_INTEGER, GL_UNSIGNED_BYTE, vram);
}

// Standard OBJ palette (256 entries) + the 16 ext-pal slots, BGR555→5_5_5_1
// swizzled at upload (no GL_UNSIGNED_SHORT_1_5_5_5_REV on GLES3). Matches the
// PalTex_OBJ layout (row 0 = standard, rows 1.. = ext-pal).
void GLRenderer2D::UploadOBJPalette(Unit* unit)
{
    const int num = unit->Num;

    // Force a full re-copy every frame (see UploadBGPalette for the rationale):
    // a static OBJ extended palette can be dropped by the once-per-frame
    // incremental dirty consume, leaving ext-pal sprites mis-coloured.
    if (num == 0)
    {
        GPU::VRAMDirty_AOBJExtPal.Reset();
        auto epDirty = GPU::VRAMDirty_AOBJExtPal.DeriveState(&GPU::VRAMMap_AOBJExtPal);
        GPU::MakeVRAMFlat_AOBJExtPalCoherent(epDirty);
    }
    else
    {
        GPU::VRAMDirty_BOBJExtPal.Reset();
        auto epDirty = GPU::VRAMDirty_BOBJExtPal.DeriveState(&GPU::VRAMMap_BOBJExtPal);
        GPU::MakeVRAMFlat_BOBJExtPalCoherent(epDirty);
    }

    static u16 temp[kOBJPalEntries];
    memcpy(&temp[0], &GPU::Palette[num ? 0x600 : 0x200], 256 * 2);
    {
        u16* pal = unit->GetOBJExtPal();
        memcpy(&temp[256], pal, 256 * 16 * 2);
    }

    // Same change-detection optimisation as UploadBGPalette: skip when the
    // source data is identical to the last uploaded version.
    if (PalOBJValid[num] &&
        memcmp(temp, PrevPalOBJ[num], sizeof(u16) * kOBJPalEntries) == 0)
        return;

    memcpy(PrevPalOBJ[num], temp, sizeof(u16) * kOBJPalEntries);
    PalOBJValid[num] = true;

    static u16 swiz[kOBJPalEntries];
    for (int i = 0; i < kOBJPalEntries; i++)
    {
        u16 s = temp[i];
        swiz[i] = ((s & 0x001F) << 11) | ((s & 0x03E0) << 1) | ((s & 0x7C00) >> 9) | ((s >> 15) & 0x1);
    }

    glBindTexture(GL_TEXTURE_2D, PalTex_OBJ[num]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1 + 16,
                    GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, swiz);
}

// Rasterise each non-bitmap OBJ into the 1024×512 atlas (SpriteTex[num]).
void GLRenderer2D::PrerenderSprites(Unit* unit)
{
    const int num = unit->Num;

    u16* vtxbuf = SpritePreVtxData;
    int vtxnum = 0;

    for (int i = 0; i < NumSprites[num]; i++)
    {
        auto& sprite = SpriteConfig[num].uOAM[i];
        if (sprite.Type >= 3)   // bitmap/capture sprite — composited directly
            continue;

        *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
        *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
        *vtxbuf++ = 1; *vtxbuf++ = 1; *vtxbuf++ = i;
        *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
        *vtxbuf++ = 0; *vtxbuf++ = 0; *vtxbuf++ = i;
        *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
        vtxnum += 6;
    }

    if (vtxnum == 0) return;

    glUseProgram(SpritePreShader[2]);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);

    glBindBufferBase(GL_UNIFORM_BUFFER, 21, SpriteConfigUBO[num]);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, SpriteFB[num]);
    glViewport(0, 0, 1024, 512);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, VRAMTex_OBJ[num]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, PalTex_OBJ[num]);

    glBindBuffer(GL_ARRAY_BUFFER, SpritePreVtxBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vtxnum * 3 * sizeof(u16), SpritePreVtxData);
    glBindVertexArray(SpritePreVtxArray);
    glDrawArrays(GL_TRIANGLES, 0, vtxnum);

    // ── OBJ-PROBE (one-shot, engine A): diagnose bluish sprite colours ──
    // Samples the rendered atlas at each sprite's CENTRE (body colour, not the
    // transparent corner), and dumps the raw NDS OBJ palette straight from
    // memory (reliable) with the same NDS→RGB conversion the renderer uses, so
    // atlas colours can be compared to ground truth. Remove once fixed.
    {
        static bool objProbe[2] = { false, false };
        if (YAGE_MELONDS_GL_DIAG && !objProbe[num] && num == 0 &&
            NumSprites[num] > 0)
        {
            objProbe[num] = true;
            MELONDS_2D_LOG("OBJ-PROBE u%d DispCnt=%08X objExtPalEn=%d NumSpr=%d",
                           num, unit->DispCnt, !!(unit->DispCnt & (1u << 31)),
                           NumSprites[num]);

            // Raw standard OBJ palette (engine A @ 0x200), first 8 entries.
            // NDS BGR555: r=bits0-4, g=5-9, b=10-14. Logs raw + 8-bit RGB.
            const u16* objpal = (const u16*)&GPU::Palette[0x200];
            for (int e = 0; e < 8; e++)
            {
                u16 c = objpal[e];
                int r = ((c)       & 0x1F) << 3;
                int g = ((c >> 5)  & 0x1F) << 3;
                int b = ((c >> 10) & 0x1F) << 3;
                MELONDS_2D_LOG("OBJ-PROBE rawpal[%d]=0x%04X -> rgb=(%d,%d,%d)", e, c, r, g, b);
            }

            glFinish();
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, SpriteFB[num]);
            glReadBuffer(GL_COLOR_ATTACHMENT0);

            for (int s = 0; s < NumSprites[num] && s < 12; s++)
            {
                auto& o = SpriteConfig[num].uOAM[s];
                // centre of this sprite's drawn content within its 64×64 cell
                int cx = ((s & 0xF) * 64) + (o.Size[0] >> 1);
                int cy = ((s >> 4) * 64) + (o.Size[1] >> 1);
                GLubyte ap[4] = {0};
                glReadPixels(cx, cy, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, ap);
                MELONDS_2D_LOG("OBJ-PROBE spr[%d] Type=%u PalOff=%u OBJMode=%u "
                               "Size=%dx%d atlasCtr=(%u,%u,%u,%u)",
                               s, o.Type, o.PalOffset, o.OBJMode,
                               o.Size[0], o.Size[1], ap[0], ap[1], ap[2], ap[3]);
            }
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        }
    }
}

// Composite the atlas into the OBJ layer for [LastSpriteLine, line). Two mosaic
// passes then an opaque pass; priority resolved via depth.
void GLRenderer2D::DoRenderSprites(int line, Unit* unit)
{
    const int num = unit->Num;
    int ystart = LastSpriteLine[num];
    int yend = line;

    glUseProgram(SpriteShader[2]);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);

    glBindBufferBase(GL_UNIFORM_BUFFER, 21, SpriteConfigUBO[num]);
    glBindBufferBase(GL_UNIFORM_BUFFER, 24, SpriteScanlineConfigUBO[num]);

    glBindBuffer(GL_UNIFORM_BUFFER, SpriteScanlineConfigUBO[num]);
    glBufferSubData(GL_UNIFORM_BUFFER,
                    ystart * sizeof(s32),
                    (yend - ystart) * sizeof(s32),
                    &SpriteScanlineConfig[num].uMosaicLine[ystart]);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, OBJLayerFB[num]);
    glViewport(0, 0, ScreenW, ScreenH);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, SpriteTex[num]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D_ARRAY, DummyTexArray);   // Capture128 (deferred)
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D_ARRAY, DummyTexArray);   // Capture256 (deferred)

    glEnable(GL_SCISSOR_TEST);
    glScissor(0, ystart * ScaleFactor, ScreenW, (yend - ystart) * ScaleFactor);

    // Two passes for mosaic: mosaic flags get set for transparent pixels too,
    // and priority is only checked against opaque pixels (see upstream).
    //
    // ⚠ GLES3.0 limitation: the per-attachment colour masks below (glColorMaski
    // with idx>0) are honoured ONLY for attachment 0 by the compat shim — real
    // independent draw-buffer masking needs GL_EXT_draw_buffers_indexed
    // (GLES3.2). The flags buffer (attachment 1) is therefore not masked
    // per-channel here. To resolve at C4 when this path drives pixels (detect
    // the EXT and use glColorMaskiEXT, or restructure the flags accumulation).
    glClearColor(0, 0, 0, 0);
    glClearDepthf(1.0f);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glColorMaski(0, GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glDepthMask(GL_FALSE);

    if (SpriteUseMosaic[num])
    {
        glUniform1i(SpriteRenderTransULoc, 1);
        glColorMaski(1, GL_FALSE, GL_TRUE, GL_FALSE, GL_TRUE);

        RenderSprites(false, ystart, yend, unit);
    }

    glUniform1i(SpriteRenderTransULoc, 0);
    glColorMaski(1, GL_FALSE, GL_FALSE, GL_TRUE, GL_FALSE);

    RenderSprites(true, ystart, yend, unit);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glColorMaski(1, GL_TRUE, GL_TRUE, GL_FALSE, GL_TRUE);

    RenderSprites(false, ystart, yend, unit);

    glDisable(GL_SCISSOR_TEST);
}

// One sprite draw pass over [ystart, yend). `window` selects OBJ-window sprites
// (OBJMode == 2) vs the rest. Each visible OBJ contributes 1–2 quads (wrap).
void GLRenderer2D::RenderSprites(bool window, int ystart, int yend, Unit* unit)
{
    const int num = unit->Num;

    if (window)
    {
        if (!(unit->DispCnt & (1 << 15)))
            return;
    }

    u16* vtxbuf = SpriteVtxData;
    int vtxnum = 0;

    for (int i = 0; i < NumSprites[num]; i++)
    {
        auto& sprite = SpriteConfig[num].uOAM[i];

        bool iswin = (sprite.OBJMode == 2);
        if (iswin != window)
            continue;

        s32 xpos = (s32)sprite.Position[0];
        s32 ypos = (s32)sprite.Position[1];
        s32 boundwidth = sprite.BoundSize[0];
        s32 boundheight = sprite.BoundSize[1];

        bool yc0 = ((ypos + boundheight) > ystart) && (ypos < yend);
        bool yc1 = (((ypos & 0xFF) + boundheight) > ystart) && ((ypos & 0xFF) < yend);

        if (yc0)
        {
            s32 x0 = xpos, x1 = xpos + boundwidth;
            s32 y0 = ypos, y1 = ypos + boundheight;

            *vtxbuf++ = x0; *vtxbuf++ = y1; *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y0; *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y1; *vtxbuf++ = 1; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x0; *vtxbuf++ = y1; *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x0; *vtxbuf++ = y0; *vtxbuf++ = 0; *vtxbuf++ = 0; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y0; *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
            vtxnum += 6;
        }

        if (yc1)
        {
            ypos &= 0xFF;
            s32 x0 = xpos, x1 = xpos + boundwidth;
            s32 y0 = ypos, y1 = ypos + boundheight;

            *vtxbuf++ = x0; *vtxbuf++ = y1; *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y0; *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y1; *vtxbuf++ = 1; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x0; *vtxbuf++ = y1; *vtxbuf++ = 0; *vtxbuf++ = 1; *vtxbuf++ = i;
            *vtxbuf++ = x0; *vtxbuf++ = y0; *vtxbuf++ = 0; *vtxbuf++ = 0; *vtxbuf++ = i;
            *vtxbuf++ = x1; *vtxbuf++ = y0; *vtxbuf++ = 1; *vtxbuf++ = 0; *vtxbuf++ = i;
            vtxnum += 6;
        }
    }

    if (vtxnum == 0) return;

    glBindBuffer(GL_ARRAY_BUFFER, SpriteVtxBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vtxnum * 5 * sizeof(u16), SpriteVtxData);
    glBindVertexArray(SpriteVtxArray);
    glDrawArrays(GL_TRIANGLES, 0, vtxnum);
}

// ── Compositor (C4.1) ──────────────────────────────────────────────────────
// Ported from upstream GLRenderer2D::{UpdateCompositorConfig,RenderScreen},
// adapted to the fork's per-Unit API. The cached register fields upstream keeps
// (LayerEnable/OBJEnable/ForcedBlank) are derived from unit->DispCnt here.
// Not called during software passthrough — go live at C4.2.

void GLRenderer2D::UpdateCompositorConfig(Unit* unit)
{
    const int num = unit->Num;
    auto& cfg = CompositorConfig[num];

    // DISPCNT bits 8-12 = BG0-3 + OBJ enable.
    u32 layerEnable = (unit->DispCnt >> 8) & 0x1F;

    for (int i = 0; i < 4; i++)
        cfg.uBGPrio[i] = (u32)-1;

    for (int layer = 0; layer < 4; layer++)
    {
        if (!(layerEnable & (1u << layer)))
            continue;
        cfg.uBGPrio[layer] = unit->BGCnt[layer] & 0x3;
    }

    // One-shot compositor dump on first frame with at least one BG layer enabled
    // (logged after priority computation so BGpri shows current-frame values).
    static bool compDumped[2] = { false, false };
    if (YAGE_MELONDS_GL_DIAG && !compDumped[num] &&
        (layerEnable & 0x0F) != 0)
    {
        compDumped[num] = true;
        MELONDS_2D_LOG("COMP-DBG u%d (BG-enabled): DispCnt=%08X layerEnable=0x%X BGpri=%d/%d/%d/%d OBJen=%d",
                       num, unit->DispCnt, layerEnable,
                       cfg.uBGPrio[0], cfg.uBGPrio[1], cfg.uBGPrio[2], cfg.uBGPrio[3],
                       !!(layerEnable & (1u << 4)));
    }

    cfg.uEnableOBJ    = !!(layerEnable & (1u << 4));
    cfg.uEnable3D     = !!(unit->DispCnt & (1u << 3));
    cfg.uBlendCnt     = unit->BlendCnt;
    cfg.uBlendEffect  = (unit->BlendCnt >> 6) & 0x3;
    cfg.uBlendCoef[0] = unit->EVA;
    cfg.uBlendCoef[1] = unit->EVB;
    cfg.uBlendCoef[2] = unit->EVY;

    glBindBuffer(GL_UNIFORM_BUFFER, CompositorConfigUBO[num]);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(sCompositorConfig), &cfg);
}

void GLRenderer2D::RenderScreen(int ystart, int yend, Unit* unit)
{
    const int num = unit->Num;

    // ── ARCH-DBG: architecture-state probe (remove once feature gaps closed) ──
    // Logs the display-engine features the GL2D renderer does NOT yet implement
    // (alt display modes, display capture) so we can tell which test games hit
    // them. Fires on the first frame, on any dispmode/capture-enable CHANGE, and
    // as a heartbeat every 300 frames per unit.
    if (YAGE_MELONDS_GL_DIAG)
    {
        static u32  lastDispMode[2] = { 0xFFFFFFFF, 0xFFFFFFFF };
        static u32  lastCapEn[2]    = { 0xFFFFFFFF, 0xFFFFFFFF };
        static int  archFrame[2]    = { 0, 0 };
        u32 dispmode = (unit->DispCnt >> 16) & (num ? 0x1u : 0x3u);
        u32 capEn    = (num == 0) ? (unit->CaptureCnt >> 31) : 0;
        bool changed = (dispmode != lastDispMode[num]) || (capEn != lastCapEn[num]);
        if (changed || (archFrame[num] % 300) == 0)
        {
            const char* dmName = (dispmode==0)?"OFF":(dispmode==1)?"NORMAL":
                                 (dispmode==2)?"VRAM":"FIFO";
            u32 cc = unit->CaptureCnt;
            static const char* capSizeName[4] = {"128x128","256x64","256x128","256x192"};
            MELONDS_2D_LOG("ARCH u%d f=%d dispmode=%u(%s) DispCnt=%08X "
                           "mbright=%u/%u%s",
                           num, archFrame[num], dispmode, dmName, unit->DispCnt,
                           (unit->MasterBrightness >> 14) & 0x3,
                           unit->MasterBrightness & 0x1F,
                           changed ? " [CHANGED]" : "");
            if (num == 0)
                MELONDS_2D_LOG("ARCH u0 capture: en=%u srcA=%s srcB=%s capsrc=%u "
                               "size=%s dstblk=%u rdblk=%u lcdc=0x%X fifo=%d",
                               capEn,
                               (cc & (1u<<24)) ? "3Donly" : "2D+3D",
                               (cc & (1u<<25)) ? "mainmem" : "VRAM",
                               (cc >> 29) & 0x3,
                               capSizeName[(cc >> 20) & 0x3],
                               (cc >> 16) & 0x3,
                               (unit->DispCnt >> 18) & 0x3,
                               GPU::VRAMMap_LCDC,
                               (int)unit->UsesFIFO());
        }
        lastDispMode[num] = dispmode;
        lastCapEn[num]    = capEn;
        archFrame[num]++;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, OutputFB[num]);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glColorMaski(0, GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glDepthMask(GL_FALSE);

    glViewport(0, 0, ScreenW, ScreenH);

    glEnable(GL_SCISSOR_TEST);
    glScissor(0, ystart * ScaleFactor, ScreenW, (yend - ystart) * ScaleFactor);

    bool forcedBlank  = !!(unit->DispCnt & (1u << 7));
    bool unitEnabled  = unit->Enabled;
    // DispCnt[17:16] = display mode: 0=off(black), 1=normal, 2=VRAM(A-only),
    // 3=FIFO(A-only). Mode 0 disables the screen entirely (independent of
    // forced-blank bit 7). Common during scene transitions; output is black.
    u32  dispmode     = (unit->DispCnt >> 16) & (num ? 0x1u : 0x3u);

    if (forcedBlank || !unitEnabled || dispmode == 0)
    {
        // forced blank (bit 7) → white per NDS spec.
        // screen off (dispmode=0) or engine disabled → black.
        if (forcedBlank)
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        else
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        return;
    }

    glUseProgram(CompositorShader[2]);

    glBindBufferBase(GL_UNIFORM_BUFFER, 20, LayerConfigUBO[num]);
    glBindBufferBase(GL_UNIFORM_BUFFER, 22, ScanlineConfigUBO[num]);
    glBindBufferBase(GL_UNIFORM_BUFFER, 23, CompositorConfigUBO[num]);

    glBindBuffer(GL_UNIFORM_BUFFER, ScanlineConfigUBO[num]);
    glBufferSubData(GL_UNIFORM_BUFFER,
                    ystart * sizeof(sScanline),
                    (yend - ystart) * sizeof(sScanline),
                    &ScanlineConfig[num].uScanline[ystart]);

    UpdateCompositorConfig(unit);

    for (int i = 0; i < 4; i++)
    {
        glActiveTexture(GL_TEXTURE0 + i);

        // BG0 samples the GPU3D output when the 3D layer is enabled.
        if ((i == 0) && (unit->DispCnt & (1u << 3)))
        {
            Output3DTex = (GLuint)GPU::Get3DOutputTexture();
            bool bound3D = GPU::Bind3DOutputTexture();
            if (!bound3D)
                glBindTexture(GL_TEXTURE_2D, 0);

            static bool threeDDumped[2] = { false, false };
            if (YAGE_MELONDS_GL_DIAG && !threeDDumped[num])
            {
                threeDDumped[num] = true;
                MELONDS_2D_LOG("3D-HOOK u%d DispCnt=%08X tex=%u bound=%d BG0prio=%u blend=%04X",
                               num, unit->DispCnt, Output3DTex, (int)bound3D,
                               CompositorConfig[num].uBGPrio[0], unit->BlendCnt);
            }

            // Opening-scene GPU3D texture probe: fires at frame 120 of the opening
            // (DispCnt==0x00011F08) to sample both FrontBuffer and FrontBuffer^1.
            // Logs: content (alpha / RGB) at 5 y-positions for BOTH buffers.
            static int openingFrameCount[2] = { 0, 0 };
            static bool openingProbeDone[2] = { false, false };
            if (unit->DispCnt == 0x00011F08) openingFrameCount[num]++;
            if (YAGE_MELONDS_GL_DIAG &&
                !openingProbeDone[num] && openingFrameCount[num] >= 120
                && unit->DispCnt == 0x00011F08 && bound3D && Output3DTex != 0)
            {
                openingProbeDone[num] = true;
                // Access GPU3D renderer to sample both FramebufferTex slots.
                auto* gl3d = GPU3D::CurrentRenderer
                    ? reinterpret_cast<GPU3D::GLRenderer*>(GPU3D::CurrentRenderer.get())
                    : nullptr;
                if (gl3d)
                {
                    GLuint fbTex[2] = {
                        gl3d->GetAccelFrameTexture(),     // FrontBuffer^1 (current)
                        0,
                    };
                    // Determine the OTHER slot (FrontBuffer instead of FrontBuffer^1)
                    // by noting which tex is bound and picking the companion.
                    GLuint tex0 = gl3d->GetFramebufferTex(0);
                    GLuint tex1 = gl3d->GetFramebufferTex(1);
                    fbTex[1] = (fbTex[0] == tex0) ? tex1 : tex0;

                    static const int probeYs[5] = { 20, 60, 96, 130, 170 };
                    for (int bi = 0; bi < 2; bi++)
                    {
                        GLuint probeFBO;
                        glGenFramebuffers(1, &probeFBO);
                        GLint prevRFBO = 0;
                        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRFBO);
                        glBindFramebuffer(GL_READ_FRAMEBUFFER, probeFBO);
                        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                               GL_TEXTURE_2D, fbTex[bi], 0);
                        GLenum fbs = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
                        if (fbs == GL_FRAMEBUFFER_COMPLETE)
                        {
                            glReadBuffer(GL_COLOR_ATTACHMENT0);
                            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                            char logbuf[256];
                            int off = snprintf(logbuf, sizeof(logbuf),
                                               "3D-SCAN u%d buf=%s tex=%u:",
                                               num,
                                               bi==0 ? "FB^1" : "FB",
                                               fbTex[bi]);
                            for (int yi = 0; yi < 5; yi++)
                            {
                                GLubyte px[4] = {0};
                                glReadPixels(128, probeYs[yi], 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
                                off += snprintf(logbuf+off, sizeof(logbuf)-off,
                                                " y%d=(%u,%u,%u,%u)",
                                                probeYs[yi], px[0], px[1], px[2], px[3]);
                            }
                            MELONDS_2D_LOG("%s", logbuf);
                        }
                        glDeleteFramebuffers(1, &probeFBO);
                        glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)prevRFBO);
                    }
                }
            }

        }
        else
            glBindTexture(GL_TEXTURE_2D, BGLayerTex[num][i]);

        // ⚠ GLES3.0 has no GL_CLAMP_TO_BORDER (GLES3.2 / EXT_texture_border_clamp).
        // Upstream uses CLAMP_TO_BORDER so out-of-bounds affine/rotscale reads
        // come back transparent; CLAMP_TO_EDGE smears the edge texel instead.
        // To resolve at C4.2 (detect the EXT, or bounds-check in the FS).
        GLint wrapmode = LayerConfig[num].uBGConfig[i].Clamp ? GL_CLAMP_TO_EDGE : GL_REPEAT;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapmode);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrapmode);
    }

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D_ARRAY, OBJLayerTex[num]);
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D_ARRAY, DummyTexArray);   // Capture128 (deferred)
    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D_ARRAY, DummyTexArray);   // Capture256 (deferred)
    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, MosaicTex);

    glBindBuffer(GL_ARRAY_BUFFER, RectVtxBuffer);
    glBindVertexArray(RectVtxArray);
    glDrawArrays(GL_TRIANGLES, 0, 2 * 3);

    glDisable(GL_SCISSOR_TEST);

    // ── LAYER-PROBE (one-shot, engine A): separate the wash by layer ──
    // The sprite atlas is already proven correct, but the final output is washed
    // pastel. This reads THREE layers at the same points to localise the wash:
    //   OBJ  = OBJLayerTex[0]      (composited sprite colour, real RGB)
    //   3D   = GPU3D output tex    (BG0 source, real RGB)
    //   OUT  = OutputTex           (final; stored .bgr → logged un-swizzled as RGB)
    // Scans a vertical centre column (maps the scene + resolves any Y-flip) plus
    // the character row. Remove once the colour issue is fixed.
    if (num == 0)
    {
        static int outFrame = 0;
        static bool outProbe = false;
        if (unit->DispCnt == 0x00011F08 || (unit->DispCnt & 0x7) == 5) outFrame++;
        if (YAGE_MELONDS_GL_DIAG && !outProbe && outFrame >= 200)
        {
            outProbe = true;
            glFinish();
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

            GLuint tmpFBO; glGenFramebuffers(1, &tmpFBO);
            GLuint tex3d = (GLuint)GPU::Get3DOutputTexture();
            MELONDS_2D_LOG("LAYER-PROBE tex3d=%u (0=null read invalid)", tex3d);

            // Ground truth: raw BG palettes straight from memory (NDS BGR555).
            // If these are vibrant but OUT is washed → bug is in BG rendering.
            const u16* bgpal = (const u16*)&GPU::Palette[0x000];   // engine A BG std pal
            for (int e = 0; e < 8; e++)
            {
                u16 c = bgpal[e];
                MELONDS_2D_LOG("LAYER-PROBE bgStdPal[%d]=0x%04X rgb=(%d,%d,%d)", e, c,
                               (c & 0x1F) << 3, ((c >> 5) & 0x1F) << 3, ((c >> 10) & 0x1F) << 3);
            }
            // BG extended palettes (mode 5 hills use these): slots 0-3, pal 0, a few entries.
            for (int s = 0; s < 4; s++)
            {
                u16* ep = unit->GetBGExtPal(s, 0);
                if (!ep) { MELONDS_2D_LOG("LAYER-PROBE bgExtPal slot%d = NULL", s); continue; }
                MELONDS_2D_LOG("LAYER-PROBE bgExtPal slot%d pal0 e1..4: "
                               "0x%04X 0x%04X 0x%04X 0x%04X", s, ep[1], ep[2], ep[3], ep[4]);
            }
            // Read the prerendered BG hill layers directly (BGLayerTex[1..3]) at a
            // couple of texture coords — are they washed at the prerender stage?
            for (int L = 1; L <= 3; L++)
            {
                glBindFramebuffer(GL_READ_FRAMEBUFFER, BGLayerFB[num][L]);
                glReadBuffer(GL_COLOR_ATTACHMENT0);
                GLubyte a[4]={0}, b[4]={0};
                glReadPixels(128, 128, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, a);
                glReadPixels(64, 200, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, b);
                MELONDS_2D_LOG("LAYER-PROBE BGLayerTex[%d] (128,128)=(%u,%u,%u,%u) (64,200)=(%u,%u,%u,%u)",
                               L, a[0],a[1],a[2],a[3], b[0],b[1],b[2],b[3]);
            }

            // points: a centre column (y sweep) then the character row (x sweep)
            static const int pts[][2] = {
                {128, 16}, {128, 48}, {128, 80}, {128, 112}, {128, 144}, {128, 176},
                {26, 150}, {58, 150}, {85, 150}, {190, 150},
            };
            const int NP = 10;
            for (int p = 0; p < NP; p++)
            {
                int x = pts[p][0] * ScaleFactor;
                int y = pts[p][1] * ScaleFactor;
                GLubyte obj[4] = {0}, d3[4] = {0}, out[4] = {0};

                // OBJ sprite layer (2D array layer 0)
                glBindFramebuffer(GL_READ_FRAMEBUFFER, tmpFBO);
                glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                          OBJLayerTex[num], 0, 0);
                if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
                { glReadBuffer(GL_COLOR_ATTACHMENT0); glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, obj); }

                // 3D layer (BG0 source)
                if (tex3d)
                {
                    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                           GL_TEXTURE_2D, tex3d, 0);
                    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
                    { glReadBuffer(GL_COLOR_ATTACHMENT0); glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, d3); }
                }

                // final OutputTex (stored .bgr → un-swizzle to real RGB for logging)
                glBindFramebuffer(GL_READ_FRAMEBUFFER, OutputFB[num]);
                glReadBuffer(GL_COLOR_ATTACHMENT0);
                glReadPixels(x, y, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);

                MELONDS_2D_LOG("LAYER-PROBE (%3d,%3d) OBJ=(%u,%u,%u,%u) 3D=(%u,%u,%u,%u) OUTrgb=(%u,%u,%u)",
                               pts[p][0], pts[p][1],
                               obj[0],obj[1],obj[2],obj[3],
                               d3[0],d3[1],d3[2],d3[3],
                               out[2],out[1],out[0]);   // un-swizzle .bgr→rgb
            }
            glDeleteFramebuffers(1, &tmpFBO);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        }
    }
}

// ── Rendering: the GPU 2D pipeline drives the frame (C4.2) ─────────────────
// The fork's GPU calls, per frame: CheckWindows(line) (in StartScanline, before
// us) then DrawScanline(line,A), DrawScanline(line,B) for line 0..191;
// DrawSprites(line+1,*) one scanline ahead; VBlankEnd(A,B) at the start of the
// next frame. We capture each scanline's register state in DrawScanline /
// DrawSprites and run the whole GPU pipeline at the last visible scanline
// (line 191), once per unit, into OutputTex — then read it back into the CPU
// backbuffer for the libretro (software-upload) display path. The GL context is
// current here: the GL 3D renderer renders mid-RunFrame too (at VCount==215).

void GLRenderer2D::SetFramebuffer(u32* unitA, u32* unitB)
{
    Framebuffer[0] = unitA;
    Framebuffer[1] = unitB;

    int screenForUnit[2] = { -1, -1 };
    for (int fb = 0; fb < 2; fb++)
    {
        for (int screen = 0; screen < 2; screen++)
        {
            if (unitA && unitA == GPU::Framebuffer[fb][screen])
                screenForUnit[0] = screen;
            if (unitB && unitB == GPU::Framebuffer[fb][screen])
                screenForUnit[1] = screen;
        }
    }

    int newScreenUnit[2] = { OutputScreenUnit[0], OutputScreenUnit[1] };
    if (screenForUnit[0] >= 0)
        newScreenUnit[screenForUnit[0]] = 0;
    if (screenForUnit[1] >= 0)
        newScreenUnit[screenForUnit[1]] = 1;

    static int lastLogScreenUnit[2] = { -1, -1 };
    if (newScreenUnit[0] != OutputScreenUnit[0] ||
        newScreenUnit[1] != OutputScreenUnit[1])
    {
        OutputScreenUnit[0] = newScreenUnit[0];
        OutputScreenUnit[1] = newScreenUnit[1];
    }
    if (lastLogScreenUnit[0] != OutputScreenUnit[0] ||
        lastLogScreenUnit[1] != OutputScreenUnit[1])
    {
        lastLogScreenUnit[0] = OutputScreenUnit[0];
        lastLogScreenUnit[1] = OutputScreenUnit[1];
        MELONDS_2D_LOG("PRES-MAP top=u%d bottom=u%d", OutputScreenUnit[0], OutputScreenUnit[1]);
    }
}

void GLRenderer2D::BindOutputTexture(int unit)
{
    glBindTexture(GL_TEXTURE_2D, OutputTex[unit & 1]);
}

void GLRenderer2D::BindOutputTextureForScreen(int screen)
{
    BindOutputTexture(OutputScreenUnit[screen & 1]);
}

void GLRenderer2D::DrawScanline(u32 line, Unit* unit)
{
    if (line >= 192) return;

    // Capture this scanline's BG scroll / rotscale / window / mosaic / back
    // colour. Safe now that the SoftRenderer is out of the loop: CheckWindows()
    // (run by the GPU before us) sets Win0/1Active and we are its only
    // renderer-side consumer.
    UpdateScanlineConfig((int)line, unit);

    // All 192 scanlines captured for this unit → run the GPU pipeline.
    if (line == 191)
        RenderFrame(unit);
}

void GLRenderer2D::DrawSprites(u32 line, Unit* unit)
{
    if (line >= 192) return;
    // OBJ mosaic-snapped line (fork's equivalent of upstream GPU2D.OBJMosaicLine);
    // consumed by the Sprite FS during DoRenderSprites.
    SpriteScanlineConfig[unit->Num].uMosaicLine[line] = (s32)unit->OBJMosaicY;
}

void GLRenderer2D::VBlankEnd(Unit* unitA, Unit* unitB)
{
    // Display capture (2D engine → VRAM) is deferred (v1); the per-frame work
    // runs at DrawScanline(191) → RenderFrame.
    (void)unitA;
    (void)unitB;
}

void GLRenderer2D::RenderFrame(Unit* unit)
{
    const int num = unit->Num;

    GL2D_CHK("RenderFrame:enter");   // drain errors inherited from 3D/libretro

    // BG layers: derive config + upload referenced VRAM/palette + pre-render
    // each active layer into its AllBGLayer texture.
    PrerenderBGLayers(unit);   GL2D_CHK("PrerenderBGLayers");

    // Sprites: scan OAM, upload OBJ VRAM/palette, rasterise the atlas, then
    // composite the OBJ layer for all 192 scanlines.
    UpdateOAM(0, 192, unit);    GL2D_CHK("UpdateOAM");
    UploadOBJVRAM(unit);        GL2D_CHK("UploadOBJVRAM");
    UploadOBJPalette(unit);     GL2D_CHK("UploadOBJPalette");
    PrerenderSprites(unit);     GL2D_CHK("PrerenderSprites");
    LastSpriteLine[num] = 0;
    DoRenderSprites(192, unit);  GL2D_CHK("DoRenderSprites");

    // Composite BG + OBJ (+ GPU3D as BG0) into OutputTex.
    LastLine[num] = 0;
    RenderScreen(0, 192, unit);  GL2D_CHK("RenderScreen");

    // Present is handled by libretro/opengl.cpp sampling OutputTex directly.
    // Do not glReadPixels here: several GLES drivers reject readback from these
    // render targets and the CPU framebuffer is not part of the GL2D hot path.
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    GL2D_CHK("RenderFrame:present-ready");

    // TEMP one-shot per-unit state dump: tells us whether a black screen is a
    // forced-blank/disabled engine, no active layers, or a genuinely empty
    // composite (centerPx = middle pixel of the read-back screen).
    static bool dumped[2] = { false, false };
    if (YAGE_MELONDS_GL_DIAG && !dumped[num])
    {
        dumped[num] = true;
        MELONDS_2D_LOG("2D-STATE u%d DispCnt=%08X En=%d fblank=%d BGAct=%X NumSpr=%d "
                       "Out3D=%u OutFB=%u OutTex=%u",
                       num, unit->DispCnt, (int)unit->Enabled,
                       (int)!!(unit->DispCnt & (1u << 7)), BGLayerActive[num],
                       NumSprites[num], Output3DTex, OutputFB[num], OutputTex[num]);
    }
}

}
