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

    void BindOutputTexture(int unit);
    void BindOutputTextureForScreen(int screen);
    // M27: execute any deferred RenderFrame(s) queued by DrawScanline(191).
    // Called from the libretro GL present thread so the compositor runs with
    // the GL context current on that thread.
    void RenderPending();

    // Size the (upscaled) OBJ render target + compositor output textures to
    // `scale`×. Called by GPU::SetRenderSettings when the GL scale option
    // changes so OutputTex stays in sync with the GPU3D framebuffer size.
    void SetScaleFactor(int scale);

private:
    // ── GL setup (C2.1) ───────────────────────────────────────────────────
    bool InitShaders();
    bool InitResources();   // textures / FBOs / UBOs for both units
    void DeInitGL();

    // ── Register capture (C2.2) ───────────────────────────────────────────
    // Capture per-scanline DS 2D register state for `unit` into
    // ScanlineConfig[unit->Num].uScanline[line]. NOTE: mutates the unit's
    // Win0Active/Win1Active window-tracking state, which the SoftRenderer also
    // owns — so this is only safe to call once the GPU render path has replaced
    // the software passthrough (C4); not wired during passthrough.
    void UpdateScanlineConfig(int line, Unit* unit);

    // Derive per-BG layer config (size/type/offsets/palette) + VRAM ranges from
    // the control registers; selects the AllBGLayer texture/FB per layer and
    // uploads LayerConfig[unit->Num] to its UBO. Display-capture BG types 7/8
    // are deferred (treated as direct-colour bitmap, Type 5) until C4.
    void UpdateLayerConfig(Unit* unit);

    // ── Background prerender (C2.3) ───────────────────────────────────────
    // Build the intermediate BG layer textures for `unit`: derive the layer
    // config, upload the VRAM regions + palette the active layers reference,
    // then run the LayerPre program once per active text/affine/extended/bitmap
    // layer into its selected AllBGLayer texture. The 3D layer (Type 6) is
    // skipped here (the compositor samples GPU3D's output at C4). Like the
    // config-capture helpers above, this is NOT yet wired into the render path
    // — the software passthrough still drives displayed pixels until C4.
    void PrerenderBGLayers(Unit* unit);
    void PrerenderLayer(int layer, Unit* unit);
    void UploadBGVRAM(Unit* unit);     // VRAM regions referenced by active layers
    void UploadBGPalette(Unit* unit);  // standard + extended BG palette (swizzled)

    // ── Sprites (C3.1) ────────────────────────────────────────────────────
    // Ported from upstream GLRenderer2D's OBJ pipeline, adapted to the fork's
    // per-Unit API. Like the BG helpers above, these are compiled + validated
    // but NOT yet wired into the render path — software passthrough still drives
    // displayed pixels until C4. OBJ display-capture (sprite Type 3/4) is
    // deferred (the fork has no GetCaptureInfo_OBJ), mirroring the BG capture
    // deferral; bitmap sprites stay direct-colour (Type 2) until C4.
    //
    // Scan OAM for `unit`, build the per-OBJ config (SpriteConfig[unit->Num]),
    // and upload it to the sprite UBO. Sets NumSprites / SpriteUseMosaic.
    void UpdateOAM(int ystart, int yend, Unit* unit);
    // Rasterise each non-bitmap OBJ into the 1024×512 sprite atlas (SpriteTex).
    void PrerenderSprites(Unit* unit);
    void UploadOBJVRAM(Unit* unit);    // OBJ VRAM (whole) → integer texture
    void UploadOBJPalette(Unit* unit); // standard + ext OBJ palette (swizzled)
    // Composite the atlas into the OBJ layer (OBJLayerTex) for lines
    // [LastSpriteLine, line): two-pass mosaic + window/opaque passes via depth.
    void DoRenderSprites(int line, Unit* unit);
    // One sprite draw pass (window=OBJ-window sprites, else normal sprites).
    void RenderSprites(bool window, int ystart, int yend, Unit* unit);

    // ── Compositor (C4.1) ─────────────────────────────────────────────────
    // Derive the per-frame compositor config (BG priorities, OBJ/3D enable,
    // blend mode + coefficients) from `unit`'s registers and upload it.
    void UpdateCompositorConfig(Unit* unit);
    // Composite BG + OBJ layers for scanlines [ystart, yend) into OutputTex,
    // applying windows / colour effects / mosaic / back colour. Samples the
    // GPU3D output (Output3DTex) as BG0 when the 3D layer is enabled. This is
    // where the GPU 2D pipeline produces visible pixels — driven at C4.2.
    void RenderScreen(int ystart, int yend, Unit* unit);

    // ── Frame driver (C4.2) ───────────────────────────────────────────────
    // Run the whole GPU 2D pipeline for `unit` for the frame whose per-scanline
    // config has been captured (BG prerender → sprite prerender + composite →
    // screen composite → OutputTex). Called from DrawScanline at the last
    // visible scanline. The libretro GL present path samples OutputTex directly;
    // there is no GPU→CPU readback in the hot path.
    void RenderFrame(Unit* unit);

    bool GLReady = false;

    // LayerPre program (shared between units). The fork's
    // OpenGL::BuildShaderProgram fills a GLuint[3] = {vs, fs, program}.
    GLuint LayerPreShader[3] = {0, 0, 0};
    GLint  LayerPreCurBGULoc = -1;

    // fullscreen [0,1] rect that drives the LayerPre program (shared, not
    // per-unit). vPosition spans 0..1; the VS maps it to clip space and to
    // layer-sized texcoords.
    GLuint RectVtxBuffer = 0;
    GLuint RectVtxArray  = 0;

    // ── Sprite programs (C3.1, shared between units) ──────────────────────
    GLuint SpritePreShader[3]   = {0, 0, 0};   // rasterise OBJ → atlas
    GLuint SpriteShader[3]      = {0, 0, 0};   // composite atlas → OBJ layer
    GLint  SpriteRenderTransULoc = -1;

    // ── Compositor program (C4.1, shared between units) ───────────────────
    GLuint CompositorShader[3]  = {0, 0, 0};   // BG+OBJ → OutputTex
    GLint  CompositorScaleULoc  = -1;

    // OBJ→atlas vertex stream (2× pos + sprite index, per OBJ), and the
    // atlas→OBJ-layer stream (2× pos + 2× texcoord + index). Shared scratch
    // reused per unit (each unit is processed to completion before the next).
    GLuint SpritePreVtxBuffer = 0;
    GLuint SpritePreVtxArray  = 0;
    u16*   SpritePreVtxData   = nullptr;
    GLuint SpriteVtxBuffer    = 0;
    GLuint SpriteVtxArray     = 0;
    u16*   SpriteVtxData      = nullptr;

    // 16×256 mosaic lookup texture (R8I), shared. Built in InitShaders.
    GLuint MosaicTex = 0;

    // 1×1 dummy 2D-array bound to the Sprite FS's deferred capture samplers
    // (Capture128/256Tex) so DoRenderSprites stays GL-valid until C4 adds
    // real display-capture textures.
    GLuint DummyTexArray = 0;

    // upscale factor for the OBJ render target + compositor output.
    int ScaleFactor = 1;
    int ScreenW = 256;
    int ScreenH = 192;

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
        u32 MasterBright;       // reg 0x6C, applied to the final composited pixel
        s32 WinPos[4];
        u32 BGMosaicEnable[4];
        s32 MosaicSize[4];
    };
    struct sScanlineConfig
    {
        sScanline uScanline[192];
    } ScanlineConfig[2];

    // std140 layout for the sprite shaders' ubSpriteConfig (matches the
    // sOAM/ubSpriteConfig block in GPU2D_OpenGL_shaders.h). The shader reads
    // Flip/Mosaic as bvec2/bool — stored as 0/1 ints here. sOAM is 64 bytes
    // (std140 array stride 64); uRotscale[32] ivec4 follows uVRAMMask+pad[3].
    struct sSpriteConfig
    {
        u32 uVRAMMask;
        u32 __pad0[3];
        s32 uRotscale[32][4];
        struct sOAM
        {
            s32 Position[2];
            s32 Flip[2];
            s32 Size[2];
            s32 BoundSize[2];
            u32 OBJMode;
            u32 Type;
            u32 PalOffset;
            u32 TileOffset;
            u32 TileStride;
            u32 Rotscale;
            u32 BGPrio;
            u32 Mosaic;
        } uOAM[128];
    } SpriteConfig[2];

    // per-scanline OBJ mosaic line (the mosaic-snapped Y for each scanline).
    // Shader declares this packed as ivec4[48]; s32[192] is the same 768 bytes.
    struct sSpriteScanlineConfig
    {
        s32 uMosaicLine[192];
    } SpriteScanlineConfig[2];

    int  NumSprites[2]     = {0, 0};
    bool SpriteUseMosaic[2] = {false, false};

    // std140 layout for the compositor's ubCompositorConfig (matches the block
    // in GPU2D_OpenGL_shaders.h: ivec4 uBGPrio; bool×2; int×2; ivec3 coef).
    struct sCompositorConfig
    {
        u32 uBGPrio[4];
        u32 uEnableOBJ;
        u32 uEnable3D;
        u32 uBlendCnt;
        u32 uBlendEffect;
        u32 uBlendCoef[4];   // [0]=EVA [1]=EVB [2]=EVY ([3] pads to vec4)
    } CompositorConfig[2];

    // ── Per-unit GL resources (indexed by unit->Num) ──────────────────────
    GLuint VRAMTex_BG[2]        = {0, 0};
    GLuint PalTex_BG[2]         = {0, 0};
    GLuint LayerConfigUBO[2]    = {0, 0};
    GLuint ScanlineConfigUBO[2] = {0, 0};

    // pre-rendered BG layer textures: all possible sizes (22), per unit.
    GLuint AllBGLayerTex[2][22] = {};
    GLuint AllBGLayerFB[2][22]  = {};

    // the AllBGLayer texture/FB selected for each of the 4 active BG layers,
    // and the VRAM tile/map ranges each layer reads (for upload in C2.3).
    GLuint BGLayerTex[2][4] = {};
    GLuint BGLayerFB[2][4]  = {};
    u32    BGVRAMRange[2][4][4] = {};

    // bitmask of BG layers that UpdateLayerConfig prepared a prerendered layer
    // texture for (i.e. active, non-3D). Drives which layers PrerenderBGLayers
    // uploads VRAM for and dispatches the LayerPre program over.
    u32    BGLayerActive[2] = {0, 0};

    // ── Per-unit OBJ GL resources (indexed by unit->Num) ──────────────────
    GLuint VRAMTex_OBJ[2]            = {0, 0};   // raw OBJ VRAM, R8UI
    GLuint PalTex_OBJ[2]             = {0, 0};   // std + 16 ext-pal, RGB5_A1
    GLuint SpriteConfigUBO[2]        = {0, 0};
    GLuint SpriteScanlineConfigUBO[2] = {0, 0};

    // 1024×512 RGBA8 atlas (16×8 grid of 64×64 cells) the OBJs prerender into.
    GLuint SpriteTex[2] = {0, 0};
    GLuint SpriteFB[2]  = {0, 0};

    // upscaled OBJ layer: 2D-array, layer 0 = colour, layer 1 = flags (MRT),
    // plus a depth texture for priority resolution.
    GLuint OBJLayerTex[2]  = {0, 0};
    GLuint OBJDepthTex[2]  = {0, 0};
    GLuint OBJLayerFB[2]   = {0, 0};

    // first sprite scanline composited so far this frame (per unit).
    int    LastSpriteLine[2] = {0, 0};

    // ── Compositor output, per unit (C4.1) ────────────────────────────────
    GLuint CompositorConfigUBO[2] = {0, 0};
    GLuint OutputTex[2] = {0, 0};   // final composited screen (RGBA8, scaled)
    GLuint OutputFB[2]  = {0, 0};
    int    OutputScreenUnit[2] = {0, 1};
    // first scanline not yet composited this frame (per unit).
    int    LastLine[2]  = {0, 0};

    // GPU3D's colour output, sampled as BG0 when the 3D layer is enabled. Set
    // by the driver at C4.2 (= GPU3D_OpenGL FramebufferTex[GPU::FrontBuffer]);
    // 0 until then. RenderScreen binds it to BGLayerTex[0] when DispCnt&(1<<3).
    GLuint Output3DTex = 0;

    // ── Palette upload caching (frame-to-frame change detection) ──────────
    // Palettes are often static between frames (e.g. during scroll-only
    // animations). The GPU upload (swizzle loop + glTexSubImage2D) is skipped
    // when the source palette data hasn't changed since the last upload.
    // bionic's memcmp is NEON-vectorised so the check is cheap (~7 µs per
    // buffer) vs the upload (~350 µs per unit). The previous-frame buffers
    // are always invalidated on Init to force the first frame to upload.
    static constexpr int kBGPalEntries  = 256 * (1 + 4 * 16); // std + 4×16 ext-pal rows
    static constexpr int kOBJPalEntries = 256 * (1 + 16);     // std + 16 ext-pal rows
    static constexpr int kOBJVRAMBytes  = 256 * 1024;         // engine A max OBJ VRAM
    u16 PrevPalBG[2][kBGPalEntries]   = {};
    u16 PrevPalOBJ[2][kOBJPalEntries] = {};
    bool PalBGValid[2]  = {false, false};
    bool PalOBJValid[2] = {false, false};
    u8   PrevOBJVRAM[2][kOBJVRAMBytes] = {};
    u32  PrevOBJVRAMBytes[2]           = {0, 0};
    bool OBJVRAMValid[2]               = {false, false};

    // C1/C2 passthrough backend (correctness fallback until C4 closes the loop).
    SoftRenderer Soft;
};

}
