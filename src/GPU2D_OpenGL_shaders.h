/*
    GPU2D_OpenGL_shaders.h — GLSL ES 3.00 shaders for the GPU 2D renderer.

    Ported from upstream melonDS src/OpenGL_shaders/2D*.glsl (desktop GLSL #version
    140) to GLSL ES 3.00 for the libretro/Android GLES3 build:
      * #version 300 es + explicit precision (k2DShaderHeader)
      * strict int->float fixes (GLSL ES 3.00 has no implicit int->float):
          - `/ 63`        -> `/ 63.0`
          - `? 0 : 1`     -> `? 0.0 : 1.0`   (float alpha assignments)
          - `vPosition*2` -> `vPosition*2.0`, `-1` -> `-1.0`, vec4 int args -> .0

    Phase C2 stages the LayerPre (background) shaders. C3.0 adds the sprite
    shaders (SpritePre + Sprite); the compositor shader is added in C4.

    This file is part of melonDS (GPLv3); see the project license.
*/

#pragma once

#define k2DShaderHeader \
    "#version 300 es\n" \
    "precision highp float;\n" \
    "precision highp int;\n" \
    "precision highp sampler2D;\n" \
    "precision highp usampler2D;\n"

// The Sprite fragment shader additionally samples the display-capture texture
// arrays (sampler2DArray), which need an explicit precision in GLSL ES 3.00.
#define k2DSpriteShaderHeader k2DShaderHeader \
    "precision highp sampler2DArray;\n"

// The Compositor fragment shader samples the OBJ/capture arrays (sampler2DArray)
// and the integer mosaic LUT (isampler2D) — both need explicit precision in
// GLSL ES 3.00.
#define k2DCompositorShaderHeader k2DShaderHeader \
    "precision highp sampler2DArray;\n" \
    "precision highp isampler2D;\n"

namespace GPU2D
{

// ── Background layer pre-pass ─────────────────────────────────────────────
// Renders one BG layer (text 16/256-colour, affine, extended tile/bitmap/direct)
// into an intermediate texture by sampling VRAM (integer texture) + palette.

static const char* const k2DLayerPreVS = k2DShaderHeader R"(
struct sBGConfig
{
    ivec2 Size;
    int Type;
    int PalOffset;
    int TileOffset;
    int MapOffset;
    bool Clamp;
};

layout(std140) uniform ubBGConfig
{
    int uVRAMMask;
    sBGConfig uBGConfig[4];
};

uniform int uCurBG;

in vec2 vPosition;

smooth out vec2 fTexcoord;

void main()
{
    gl_Position = vec4((vPosition * 2.0) - 1.0, 0.0, 1.0);
    fTexcoord = vPosition * vec2(uBGConfig[uCurBG].Size);
}
)";

static const char* const k2DLayerPreFS = k2DShaderHeader R"(
uniform usampler2D VRAMTex;
uniform sampler2D PalTex;

struct sBGConfig
{
    ivec2 Size;
    int Type;
    int PalOffset;
    int TileOffset;
    int MapOffset;
    bool Clamp;
};

layout(std140) uniform ubBGConfig
{
    int uVRAMMask;
    sBGConfig uBGConfig[4];
};

uniform int uCurBG;

smooth in vec2 fTexcoord;

out vec4 oColor;

vec4 GetBGPalEntry(int layer, int pal, int id)
{
    ivec2 coord = ivec2(id, uBGConfig[layer].PalOffset + pal);
    vec4 col = texelFetch(PalTex, coord, 0);
    col.rgb *= (62.0/63.0);
    col.g += (col.a * 1.0/63.0);
    return col;
}

int VRAMRead8(int addr)
{
    ivec2 coord = ivec2(addr & 0x3FF, (addr >> 10) & uVRAMMask);
    int val = int(texelFetch(VRAMTex, coord, 0).r);
    return val;
}

int VRAMRead16(int addr)
{
    ivec2 coord = ivec2(addr & 0x3FF, (addr >> 10) & uVRAMMask);
    int lo = int(texelFetch(VRAMTex, coord, 0).r);
    int hi = int(texelFetch(VRAMTex, coord+ivec2(1,0), 0).r);
    return lo | (hi << 8);
}

vec4 GetBGLayerPixel(int layer, ivec2 coord)
{
    vec4 ret = vec4(0.0);

    if (uBGConfig[layer].Type == 0)
    {
        // text - 16-color tiles
        int mapoffset = uBGConfig[layer].MapOffset +
            (((coord.x >> 3) & 0x1F) << 1) +
            (((coord.y >> 3) & 0x1F) << 6);

        if (uBGConfig[layer].Size.y == 512)
        {
            if (uBGConfig[layer].Size.x == 512)
                mapoffset += (((coord.x >> 8) & 0x1) << 11) + (((coord.y >> 8) & 0x1) << 12);
            else
                mapoffset += (((coord.y >> 8) & 0x1) << 11);
        }
        else if (uBGConfig[layer].Size.x == 512)
        {
            mapoffset += (((coord.x >> 8) & 0x1) << 11);
        }

        int mapval = VRAMRead16(mapoffset);
        int tileoffset = (uBGConfig[layer].TileOffset << 1) + ((mapval & 0x3FF) << 6);

        if ((mapval & (1<<10)) != 0) tileoffset += (7 - (coord.x & 0x7));
        else                         tileoffset += (coord.x & 0x7);
        if ((mapval & (1<<11)) != 0) tileoffset += ((7 - (coord.y & 0x7)) << 3);
        else                         tileoffset += ((coord.y & 0x7) << 3);

        int col = VRAMRead8(tileoffset >> 1);
        if ((tileoffset & 0x1) != 0) col >>= 4;
        else                         col &= 0xF;
        col += ((mapval >> 12) << 4);

        ret = GetBGPalEntry(layer, 0, col);
        ret.a = ((col & 0xF) == 0) ? 0.0 : 1.0;
    }
    else if (uBGConfig[layer].Type == 1)
    {
        // text - 256-color tiles
        int mapoffset = uBGConfig[layer].MapOffset +
            (((coord.x >> 3) & 0x1F) << 1) +
            (((coord.y >> 3) & 0x1F) << 6);

        if (uBGConfig[layer].Size.y == 512)
        {
            if (uBGConfig[layer].Size.x == 512)
                mapoffset += (((coord.x >> 8) & 0x1) << 11) + (((coord.y >> 8) & 0x1) << 12);
            else
                mapoffset += (((coord.y >> 8) & 0x1) << 11);
        }
        else if (uBGConfig[layer].Size.x == 512)
        {
            mapoffset += (((coord.x >> 8) & 0x1) << 11);
        }

        int mapval = VRAMRead16(mapoffset);
        int tileoffset = uBGConfig[layer].TileOffset + ((mapval & 0x3FF) << 6);

        if ((mapval & (1<<10)) != 0) tileoffset += (7 - (coord.x & 0x7));
        else                         tileoffset += (coord.x & 0x7);
        if ((mapval & (1<<11)) != 0) tileoffset += ((7 - (coord.y & 0x7)) << 3);
        else                         tileoffset += ((coord.y & 0x7) << 3);

        int col = VRAMRead8(tileoffset);
        int pal = (uBGConfig[layer].PalOffset != 0) ? (mapval >> 12) : 0;

        ret = GetBGPalEntry(layer, pal, col);
        ret.a = (col == 0) ? 0.0 : 1.0;
    }
    else if (uBGConfig[layer].Type == 2)
    {
        // affine - 256 color tiles
        int mapoffset = uBGConfig[layer].MapOffset +
            (coord.x >> 3) +
            ((coord.y >> 3) * (uBGConfig[layer].Size.x >> 3));

        int mapval = VRAMRead8(mapoffset);
        int tileoffset = uBGConfig[layer].TileOffset + (mapval << 6);
        tileoffset += ((coord.y & 0x7) << 3);
        tileoffset += (coord.x & 0x7);

        int col = VRAMRead8(tileoffset);
        ret = GetBGPalEntry(layer, 0, col);
        ret.a = (col == 0) ? 0.0 : 1.0;
    }
    else if (uBGConfig[layer].Type == 3)
    {
        // extended - 256 color tiles
        int mapoffset = uBGConfig[layer].MapOffset +
            (((coord.x >> 3) +
            ((coord.y >> 3) * (uBGConfig[layer].Size.x >> 3))) << 1);

        int mapval = VRAMRead16(mapoffset);
        int tileoffset = uBGConfig[layer].TileOffset + ((mapval & 0x3FF) << 6);

        if ((mapval & (1<<10)) != 0) tileoffset += (7 - (coord.x & 0x7));
        else                         tileoffset += (coord.x & 0x7);
        if ((mapval & (1<<11)) != 0) tileoffset += ((7 - (coord.y & 0x7)) << 3);
        else                         tileoffset += ((coord.y & 0x7) << 3);

        int col = VRAMRead8(tileoffset);
        int pal = (uBGConfig[layer].PalOffset != 0) ? (mapval >> 12) : 0;

        ret = GetBGPalEntry(layer, pal, col);
        ret.a = (col == 0) ? 0.0 : 1.0;
    }
    else if (uBGConfig[layer].Type == 4)
    {
        // extended - 256 color bitmap
        int mapoffset = uBGConfig[layer].MapOffset +
            coord.x +
            (coord.y * uBGConfig[layer].Size.x);

        int col = VRAMRead8(mapoffset);
        ret = GetBGPalEntry(layer, 0, col);
        ret.a = (col == 0) ? 0.0 : 1.0;
    }
    else if (uBGConfig[layer].Type == 5)
    {
        // extended - direct color bitmap
        int mapoffset = uBGConfig[layer].MapOffset +
            ((coord.x +
            (coord.y * uBGConfig[layer].Size.x)) << 1);

        int col = VRAMRead16(mapoffset);
        ret.r = float((col << 1) & 0x3E) / 63.0;
        ret.g = float((col >> 4) & 0x3E) / 63.0;
        ret.b = float((col >> 9) & 0x3E) / 63.0;
        ret.a = float(col >> 15);
    }

    return ret;
}

void main()
{
    oColor = GetBGLayerPixel(uCurBG, ivec2(fTexcoord));
}
)";

// ── Sprite pre-pass (C3.0) ────────────────────────────────────────────────
// Rasterises each active OBJ into a 1024×512 atlas texture (16×8 grid of 64×64
// cells, one per sprite index) by sampling OBJ VRAM + palette. Vertex attribs
// are integer (glVertexAttribIPointer): vPosition (ivec2, the 0/1 quad corner)
// and vSpriteIndex (int). Ported from upstream 2DSpritePre{VS,FS}.glsl; the
// std140 sOAM/ubSpriteConfig layout mirrors GLRenderer2D::sSpriteConfig.

#define k2DSpriteStructs R"(
struct sOAM
{
    ivec2 Position;
    bvec2 Flip;
    ivec2 Size;
    ivec2 BoundSize;
    int OBJMode;
    int Type;
    int PalOffset;
    int TileOffset;
    int TileStride;
    int Rotscale;
    int BGPrio;
    bool Mosaic;
};

layout(std140) uniform ubSpriteConfig
{
    int uVRAMMask;
    ivec4 uRotscale[32];
    sOAM uOAM[128];
};
)"

static const char* const k2DSpritePreVS = k2DShaderHeader k2DSpriteStructs R"(
in ivec2 vPosition;
in int vSpriteIndex;

flat out int fSpriteIndex;
smooth out vec2 fTexcoord;

void main()
{
    ivec2 sprpos = ivec2((vSpriteIndex & 0xF) * 64, (vSpriteIndex >> 4) * 64);
    ivec2 sprsize = uOAM[vSpriteIndex].Size;
    vec2 vtxpos = vec2(sprpos) + (vec2(vPosition) * vec2(sprsize));
    vec2 fbsize = vec2(1024.0, 512.0);

    gl_Position = vec4(((vtxpos * 2.0) / fbsize) - 1.0, 0.0, 1.0);
    fSpriteIndex = vSpriteIndex;
    fTexcoord = vec2(vPosition) * vec2(sprsize);
}
)";

static const char* const k2DSpritePreFS = k2DShaderHeader k2DSpriteStructs R"(
uniform usampler2D VRAMTex;
uniform sampler2D PalTex;

flat in int fSpriteIndex;
smooth in vec2 fTexcoord;

out vec4 oColor;

vec4 GetOBJPalEntry(int pal, int id)
{
    ivec2 coord = ivec2(id, pal);
    vec4 col = texelFetch(PalTex, coord, 0);
    col.rgb *= (62.0/63.0);
    col.g += (col.a * 1.0/63.0);
    return col;
}

int VRAMRead8(int addr)
{
    ivec2 coord = ivec2(addr & 0x3FF, (addr >> 10) & uVRAMMask);
    int val = int(texelFetch(VRAMTex, coord, 0).r);
    return val;
}

int VRAMRead16(int addr)
{
    ivec2 coord = ivec2(addr & 0x3FF, (addr >> 10) & uVRAMMask);
    int lo = int(texelFetch(VRAMTex, coord, 0).r);
    int hi = int(texelFetch(VRAMTex, coord+ivec2(1,0), 0).r);
    return lo | (hi << 8);
}

vec4 GetSpritePixel(int sprite, ivec2 coord)
{
    vec4 ret;

    if (uOAM[sprite].Type == 0)
    {
        // 16-color
        int tileoffset = uOAM[sprite].TileOffset +
            ((coord.x >> 3) * 32) +
            ((coord.y >> 3) * uOAM[sprite].TileStride) +
            ((coord.x & 0x7) >> 1) +
            ((coord.y & 0x7) << 2);

        int col = VRAMRead8(tileoffset);
        if ((coord.x & 1) != 0) col >>= 4;
        else                    col &= 0xF;
        col += uOAM[sprite].PalOffset;

        ret = GetOBJPalEntry(0, col);
        ret.a = ((col & 0xF) == 0) ? 0.0 : 1.0;
    }
    else if (uOAM[sprite].Type == 1)
    {
        // 256-color
        int tileoffset = uOAM[sprite].TileOffset +
            ((coord.x >> 3) * 64) +
            ((coord.y >> 3) * uOAM[sprite].TileStride) +
             (coord.x & 0x7) +
            ((coord.y & 0x7) << 3);

        int col = VRAMRead8(tileoffset);

        ret = GetOBJPalEntry(uOAM[sprite].PalOffset, col);
        ret.a = (col == 0) ? 0.0 : 1.0;
    }
    else //if (uOAM[sprite].Type == 2)
    {
        // direct color bitmap
        int tileoffset = uOAM[sprite].TileOffset +
            (coord.x * 2) +
            (coord.y * uOAM[sprite].TileStride);

        int col = VRAMRead16(tileoffset);

        ret.r = float((col << 1) & 0x3E) / 63.0;
        ret.g = float((col >> 4) & 0x3E) / 63.0;
        ret.b = float((col >> 9) & 0x3E) / 63.0;
        ret.a = float(col >> 15);
    }

    return ret;
}

void main()
{
    oColor = GetSpritePixel(fSpriteIndex, ivec2(fTexcoord));
}
)";

// ── Sprite composite (C3.0 shader; dispatched at C4) ──────────────────────
// Composites the sprite atlas into the OBJ layer at screen scale, applying
// rot/scale (sprite-centre coords), priority (depth), two-pass mosaic, OBJ
// window/semi-transparent/bitmap modes, and the display-capture bitmap OBJ
// types (3/4 → Capture256Tex). MRT: oColor + oFlags. uMosaicLine is declared
// ivec4[48] (packed) to match the tightly-packed s32[192] on the C++ side.
// Ported from upstream 2DSprite{VS,FS}.glsl.

static const char* const k2DSpriteVS = k2DShaderHeader k2DSpriteStructs R"(
in ivec2 vPosition;
in ivec2 vTexcoord;
in int vSpriteIndex;

flat out int fSpriteIndex;
smooth out vec2 fPosition;
smooth out vec2 fTexcoord;

void main()
{
    vec2 sprsize = vec2(uOAM[vSpriteIndex].BoundSize);
    vec2 fbsize = vec2(256.0, 192.0);

    int totalprio = (uOAM[vSpriteIndex].BGPrio * 128) + vSpriteIndex;
    float z = float(totalprio) / 512.0;
    gl_Position = vec4(((vec2(vPosition) * 2.0) / fbsize) - 1.0, z, 1.0);
    fPosition = vec2(vPosition);
    fSpriteIndex = vSpriteIndex;

    if (uOAM[vSpriteIndex].Rotscale == -1)
    {
        vec2 tmp = vec2(vTexcoord) * sprsize;
        fTexcoord = mix(tmp, (sprsize - tmp), uOAM[vSpriteIndex].Flip);
    }
    else
        fTexcoord = (vec2(vTexcoord) * sprsize) - (sprsize / 2.0);
}
)";

static const char* const k2DSpriteFS = k2DSpriteShaderHeader k2DSpriteStructs R"(
uniform sampler2D SpriteTex;
uniform sampler2DArray Capture128Tex;
uniform sampler2DArray Capture256Tex;

layout(std140) uniform ubSpriteScanlineConfig
{
    ivec4 uMosaicLine[48];
};

uniform bool uRenderTransparent;

flat in int fSpriteIndex;
smooth in vec2 fPosition;
smooth in vec2 fTexcoord;

layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oFlags;

vec4 GetSpritePixel(int sprite, vec2 coord)
{
    ivec2 basecoord = ivec2((sprite & 0xF) * 64, (sprite >> 4) * 64);
    return texelFetch(SpriteTex, basecoord + ivec2(coord), 0);
}

void main()
{
    vec4 col, flags = vec4(0.0);
    vec2 coord = fTexcoord;

    if (uOAM[fSpriteIndex].Mosaic)
    {
        int line = int(fPosition.y);
        int mosline = uMosaicLine[line>>2][line&0x3];

        float ymin = 0.0;
        if (uOAM[fSpriteIndex].Rotscale != -1)
            ymin = -float(uOAM[fSpriteIndex].Size.y) / 2.0;

        float mosy = coord.y - float(line - mosline);
        if (coord.y >= ymin)
            coord.y = max(mosy, ymin);
    }

    if (uOAM[fSpriteIndex].Rotscale != -1)
    {
        // rotscale sprite — fTexcoord is based on the sprite centre
        vec2 sprsize = vec2(uOAM[fSpriteIndex].Size);
        vec4 rotscale = vec4(uRotscale[uOAM[fSpriteIndex].Rotscale]) / 256.0;
        mat2 rsmatrix = mat2(rotscale.xy, rotscale.zw);
        coord = (coord * rsmatrix) + (sprsize / 2.0);
        if (any(lessThan(coord, vec2(0.0)))) discard;
        if (any(greaterThanEqual(coord, sprsize))) discard;
    }

    if (uRenderTransparent)
    {
        // set BG priority and mosaic flags for transparent pixels
        if (uOAM[fSpriteIndex].Mosaic)
            flags.g = 1.0;

        flags.a = float(uOAM[fSpriteIndex].BGPrio) / 255.0;

        oColor = vec4(0.0);
        oFlags = flags;
        return;
    }

    if (uOAM[fSpriteIndex].Type == 3)
    {
        coord += vec2(ivec2(uOAM[fSpriteIndex].TileOffset) >> ivec2(1, 8));
        coord *= (1.0/128.0);
        col = texture(Capture256Tex, vec3(fract(coord), float(uOAM[fSpriteIndex].TileStride)));
    }
    else if (uOAM[fSpriteIndex].Type == 4)
    {
        coord += vec2(ivec2(uOAM[fSpriteIndex].TileOffset) >> ivec2(1, 9));
        coord *= (1.0/256.0);
        col = texture(Capture256Tex, vec3(fract(coord), float(uOAM[fSpriteIndex].TileStride)));
    }
    else
    {
        col = GetSpritePixel(fSpriteIndex, coord);
    }

    if (col.a == 0.0) discard;

    // oFlags: r = sprite blending flag, g = mosaic, b = OBJ window, a = BG prio
    if (uOAM[fSpriteIndex].OBJMode == 2)
    {
        // OBJ window (OBJ mosaic doesn't apply to window sprites)
        flags.b = 1.0;
    }
    else
    {
        if (uOAM[fSpriteIndex].OBJMode == 1)
        {
            // semi-transparent sprite
            flags.r = 1.0 / 255.0;
        }
        else if (uOAM[fSpriteIndex].OBJMode == 3)
        {
            // bitmap sprite
            col.a = float(uOAM[fSpriteIndex].PalOffset) / 31.0;
            flags.r = 2.0 / 255.0;
        }

        if (uOAM[fSpriteIndex].Mosaic)
            flags.g = 1.0;

        flags.a = float(uOAM[fSpriteIndex].BGPrio) / 255.0;
    }

    oColor = col;
    oFlags = flags;
}
)";

// ── Compositor (C4.0) ─────────────────────────────────────────────────────
// Combines the 4 pre-rendered BG layers + the OBJ layer into the final screen,
// applying per-pixel priority, windows (Win0/1/OBJ/outside), colour special
// effects (alpha blend, brightness inc/dec, 3D/semi-transparent/bitmap-OBJ
// blends), horizontal BG/OBJ mosaic, and the back colour. BG0 is sampled from
// whatever is bound to BGLayerTex[0] — the GPU3D output texture when the 3D
// layer is enabled (wired by RenderScreen at C4). Ported from upstream
// 2DCompositor{VS,FS}.glsl (desktop GLSL 140) to GLSL ES 3.00:
//   * #version 300 es + precision (incl. sampler2DArray + isampler2D).
//   * vec2 * uScaleFactor → * float(uScaleFactor); / 256 → / 256.0;
//     bgpos.y += MapOffset(int) → += float(MapOffset); int-literal vec
//     constructors → .0 (vec4(0.0) etc.).
// Display-capture BG (Type ≥ 7) stays dead in the fork (UpdateLayerConfig caps
// BG types at 5/6), matching the BG/OBJ capture deferral; the Capture* samplers
// are bound to a dummy until capture lands. Dispatched at C4.

static const char* const k2DCompositorVS = k2DShaderHeader R"(
uniform int uScaleFactor;

in vec2 vPosition;

smooth out vec4 fTexcoord;

void main()
{
    gl_Position = vec4((vPosition * 2.0) - 1.0, 0.0, 1.0);
    fTexcoord.xy = vPosition * vec2(256.0, 192.0);
    fTexcoord.zw = fTexcoord.xy * float(uScaleFactor);
}
)";

static const char* const k2DCompositorFS = k2DCompositorShaderHeader R"(
uniform sampler2D BGLayerTex[4];
uniform sampler2DArray OBJLayerTex;
uniform sampler2DArray Capture128Tex;
uniform sampler2DArray Capture256Tex;
uniform isampler2D MosaicTex;

struct sBGConfig
{
    ivec2 Size;
    int Type;
    int PalOffset;
    int TileOffset;
    int MapOffset;
    bool Clamp;
};

layout(std140) uniform ubBGConfig
{
    int uVRAMMask;
    sBGConfig uBGConfig[4];
};

struct sScanline
{
    ivec2 BGOffset[4];
    ivec4 BGRotscale[2];
    int BackColor;
    uint WinRegs;
    int WinMask;
    int MasterBright;
    ivec4 WinPos;
    bvec4 BGMosaicEnable;
    ivec4 MosaicSize;
};

layout(std140) uniform ubScanlineConfig
{
    sScanline uScanline[192];
};

layout(std140) uniform ubCompositorConfig
{
    ivec4 uBGPrio;
    bool uEnableOBJ;
    bool uEnable3D;
    int uBlendCnt;
    int uBlendEffect;
    ivec3 uBlendCoef;
};

uniform int uScaleFactor;

smooth in vec4 fTexcoord;

out vec4 oColor;

int MosaicX = 0;

ivec3 ConvertColor(int col)
{
    ivec3 ret;
    ret.r = (col & 0x1F) << 1;
    ret.g = ((col & 0x3E0) >> 4) | (col >> 15);
    ret.b = (col & 0x7C00) >> 9;
    return ret;
}

vec4 BG0Fetch(vec2 coord)
{
    vec4 col = texture(BGLayerTex[0], coord);
    return uEnable3D ? col.bgra : col;
}
vec4 BG1Fetch(vec2 coord) { return texture(BGLayerTex[1], coord); }
vec4 BG2Fetch(vec2 coord) { return texture(BGLayerTex[2], coord); }
vec4 BG3Fetch(vec2 coord) { return texture(BGLayerTex[3], coord); }

vec4 BG0CalcAndFetch(vec2 coord, int line)
{
    ivec2 bgoffset = uScanline[line].BGOffset[0];
    vec2 bgpos = vec2(bgoffset.xy) + coord;

    if (uEnable3D)
    {
        ivec2 texsize = uBGConfig[0].Size * uScaleFactor;
        // GPU3D vertex shader maps NDS screen y=0 (top) → NDC y=-1 → texture y=0.
        // So texture y == NDS scanline index directly. bgpos.y = fract(fTexcoord.y) ≈ 0
        // for every scanline — using it would always sample row 0, hiding the castle.
        ivec2 texcoord = ivec2(
            int(floor(bgpos.x * float(uScaleFactor))),
            line * uScaleFactor
        );
        texcoord = clamp(texcoord, ivec2(0), texsize - ivec2(1));

        vec4 col = texelFetch(BGLayerTex[0], texcoord, 0);
        // GPU3D writes BGR into its RGBA target (FinalColor() returns col.bgra),
        // matching the old GL compositor path in GPU_OpenGL_shaders.h. Convert it
        // back to RGB here so BG0 enters the GL2D compositor in the same order as
        // BG1-3 and OBJ before CompositeLayers() does the final screen swizzle.
        return col.bgra;
    }

    if (uScanline[line].BGMosaicEnable[0])
        bgpos = floor(bgpos) - vec2(MosaicX, 0);

    return BG0Fetch(bgpos / vec2(uBGConfig[0].Size));
}

vec4 BG1CalcAndFetch(vec2 coord, int line)
{
    ivec2 bgoffset = uScanline[line].BGOffset[1];
    vec2 bgpos = vec2(bgoffset.xy) + coord;

    if (uScanline[line].BGMosaicEnable[1])
        bgpos = floor(bgpos) - vec2(MosaicX, 0);

    return BG1Fetch(bgpos / vec2(uBGConfig[1].Size));
}

vec4 BG2CalcAndFetch(vec2 coord, int line)
{
    ivec2 bgoffset = uScanline[line].BGOffset[2];
    vec2 bgpos;
    if (uBGConfig[2].Type >= 2)
    {
        // rotscale BG
        bgpos = vec2(bgoffset.xy) / 256.0;
        vec4 rotscale = vec4(uScanline[line].BGRotscale[0]) / 256.0;
        mat2 rsmatrix = mat2(rotscale.xy, rotscale.zw);
        bgpos = bgpos + (coord * rsmatrix);
    }
    else
    {
        // text-mode BG
        bgpos = vec2(bgoffset.xy) + coord;
    }

    if (uScanline[line].BGMosaicEnable[2])
        bgpos = floor(bgpos) - vec2(MosaicX, 0);

    if (uBGConfig[2].Type >= 7)
    {
        // hi-res capture
        bgpos.y += float(uBGConfig[2].MapOffset);
        vec3 capcoord = vec3(bgpos / vec2(uBGConfig[2].Size), uBGConfig[2].TileOffset);

        if (uBGConfig[2].Clamp)
        {
            if (any(lessThan(capcoord.xy, vec2(0.0))) || any(greaterThanEqual(capcoord.xy, vec2(1.0))))
                return vec4(0.0);
        }

        if (uBGConfig[2].Type == 7)
            return texture(Capture128Tex, capcoord);
        else
            return texture(Capture256Tex, capcoord);
    }

    return BG2Fetch(bgpos / vec2(uBGConfig[2].Size));
}

vec4 BG3CalcAndFetch(vec2 coord, int line)
{
    ivec2 bgoffset = uScanline[line].BGOffset[3];
    vec2 bgpos;
    if (uBGConfig[3].Type >= 2)
    {
        // rotscale BG
        bgpos = vec2(bgoffset.xy) / 256.0;
        vec4 rotscale = vec4(uScanline[line].BGRotscale[1]) / 256.0;
        mat2 rsmatrix = mat2(rotscale.xy, rotscale.zw);
        bgpos = bgpos + (coord * rsmatrix);
    }
    else
    {
        // text-mode BG
        bgpos = vec2(bgoffset.xy) + coord;
    }

    if (uScanline[line].BGMosaicEnable[3])
        bgpos = floor(bgpos) - vec2(MosaicX, 0);

    if (uBGConfig[3].Type >= 7)
    {
        // hi-res capture
        bgpos.y += float(uBGConfig[3].MapOffset);
        vec3 capcoord = vec3(bgpos / vec2(uBGConfig[3].Size), uBGConfig[3].TileOffset);

        if (uBGConfig[3].Clamp)
        {
            if (any(lessThan(capcoord.xy, vec2(0.0))) || any(greaterThanEqual(capcoord.xy, vec2(1.0))))
                return vec4(0.0);
        }

        if (uBGConfig[3].Type == 7)
            return texture(Capture128Tex, capcoord);
        else
            return texture(Capture256Tex, capcoord);
    }

    return BG3Fetch(bgpos / vec2(uBGConfig[3].Size));
}

void CalcSpriteMosaic(in ivec2 coord, out ivec4 objflags, out vec4 objcolor)
{
    for (int i = 0; i < 16; i++)
    {
        ivec2 curpos = ivec2(coord.x - 15 + i, coord.y);

        if (curpos.x < 0)
        {
            objflags = ivec4(0);
            objcolor = vec4(0.0);
        }
        else
        {
            int mosx = texelFetch(MosaicTex, ivec2(curpos.x, uScanline[curpos.y].MosaicSize.z), 0).r;
            vec4 color = texelFetch(OBJLayerTex, ivec3(curpos * uScaleFactor, 0), 0);
            ivec4 flags = ivec4(texelFetch(OBJLayerTex, ivec3(curpos * uScaleFactor, 1), 0) * 255.0);

            bool latch = false;
            if (mosx == 0)              latch = true;
            else if (flags.g == 0)      latch = true;
            else if (objflags.g == 0)   latch = true;
            else if (flags.a < objflags.a) latch = true;

            if (latch)
            {
                objflags = flags;
                objcolor = color;
            }
        }
    }
}

vec4 CompositeLayers()
{
    ivec2 coord = ivec2(fTexcoord.zw);
    vec2 bgcoord = vec2(fTexcoord.x, fract(fTexcoord.y));
    int xpos = int(fTexcoord.x);
    int line = int(fTexcoord.y);

    if (uScanline[line].MosaicSize.x > 0)
        MosaicX = texelFetch(MosaicTex, ivec2(bgcoord.x, uScanline[line].MosaicSize.x), 0).r;

    ivec4 col1 = ivec4(ConvertColor(uScanline[line].BackColor), 0x20);
    int mask1 = 0x20;
    ivec4 col2 = ivec4(0);
    int mask2 = 0;
    bool specialcase = false;

    vec4 layercol[6];
    layercol[0] = BG0CalcAndFetch(bgcoord, line);
    layercol[1] = BG1CalcAndFetch(bgcoord, line);
    layercol[2] = BG2CalcAndFetch(bgcoord, line);
    layercol[3] = BG3CalcAndFetch(bgcoord, line);

    ivec4 objflags;
    if (uScanline[line].MosaicSize.z > 0)
    {
        CalcSpriteMosaic(ivec2(fTexcoord.xy), objflags, layercol[4]);
    }
    else
    {
        layercol[4] = texelFetch(OBJLayerTex, ivec3(coord, 0), 0);
        layercol[5] = texelFetch(OBJLayerTex, ivec3(coord, 1), 0);
        objflags = ivec4(layercol[5] * 255.0);
    }

    int winmask = uScanline[line].WinMask;
    bool inside_win0, inside_win1;

    if (xpos < uScanline[line].WinPos[0])
        inside_win0 = ((winmask & (1<<0)) != 0);
    else if (xpos < uScanline[line].WinPos[1])
        inside_win0 = ((winmask & (1<<1)) != 0);
    else
        inside_win0 = ((winmask & (1<<2)) != 0);

    if (xpos < uScanline[line].WinPos[2])
        inside_win1 = ((winmask & (1<<3)) != 0);
    else if (xpos < uScanline[line].WinPos[3])
        inside_win1 = ((winmask & (1<<4)) != 0);
    else
        inside_win1 = ((winmask & (1<<5)) != 0);

    uint winregs = uScanline[line].WinRegs;
    uint winsel = winregs;
    if (objflags.b > 0)  winsel = winregs >> 8;
    if (inside_win1)     winsel = winregs >> 16;
    if (inside_win0)     winsel = winregs >> 24;

    for (int prio = 3; prio >= 0; prio--)
    {
        for (int bg = 3; bg >= 0; bg--)
        {
            if ((uBGPrio[bg] == prio) && (layercol[bg].a > 0.0) && ((winsel & (1u << uint(bg))) != 0u))
            {
                col2 = col1;
                mask2 = mask1 << 8;
                col1 = ivec4(layercol[bg] * 255.0) >> ivec4(2, 2, 2, 3);
                mask1 = (1 << bg);
                specialcase = (bg == 0) && uEnable3D;
            }
        }

        if (uEnableOBJ && (objflags.a == prio) && (layercol[4].a > 0.0) && ((winsel & (1u << 4)) != 0u))
        {
            col2 = col1;
            mask2 = mask1 << 8;
            col1 = ivec4(layercol[4] * 255.0) >> ivec4(2, 2, 2, 3);
            mask1 = (1 << 4);
            specialcase = (objflags.r != 0);
        }
    }

    int effect = 0;
    int eva, evb, evy = uBlendCoef[2];

    if (specialcase && (uBlendCnt & mask2) != 0)
    {
        if (mask1 == (1 << 0))
        {
            // 3D layer blending
            effect = 4;
            eva = (col1.a & 0x1F) + 1;
            evb = 32 - eva;
        }
        else if (objflags.r == 1)
        {
            // semi-transparent sprite
            effect = 1;
            eva = uBlendCoef[0];
            evb = uBlendCoef[1];
        }
        else //if (objflags.r == 2)
        {
            // bitmap sprite
            effect = 1;
            eva = col1.a;
            evb = 16 - eva;
        }
    }
    else if (((uBlendCnt & mask1) != 0) && ((winsel & (1u << 5)) != 0u))
    {
        effect = uBlendEffect;
        if (effect == 1)
        {
            if ((uBlendCnt & mask2) != 0)
            {
                eva = uBlendCoef[0];
                evb = uBlendCoef[1];
            }
            else
                effect = 0;
        }
    }

    if (effect == 1)
    {
        // blending
        col1 = ((col1 * eva) + (col2 * evb) + 0x8) >> 4;
        col1 = min(col1, 0x3F);
    }
    else if (effect == 2)
    {
        // brightness up
        col1 = col1 + ((((0x3F - col1) * evy) + 0x8) >> 4);
    }
    else if (effect == 3)
    {
        // brightness down
        col1 = col1 - (((col1 * evy) + 0x7) >> 4);
    }
    else if (effect == 4)
    {
        // 3D layer blending
        col1 = ((col1 * eva) + (col2 * evb) + 0x10) >> 5;
    }

    // Master brightness (reg 0x6C) — final whole-screen fade, applied after all
    // BG/OBJ/blend compositing. Mirrors SoftRenderer::ColorBrightnessUp/Down:
    // up   = c + (((0x3F - c) * f) >> 4),  down = c - ((c * f) >> 4),  f clamped 16.
    // (No +0x8/+0x7 rounding — that is the BLDY effect above, not master bright.)
    int mbMode   = (uScanline[line].MasterBright >> 14) & 0x3;
    int mbFactor = uScanline[line].MasterBright & 0x1F;
    if (mbFactor > 16) mbFactor = 16;
    if (mbMode == 1)
        col1.rgb = col1.rgb + (((ivec3(0x3F) - col1.rgb) * mbFactor) >> 4);
    else if (mbMode == 2)
        col1.rgb = col1.rgb - ((col1.rgb * mbFactor) >> 4);
    col1.rgb = clamp(col1.rgb, ivec3(0), ivec3(0x3F));

    // Swizzle RGB→BGR so the RGBA bytes in Framebuffer[] match the BGRA layout
    // the libretro screen shader expects (it does .bgr to correct SoftRenderer's
    // BGRA buffer uploaded as GL_RGBA). Our glReadPixels(GL_RGBA) path must match.
    return vec4(vec3(col1.bgr << 2) / 255.0, 1.0);
}

void main()
{
    oColor = CompositeLayers();
}
)";

}
