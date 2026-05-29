/*
    GPU2D_OpenGL_shaders.h — GLSL ES 3.00 shaders for the GPU 2D renderer.

    Ported from upstream melonDS src/OpenGL_shaders/2D*.glsl (desktop GLSL #version
    140) to GLSL ES 3.00 for the libretro/Android GLES3 build:
      * #version 300 es + explicit precision (k2DShaderHeader)
      * strict int->float fixes (GLSL ES 3.00 has no implicit int->float):
          - `/ 63`        -> `/ 63.0`
          - `? 0 : 1`     -> `? 0.0 : 1.0`   (float alpha assignments)
          - `vPosition*2` -> `vPosition*2.0`, `-1` -> `-1.0`, vec4 int args -> .0

    Phase C2 stages the LayerPre (background) shaders. Sprite/compositor shaders
    are added in C3/C4.

    This file is part of melonDS (GPLv3); see the project license.
*/

#pragma once

#define k2DShaderHeader \
    "#version 300 es\n" \
    "precision highp float;\n" \
    "precision highp int;\n" \
    "precision highp sampler2D;\n" \
    "precision highp usampler2D;\n"

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

}
