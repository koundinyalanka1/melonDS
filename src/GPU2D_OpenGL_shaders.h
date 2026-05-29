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

}
