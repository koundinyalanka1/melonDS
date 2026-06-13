// M31 NEON fixed-point parity harness.
//
// Verifies the GPU3D.cpp NEON paths added in M29/M31 are BIT-EXACT against the
// scalar 20.12 fixed-point reference over randomized inputs, including
// overflow-adversarial magnitudes (INT32_MIN/MAX, +/-0x1000, products that
// overflow 32 bits and rely on the widening s64 accumulate + truncating >>12).
//
// Self-contained: copies of both implementations live here so the harness can
// be built without the melonDS tree's headers. If you change the scalar or
// NEON code in GPU3D.cpp, update the copies here and re-run.
//
// Build & run on any ARM host with NEON (Apple Silicon Mac, the Android TV via
// adb shell, a Pi, ...):
//   armv7:   clang++ -O2 -mfpu=neon -o neon_fixed_test neon_fixed_test.cpp && ./neon_fixed_test
//   aarch64: clang++ -O2 -o neon_fixed_test neon_fixed_test.cpp && ./neon_fixed_test
//
// Expected output: "ALL PASS" with 0 mismatches for every function.

#include <arm_neon.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

typedef int32_t s32;
typedef int64_t s64;

// ── scalar references (verbatim from GPU3D.cpp) ─────────────────────────────
static void MatrixMult3x3_ref(s32* m, const s32* s)
{
    s32 tmp[12];
    memcpy(tmp, m, 12*4);
    m[0] = ((s64)s[0]*tmp[0] + (s64)s[1]*tmp[4] + (s64)s[2]*tmp[8]) >> 12;
    m[1] = ((s64)s[0]*tmp[1] + (s64)s[1]*tmp[5] + (s64)s[2]*tmp[9]) >> 12;
    m[2] = ((s64)s[0]*tmp[2] + (s64)s[1]*tmp[6] + (s64)s[2]*tmp[10]) >> 12;
    m[3] = ((s64)s[0]*tmp[3] + (s64)s[1]*tmp[7] + (s64)s[2]*tmp[11]) >> 12;
    m[4] = ((s64)s[3]*tmp[0] + (s64)s[4]*tmp[4] + (s64)s[5]*tmp[8]) >> 12;
    m[5] = ((s64)s[3]*tmp[1] + (s64)s[4]*tmp[5] + (s64)s[5]*tmp[9]) >> 12;
    m[6] = ((s64)s[3]*tmp[2] + (s64)s[4]*tmp[6] + (s64)s[5]*tmp[10]) >> 12;
    m[7] = ((s64)s[3]*tmp[3] + (s64)s[4]*tmp[7] + (s64)s[5]*tmp[11]) >> 12;
    m[8] = ((s64)s[6]*tmp[0] + (s64)s[7]*tmp[4] + (s64)s[8]*tmp[8]) >> 12;
    m[9] = ((s64)s[6]*tmp[1] + (s64)s[7]*tmp[5] + (s64)s[8]*tmp[9]) >> 12;
    m[10] = ((s64)s[6]*tmp[2] + (s64)s[7]*tmp[6] + (s64)s[8]*tmp[10]) >> 12;
    m[11] = ((s64)s[6]*tmp[3] + (s64)s[7]*tmp[7] + (s64)s[8]*tmp[11]) >> 12;
}

static void MatrixScale_ref(s32* m, const s32* s)
{
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++)
            m[r*4+c] = ((s64)s[r]*m[r*4+c]) >> 12;
}

static void MatrixTranslate_ref(s32* m, const s32* s)
{
    m[12] += ((s64)s[0]*m[0] + (s64)s[1]*m[4] + (s64)s[2]*m[8]) >> 12;
    m[13] += ((s64)s[0]*m[1] + (s64)s[1]*m[5] + (s64)s[2]*m[9]) >> 12;
    m[14] += ((s64)s[0]*m[2] + (s64)s[1]*m[6] + (s64)s[2]*m[10]) >> 12;
    m[15] += ((s64)s[0]*m[3] + (s64)s[1]*m[7] + (s64)s[2]*m[11]) >> 12;
}

static void VertexTransform_ref(const s32* cm, const s32* v, s32* out)
{
    s64 vertex[4] = { v[0], v[1], v[2], 0x1000 };
    out[0] = (vertex[0]*cm[0] + vertex[1]*cm[4] + vertex[2]*cm[8] + vertex[3]*cm[12]) >> 12;
    out[1] = (vertex[0]*cm[1] + vertex[1]*cm[5] + vertex[2]*cm[9] + vertex[3]*cm[13]) >> 12;
    out[2] = (vertex[0]*cm[2] + vertex[1]*cm[6] + vertex[2]*cm[10] + vertex[3]*cm[14]) >> 12;
    out[3] = (vertex[0]*cm[3] + vertex[1]*cm[7] + vertex[2]*cm[11] + vertex[3]*cm[15]) >> 12;
}

// ── NEON implementations (verbatim from GPU3D.cpp M31 paths) ────────────────
static void MatrixMult3x3_neon(s32* m, const s32* s)
{
    s32 tmpn[12];
    memcpy(tmpn, m, 12*4);
    const int32x2_t t[6] = {
        vld1_s32(&tmpn[0]), vld1_s32(&tmpn[2]),
        vld1_s32(&tmpn[4]), vld1_s32(&tmpn[6]),
        vld1_s32(&tmpn[8]), vld1_s32(&tmpn[10]),
    };
    for (int r = 0; r < 3; r++)
    {
        int64x2_t lo = vmull_n_s32(t[0], s[r*3+0]);
        int64x2_t hi = vmull_n_s32(t[1], s[r*3+0]);
        lo = vmlal_n_s32(lo, t[2], s[r*3+1]); hi = vmlal_n_s32(hi, t[3], s[r*3+1]);
        lo = vmlal_n_s32(lo, t[4], s[r*3+2]); hi = vmlal_n_s32(hi, t[5], s[r*3+2]);
        vst1_s32(&m[r*4+0], vshrn_n_s64(lo, 12));
        vst1_s32(&m[r*4+2], vshrn_n_s64(hi, 12));
    }
}

static void MatrixScale_neon(s32* m, const s32* s)
{
    for (int r = 0; r < 3; r++)
    {
        int64x2_t lo = vmull_n_s32(vld1_s32(&m[r*4+0]), s[r]);
        int64x2_t hi = vmull_n_s32(vld1_s32(&m[r*4+2]), s[r]);
        vst1_s32(&m[r*4+0], vshrn_n_s64(lo, 12));
        vst1_s32(&m[r*4+2], vshrn_n_s64(hi, 12));
    }
}

static void MatrixTranslate_neon(s32* m, const s32* s)
{
    int64x2_t lo = vmull_n_s32(vld1_s32(&m[0]), s[0]);
    int64x2_t hi = vmull_n_s32(vld1_s32(&m[2]), s[0]);
    lo = vmlal_n_s32(lo, vld1_s32(&m[4]), s[1]);
    hi = vmlal_n_s32(hi, vld1_s32(&m[6]), s[1]);
    lo = vmlal_n_s32(lo, vld1_s32(&m[8]), s[2]);
    hi = vmlal_n_s32(hi, vld1_s32(&m[10]), s[2]);
    vst1_s32(&m[12], vadd_s32(vld1_s32(&m[12]), vshrn_n_s64(lo, 12)));
    vst1_s32(&m[14], vadd_s32(vld1_s32(&m[14]), vshrn_n_s64(hi, 12)));
}

static void VertexTransform_neon(const s32* cm, const s32* v, s32* out)
{
    int64x2_t lo = vmull_n_s32(vld1_s32(&cm[0]), v[0]);
    int64x2_t hi = vmull_n_s32(vld1_s32(&cm[2]), v[0]);
    lo = vmlal_n_s32(lo, vld1_s32(&cm[4]), v[1]);
    hi = vmlal_n_s32(hi, vld1_s32(&cm[6]), v[1]);
    lo = vmlal_n_s32(lo, vld1_s32(&cm[8]), v[2]);
    hi = vmlal_n_s32(hi, vld1_s32(&cm[10]), v[2]);
    lo = vmlal_n_s32(lo, vld1_s32(&cm[12]), 0x1000);
    hi = vmlal_n_s32(hi, vld1_s32(&cm[14]), 0x1000);
    vst1_s32(&out[0], vshrn_n_s64(lo, 12));
    vst1_s32(&out[2], vshrn_n_s64(hi, 12));
}

// ── harness ──────────────────────────────────────────────────────────────────
int main()
{
    std::mt19937 rng(0xD55EEDu);
    // Mix of realistic 20.12 magnitudes and adversarial extremes.
    auto rnd = [&](int phase) -> s32 {
        switch (phase % 5)
        {
        case 0: return (s32)rng();                                    // full range
        case 1: return (s32)(rng() & 0x3FFF) - 0x2000;                // small fixed
        case 2: return (rng() & 1) ? INT32_MAX : INT32_MIN;           // extremes
        case 3: return (rng() & 1) ? 0x1000 : -0x1000;                // +/-1.0
        default: return (s32)(rng() & 0xFFFFFF) - 0x800000;           // mid
        }
    };

    const int N = 1000000;
    long fails3x3 = 0, failsScale = 0, failsTrans = 0, failsVtx = 0;

    for (int i = 0; i < N; i++)
    {
        s32 m0[16], s[12], v[3];
        for (int k = 0; k < 16; k++) m0[k] = rnd(i + k);
        for (int k = 0; k < 12; k++) s[k]  = rnd(i + k + 7);
        for (int k = 0; k < 3; k++)  v[k]  = rnd(i + k + 3);

        s32 a[16], b[16];

        memcpy(a, m0, sizeof(a)); memcpy(b, m0, sizeof(b));
        MatrixMult3x3_ref(a, s); MatrixMult3x3_neon(b, s);
        if (memcmp(a, b, sizeof(a))) fails3x3++;

        memcpy(a, m0, sizeof(a)); memcpy(b, m0, sizeof(b));
        MatrixScale_ref(a, s); MatrixScale_neon(b, s);
        if (memcmp(a, b, sizeof(a))) failsScale++;

        memcpy(a, m0, sizeof(a)); memcpy(b, m0, sizeof(b));
        MatrixTranslate_ref(a, s); MatrixTranslate_neon(b, s);
        if (memcmp(a, b, sizeof(a))) failsTrans++;

        s32 oa[4], ob[4];
        VertexTransform_ref(m0, v, oa); VertexTransform_neon(m0, v, ob);
        if (memcmp(oa, ob, sizeof(oa))) failsVtx++;
    }

    printf("MatrixMult3x3:   %ld mismatches / %d\n", fails3x3, N);
    printf("MatrixScale:     %ld mismatches / %d\n", failsScale, N);
    printf("MatrixTranslate: %ld mismatches / %d\n", failsTrans, N);
    printf("VertexTransform: %ld mismatches / %d\n", failsVtx, N);
    const bool pass = !fails3x3 && !failsScale && !failsTrans && !failsVtx;
    printf(pass ? "ALL PASS\n" : "FAILURES — NEON paths are NOT bit-exact\n");
    return pass ? 0 : 1;
}
