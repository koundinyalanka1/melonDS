/*
    Copyright 2016-2021 Arisotura, RSDuck

    This file is part of melonDS.

    melonDS is free software: you can redistribute it and/or modify it under
    the terms of the GNU General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option)
    any later version.

    melonDS is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
    FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with melonDS. If not, see http://www.gnu.org/licenses/.
*/

#ifndef ARMJIT_MEMORY
#define ARMJIT_MEMORY

#include "types.h"

#include "ARM.h"

namespace ARMJIT_Memory
{

extern void* FastMem9Start;
extern void* FastMem7Start;

void Init();
void DeInit();

void Reset();

enum
{
    memregion_Other = 0,
    memregion_ITCM,
    memregion_DTCM,
    memregion_BIOS9,
    memregion_MainRAM,
    memregion_SharedWRAM,
    memregion_IO9,
    memregion_VRAM,
    memregion_BIOS7,
    memregion_WRAM7,
    memregion_IO7,
    memregion_Wifi,
    memregion_VWRAM,

    // DSi
    memregion_BIOS9DSi,
    memregion_BIOS7DSi,
    memregion_NewSharedWRAM_A,
    memregion_NewSharedWRAM_B,
    memregion_NewSharedWRAM_C,

    memregions_Count
};

int ClassifyAddress9(u32 addr);
int ClassifyAddress7(u32 addr);

bool GetMirrorLocation(int region, u32 num, u32 addr, u32& memoryOffset, u32& mirrorStart, u32& mirrorSize);
u32 LocaliseAddress(int region, u32 num, u32 addr);

bool IsFastmemCompatible(int region);

void RemapDTCM(u32 newBase, u32 newSize);
void RemapSWRAM();
void RemapNWRAM(int num);

void SetCodeProtection(int region, u32 offset, bool protect);

void* GetFuncForAddr(ARM* cpu, u32 addr, bool store, int size);

/* ── M31: A32 fastmem-lite page tables ────────────────────────────────────
 * Per-CPU, per-direction 16 KB-granule guest→host translation tables covering
 * guest addresses [0, 0x04000000) — every direct-safe region (ITCM, DTCM,
 * MainRAM, Shared WRAM, WRAM7, BIOS7) lives below 0x04000000; everything at or
 * above it (IO/VRAM/...) is never direct-safe and is deflected by the JIT's
 * existing high-address fast exit.
 *
 * Entry semantics: table[addr >> 14] is either 0 ("not direct — take the
 * helper") or (hostPagePtr - guestPageBase), so the JIT computes
 *      hostEA = entry + guestAddr
 * with a single ADD — no masking, no per-region guards, and DTCM-overlap
 * correctness BY CONSTRUCTION (the table is rebuilt from ClassifyAddress9/7,
 * which resolves DTCM before MainRAM; pages straddling a non-page-aligned
 * DTCM window are left 0).
 *
 * Write entries are additionally 0 for read-only regions (BIOS7) and for any
 * page whose 16 KB contains compiled code (so SMC invalidation always runs in
 * the helper). FastMemLiteRebuild() must be called after anything that changes
 * the guest memory map or code protection: ARMJIT_Memory::Reset, RemapDTCM /
 * RemapSWRAM / RemapNWRAM consumers (CP15 TCM updates, WRAMCNT writes), and
 * SetCodeProtection transitions. Single-threaded: only the emulation thread
 * reads or rebuilds the tables. */
constexpr u32 kFastMemLitePageShift = 14;
constexpr u32 kFastMemLitePages     = 0x04000000 >> kFastMemLitePageShift; // 4096
extern u32 FastMemLiteRead[2][kFastMemLitePages];
extern u32 FastMemLiteWrite[2][kFastMemLitePages];
void FastMemLiteRebuild();

}

#endif