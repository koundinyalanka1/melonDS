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

#ifndef ARMJIT_A32_COMPILER_H
#define ARMJIT_A32_COMPILER_H

#include "../ARMJIT.h"
#include "../ARMJIT_Internal.h"

namespace ARMJIT
{

class Compiler
{
public:
    Compiler();
    ~Compiler();

    void Reset();

    JitBlockEntry CompileBlock(ARM* cpu, bool thumb, FetchedInstr instrs[],
                               int instrsCount, bool hasMemoryInstr);

    bool CanCompile(bool thumb, u16 kind);

    JitBlockEntry AddEntryOffset(u32 offset)
    {
        return (JitBlockEntry)(CodeStart + offset);
    }

    u32 SubEntryOffset(JitBlockEntry entry)
    {
        return (u8*)entry - CodeStart;
    }

    bool IsJITFault(u8* pc);
    u8* RewriteMemAccess(u8* pc);

    // ── M31: cross-block direct linking ──────────────────────────────────────
    // A block whose tail is an unconditional static branch ends in a "link
    // tail": the same StopExecution + event-horizon checks as the M18
    // self-link, then `LDR ip,[pc,#..]; BX ip` through a patchable data
    // literal.  Unlinked, the literal holds &ARM_Ret (normal dispatcher
    // round-trip); linked, it holds the target block's EntryPoint.  Patching
    // is a single u32 data store (the literal is data-loaded, so no icache
    // maintenance).  ARMJIT.cpp owns registration/backpatch/unpatch.
    struct LinkSiteRecord
    {
        u8* literal;     // address of the patchable .word inside the link tail
        u32 targetAddr;  // guest block address the tail branches to (ARM9)
    };
    // Moves the sites recorded while compiling the last block into `out`
    // (at most `max`); returns the count and clears the internal list.
    int TakeLinkSites(LinkSiteRecord* out, int max);

private:
    // ── Code buffer ──────────────────────────────────────────────────────────
    u8* CodeStart = nullptr;
    u8* CodePtr   = nullptr;
    u32 CodeSize  = 0;

    // ── M29 Phase 7: generational code-cache wrap-around ──────────────────────
    // The code buffer is split into kCodeGenCount equal generations.  Compilation
    // bump-allocates within the current generation; when a block won't fit, we
    // advance to the next generation, evict only that generation's blocks, and
    // reuse it — keeping the other (more recent) generations alive instead of a
    // full ResetBlockCache().  A block never straddles a generation boundary, so
    // a block's host EntryPoint uniquely identifies its generation.
    static constexpr u32 kCodeGenCount = 4;
    u32 m_curGen = 0;
    u8* GenBase(u32 g) const { return CodeStart + (CodeSize / kCodeGenCount) * g; }
    u8* GenEnd(u32 g)  const { return (g == kCodeGenCount - 1)
                                      ? (CodeStart + CodeSize)
                                      : (CodeStart + (CodeSize / kCodeGenCount) * (g + 1)); }

    // ── Block-local constant-propagation state ────────────────────────────────
    // Tracks which guest registers have a statically known value within the
    // current compiled block (reset at CompileBlock entry).
    bool m_regKnown[16];
    u32  m_knownRegVals[16];

    // ── M29 Phase 1: write-back guest-register cache (R5–R11) ─────────────────
    // Maps guest reg index → host callee-saved reg (R5–R11) or -1 when unmapped.
    // A cached reg's authoritative value lives in the host reg; m_rcDirty marks
    // ones whose host value hasn't been written back to cpu->R[] yet.  Active only
    // across the leading run of cache-safe (pure-ALU) instructions; dropped at the
    // first barrier (memory/branch/helper/CPSR op) — see IsRegCacheBarrierKind.
    int8_t m_rcHostReg[16];
    u16    m_rcDirty   = 0;   // guest regs whose cached host value is unspilled
    bool   m_rcPoison  = false; // a helper ran this instr while the cache was live
    // M31: cache is being held LIVE across a single-transfer memory op — guest
    // register writes bypass the cache (write straight to cpu->R[]) so the
    // native fast path and the helper fallback path converge; the CompileBlock
    // merge logic reloads the written cached regs afterwards.
    bool   m_rcMemSpan = false;
    // M31 mem-span helpers (CompileBlock loop)
    bool IsMemSpanKind(bool thumb, const FetchedInstr& instr) const;
    void ReloadCachedRegsWrittenBy(bool thumb, const FetchedInstr& instr);

    // ── M29 Phase 2: lazy flag evaluation ─────────────────────────────────────
    // After a flag-setting ALU op the result stays in the host CPSR; we only copy
    // NZCV into the guest CPSR when a consumer needs it.  m_flagsDirty means the
    // host CPSR holds guest flags (mask m_flagsDirtyMask) not yet spilled.
    bool m_flagsDirty     = false;
    u32  m_flagsDirtyMask = 0;

    // ── Cycle batching accumulator ────────────────────────────────────────────
    int m_pendingCycles;

    // ── Native emission ───────────────────────────────────────────────────────
    bool TryEmitNative(const FetchedInstr& instr, bool thumb, ARM* cpu, bool tailReturn);
    bool TryEmitThumbNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn);
    // M28: native Thumb register-base immediate-offset single transfer.
    // M29 Phase 5: offsetReg >= 0 selects the register-offset form (base = R[rn] +
    // R[offsetReg], the `offset` immediate is then ignored) used by tk_*_REG.
    // M31: sign=true sign-extends the loaded byte/half (tk_LDRSB_REG/tk_LDRSH_REG).
    // Misaligned halves still take the helper (ROR semantics) via the align guard.
    bool TryEmitThumbMemImmDirect(const FetchedInstr& instr, ARM* cpu, bool tailReturn,
                                  int rd, int rn, u32 offset, int size, bool load,
                                  int offsetReg = -1, bool sign = false);
    // M28: native Thumb block transfer (PUSH/POP/STMIA/LDMIA).
    bool TryEmitThumbBlockTransfer(const FetchedInstr& instr, ARM* cpu, bool tailReturn);
    bool TryEmitArmNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn);

    // ARM memory emission helpers
    bool TryEmitArmDecodedMemoryHelper(const FetchedInstr& instr, ARM* cpu, bool tailReturn,
                                       bool load, bool wordTransfer, bool sign, bool post,
                                       bool imm, int size);
    bool TryEmitArmNativeMemory(const FetchedInstr& instr, ARM* cpu, bool tailReturn);
    bool TryEmitArmNativeLDM(const FetchedInstr& instr, ARM* cpu, bool tailReturn);
    bool TryEmitArmNativeSTM(const FetchedInstr& instr, ARM* cpu, bool tailReturn);

    // ── Code emitters ─────────────────────────────────────────────────────────
    void EmitU32(u32 value);
    void EmitInterpreterFallback(const FetchedInstr& instr, bool thumb, bool tailReturn);
    void EmitHelperCallWithDesc(const void* helper, const void* desc, u32 descSize, bool tailReturn);
    void EmitHelperCallLoadedPC(int pcReg, bool restoreCPSR);
    void EmitHelperCall1(const void* helper, u32 arg1Imm);
    void EmitHelperCall2(const void* helper, int arg1Reg, u32 arg2Imm);
    void EmitHelperCall3(const void* helper, int arg1Reg, int arg2Reg, u32 arg3Imm);
    void EmitReturn();
    void EmitFlushPendingCycles();
    u8*  EmitBranchPlaceholder(u32 cond);
    void PatchBranch(u8* branch, u8* target);
    void EmitLoadImm(int reg, u32 value);
    void EmitLoadReg(int dst, int base, u32 offset);
    void EmitStoreReg(int src, int base, u32 offset);
    void EmitLoadRegOffset(int dst, int base, int offsetReg);
    void EmitLoadRegOffsetLsl2(int dst, int base, int idxReg);   // M31 page-table fetch
    void EmitStoreRegOffset(int src, int base, int offsetReg);
    void EmitLoadByteRegOffset(int dst, int base, int offsetReg);
    void EmitStoreByteRegOffset(int src, int base, int offsetReg);
    void EmitLoadHalfRegOffset(int dst, int base, int offsetReg);
    void EmitStoreHalfRegOffset(int src, int base, int offsetReg);
    void EmitAddReg(int dst, int lhs, int rhs, bool setFlags);
    void EmitSubReg(int dst, int lhs, int rhs, bool setFlags);
    void EmitAndReg(int dst, int lhs, int rhs);
    void EmitEorReg(int dst, int lhs, int rhs);
    void EmitOrrReg(int dst, int lhs, int rhs);
    void EmitBicReg(int dst, int lhs, int rhs);
    void EmitMvnReg(int dst, int src);
    void EmitMulReg(int dst, int lhs, int rhs, bool setFlags);
    void EmitMovRegShiftImm(int dst, int src, int shiftType, u32 shift, bool setFlags);
    void EmitMovRegShiftReg(int dst, int src, int shiftType, int shiftReg, bool setFlags);
    void EmitCmpImm(int lhs, u32 imm);
    void EmitCmpReg(int lhs, int rhs);
    void EmitMRS(int dst);
    void EmitLoadGuestCPSRFlags();
    void EmitGuestLoad(int dst, int guestReg);
    void EmitGuestStore(int guestReg, int src);
    void EmitCondGuestStore(int guestReg, int src, u32 cond);

    // ── M29 Phase 1 register-cache helpers ────────────────────────────────────
    void SaveReg(int guestReg);                 // STR host[guestReg] → cpu->R[]
    void LoadReg(int guestReg);                 // LDR cpu->R[] → host[guestReg]
    void FlushRegAllocForHelper();              // spill all dirty cached regs
    void DropRegAlloc();                        // forget all mappings (compile-time)
    void RegAllocPrepareBlock(bool thumb, FetchedInstr instrs[], int emitCount);
    bool IsRegCacheBarrierKind(bool thumb, const FetchedInstr& instr);
    void OnHelperEmitted();                     // flush flags+regs before a helper

    // ── M29 Phase 2 lazy-flag helpers ─────────────────────────────────────────
    void MarkHostFlagsDirty(u32 mask);          // defer the host→guest flag copy
    void EmitFlushDirtyFlags();                 // spill deferred flags if pending
    void EmitStoreR15(u32 r15);
    void EmitCopyNZFromReg(int reg);
    void EmitCopyHostFlags(u32 mask);
    void EmitAddCycles(const FetchedInstr& instr, bool thumb, ARM* cpu);
    void EmitAddCyclesConst(u32 cycles);
    void EmitJumpToConst(ARM* cpu, u32 target, bool sourceThumb,
                         bool thumbTarget, bool tailReturn,
                         bool allowLink = false);   // M31: tail may direct-link
    void EmitIncrementCounter(u32* counter);

    // ── M31 cross-block link tail ─────────────────────────────────────────────
    // Emits the patchable direct-link tail (ARM9 only) and records the literal
    // in m_linkSites.  blockAddr = guest address of the first instruction of
    // the target block (newPC - pipeline offset).
    void EmitLinkTail(ARM* cpu, u32 blockAddr);
    static constexpr int kMaxLinkSites = 4;
    LinkSiteRecord m_linkSites[kMaxLinkSites];
    int m_linkSiteCount = 0;

    // ── Direct safe-region helpers ────────────────────────────────────────────
    bool IsDirectSafeDataRegion(ARM* cpu, int region, u32 compileAddr);
    void EmitDirectSafeRegionAddressGuard(int region, ARM* cpu, u32 compileAddr, int addrReg,
                                          u8** fallbackBranches, int& fallbackBranchCount,
                                          bool checkDTCM = false);
    void EmitDirectSafeRegionBase(int region, ARM* cpu, int baseReg);
    void EmitDirectSafeRegionSetup(int region, ARM* cpu, u32 compileAddr, int addrReg, int offsetReg,
                                   int baseReg, int size, u8** fallbackBranches, int& fallbackBranchCount);
    void EmitDirectSafeRegionRangeSetup(int region, ARM* cpu, u32 compileAddr, int addrReg,
                                        int offsetReg, int baseReg, u32 byteCount,
                                        u8** fallbackBranches, int& fallbackBranchCount);
    void EmitCodeBitmapGuard(int region, int offsetReg, u8** fallbackBranches, int& fallbackBranchCount);

    // ── M31 fastmem-lite ──────────────────────────────────────────────────────
    // Emits the 16 KB page-table lookup for the guest address in R0:
    //   R3 = R0 >> 14; R12 = table[R3]; R12 == 0 → append helper-miss branch.
    // On the fall-through path R12 holds (hostPage - guestPage), so the host
    // effective address is simply [R12 + R0].  Caller must already have emitted
    // the alignment guard and the >= 0x04000000 fast exit.
    void EmitFastMemLiteLookup(ARM* cpu, bool store,
                               u8** helperBranches, int& helperBranchCount);

    // ── M18 self-link tail ────────────────────────────────────────────────────
    void EmitSelfLinkTail(ARM* cpu, u8* blockStart, u32 branchCycles);

    // ── Utility ───────────────────────────────────────────────────────────────
    u32  CodeOffset() const;
    void FlushIcache(u8* start, u8* end);
};

}

#endif
