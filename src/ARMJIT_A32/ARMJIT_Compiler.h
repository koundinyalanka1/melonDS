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

private:
    // ── Code buffer ──────────────────────────────────────────────────────────
    u8* CodeStart = nullptr;
    u8* CodePtr   = nullptr;
    u32 CodeSize  = 0;

    // ── Block-local constant-propagation state ────────────────────────────────
    // Tracks which guest registers have a statically known value within the
    // current compiled block (reset at CompileBlock entry).
    bool m_regKnown[16];
    u32  m_knownRegVals[16];

    // ── Block-local register cache (R5–R10) ───────────────────────────────────
    // For pure-source ARM guest registers, maps guest reg index → host reg or -1.
    int8_t m_rcHostReg[16];

    // ── Cycle batching accumulator ────────────────────────────────────────────
    int m_pendingCycles;

    // ── Native emission ───────────────────────────────────────────────────────
    bool TryEmitNative(const FetchedInstr& instr, bool thumb, ARM* cpu, bool tailReturn);
    bool TryEmitThumbNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn);
    // M28: native Thumb register-base immediate-offset single transfer.
    bool TryEmitThumbMemImmDirect(const FetchedInstr& instr, ARM* cpu, bool tailReturn,
                                  int rd, int rn, u32 offset, int size, bool load);
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
    void EmitStoreR15(u32 r15);
    void EmitCopyNZFromReg(int reg);
    void EmitCopyHostFlags(u32 mask);
    void EmitAddCycles(const FetchedInstr& instr, bool thumb, ARM* cpu);
    void EmitAddCyclesConst(u32 cycles);
    void EmitJumpToConst(ARM* cpu, u32 target, bool sourceThumb,
                         bool thumbTarget, bool tailReturn);
    void EmitIncrementCounter(u32* counter);

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

    // ── M18 self-link tail ────────────────────────────────────────────────────
    void EmitSelfLinkTail(ARM* cpu, u8* blockStart, u32 branchCycles);

    // ── Utility ───────────────────────────────────────────────────────────────
    u32  CodeOffset() const;
    void FlushIcache(u8* start, u8* end);
};

}

#endif
