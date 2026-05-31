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
    u8* CodeStart = nullptr;
    u8* CodePtr = nullptr;
    u32 CodeSize = 0;

    bool TryEmitNative(const FetchedInstr& instr, bool thumb, ARM* cpu, bool tailReturn);
    bool TryEmitThumbNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn);

    void EmitU32(u32 value);
    void EmitInterpreterFallback(const FetchedInstr& instr, bool thumb, bool tailReturn);
    void EmitHelperCallWithDesc(const void* helper, const void* desc, u32 descSize, bool tailReturn);
    void EmitReturn();
    void EmitLoadImm(int reg, u32 value);
    void EmitLoadReg(int dst, int base, u32 offset);
    void EmitStoreReg(int src, int base, u32 offset);
    void EmitAddReg(int dst, int lhs, int rhs, bool setFlags);
    void EmitSubReg(int dst, int lhs, int rhs, bool setFlags);
    void EmitAndReg(int dst, int lhs, int rhs);
    void EmitEorReg(int dst, int lhs, int rhs);
    void EmitOrrReg(int dst, int lhs, int rhs);
    void EmitBicReg(int dst, int lhs, int rhs);
    void EmitMvnReg(int dst, int src);
    void EmitMovRegShiftImm(int dst, int src, int shiftType, u32 shift, bool setFlags);
    void EmitCmpImm(int lhs, u32 imm);
    void EmitMRS(int dst);
    void EmitGuestLoad(int dst, int guestReg);
    void EmitGuestStore(int guestReg, int src);
    void EmitStoreR15(u32 r15);
    void EmitCopyNZFromReg(int reg);
    void EmitCopyHostFlags(u32 mask);
    void EmitAddCycles(const FetchedInstr& instr, bool thumb, ARM* cpu);
    u32 CodeOffset() const;
    void FlushIcache(u8* start, u8* end);
};

}

#endif
