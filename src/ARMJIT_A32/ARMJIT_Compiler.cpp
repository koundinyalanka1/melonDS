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

#include "ARMJIT_Compiler.h"

#include "../ARMInterpreter.h"
#include "../Config.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

extern "C" void ARM_Ret();

namespace ARMJIT
{

namespace
{

const int JitMemSize = 16 * 1024 * 1024;
const int MaxInstrsPerBlock = 8;
alignas(4096) u8 JitMem[JitMemSize];

enum HostReg
{
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R12 = 12,
};

enum ARMDataOp
{
    DP_AND = 0,
    DP_EOR = 1,
    DP_SUB = 2,
    DP_ADD = 4,
    DP_ORR = 12,
    DP_BIC = 14,
    DP_MVN = 15,
};

const u32 CPSR_NZMask = 0xC0000000;
const u32 CPSR_NZCMask = 0xE0000000;
const u32 CPSR_NZCVMask = 0xF0000000;

enum ShiftType
{
    Shift_LSL = 0,
    Shift_LSR = 1,
    Shift_ASR = 2,
    Shift_ROR = 3,
};

struct A32InstrDesc
{
    u32 instr;
    u32 r15;
    u32 codeCycles;
    u32 kind;
    u32 thumb;
};

struct A32BranchDesc
{
    u32 r15;
    u32 codeCycles;
    u32 target;
    u32 aux;
};

u32 DpReg(ARMDataOp op, bool setFlags, int rn, int rd, int rm)
{
    return 0xE0000000
        | ((u32)op << 21)
        | (setFlags ? (1u << 20) : 0)
        | ((u32)rn << 16)
        | ((u32)rd << 12)
        | (u32)rm;
}

u32 LdrImm(int rd, int rn, u32 offset)
{
    assert(offset <= 0xFFF);
    return 0xE5900000 | ((u32)rn << 16) | ((u32)rd << 12) | offset;
}

u32 StrImm(int rd, int rn, u32 offset)
{
    assert(offset <= 0xFFF);
    return 0xE5800000 | ((u32)rn << 16) | ((u32)rd << 12) | offset;
}

void RunInterpreterInstr(ARM* cpu, const A32InstrDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CurInstr = desc->instr;
    cpu->CodeCycles = desc->codeCycles;

    if (desc->thumb)
    {
        InterpretTHUMB[desc->kind](cpu);
        return;
    }

    if (desc->kind == ARMInstrInfo::ak_BLX_IMM || cpu->CheckCondition(desc->instr >> 28))
    {
        InterpretARM[desc->kind](cpu);
        return;
    }

    cpu->AddCycles_C();
}

void RunThumbBranch(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->JumpTo(desc->target);
}

void RunThumbCondBranch(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;

    if (cpu->CheckCondition(desc->aux))
        cpu->JumpTo(desc->target);
    else
        cpu->AddCycles_C();
}

void RunThumbBX(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->JumpTo(cpu->R[desc->aux & 0xF]);
}

void RunThumbBLXReg(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;

    u32 lr = cpu->R[15] - 1;
    cpu->JumpTo(cpu->R[desc->aux & 0xF]);
    cpu->R[14] = lr;
}

void RunThumbBLLong(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->AddCycles_C();
    cpu->R[15] += 2;
    cpu->R[14] = desc->aux;
    cpu->JumpTo(desc->target);
}

void RunThumbMemInstr(ARM* cpu, const A32InstrDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;

    const u32 instr = desc->instr;

    switch (desc->kind)
    {
    case ARMInstrInfo::tk_LDR_PCREL:
    {
        u32 addr = (cpu->R[15] & ~0x2) + ((instr & 0xFF) << 2);
        cpu->DataRead32(addr, &cpu->R[(instr >> 8) & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STR_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataWrite32(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_STRB_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataWrite8(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDR_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        u32 val;
        cpu->DataRead32(addr, &val);
        cpu->R[instr & 0x7] = ROR(val, 8 * (addr & 0x3));
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_LDRB_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataRead8(addr, &cpu->R[instr & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STRH_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataWrite16(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDRSB_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataRead8(addr, &cpu->R[instr & 0x7]);
        cpu->R[instr & 0x7] = (s32)(s8)cpu->R[instr & 0x7];
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_LDRH_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataRead16(addr, &cpu->R[instr & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_LDRSH_REG:
    {
        u32 addr = cpu->R[(instr >> 3) & 0x7] + cpu->R[(instr >> 6) & 0x7];
        cpu->DataRead16(addr, &cpu->R[instr & 0x7]);
        cpu->R[instr & 0x7] = (s32)(s16)cpu->R[instr & 0x7];
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STR_IMM:
    {
        u32 addr = ((instr >> 4) & 0x7C) + cpu->R[(instr >> 3) & 0x7];
        cpu->DataWrite32(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDR_IMM:
    {
        u32 addr = ((instr >> 4) & 0x7C) + cpu->R[(instr >> 3) & 0x7];
        u32 val;
        cpu->DataRead32(addr, &val);
        cpu->R[instr & 0x7] = ROR(val, 8 * (addr & 0x3));
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STRB_IMM:
    {
        u32 addr = ((instr >> 6) & 0x1F) + cpu->R[(instr >> 3) & 0x7];
        cpu->DataWrite8(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDRB_IMM:
    {
        u32 addr = ((instr >> 6) & 0x1F) + cpu->R[(instr >> 3) & 0x7];
        cpu->DataRead8(addr, &cpu->R[instr & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STRH_IMM:
    {
        u32 addr = ((instr >> 5) & 0x3E) + cpu->R[(instr >> 3) & 0x7];
        cpu->DataWrite16(addr, cpu->R[instr & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDRH_IMM:
    {
        u32 addr = ((instr >> 5) & 0x3E) + cpu->R[(instr >> 3) & 0x7];
        cpu->DataRead16(addr, &cpu->R[instr & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STR_SPREL:
    {
        u32 addr = ((instr << 2) & 0x3FC) + cpu->R[13];
        cpu->DataWrite32(addr, cpu->R[(instr >> 8) & 0x7]);
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDR_SPREL:
    {
        u32 addr = ((instr << 2) & 0x3FC) + cpu->R[13];
        cpu->DataRead32(addr, &cpu->R[(instr >> 8) & 0x7]);
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_PUSH:
    {
        int nregs = 0;
        for (int i = 0; i < 8; i++)
        {
            if (instr & (1 << i))
                nregs++;
        }
        if (instr & (1 << 8))
            nregs++;

        bool first = true;
        u32 base = cpu->R[13] - (nregs << 2);
        cpu->R[13] = base;

        for (int i = 0; i < 8; i++)
        {
            if (instr & (1 << i))
            {
                if (first) cpu->DataWrite32(base, cpu->R[i]);
                else       cpu->DataWrite32S(base, cpu->R[i]);
                first = false;
                base += 4;
            }
        }

        if (instr & (1 << 8))
        {
            if (first) cpu->DataWrite32(base, cpu->R[14]);
            else       cpu->DataWrite32S(base, cpu->R[14]);
        }

        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_POP:
    {
        u32 base = cpu->R[13];
        bool first = true;

        for (int i = 0; i < 8; i++)
        {
            if (instr & (1 << i))
            {
                if (first) cpu->DataRead32(base, &cpu->R[i]);
                else       cpu->DataRead32S(base, &cpu->R[i]);
                first = false;
                base += 4;
            }
        }

        if (instr & (1 << 8))
        {
            u32 pc;
            if (first) cpu->DataRead32(base, &pc);
            else       cpu->DataRead32S(base, &pc);
            if (cpu->Num == 1)
                pc |= 0x1;
            cpu->JumpTo(pc);
            base += 4;
        }

        cpu->R[13] = base;
        cpu->AddCycles_CDI();
        break;
    }

    case ARMInstrInfo::tk_STMIA:
    {
        u32 baseReg = (instr >> 8) & 0x7;
        u32 base = cpu->R[baseReg];
        bool first = true;

        for (int i = 0; i < 8; i++)
        {
            if (instr & (1 << i))
            {
                if (first) cpu->DataWrite32(base, cpu->R[i]);
                else       cpu->DataWrite32S(base, cpu->R[i]);
                first = false;
                base += 4;
            }
        }

        cpu->R[baseReg] = base;
        cpu->AddCycles_CD();
        break;
    }

    case ARMInstrInfo::tk_LDMIA:
    {
        u32 baseReg = (instr >> 8) & 0x7;
        u32 base = cpu->R[baseReg];
        bool first = true;

        for (int i = 0; i < 8; i++)
        {
            if (instr & (1 << i))
            {
                if (first) cpu->DataRead32(base, &cpu->R[i]);
                else       cpu->DataRead32S(base, &cpu->R[i]);
                first = false;
                base += 4;
            }
        }

        if (!(instr & (1 << baseReg)))
            cpu->R[baseReg] = base;
        cpu->AddCycles_CDI();
        break;
    }

    default:
        assert(false);
        break;
    }
}

}

Compiler::Compiler()
{
    u64 pageSize = sysconf(_SC_PAGE_SIZE);
    u8* pageAligned = (u8*)(((u64)JitMem + pageSize - 1) & ~(pageSize - 1));
    u64 alignedSize = (((u64)JitMem + sizeof(JitMem)) & ~(pageSize - 1)) - (u64)pageAligned;

    int ok = mprotect(pageAligned, alignedSize, PROT_EXEC | PROT_READ | PROT_WRITE);
    assert(ok == 0);

    CodeStart = pageAligned;
    CodePtr = CodeStart;
    CodeSize = (u32)alignedSize;

    printf("melonDS JIT: armeabi-v7a/AArch32 JIT backend active (Thumb ALU/address native slice + branch/memory helpers; 8-instr straight-line blocks; fastmem disabled)\n");
}

Compiler::~Compiler()
{
}

void Compiler::Reset()
{
    memset(CodeStart, 0, CodeSize);
    CodePtr = CodeStart;
}

bool Compiler::CanCompile(bool thumb, u16 kind)
{
    return thumb ? kind < ARMInstrInfo::tk_Count : kind < ARMInstrInfo::ak_Count;
}

JitBlockEntry Compiler::CompileBlock(ARM* cpu, bool thumb, FetchedInstr instrs[],
                                     int instrsCount, bool hasMemoryInstr)
{
    (void)hasMemoryInstr;

    if (instrsCount <= 0)
        return nullptr;

    const u32 bytesNeeded = 8192;
    if (CodeSize - CodeOffset() < bytesNeeded)
    {
        printf("AArch32 JIT memory full, resetting...\n");
        ResetBlockCache();
    }

    u8* start = CodePtr;
    int emitCount = instrsCount < MaxInstrsPerBlock ? instrsCount : MaxInstrsPerBlock;

    for (int i = 0; i < emitCount; i++)
    {
        const FetchedInstr& instr = instrs[i];
        bool forceExit = instr.Info.EndBlock || instr.Info.Branches() || instr.BranchFlags != 0;
        bool tailReturn = forceExit || i == (emitCount - 1);

        if (!TryEmitNative(instr, thumb, cpu, tailReturn))
            EmitInterpreterFallback(instr, thumb, tailReturn);

        if (tailReturn)
            break;
    }

    FlushIcache(start, CodePtr);
    return (JitBlockEntry)start;
}

bool Compiler::IsJITFault(u8* pc)
{
    (void)pc;
    return false;
}

u8* Compiler::RewriteMemAccess(u8* pc)
{
    return pc;
}

bool Compiler::TryEmitNative(const FetchedInstr& instr, bool thumb, ARM* cpu, bool tailReturn)
{
    u8* start = CodePtr;

    if (thumb && TryEmitThumbNative(instr, cpu, tailReturn))
        return true;

    CodePtr = start;
    return false;
}

bool Compiler::TryEmitThumbNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    const u32 op = instr.Instr;

    switch (instr.Info.Kind)
    {
    case ARMInstrInfo::tk_LSL_IMM:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;
        u32 shift = (op >> 6) & 0x1F;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, src);
        EmitMovRegShiftImm(R2, R0, Shift_LSL, shift, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(shift ? CPSR_NZCMask : CPSR_NZMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_LSR_IMM:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;
        u32 shift = (op >> 6) & 0x1F;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, src);
        EmitMovRegShiftImm(R2, R0, Shift_LSR, shift, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ASR_IMM:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;
        u32 shift = (op >> 6) & 0x1F;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, src);
        EmitMovRegShiftImm(R2, R0, Shift_ASR, shift, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_MOV_IMM:
    {
        u32 dst = (op >> 8) & 0x7;
        u32 imm = op & 0xFF;

        EmitStoreR15(instr.Addr + 4);
        EmitLoadImm(R2, imm);
        EmitGuestStore(dst, R2);
        EmitCmpImm(R2, 0);
        EmitCopyHostFlags(CPSR_NZMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_CMP_IMM:
    {
        u32 src = (op >> 8) & 0x7;
        u32 imm = op & 0xFF;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, src);
        EmitLoadImm(R1, imm);
        EmitSubReg(R2, R0, R1, true);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_IMM:
    {
        u32 dst = (op >> 8) & 0x7;
        u32 imm = op & 0xFF;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitLoadImm(R1, imm);
        EmitAddReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_SUB_IMM:
    {
        u32 dst = (op >> 8) & 0x7;
        u32 imm = op & 0xFF;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitLoadImm(R1, imm);
        EmitSubReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_REG_:
    {
        u32 dst = op & 0x7;
        u32 lhs = (op >> 3) & 0x7;
        u32 rhs = (op >> 6) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, lhs);
        EmitGuestLoad(R1, rhs);
        EmitAddReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_SUB_REG_:
    {
        u32 dst = op & 0x7;
        u32 lhs = (op >> 3) & 0x7;
        u32 rhs = (op >> 6) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, lhs);
        EmitGuestLoad(R1, rhs);
        EmitSubReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_IMM_:
    {
        u32 dst = op & 0x7;
        u32 lhs = (op >> 3) & 0x7;
        u32 imm = (op >> 6) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, lhs);
        EmitLoadImm(R1, imm);
        EmitAddReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_SUB_IMM_:
    {
        u32 dst = op & 0x7;
        u32 lhs = (op >> 3) & 0x7;
        u32 imm = (op >> 6) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, lhs);
        EmitLoadImm(R1, imm);
        EmitSubReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_AND_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitAndReg(R2, R0, R1);
        EmitGuestStore(dst, R2);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_EOR_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitEorReg(R2, R0, R1);
        EmitGuestStore(dst, R2);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_TST_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitAndReg(R2, R0, R1);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_NEG_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitLoadImm(R0, 0);
        EmitGuestLoad(R1, src);
        EmitSubReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_CMP_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitSubReg(R2, R0, R1, true);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_CMN_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitAddReg(R2, R0, R1, true);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ORR_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitOrrReg(R2, R0, R1);
        EmitGuestStore(dst, R2);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_BIC_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitBicReg(R2, R0, R1);
        EmitGuestStore(dst, R2);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_MVN_REG:
    {
        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R1, src);
        EmitMvnReg(R2, R1);
        EmitGuestStore(dst, R2);
        EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_HIREG:
    {
        u32 dst = (op & 0x7) | ((op >> 4) & 0x8);
        u32 src = (op >> 3) & 0xF;

        if (dst == 15)
            return false;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitAddReg(R2, R0, R1, false);
        EmitGuestStore(dst, R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_CMP_HIREG:
    {
        u32 dst = (op & 0x7) | ((op >> 4) & 0x8);
        u32 src = (op >> 3) & 0xF;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitSubReg(R2, R0, R1, true);
        EmitCopyHostFlags(CPSR_NZCVMask);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_MOV_HIREG:
    {
        u32 dst = (op & 0x7) | ((op >> 4) & 0x8);
        u32 src = (op >> 3) & 0xF;

        if (dst == 15 || (op & 0xFFFF) == 0x46E4)
            return false;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, src);
        EmitGuestStore(dst, R0);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_PCREL:
    {
        u32 dst = (op >> 8) & 0x7;
        u32 value = ((instr.Addr + 4) & ~2u) + ((op & 0xFF) << 2);

        EmitStoreR15(instr.Addr + 4);
        EmitLoadImm(R0, value);
        EmitGuestStore(dst, R0);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_SPREL:
    {
        u32 dst = (op >> 8) & 0x7;
        u32 imm = (op & 0xFF) << 2;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, 13);
        EmitLoadImm(R1, imm);
        EmitAddReg(R2, R0, R1, false);
        EmitGuestStore(dst, R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_ADD_SP:
    {
        u32 imm = (op & 0x7F) << 2;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, 13);
        EmitLoadImm(R1, imm);
        if (op & (1 << 7))
            EmitSubReg(R2, R0, R1, false);
        else
            EmitAddReg(R2, R0, R1, false);
        EmitGuestStore(13, R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_LDR_PCREL:
    case ARMInstrInfo::tk_STR_REG:
    case ARMInstrInfo::tk_STRB_REG:
    case ARMInstrInfo::tk_LDR_REG:
    case ARMInstrInfo::tk_LDRB_REG:
    case ARMInstrInfo::tk_STRH_REG:
    case ARMInstrInfo::tk_LDRSB_REG:
    case ARMInstrInfo::tk_LDRH_REG:
    case ARMInstrInfo::tk_LDRSH_REG:
    case ARMInstrInfo::tk_STR_IMM:
    case ARMInstrInfo::tk_LDR_IMM:
    case ARMInstrInfo::tk_STRB_IMM:
    case ARMInstrInfo::tk_LDRB_IMM:
    case ARMInstrInfo::tk_STRH_IMM:
    case ARMInstrInfo::tk_LDRH_IMM:
    case ARMInstrInfo::tk_STR_SPREL:
    case ARMInstrInfo::tk_LDR_SPREL:
    case ARMInstrInfo::tk_PUSH:
    case ARMInstrInfo::tk_POP:
    case ARMInstrInfo::tk_STMIA:
    case ARMInstrInfo::tk_LDMIA:
    {
        A32InstrDesc desc;
        desc.instr = instr.Instr;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.kind = instr.Info.Kind;
        desc.thumb = 1;

        EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::tk_BCOND:
    {
        s32 offset = (s32)(op << 24) >> 23;
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.target = desc.r15 + offset + 1;
        desc.aux = (op >> 8) & 0xF;

        EmitHelperCallWithDesc((const void*)&RunThumbCondBranch, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::tk_BX:
    {
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.target = 0;
        desc.aux = (op >> 3) & 0xF;

        EmitHelperCallWithDesc((const void*)&RunThumbBX, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::tk_BLX_REG:
    {
        if (cpu->Num == 1)
            return false;

        A32BranchDesc desc;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.target = 0;
        desc.aux = (op >> 3) & 0xF;

        EmitHelperCallWithDesc((const void*)&RunThumbBLXReg, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::tk_B:
    {
        s32 offset = (s32)((op & 0x7FF) << 21) >> 20;
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.target = desc.r15 + offset + 1;
        desc.aux = 0;

        EmitHelperCallWithDesc((const void*)&RunThumbBranch, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::tk_BL_LONG:
    {
        u32 first = op & 0xFFFF;
        u32 second = (op >> 16) & 0xFFFF;
        s32 offset1 = (s32)((first & 0x7FF) << 21) >> 9;
        u32 target = (instr.Addr + 4) + offset1 + ((second & 0x7FF) << 1);
        if ((cpu->Num == 1) || (second & (1 << 12)))
            target |= 1;

        A32BranchDesc desc;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.target = target;
        desc.aux = (instr.Addr + 4) | 1;

        EmitHelperCallWithDesc((const void*)&RunThumbBLLong, &desc, sizeof(desc), tailReturn);
        return true;
    }

    default:
        return false;
    }
}

void Compiler::EmitU32(u32 value)
{
    assert(CodeOffset() + sizeof(value) <= CodeSize);
    memcpy(CodePtr, &value, sizeof(value));
    CodePtr += sizeof(value);
}

void Compiler::EmitInterpreterFallback(const FetchedInstr& instr, bool thumb, bool tailReturn)
{
    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + (thumb ? 4 : 8);
    desc.codeCycles = instr.CodeCycles;
    desc.kind = instr.Info.Kind;
    desc.thumb = thumb ? 1 : 0;

    EmitHelperCallWithDesc((const void*)&RunInterpreterInstr, &desc, sizeof(desc), tailReturn);
}

void Compiler::EmitHelperCallWithDesc(const void* helper, const void* desc, u32 descSize, bool tailReturn)
{
    assert((descSize & 0x3) == 0);

    if (tailReturn)
    {
        // r4 is loaded with ARM* by ARM_Dispatch and is callee-saved by AAPCS.
        EmitU32(0xE1A00004); // mov r0, r4
        EmitU32(0xE28F1014); // add r1, pc, #20 ; descriptor starts after literals
        EmitU32(0xE59FC008); // ldr ip, [pc, #8]
        EmitU32(0xE12FFF3C); // blx ip
        EmitU32(0xE59FC004); // ldr ip, [pc, #4]
        EmitU32(0xE12FFF1C); // bx ip
        EmitU32((u32)(uintptr_t)helper);
        EmitU32((u32)(uintptr_t)&ARM_Ret);
    }
    else
    {
        EmitU32(0xE1A00004); // mov r0, r4
        EmitU32(0xE28F100C); // add r1, pc, #12 ; descriptor starts after helper literal
        EmitU32(0xE59FC004); // ldr ip, [pc, #4]
        EmitU32(0xE12FFF3C); // blx ip
        EmitU32(0xEA000000 | (descSize / 4)); // b after descriptor
        EmitU32((u32)(uintptr_t)helper);
    }

    assert(CodeOffset() + descSize <= CodeSize);
    memcpy(CodePtr, desc, descSize);
    CodePtr += descSize;
}

void Compiler::EmitReturn()
{
    EmitU32(0xE59FC000); // ldr ip, [pc]
    EmitU32(0xE12FFF1C); // bx ip
    EmitU32((u32)(uintptr_t)&ARM_Ret);
}

void Compiler::EmitLoadImm(int reg, u32 value)
{
    EmitU32(0xE3000000
        | (((value >> 12) & 0xF) << 16)
        | ((u32)reg << 12)
        | (value & 0xFFF)); // movw reg, #lo16

    if (value >> 16)
    {
        u32 high = value >> 16;
        EmitU32(0xE3400000
            | (((high >> 12) & 0xF) << 16)
            | ((u32)reg << 12)
            | (high & 0xFFF)); // movt reg, #hi16
    }
}

void Compiler::EmitLoadReg(int dst, int base, u32 offset)
{
    EmitU32(LdrImm(dst, base, offset));
}

void Compiler::EmitStoreReg(int src, int base, u32 offset)
{
    EmitU32(StrImm(src, base, offset));
}

void Compiler::EmitAddReg(int dst, int lhs, int rhs, bool setFlags)
{
    EmitU32(DpReg(DP_ADD, setFlags, lhs, dst, rhs));
}

void Compiler::EmitSubReg(int dst, int lhs, int rhs, bool setFlags)
{
    EmitU32(DpReg(DP_SUB, setFlags, lhs, dst, rhs));
}

void Compiler::EmitAndReg(int dst, int lhs, int rhs)
{
    EmitU32(DpReg(DP_AND, false, lhs, dst, rhs));
}

void Compiler::EmitEorReg(int dst, int lhs, int rhs)
{
    EmitU32(DpReg(DP_EOR, false, lhs, dst, rhs));
}

void Compiler::EmitOrrReg(int dst, int lhs, int rhs)
{
    EmitU32(DpReg(DP_ORR, false, lhs, dst, rhs));
}

void Compiler::EmitBicReg(int dst, int lhs, int rhs)
{
    EmitU32(DpReg(DP_BIC, false, lhs, dst, rhs));
}

void Compiler::EmitMvnReg(int dst, int src)
{
    EmitU32(DpReg(DP_MVN, false, 0, dst, src));
}

void Compiler::EmitMovRegShiftImm(int dst, int src, int shiftType, u32 shift, bool setFlags)
{
    assert(shift <= 0x1F);
    EmitU32(0xE1A00000
        | (setFlags ? (1u << 20) : 0)
        | ((u32)dst << 12)
        | (shift << 7)
        | ((u32)shiftType << 5)
        | (u32)src);
}

void Compiler::EmitCmpImm(int lhs, u32 imm)
{
    assert(imm <= 0xFF);
    EmitU32(0xE3500000 | ((u32)lhs << 16) | imm);
}

void Compiler::EmitMRS(int dst)
{
    EmitU32(0xE10F0000 | ((u32)dst << 12)); // mrs dst, cpsr
}

void Compiler::EmitGuestLoad(int dst, int guestReg)
{
    EmitLoadReg(dst, R4, offsetof(ARM, R) + guestReg * 4);
}

void Compiler::EmitGuestStore(int guestReg, int src)
{
    EmitStoreReg(src, R4, offsetof(ARM, R) + guestReg * 4);
}

void Compiler::EmitStoreR15(u32 r15)
{
    EmitLoadImm(R0, r15);
    EmitGuestStore(15, R0);
}

void Compiler::EmitCopyNZFromReg(int reg)
{
    EmitCmpImm(reg, 0);
    EmitCopyHostFlags(CPSR_NZMask);
}

void Compiler::EmitCopyHostFlags(u32 mask)
{
    EmitMRS(R3);
    EmitLoadReg(R0, R4, offsetof(ARM, CPSR));
    EmitLoadImm(R1, ~mask);
    EmitAndReg(R0, R0, R1);
    EmitLoadImm(R1, mask);
    EmitAndReg(R3, R3, R1);
    EmitOrrReg(R0, R0, R3);
    EmitStoreReg(R0, R4, offsetof(ARM, CPSR));
}

void Compiler::EmitAddCycles(const FetchedInstr& instr, bool thumb, ARM* cpu)
{
    u32 r15 = instr.Addr + (thumb ? 4 : 8);
    u32 cycles = 0;

    if (cpu->Num == 0)
        cycles = (r15 & 0x2) ? 0 : instr.CodeCycles;
    else
        cycles = NDS::ARM7MemTimings[instr.CodeCycles][thumb ? 1 : 3];

    if (!cycles)
        return;

    EmitLoadReg(R0, R4, offsetof(ARM, Cycles));
    EmitLoadImm(R1, cycles);
    EmitAddReg(R0, R0, R1, false);
    EmitStoreReg(R0, R4, offsetof(ARM, Cycles));
}

u32 Compiler::CodeOffset() const
{
    return (u32)(CodePtr - CodeStart);
}

void Compiler::FlushIcache(u8* start, u8* end)
{
    __builtin___clear_cache((char*)start, (char*)end);
}

}
