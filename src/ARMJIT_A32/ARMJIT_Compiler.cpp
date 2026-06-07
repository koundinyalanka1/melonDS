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
#include "../DMA.h"
#include "../GPU3D.h"

#include <algorithm>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif

extern "C" void ARM_Ret();

namespace ARMJIT
{

namespace
{

const int JitMemSize = 16 * 1024 * 1024;
// Upper safety cap on instructions emitted per block. The *effective* block
// length is Config::JIT_MaxBlockSize (set from the melonds_jit_block_size core
// option and clamped to <=128 in ARMJIT.cpp::CompileBlock). This constant must
// stay >= that clamp so it never silently truncates a configured block.
const int MaxInstrsPerBlock = 128;
const bool EnableEmittedMainRAMLoadStore = true;
const bool EnableRuntimeDirectRegionChain = true;
// M28 (partial): native emission for Thumb register-base immediate-offset single
// transfers (tk_{LDR,STR,LDRB,STRB,LDRH,STRH}_IMM). These are the most common
// Thumb memory ops in ARM9 game code and previously always took the full
// RunThumbMemInstr AAPCS helper call. Direct path reuses the (DTCM-safe)
// EmitDirectSafeRegionSetup candidate chain; any guard miss falls back to the
// helper, so correctness is preserved. Flip false to revert to helper-only.
const bool EnableThumbMemImmDirect = true;
// M28 (partial): native emission for Thumb block transfers — PUSH/POP (function
// prologue/epilogue) and STMIA/LDMIA. Reuses the ARM LDM/STM range machinery
// (EmitDirectSafeRegionRangeSetup, DTCM-safe). POP-with-PC is a block-exiting
// return; handled natively on ARM9 only (RunArmLoadedPC matches the interpreter
// there — on ARM7 a Thumb POP-PC must set bit0, which that helper clears, so ARM7
// POP-PC stays on RunThumbMemInstr). Any guard miss falls back to the helper.
const bool EnableThumbBlockTransfer = true;
const bool EnableRegAllocator = true;
#ifndef A32JIT_PROFILE
#define A32JIT_PROFILE 0
#endif
const bool EnableA32JitProfiling = A32JIT_PROFILE != 0;
alignas(4096) u8 JitMem[JitMemSize];

// M18: always-on (production) self-link emission counter, independent of the
// heavyweight A32JIT_PROFILE event path. Lets a normal build confirm in logcat
// that the self-link tail is actually emitted. Bumped once per compiled
// self-loop block — off the hot path (blocks compile once, run many times).
u32 g_M18SelfLinkEmitted = 0;
u32 g_M18FastBranchEmitted = 0;

// Mirrors ARMJIT_A64/ARMJIT_Branch.cpp and CP15.cpp::kCodeCacheTiming.
const int A32CodeCacheTiming = 3;

enum HostReg
{
    R0 = 0,
    R1 = 1,
    R2 = 2,
    R3 = 3,
    R4 = 4,
    R5 = 5,
    R6 = 6,
    R7 = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R12 = 12,
};

enum ArmCond
{
    Cond_EQ = 0x0,
    Cond_NE = 0x1,
    Cond_HS = 0x2,
    Cond_LO = 0x3,
    Cond_AL = 0xE,
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
const uintptr_t ARMOffsetBase = 0x1000;

#ifdef __ANDROID__
#define A32JIT_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "melonDS-JIT", __VA_ARGS__)
#else
#define A32JIT_LOGI(...) do { printf(__VA_ARGS__); printf("\n"); } while (0)
#endif

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

enum A32MemHelperReason
{
    MemReason_Thumb = 0,
    MemReason_CondNonAL,
    MemReason_UnsupportedKind,
    MemReason_PCRegister,
    MemReason_PCBase,
    MemReason_PCDest,
    MemReason_PCOffset,
    MemReason_RRXOffset,
    MemReason_UnsafeRegion,
    MemReason_Unaligned,
    MemReason_CodeWrite,
    MemReason_DirectEligible,
    MemReason_LDMEmpty,
    MemReason_LDMBasePC,
    MemReason_LDMBankedUser,
    MemReason_LDMRestoreCPSR,
    MemReason_LDMBaseInListWB,
    MemReason_LDMUnsafeRegion,
    MemReason_LDMUnaligned,
    MemReason_LDMRangeWrap,
    MemReason_STMBankedUser,
    MemReason_STMPCOrBase,
    MemReason_STMUnsafeRegion,
    MemReason_STMUnaligned,
    MemReason_STMRangeWrap,
    MemReason_STMCodeWrite,
    MemReason_Count,
};

template <typename Class, typename Member>
u32 CpuMemberOffset(Member Class::* member)
{
    const Class* cpu = reinterpret_cast<const Class*>(ARMOffsetBase);
    return (u32)(reinterpret_cast<uintptr_t>(&(cpu->*member)) - ARMOffsetBase);
}

template <typename Member>
u32 ArmMemberOffset(Member ARM::* member)
{
    return CpuMemberOffset<ARM>(member);
}

u32 AddressRangeCodeOffset()
{
    return sizeof(TinyVector<JitBlock*>);
}

u32 ARM9CodeFetchCyclesForJump(ARMv5* cpu, u32 addr, bool branch, u32 regionCodeCycles)
{
    if (addr < cpu->ITCMSize)
        return 1;

    u32 cycles = regionCodeCycles;
    if (cycles == 0xFF)
        cycles = (branch || !(addr & 0x1F)) ? A32CodeCacheTiming : 1;
    return cycles;
}

u32 ARM9JumpCycles(ARMv5* cpu, u32 target, bool thumbTarget,
                   u32& newPC, u32& regionCodeCycles)
{
    regionCodeCycles = cpu->MemTimings[target >> 12][0];

    if (thumbTarget)
    {
        u32 addr = target & ~0x1u;
        newPC = addr + 2;

        if (addr & 0x2)
        {
            return ARM9CodeFetchCyclesForJump(cpu, addr - 2, true, regionCodeCycles) +
                   ARM9CodeFetchCyclesForJump(cpu, addr + 2, false, regionCodeCycles);
        }

        return ARM9CodeFetchCyclesForJump(cpu, addr, true, regionCodeCycles);
    }

    u32 addr = target & ~0x3u;
    newPC = addr + 4;
    return ARM9CodeFetchCyclesForJump(cpu, addr, true, regionCodeCycles) +
           ARM9CodeFetchCyclesForJump(cpu, addr + 4, false, regionCodeCycles);
}

u32 ARM7JumpCycles(u32 target, bool thumbTarget, u32& newPC,
                   u32& codeRegion, u32& codeCycles)
{
    codeRegion = target >> 24;
    codeCycles = target >> 15;

    if (thumbTarget)
    {
        u32 addr = target & ~0x1u;
        newPC = addr + 2;
        return NDS::ARM7MemTimings[codeCycles][0] +
               NDS::ARM7MemTimings[codeCycles][1];
    }

    u32 addr = target & ~0x3u;
    newPC = addr + 4;
    return NDS::ARM7MemTimings[codeCycles][2] +
           NDS::ARM7MemTimings[codeCycles][3];
}

struct A32JitProfileCounters
{
    u64 Events;
    u64 ArmInterp;
    u64 ThumbInterp;
    u64 ArmMem;
    u64 ThumbMem;
    u64 FastMemRead;
    u64 FastMemWrite;
    u32 DirectMemRead;
    u32 DirectMemWrite;
    u32 DirectMemNarrowRead;
    u32 DirectMemNarrowWrite;
    u32 DirectMemRegRead;
    u32 DirectMemRegWrite;
    u32 DirectMemPostRead;
    u32 DirectMemPostWrite;
    u32 DirectMemWritebackRead;
    u32 DirectMemWritebackWrite;
    u32 DirectMemLDM;
    u32 DirectMemSTM;
    // M16 pcrel-base propagation (compile-time — one per compiled instruction)
    u32 PcrelConstKnown;
    u32 PcrelConstUnsafe;
    u32 PcrelConstRegionMainRAM;
    u32 PcrelConstRegionITCM;
    u32 PcrelConstRegionWRAM7;
    u32 PcrelConstRegionShared;
    // M16 ALU immediate propagation (compile-time — one per propagated ALU instruction)
    u32 AluImmPropKnown;
    // M18 self-link: ARM9 ARM-mode backward self-loop direct re-entry (compile-time)
    u32 SelfLinkEmitted;
    // M16 known-base fast path (runtime — one per block execution)
    u32 KnownBaseFastpath;
    u32 KnownBaseGuardFail;
    u32 KnownBaseRegionMainRAM;
    u32 KnownBaseRegionITCM;
    u32 KnownBaseRegionDTCM;
    u32 KnownBaseRegionWRAM7;
    u32 KnownBaseRegionShared;
    // M17 decoded single-transfer helpers.
    u32 DecodedMemRead8;
    u32 DecodedMemRead16;
    u32 DecodedMemRead32;
    u32 DecodedMemWrite8;
    u32 DecodedMemWrite16;
    u32 DecodedMemWrite32;
    u32 DecodedMemUnsafeARM9;
    u32 DecodedMemUnsafeARM7;
    u32 DecodedMemUnsafeIO9;
    u32 DecodedMemUnsafeIO7;
    u32 DecodedMemUnsafeVRAM;
    u32 DecodedMemUnsafeVWRAM;
    u32 DecodedMemUnsafeBIOS;
    u32 DecodedMemUnsafeOther;
    u32 DecodedMemDirectEligible;
    u32 DecodedMemDirectIORead;
    u32 DecodedMemDirectIOWrite;
    u32 DecodedMemDirectGPU3DRead;
    u32 DecodedMemDirectGPU3DWrite;
    u32 DecodedMemDirectIRQMemRead;
    u32 DecodedMemDirectIRQMemWrite;
    u32 DecodedMemDirectDivSqrtRead;
    u32 DecodedMemDirectDivSqrtWrite;
    u32 DecodedMemDirectDMARead;
    u32 DecodedMemDirectDMAWrite;
    u32 DecodedMemDirectTimerRead;
    u32 DecodedMemDirectTimerWrite;
    u32 DecodedMemDirectInputRead;
    u32 DecodedMemDirectInputWrite;
    // Runtime direct-safe helper calls that missed compile-time direct emission.
    u32 DirectEligibleSingle;
    u32 DirectEligibleLDM;
    u32 DirectEligibleSTM;
    u32 DirectEligibleARM9;
    u32 DirectEligibleARM7;
    u32 DirectEligibleRegionMainRAM;
    u32 DirectEligibleRegionITCM;
    u32 DirectEligibleRegionDTCM;
    u32 DirectEligibleRegionWRAM7;
    u32 DirectEligibleRegionShared;
    u32 DirectEligibleLoad;
    u32 DirectEligibleStore;
    u32 DirectEligibleByte;
    u32 DirectEligibleHalf;
    u32 DirectEligibleWord;
    u32 DirectEligibleSingleRegOffset;
    u32 DirectEligibleSingleImmOffset;
    u32 DirectEligibleSinglePost;
    u32 DirectEligibleWriteback;
    u32 DirectEligibleCond;
    u32 RuntimeDirectSingleFastpath;
    u32 RuntimeDirectSingleGuardFail;
    u32 RuntimeDirectSingleDecodedFallback;
    u32 RuntimeDirectSingleRead;
    u32 RuntimeDirectSingleWrite;
    u32 RuntimeDirectRegionMainRAM;
    u32 RuntimeDirectRegionITCM;
    u32 RuntimeDirectRegionDTCM;
    u32 RuntimeDirectRegionWRAM7;
    u32 RuntimeDirectRegionShared;
    u32 RuntimeDirectRegionBIOS7;
    u32 DecodedMemIOGPU2D;
    u32 DecodedMemIOGPU3D;
    u32 DecodedMemIODMA;
    u32 DecodedMemIOTimer;
    u32 DecodedMemIOInput;
    u32 DecodedMemIOIPC;
    u32 DecodedMemIOCart;
    u32 DecodedMemIODivSqrt;
    u32 DecodedMemIOIRQPowerMem;
    u32 DecodedMemIOSPU;
    u32 DecodedMemIOOther;
    u64 ArmInterpKinds[ARMInstrInfo::ak_Count];
    u64 ThumbInterpKinds[ARMInstrInfo::tk_Count];
    u64 ArmMemKinds[ARMInstrInfo::ak_Count];
    u64 ThumbMemKinds[ARMInstrInfo::tk_Count];
    u64 DirectEligibleArmKinds[ARMInstrInfo::ak_Count];
    u64 MemReasons[MemReason_Count];
};

const u64 JitProfileEventWindow = 1000000ULL;
A32JitProfileCounters JitProfileCounters;

void FormatArmKind(u32 kind, char* out, size_t outLen)
{
    static const char* const ALUOps[] = {
        "AND", "EOR", "SUB", "RSB", "ADD", "ADC",
        "SBC", "RSC", "ORR", "MOV", "BIC", "MVN",
    };
    static const char* const ALUForms[] = {
        "REG_LSL_IMM", "REG_LSR_IMM", "REG_ASR_IMM", "REG_ROR_IMM",
        "REG_LSL_REG", "REG_LSR_REG", "REG_ASR_REG", "REG_ROR_REG",
        "IMM",
        "REG_LSL_IMM_S", "REG_LSR_IMM_S", "REG_ASR_IMM_S", "REG_ROR_IMM_S",
        "REG_LSL_REG_S", "REG_LSR_REG_S", "REG_ASR_REG_S", "REG_ROR_REG_S",
        "IMM_S",
    };
    static const char* const TestOps[] = { "TST", "TEQ", "CMP", "CMN" };
    static const char* const TestForms[] = {
        "REG_LSL_IMM", "REG_LSR_IMM", "REG_ASR_IMM", "REG_ROR_IMM",
        "REG_LSL_REG", "REG_LSR_REG", "REG_ASR_REG", "REG_ROR_REG",
        "IMM",
    };
    static const char* const WBMemOps[] = { "STR", "STRB", "LDR", "LDRB" };
    static const char* const WBMemForms[] = {
        "REG_LSL", "REG_LSR", "REG_ASR", "REG_ROR", "IMM",
        "POST_REG_LSL", "POST_REG_LSR", "POST_REG_ASR", "POST_REG_ROR", "POST_IMM",
    };
    static const char* const HDMemOps[] = { "STRH", "LDRD", "STRD", "LDRH", "LDRSB", "LDRSH" };
    static const char* const HDMemForms[] = { "REG", "IMM", "POST_REG", "POST_IMM" };

    if (kind >= ARMInstrInfo::ak_AND_REG_LSL_IMM && kind <= ARMInstrInfo::ak_MVN_IMM_S)
    {
        const u32 idx = kind - ARMInstrInfo::ak_AND_REG_LSL_IMM;
        snprintf(out, outLen, "ak_%s_%s", ALUOps[idx / 18], ALUForms[idx % 18]);
        return;
    }

    if (kind >= ARMInstrInfo::ak_TST_REG_LSL_IMM && kind <= ARMInstrInfo::ak_CMN_IMM)
    {
        const u32 idx = kind - ARMInstrInfo::ak_TST_REG_LSL_IMM;
        snprintf(out, outLen, "ak_%s_%s", TestOps[idx / 9], TestForms[idx % 9]);
        return;
    }

    if (kind >= ARMInstrInfo::ak_STR_REG_LSL && kind <= ARMInstrInfo::ak_LDRB_POST_IMM)
    {
        const u32 idx = kind - ARMInstrInfo::ak_STR_REG_LSL;
        snprintf(out, outLen, "ak_%s_%s", WBMemOps[idx / 10], WBMemForms[idx % 10]);
        return;
    }

    if (kind >= ARMInstrInfo::ak_STRH_REG && kind <= ARMInstrInfo::ak_LDRSH_POST_IMM)
    {
        const u32 idx = kind - ARMInstrInfo::ak_STRH_REG;
        snprintf(out, outLen, "ak_%s_%s", HDMemOps[idx / 4], HDMemForms[idx % 4]);
        return;
    }

    switch (kind)
    {
    case ARMInstrInfo::ak_MUL: snprintf(out, outLen, "ak_MUL"); return;
    case ARMInstrInfo::ak_MLA: snprintf(out, outLen, "ak_MLA"); return;
    case ARMInstrInfo::ak_UMULL: snprintf(out, outLen, "ak_UMULL"); return;
    case ARMInstrInfo::ak_UMLAL: snprintf(out, outLen, "ak_UMLAL"); return;
    case ARMInstrInfo::ak_SMULL: snprintf(out, outLen, "ak_SMULL"); return;
    case ARMInstrInfo::ak_SMLAL: snprintf(out, outLen, "ak_SMLAL"); return;
    case ARMInstrInfo::ak_SMLAxy: snprintf(out, outLen, "ak_SMLAxy"); return;
    case ARMInstrInfo::ak_SMLAWy: snprintf(out, outLen, "ak_SMLAWy"); return;
    case ARMInstrInfo::ak_SMULWy: snprintf(out, outLen, "ak_SMULWy"); return;
    case ARMInstrInfo::ak_SMLALxy: snprintf(out, outLen, "ak_SMLALxy"); return;
    case ARMInstrInfo::ak_SMULxy: snprintf(out, outLen, "ak_SMULxy"); return;
    case ARMInstrInfo::ak_CLZ: snprintf(out, outLen, "ak_CLZ"); return;
    case ARMInstrInfo::ak_QADD: snprintf(out, outLen, "ak_QADD"); return;
    case ARMInstrInfo::ak_QSUB: snprintf(out, outLen, "ak_QSUB"); return;
    case ARMInstrInfo::ak_QDADD: snprintf(out, outLen, "ak_QDADD"); return;
    case ARMInstrInfo::ak_QDSUB: snprintf(out, outLen, "ak_QDSUB"); return;
    case ARMInstrInfo::ak_SWP: snprintf(out, outLen, "ak_SWP"); return;
    case ARMInstrInfo::ak_SWPB: snprintf(out, outLen, "ak_SWPB"); return;
    case ARMInstrInfo::ak_LDM: snprintf(out, outLen, "ak_LDM"); return;
    case ARMInstrInfo::ak_STM: snprintf(out, outLen, "ak_STM"); return;
    case ARMInstrInfo::ak_B: snprintf(out, outLen, "ak_B"); return;
    case ARMInstrInfo::ak_BL: snprintf(out, outLen, "ak_BL"); return;
    case ARMInstrInfo::ak_BLX_IMM: snprintf(out, outLen, "ak_BLX_IMM"); return;
    case ARMInstrInfo::ak_BX: snprintf(out, outLen, "ak_BX"); return;
    case ARMInstrInfo::ak_BLX_REG: snprintf(out, outLen, "ak_BLX_REG"); return;
    case ARMInstrInfo::ak_UNK: snprintf(out, outLen, "ak_UNK"); return;
    case ARMInstrInfo::ak_MSR_IMM: snprintf(out, outLen, "ak_MSR_IMM"); return;
    case ARMInstrInfo::ak_MSR_REG: snprintf(out, outLen, "ak_MSR_REG"); return;
    case ARMInstrInfo::ak_MRS: snprintf(out, outLen, "ak_MRS"); return;
    case ARMInstrInfo::ak_MCR: snprintf(out, outLen, "ak_MCR"); return;
    case ARMInstrInfo::ak_MRC: snprintf(out, outLen, "ak_MRC"); return;
    case ARMInstrInfo::ak_SVC: snprintf(out, outLen, "ak_SVC"); return;
    case ARMInstrInfo::ak_Nop: snprintf(out, outLen, "ak_Nop"); return;
    default: snprintf(out, outLen, "ak_%u", kind); return;
    }
}

const char* ThumbKindName(u32 kind)
{
    switch (kind)
    {
    case ARMInstrInfo::tk_LSL_IMM: return "tk_LSL_IMM";
    case ARMInstrInfo::tk_LSR_IMM: return "tk_LSR_IMM";
    case ARMInstrInfo::tk_ASR_IMM: return "tk_ASR_IMM";
    case ARMInstrInfo::tk_ADD_REG_: return "tk_ADD_REG_";
    case ARMInstrInfo::tk_SUB_REG_: return "tk_SUB_REG_";
    case ARMInstrInfo::tk_ADD_IMM_: return "tk_ADD_IMM_";
    case ARMInstrInfo::tk_SUB_IMM_: return "tk_SUB_IMM_";
    case ARMInstrInfo::tk_MOV_IMM: return "tk_MOV_IMM";
    case ARMInstrInfo::tk_CMP_IMM: return "tk_CMP_IMM";
    case ARMInstrInfo::tk_ADD_IMM: return "tk_ADD_IMM";
    case ARMInstrInfo::tk_SUB_IMM: return "tk_SUB_IMM";
    case ARMInstrInfo::tk_AND_REG: return "tk_AND_REG";
    case ARMInstrInfo::tk_EOR_REG: return "tk_EOR_REG";
    case ARMInstrInfo::tk_LSL_REG: return "tk_LSL_REG";
    case ARMInstrInfo::tk_LSR_REG: return "tk_LSR_REG";
    case ARMInstrInfo::tk_ASR_REG: return "tk_ASR_REG";
    case ARMInstrInfo::tk_ADC_REG: return "tk_ADC_REG";
    case ARMInstrInfo::tk_SBC_REG: return "tk_SBC_REG";
    case ARMInstrInfo::tk_ROR_REG: return "tk_ROR_REG";
    case ARMInstrInfo::tk_TST_REG: return "tk_TST_REG";
    case ARMInstrInfo::tk_NEG_REG: return "tk_NEG_REG";
    case ARMInstrInfo::tk_CMP_REG: return "tk_CMP_REG";
    case ARMInstrInfo::tk_CMN_REG: return "tk_CMN_REG";
    case ARMInstrInfo::tk_ORR_REG: return "tk_ORR_REG";
    case ARMInstrInfo::tk_MUL_REG: return "tk_MUL_REG";
    case ARMInstrInfo::tk_BIC_REG: return "tk_BIC_REG";
    case ARMInstrInfo::tk_MVN_REG: return "tk_MVN_REG";
    case ARMInstrInfo::tk_ADD_HIREG: return "tk_ADD_HIREG";
    case ARMInstrInfo::tk_CMP_HIREG: return "tk_CMP_HIREG";
    case ARMInstrInfo::tk_MOV_HIREG: return "tk_MOV_HIREG";
    case ARMInstrInfo::tk_ADD_PCREL: return "tk_ADD_PCREL";
    case ARMInstrInfo::tk_ADD_SPREL: return "tk_ADD_SPREL";
    case ARMInstrInfo::tk_ADD_SP: return "tk_ADD_SP";
    case ARMInstrInfo::tk_LDR_PCREL: return "tk_LDR_PCREL";
    case ARMInstrInfo::tk_STR_REG: return "tk_STR_REG";
    case ARMInstrInfo::tk_STRB_REG: return "tk_STRB_REG";
    case ARMInstrInfo::tk_LDR_REG: return "tk_LDR_REG";
    case ARMInstrInfo::tk_LDRB_REG: return "tk_LDRB_REG";
    case ARMInstrInfo::tk_STRH_REG: return "tk_STRH_REG";
    case ARMInstrInfo::tk_LDRSB_REG: return "tk_LDRSB_REG";
    case ARMInstrInfo::tk_LDRH_REG: return "tk_LDRH_REG";
    case ARMInstrInfo::tk_LDRSH_REG: return "tk_LDRSH_REG";
    case ARMInstrInfo::tk_STR_IMM: return "tk_STR_IMM";
    case ARMInstrInfo::tk_LDR_IMM: return "tk_LDR_IMM";
    case ARMInstrInfo::tk_STRB_IMM: return "tk_STRB_IMM";
    case ARMInstrInfo::tk_LDRB_IMM: return "tk_LDRB_IMM";
    case ARMInstrInfo::tk_STRH_IMM: return "tk_STRH_IMM";
    case ARMInstrInfo::tk_LDRH_IMM: return "tk_LDRH_IMM";
    case ARMInstrInfo::tk_STR_SPREL: return "tk_STR_SPREL";
    case ARMInstrInfo::tk_LDR_SPREL: return "tk_LDR_SPREL";
    case ARMInstrInfo::tk_PUSH: return "tk_PUSH";
    case ARMInstrInfo::tk_POP: return "tk_POP";
    case ARMInstrInfo::tk_LDMIA: return "tk_LDMIA";
    case ARMInstrInfo::tk_STMIA: return "tk_STMIA";
    case ARMInstrInfo::tk_BCOND: return "tk_BCOND";
    case ARMInstrInfo::tk_BX: return "tk_BX";
    case ARMInstrInfo::tk_BLX_REG: return "tk_BLX_REG";
    case ARMInstrInfo::tk_B: return "tk_B";
    case ARMInstrInfo::tk_BL_LONG_1: return "tk_BL_LONG_1";
    case ARMInstrInfo::tk_BL_LONG_2: return "tk_BL_LONG_2";
    case ARMInstrInfo::tk_UNK: return "tk_UNK";
    case ARMInstrInfo::tk_SVC: return "tk_SVC";
    case ARMInstrInfo::tk_BL_LONG: return "tk_BL_LONG";
    default: return nullptr;
    }
}

void FormatThumbKind(u32 kind, char* out, size_t outLen)
{
    const char* name = ThumbKindName(kind);
    if (name)
        snprintf(out, outLen, "%s", name);
    else
        snprintf(out, outLen, "tk_%u", kind);
}

void AppendTopKinds(char* out, size_t outLen, const u64* counts, u32 kindCount, bool thumb)
{
    u32 used[3] = { kindCount, kindCount, kindCount };
    size_t pos = 0;
    out[0] = '\0';

    for (u32 rank = 0; rank < 3; rank++)
    {
        u32 bestKind = kindCount;
        u64 bestCount = 0;

        for (u32 kind = 0; kind < kindCount; kind++)
        {
            if (kind == used[0] || kind == used[1] || kind == used[2])
                continue;
            if (counts[kind] > bestCount)
            {
                bestCount = counts[kind];
                bestKind = kind;
            }
        }

        if (!bestCount)
            break;

        used[rank] = bestKind;

        char label[64];
        if (thumb)
            FormatThumbKind(bestKind, label, sizeof(label));
        else
            FormatArmKind(bestKind, label, sizeof(label));

        int written = snprintf(out + pos, outLen - pos, "%s%s=%llu",
            pos ? " " : "", label, (unsigned long long)bestCount);
        if (written <= 0)
            break;

        pos += (size_t)written;
        if (pos >= outLen)
        {
            out[outLen - 1] = '\0';
            return;
        }
    }

    if (!pos)
        snprintf(out, outLen, "none");
}

const char* MemHelperReasonName(u32 reason)
{
    switch (reason)
    {
    case MemReason_Thumb: return "thumb";
    case MemReason_CondNonAL: return "cond";
    case MemReason_UnsupportedKind: return "unsupported";
    case MemReason_PCRegister: return "pc-reg";
    case MemReason_PCBase: return "pc-base";
    case MemReason_PCDest: return "pc-dst";
    case MemReason_PCOffset: return "pc-offset";
    case MemReason_RRXOffset: return "rrx";
    case MemReason_UnsafeRegion: return "unsafe-region";
    case MemReason_Unaligned: return "unaligned";
    case MemReason_CodeWrite: return "code-write";
    case MemReason_DirectEligible: return "direct-eligible";
    case MemReason_LDMEmpty: return "ldm-empty";
    case MemReason_LDMBasePC: return "ldm-base-pc";
    case MemReason_LDMBankedUser: return "ldm-banked-user";
    case MemReason_LDMRestoreCPSR: return "ldm-restore-cpsr";
    case MemReason_LDMBaseInListWB: return "ldm-base-wb";
    case MemReason_LDMUnsafeRegion: return "ldm-unsafe-region";
    case MemReason_LDMUnaligned: return "ldm-unaligned";
    case MemReason_LDMRangeWrap: return "ldm-range-wrap";
    case MemReason_STMBankedUser: return "stm-banked-user";
    case MemReason_STMPCOrBase: return "stm-pc-base";
    case MemReason_STMUnsafeRegion: return "stm-unsafe-region";
    case MemReason_STMUnaligned: return "stm-unaligned";
    case MemReason_STMRangeWrap: return "stm-range-wrap";
    case MemReason_STMCodeWrite: return "stm-code-write";
    default: return "unknown";
    }
}

void AppendTopReasons(char* out, size_t outLen, const u64* counts)
{
    u32 used[5] = { MemReason_Count, MemReason_Count, MemReason_Count, MemReason_Count, MemReason_Count };
    size_t pos = 0;
    out[0] = '\0';

    for (u32 rank = 0; rank < 5; rank++)
    {
        u32 bestReason = MemReason_Count;
        u64 bestCount = 0;

        for (u32 reason = 0; reason < MemReason_Count; reason++)
        {
            if (reason == used[0] || reason == used[1] || reason == used[2] ||
                reason == used[3] || reason == used[4])
                continue;
            if (counts[reason] > bestCount)
            {
                bestCount = counts[reason];
                bestReason = reason;
            }
        }

        if (!bestCount)
            break;

        used[rank] = bestReason;

        int written = snprintf(out + pos, outLen - pos, "%s%s=%llu",
            pos ? " " : "", MemHelperReasonName(bestReason), (unsigned long long)bestCount);
        if (written <= 0)
            break;

        pos += (size_t)written;
        if (pos >= outLen)
        {
            out[outLen - 1] = '\0';
            return;
        }
    }

    if (!pos)
        snprintf(out, outLen, "none");
}

void LogA32JitProfile()
{
    char armInterp[256];
    char thumbInterp[256];
    char armMem[256];
    char thumbMem[256];
    char directEligibleArm[256];
    char memReasons[256];

    AppendTopKinds(armInterp, sizeof(armInterp),
        JitProfileCounters.ArmInterpKinds, ARMInstrInfo::ak_Count, false);
    AppendTopKinds(thumbInterp, sizeof(thumbInterp),
        JitProfileCounters.ThumbInterpKinds, ARMInstrInfo::tk_Count, true);
    AppendTopKinds(armMem, sizeof(armMem),
        JitProfileCounters.ArmMemKinds, ARMInstrInfo::ak_Count, false);
    AppendTopKinds(thumbMem, sizeof(thumbMem),
        JitProfileCounters.ThumbMemKinds, ARMInstrInfo::tk_Count, true);
    AppendTopKinds(directEligibleArm, sizeof(directEligibleArm),
        JitProfileCounters.DirectEligibleArmKinds, ARMInstrInfo::ak_Count, false);
    AppendTopReasons(memReasons, sizeof(memReasons), JitProfileCounters.MemReasons);

    A32JIT_LOGI("melonDS A32 JIT stats: events=%llu interp_arm=%llu interp_thumb=%llu "
           "mem_arm=%llu mem_thumb=%llu fast_read=%llu fast_write=%llu "
           "direct_read=%u direct_write=%u direct_narrow_read=%u direct_narrow_write=%u "
           "direct_reg_read=%u direct_reg_write=%u direct_post_read=%u direct_post_write=%u "
           "direct_wb_read=%u direct_wb_write=%u direct_ldm=%u direct_stm=%u",
           (unsigned long long)JitProfileCounters.Events,
           (unsigned long long)JitProfileCounters.ArmInterp,
           (unsigned long long)JitProfileCounters.ThumbInterp,
           (unsigned long long)JitProfileCounters.ArmMem,
           (unsigned long long)JitProfileCounters.ThumbMem,
           (unsigned long long)JitProfileCounters.FastMemRead,
           (unsigned long long)JitProfileCounters.FastMemWrite,
           JitProfileCounters.DirectMemRead,
           JitProfileCounters.DirectMemWrite,
           JitProfileCounters.DirectMemNarrowRead,
           JitProfileCounters.DirectMemNarrowWrite,
           JitProfileCounters.DirectMemRegRead,
           JitProfileCounters.DirectMemRegWrite,
           JitProfileCounters.DirectMemPostRead,
           JitProfileCounters.DirectMemPostWrite,
           JitProfileCounters.DirectMemWritebackRead,
           JitProfileCounters.DirectMemWritebackWrite,
           JitProfileCounters.DirectMemLDM,
           JitProfileCounters.DirectMemSTM);
    A32JIT_LOGI("melonDS A32 JIT top: interp ARM[%s] THUMB[%s]; mem ARM[%s] THUMB[%s]",
           armInterp, thumbInterp, armMem, thumbMem);
    A32JIT_LOGI("melonDS A32 JIT mem reasons: %s", memReasons);
    A32JIT_LOGI("melonDS A32 JIT runtime-direct: "
           "single_fast:%u single_fail:%u decoded_fallback:%u read:%u write:%u "
           "region=mainram:%u itcm:%u dtcm:%u wram7:%u shared:%u bios7:%u",
           JitProfileCounters.RuntimeDirectSingleFastpath,
           JitProfileCounters.RuntimeDirectSingleGuardFail,
           JitProfileCounters.RuntimeDirectSingleDecodedFallback,
           JitProfileCounters.RuntimeDirectSingleRead,
           JitProfileCounters.RuntimeDirectSingleWrite,
           JitProfileCounters.RuntimeDirectRegionMainRAM,
           JitProfileCounters.RuntimeDirectRegionITCM,
           JitProfileCounters.RuntimeDirectRegionDTCM,
           JitProfileCounters.RuntimeDirectRegionWRAM7,
           JitProfileCounters.RuntimeDirectRegionShared,
           JitProfileCounters.RuntimeDirectRegionBIOS7);
    A32JIT_LOGI("melonDS A32 JIT direct-eligible: "
           "shape=single:%u ldm:%u stm:%u cpu=arm9:%u arm7:%u "
           "region=mainram:%u itcm:%u dtcm:%u wram7:%u shared:%u "
           "access=load:%u store:%u size=byte:%u half:%u word:%u "
           "single_mode=reg:%u imm:%u post:%u wb:%u cond:%u top ARM[%s]",
           JitProfileCounters.DirectEligibleSingle,
           JitProfileCounters.DirectEligibleLDM,
           JitProfileCounters.DirectEligibleSTM,
           JitProfileCounters.DirectEligibleARM9,
           JitProfileCounters.DirectEligibleARM7,
           JitProfileCounters.DirectEligibleRegionMainRAM,
           JitProfileCounters.DirectEligibleRegionITCM,
           JitProfileCounters.DirectEligibleRegionDTCM,
           JitProfileCounters.DirectEligibleRegionWRAM7,
           JitProfileCounters.DirectEligibleRegionShared,
           JitProfileCounters.DirectEligibleLoad,
           JitProfileCounters.DirectEligibleStore,
           JitProfileCounters.DirectEligibleByte,
           JitProfileCounters.DirectEligibleHalf,
           JitProfileCounters.DirectEligibleWord,
           JitProfileCounters.DirectEligibleSingleRegOffset,
           JitProfileCounters.DirectEligibleSingleImmOffset,
           JitProfileCounters.DirectEligibleSinglePost,
           JitProfileCounters.DirectEligibleWriteback,
           JitProfileCounters.DirectEligibleCond,
           directEligibleArm);
    A32JIT_LOGI("melonDS A32 JIT M16 pcrel: known=%u unsafe=%u region=mainram:%u itcm:%u wram7:%u shared:%u alu_prop=%u",
           JitProfileCounters.PcrelConstKnown,
           JitProfileCounters.PcrelConstUnsafe,
           JitProfileCounters.PcrelConstRegionMainRAM,
           JitProfileCounters.PcrelConstRegionITCM,
           JitProfileCounters.PcrelConstRegionWRAM7,
           JitProfileCounters.PcrelConstRegionShared,
           JitProfileCounters.AluImmPropKnown);
    A32JIT_LOGI("melonDS A32 JIT M18: self_link=%u fast_branch=%u",
           JitProfileCounters.SelfLinkEmitted,
           g_M18FastBranchEmitted);
    A32JIT_LOGI("melonDS A32 JIT M16 known-base: fastpath=%u guard_fail=%u "
           "region=mainram:%u itcm:%u dtcm:%u wram7:%u shared:%u "
           "(fail_ratio=%.1f%%)",
           JitProfileCounters.KnownBaseFastpath,
           JitProfileCounters.KnownBaseGuardFail,
           JitProfileCounters.KnownBaseRegionMainRAM,
           JitProfileCounters.KnownBaseRegionITCM,
           JitProfileCounters.KnownBaseRegionDTCM,
           JitProfileCounters.KnownBaseRegionWRAM7,
           JitProfileCounters.KnownBaseRegionShared,
           JitProfileCounters.KnownBaseFastpath
               ? 100.0 * JitProfileCounters.KnownBaseGuardFail / JitProfileCounters.KnownBaseFastpath
               : 0.0);
    A32JIT_LOGI("melonDS A32 JIT M17 decoded-mem: "
           "read8:%u read16:%u read32:%u write8:%u write16:%u write32:%u "
           "unsafe_cpu=arm9:%u arm7:%u unsafe_region=io9:%u io7:%u vram:%u vwram:%u bios:%u other:%u "
           "direct_eligible:%u direct_io_read:%u direct_io_write:%u direct_gpu3d_read:%u direct_gpu3d_write:%u "
           "direct_irqmem_read:%u direct_irqmem_write:%u direct_divsqrt_read:%u direct_divsqrt_write:%u "
           "direct_dma_read:%u direct_dma_write:%u direct_timer_read:%u direct_timer_write:%u "
           "direct_input_read:%u direct_input_write:%u "
           "io_bucket=gpu2d:%u gpu3d:%u dma:%u timer:%u input:%u ipc:%u cart:%u divsqrt:%u irqmem:%u spu:%u other:%u",
           JitProfileCounters.DecodedMemRead8,
           JitProfileCounters.DecodedMemRead16,
           JitProfileCounters.DecodedMemRead32,
           JitProfileCounters.DecodedMemWrite8,
           JitProfileCounters.DecodedMemWrite16,
           JitProfileCounters.DecodedMemWrite32,
           JitProfileCounters.DecodedMemUnsafeARM9,
           JitProfileCounters.DecodedMemUnsafeARM7,
           JitProfileCounters.DecodedMemUnsafeIO9,
           JitProfileCounters.DecodedMemUnsafeIO7,
           JitProfileCounters.DecodedMemUnsafeVRAM,
           JitProfileCounters.DecodedMemUnsafeVWRAM,
           JitProfileCounters.DecodedMemUnsafeBIOS,
           JitProfileCounters.DecodedMemUnsafeOther,
           JitProfileCounters.DecodedMemDirectEligible,
           JitProfileCounters.DecodedMemDirectIORead,
           JitProfileCounters.DecodedMemDirectIOWrite,
           JitProfileCounters.DecodedMemDirectGPU3DRead,
           JitProfileCounters.DecodedMemDirectGPU3DWrite,
           JitProfileCounters.DecodedMemDirectIRQMemRead,
           JitProfileCounters.DecodedMemDirectIRQMemWrite,
           JitProfileCounters.DecodedMemDirectDivSqrtRead,
           JitProfileCounters.DecodedMemDirectDivSqrtWrite,
           JitProfileCounters.DecodedMemDirectDMARead,
           JitProfileCounters.DecodedMemDirectDMAWrite,
           JitProfileCounters.DecodedMemDirectTimerRead,
           JitProfileCounters.DecodedMemDirectTimerWrite,
           JitProfileCounters.DecodedMemDirectInputRead,
           JitProfileCounters.DecodedMemDirectInputWrite,
           JitProfileCounters.DecodedMemIOGPU2D,
           JitProfileCounters.DecodedMemIOGPU3D,
           JitProfileCounters.DecodedMemIODMA,
           JitProfileCounters.DecodedMemIOTimer,
           JitProfileCounters.DecodedMemIOInput,
           JitProfileCounters.DecodedMemIOIPC,
           JitProfileCounters.DecodedMemIOCart,
           JitProfileCounters.DecodedMemIODivSqrt,
           JitProfileCounters.DecodedMemIOIRQPowerMem,
           JitProfileCounters.DecodedMemIOSPU,
           JitProfileCounters.DecodedMemIOOther);

    JitProfileCounters = {};
}

void RecordA32JitProfile(const A32InstrDesc* desc, bool memoryHelper)
{
    JitProfileCounters.Events++;

    if (desc->thumb)
    {
        if (memoryHelper)
        {
            JitProfileCounters.ThumbMem++;
            if (desc->kind < ARMInstrInfo::tk_Count)
                JitProfileCounters.ThumbMemKinds[desc->kind]++;
        }
        else
        {
            JitProfileCounters.ThumbInterp++;
            if (desc->kind < ARMInstrInfo::tk_Count)
                JitProfileCounters.ThumbInterpKinds[desc->kind]++;
        }
    }
    else
    {
        if (memoryHelper)
        {
            JitProfileCounters.ArmMem++;
            if (desc->kind < ARMInstrInfo::ak_Count)
                JitProfileCounters.ArmMemKinds[desc->kind]++;
        }
        else
        {
            JitProfileCounters.ArmInterp++;
            if (desc->kind < ARMInstrInfo::ak_Count)
                JitProfileCounters.ArmInterpKinds[desc->kind]++;
        }
    }

    if (JitProfileCounters.Events >= JitProfileEventWindow)
        LogA32JitProfile();
}

u32 DpReg(ARMDataOp op, bool setFlags, int rn, int rd, int rm)
{
    return 0xE0000000
        | ((u32)op << 21)
        | (setFlags ? (1u << 20) : 0)
        | ((u32)rn << 16)
        | ((u32)rd << 12)
        | (u32)rm;
}

bool IsPatchableArmAluKind(u16 kind)
{
    return (kind >= ARMInstrInfo::ak_AND_REG_LSL_IMM &&
            kind <= ARMInstrInfo::ak_MVN_IMM_S) ||
           (kind >= ARMInstrInfo::ak_TST_REG_LSL_IMM &&
            kind <= ARMInstrInfo::ak_CMN_IMM);
}

bool ArmOperand2UpdatesCarry(u32 op, bool isImmForm)
{
    if (isImmForm)
        return ((op >> 8) & 0xF) != 0;

    u32 shift = (op >> 7) & 0x1F;
    u32 shiftType = (op >> 5) & 0x3;
    return shift != 0 || shiftType != Shift_LSL;
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
    if (EnableA32JitProfiling)
        RecordA32JitProfile(desc, false);

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

// ARM-state branch helpers — mirrors the Thumb ones but for A32 semantics.

void RunArmBranch(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->JumpTo(desc->target);
}

void RunArmCondBranch(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    if (cpu->CheckCondition(desc->aux))
        cpu->JumpTo(desc->target);
    else
        cpu->AddCycles_C();
}

void RunArmBL(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->R[14] = desc->aux;  // aux = instr.Addr + 4  (return address)
    cpu->JumpTo(desc->target);
}

void RunArmBX(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    cpu->JumpTo(cpu->R[desc->aux & 0xF]);
}

void RunArmBLXReg(ARM* cpu, const A32BranchDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;
    u32 lr = desc->r15 - 4;  // = instr.Addr + 4
    cpu->JumpTo(cpu->R[desc->aux & 0xF]);
    cpu->R[14] = lr;
}

void RunArmLoadedPC(ARM* cpu, u32 pc)
{
    if (cpu->Num == 1)
        pc &= ~0x1;
    cpu->JumpTo(pc);
}

void RunArmLoadedPCRestoreCPSR(ARM* cpu, u32 pc)
{
    if (cpu->Num == 1)
        pc &= ~0x1;
    cpu->JumpTo(pc, true);
}

bool IsArmWordTransferKind(u16 kind)
{
    return (kind >= ARMInstrInfo::ak_STR_REG_LSL &&
            kind <= ARMInstrInfo::ak_LDRB_POST_IMM);
}

bool IsArmHalfTransferKind(u16 kind)
{
    return (kind >= ARMInstrInfo::ak_STRH_REG &&
            kind <= ARMInstrInfo::ak_LDRSH_POST_IMM);
}

u32 ArmImmOffset(u32 instr)
{
    u32 offset = instr & 0xFFF;
    return (instr & (1u << 23)) ? offset : (u32)-offset;
}

u32 ArmRegOffset(ARM* cpu, u32 instr)
{
    u32 offset = cpu->R[instr & 0xF];
    u32 shift = (instr >> 7) & 0x1F;

    switch ((instr >> 5) & 0x3)
    {
    case Shift_LSL:
        offset <<= shift;
        break;
    case Shift_LSR:
        offset = shift ? (offset >> shift) : 0;
        break;
    case Shift_ASR:
        offset = shift ? (u32)((s32)offset >> shift) : (u32)((s32)offset >> 31);
        break;
    case Shift_ROR:
        if (shift == 0)
            offset = (offset >> 1) | ((cpu->CPSR & 0x20000000) << 2);
        else
            offset = ROR(offset, shift);
        break;
    }

    return (instr & (1u << 23)) ? offset : (u32)-offset;
}

bool ArmWordTransferDesc(u16 kind, bool& load, bool& byte, bool& post, bool& imm)
{
    if (!IsArmWordTransferKind(kind))
        return false;

    int group = (kind - ARMInstrInfo::ak_STR_REG_LSL) / 10;
    int form = (kind - ARMInstrInfo::ak_STR_REG_LSL) % 10;

    byte = (group & 1) != 0;
    load = group >= 2;
    post = form >= 5;
    imm = (form == 4) || (form == 9);
    return true;
}

bool ArmHalfTransferDesc(u16 kind, bool& load, bool& half, bool& sign, bool& post, bool& imm)
{
    if (!IsArmHalfTransferKind(kind))
        return false;

    int group = (kind - ARMInstrInfo::ak_STRH_REG) / 4;
    int form = (kind - ARMInstrInfo::ak_STRH_REG) % 4;

    // Leave doubleword transfers on the interpreter path. They are less common
    // and have several ARM9-only corner cases.
    if (group == 1 || group == 2)
        return false;

    load = group != 0;
    half = (group == 0) || (group == 3) || (group == 5);
    sign = (group == 4) || (group == 5);
    post = form >= 2;
    imm = (form == 1) || (form == 3);
    return true;
}

u32 RegListCount(u32 rlist)
{
    return (u32)__builtin_popcount(rlist & 0xFFFF);
}

u32 AbsS32(s32 val)
{
    return val < 0 ? (u32)-val : (u32)val;
}

u32 DirectSafeRegionMask(ARM* cpu, int region)
{
    switch (region)
    {
    case ARMJIT_Memory::memregion_ITCM:
        return ITCMPhysicalSize - 1;
    case ARMJIT_Memory::memregion_DTCM:
        return DTCMPhysicalSize - 1;
    case ARMJIT_Memory::memregion_MainRAM:
        return NDS::MainRAMMask;
    case ARMJIT_Memory::memregion_SharedWRAM:
        return cpu->Num == 0 ? NDS::SWRAM_ARM9.Mask : NDS::SWRAM_ARM7.Mask;
    case ARMJIT_Memory::memregion_WRAM7:
        return NDS::ARM7WRAMSize - 1;
    case ARMJIT_Memory::memregion_BIOS7:
        return 0x3FFF; // 16 KB BIOS7
    default:
        return 0;
    }
}

u32 DirectSharedWRAMOffset(ARM* cpu)
{
    const u8* mem = cpu->Num == 0 ? NDS::SWRAM_ARM9.Mem : NDS::SWRAM_ARM7.Mem;
    return (u32)(mem - NDS::SharedWRAM);
}

u32 ArmDirectMemoryCycles(const FetchedInstr& instr, ARM* cpu, bool load)
{
    if (cpu->Num == 0)
    {
        const s32 numC = ((instr.Addr + 8) & 0x2) ? 0 : instr.CodeCycles;
        const s32 numD = instr.DataCycles;
        return (u32)std::max(numC + numD - 6, std::max(numC, numD));
    }

    s32 numC = NDS::ARM7MemTimings[instr.CodeCycles][2];
    s32 numD = instr.DataCycles;
    s32 cycles;

    if ((instr.DataRegion >> 24) == 0x02)
    {
        if (cpu->CodeRegion == 0x02)
            cycles = numC + numD;
        else
        {
            if (load)
                numC++;
            cycles = std::max(numC + numD - 3, std::max(numC, numD));
        }
    }
    else if (cpu->CodeRegion == 0x02)
    {
        if (load)
            numD++;
        cycles = std::max(numC + numD - 3, std::max(numC, numD));
    }
    else
    {
        cycles = numC + numD + (load ? 1 : 0);
    }

    return (u32)cycles;
}

bool RuntimeDirectSafeDataRegion(ARM* cpu, int region, u32 addr)
{
    switch (region)
    {
    case ARMJIT_Memory::memregion_ITCM:
        return cpu->Num == 0 && addr < ((ARMv5*)cpu)->ITCMSize;
    case ARMJIT_Memory::memregion_DTCM:
        return cpu->Num == 0 && ((ARMv5*)cpu)->DTCMSize != 0 &&
            addr >= ((ARMv5*)cpu)->DTCMBase &&
            addr < (((ARMv5*)cpu)->DTCMBase + ((ARMv5*)cpu)->DTCMSize);
    case ARMJIT_Memory::memregion_MainRAM:
        return (addr >> 24) == 0x02;
    case ARMJIT_Memory::memregion_SharedWRAM:
        if (NDS::ConsoleType != 0)
            return false;
        if (cpu->Num == 0)
            return NDS::SWRAM_ARM9.Mem != nullptr && (addr >> 24) == 0x03;
        return NDS::SWRAM_ARM7.Mem != nullptr && ((addr >> 23) == 0x06);
    case ARMJIT_Memory::memregion_WRAM7:
        return cpu->Num == 1 && NDS::ConsoleType == 0 &&
            ((addr & 0xFF800000) == 0x03000000 ||
             (addr & 0xFF800000) == 0x03800000);
    default:
        return false;
    }
}

bool RuntimeRangeOverlapsDTCM(ARM* cpu, int region, u32 addr, u32 byteCount)
{
    if (cpu->Num != 0 ||
        region == ARMJIT_Memory::memregion_ITCM ||
        region == ARMJIT_Memory::memregion_DTCM)
        return false;

    ARMv5* cpuv5 = (ARMv5*)cpu;
    if (!cpuv5->DTCMSize)
        return false;

    const u32 end = addr + byteCount;
    const u32 dtcmEnd = cpuv5->DTCMBase + cpuv5->DTCMSize;
    return addr < dtcmEnd && cpuv5->DTCMBase < end;
}

u32 RuntimeDirectOffset(ARM* cpu, int region, u32 addr)
{
    if (region == ARMJIT_Memory::memregion_DTCM)
        return addr - ((ARMv5*)cpu)->DTCMBase;

    u32 offset = addr;
    if (region == ARMJIT_Memory::memregion_SharedWRAM)
        offset += DirectSharedWRAMOffset(cpu);
    return offset & DirectSafeRegionMask(cpu, region);
}

bool RuntimeRangeWraps(ARM* cpu, int region, u32 addr, u32 byteCount)
{
    if (byteCount <= 4)
        return false;

    const u32 mask = DirectSafeRegionMask(cpu, region) & ~3u;
    const u32 lastByteOffset = byteCount - 4;
    if (lastByteOffset > mask)
        return true;

    const u32 offset = RuntimeDirectOffset(cpu, region, addr) & ~3u;
    return offset > (mask - lastByteOffset);
}

bool RuntimeRangeDirectSafe(ARM* cpu, int region, u32 addr, u32 byteCount)
{
    if (!RuntimeDirectSafeDataRegion(cpu, region, addr))
        return false;
    if (byteCount > 4 && !RuntimeDirectSafeDataRegion(cpu, region, addr + byteCount - 4))
        return false;
    if (RuntimeRangeOverlapsDTCM(cpu, region, addr, byteCount))
        return false;
    if ((addr & 0x3) != 0)
        return false;
    return !RuntimeRangeWraps(cpu, region, addr, byteCount);
}

bool RuntimeCodeBitmapHit(ARM* cpu, int region, u32 addr)
{
    if (!CodeMemRegions[region])
        return false;

    const u32 localAddr = ARMJIT_Memory::LocaliseAddress(region, cpu->Num, addr);
    AddressRange* range = &CodeMemRegions[region][(localAddr & 0x7FFFFFF) / 512];
    return (range->Code & (1 << ((localAddr & 0x1FF) / 16))) != 0;
}

bool RuntimeRangeCodeBitmapHit(ARM* cpu, int region, u32 addr, u32 regCount)
{
    for (u32 i = 0; i < regCount; i++)
    {
        if (RuntimeCodeBitmapHit(cpu, region, addr + i * 4))
            return true;
    }
    return false;
}

void RecordA32DirectEligibleCommon(ARM* cpu, const A32InstrDesc* desc,
    int region, bool load, int size, bool writeback)
{
    if (cpu->Num == 0)
        JitProfileCounters.DirectEligibleARM9++;
    else
        JitProfileCounters.DirectEligibleARM7++;

    switch (region)
    {
    case ARMJIT_Memory::memregion_MainRAM:
        JitProfileCounters.DirectEligibleRegionMainRAM++;
        break;
    case ARMJIT_Memory::memregion_ITCM:
        JitProfileCounters.DirectEligibleRegionITCM++;
        break;
    case ARMJIT_Memory::memregion_DTCM:
        JitProfileCounters.DirectEligibleRegionDTCM++;
        break;
    case ARMJIT_Memory::memregion_WRAM7:
        JitProfileCounters.DirectEligibleRegionWRAM7++;
        break;
    case ARMJIT_Memory::memregion_SharedWRAM:
        JitProfileCounters.DirectEligibleRegionShared++;
        break;
    default:
        break;
    }

    if (load)
        JitProfileCounters.DirectEligibleLoad++;
    else
        JitProfileCounters.DirectEligibleStore++;

    if (size == 1)
        JitProfileCounters.DirectEligibleByte++;
    else if (size == 2)
        JitProfileCounters.DirectEligibleHalf++;
    else
        JitProfileCounters.DirectEligibleWord++;

    if (writeback)
        JitProfileCounters.DirectEligibleWriteback++;
    if ((desc->instr >> 28) != Cond_AL)
        JitProfileCounters.DirectEligibleCond++;
    if (desc->kind < ARMInstrInfo::ak_Count)
        JitProfileCounters.DirectEligibleArmKinds[desc->kind]++;
}

void RecordA32DirectEligibleSingle(ARM* cpu, const A32InstrDesc* desc,
    int region, bool load, int size, bool registerOffset, bool post,
    bool writeback)
{
    JitProfileCounters.DirectEligibleSingle++;
    RecordA32DirectEligibleCommon(cpu, desc, region, load, size, writeback);

    if (registerOffset)
        JitProfileCounters.DirectEligibleSingleRegOffset++;
    else
        JitProfileCounters.DirectEligibleSingleImmOffset++;
    if (post)
        JitProfileCounters.DirectEligibleSinglePost++;
}

void RecordA32DirectEligibleBlock(ARM* cpu, const A32InstrDesc* desc,
    int region, bool load, bool writeback)
{
    if (load)
        JitProfileCounters.DirectEligibleLDM++;
    else
        JitProfileCounters.DirectEligibleSTM++;
    RecordA32DirectEligibleCommon(cpu, desc, region, load, 4, writeback);
}

A32MemHelperReason ClassifySingleMemHelperReason(ARM* cpu, const A32InstrDesc* desc)
{
    bool load = false;
    bool byte = false;
    bool post = false;
    bool imm = false;
    bool sign = false;
    int size = 0;
    const u32 instr = desc->instr;
    const bool wordTransfer = IsArmWordTransferKind(desc->kind);

    if (wordTransfer && ArmWordTransferDesc(desc->kind, load, byte, post, imm))
    {
        size = byte ? 1 : 4;
    }
    else
    {
        bool half = false;
        if (!ArmHalfTransferDesc(desc->kind, load, half, sign, post, imm))
            return MemReason_UnsupportedKind;
        size = half ? 2 : 1;
    }

    const u32 rn = (instr >> 16) & 0xF;
    const u32 rd = (instr >> 12) & 0xF;
    const u32 rm = instr & 0xF;
    const bool registerOffset = !imm;
    const bool writeback = (instr & (1u << 21)) != 0;
    const bool pcBaseLiteralLoad = load && rn == 15 && rd != 15 &&
        imm && !post && !writeback;
    if (rd == 15)
        return MemReason_PCDest;
    if (rn == 15 && !pcBaseLiteralLoad)
        return MemReason_PCBase;
    if (registerOffset && rm == 15)
        return MemReason_PCOffset;
    if (wordTransfer && registerOffset &&
        (((instr >> 5) & 0x3) == Shift_ROR) &&
        (((instr >> 7) & 0x1F) == 0))
        return MemReason_RRXOffset;

    u32 offset;
    if (wordTransfer)
        offset = imm ? ArmImmOffset(instr) : ArmRegOffset(cpu, instr);
    else
    {
        offset = imm ? ((instr & 0xF) | ((instr >> 4) & 0xF0))
                     : cpu->R[instr & 0xF];
        if (!(instr & (1u << 23)))
            offset = (u32)-offset;
    }
    const u32 base = pcBaseLiteralLoad ? desc->r15 : cpu->R[rn];
    const u32 addr = post ? base : base + offset;

    const int region = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(addr)
        : ARMJIT_Memory::ClassifyAddress7(addr);
    if (!RuntimeDirectSafeDataRegion(cpu, region, addr) ||
        RuntimeRangeOverlapsDTCM(cpu, region, addr, (u32)size))
        return MemReason_UnsafeRegion;
    if (size > 1 && (addr & (u32)(size - 1)))
        return MemReason_Unaligned;
    if (!load && RuntimeCodeBitmapHit(cpu, region, addr))
        return MemReason_CodeWrite;

    RecordA32DirectEligibleSingle(cpu, desc, region, load, size,
        registerOffset, post, post || writeback);
    return MemReason_DirectEligible;
}

A32MemHelperReason ClassifyLDMHelperReason(ARM* cpu, const A32InstrDesc* desc)
{
    const u32 instr = desc->instr;
    const u32 rn = (instr >> 16) & 0xF;
    const u32 rlist = instr & 0xFFFF;
    const bool hasPC = (rlist >> 15) & 1;
    const bool psrOrUserBank = (instr & (1u << 22)) != 0;
    const bool writeback = (instr & (1u << 21)) != 0;

    if (!rlist)
        return MemReason_LDMEmpty;
    if (rn == 15)
        return MemReason_LDMBasePC;
    if (psrOrUserBank && !hasPC)
        return MemReason_LDMBankedUser;
    if (psrOrUserBank && hasPC)
        return MemReason_LDMRestoreCPSR;
    if (writeback && ((rlist & 0x7FFF) & (1u << rn)))
        return MemReason_LDMBaseInListWB;

    const bool preinc = (instr & (1u << 24)) != 0;
    const bool increment = (instr & (1u << 23)) != 0;
    const u32 regCount = RegListCount(rlist);
    const u32 byteCount = regCount * 4;
    const s32 firstAddrDelta = increment
        ? (preinc ? 4 : 0)
        : -(s32)byteCount + (preinc ? 0 : 4);
    const u32 addr = cpu->R[rn] + (u32)firstAddrDelta;
    const int region = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(addr)
        : ARMJIT_Memory::ClassifyAddress7(addr);

    if (!RuntimeDirectSafeDataRegion(cpu, region, addr) ||
        (byteCount > 4 && !RuntimeDirectSafeDataRegion(cpu, region, addr + byteCount - 4)) ||
        RuntimeRangeOverlapsDTCM(cpu, region, addr, byteCount))
        return MemReason_LDMUnsafeRegion;
    if (addr & 0x3)
        return MemReason_LDMUnaligned;
    if (RuntimeRangeWraps(cpu, region, addr, byteCount))
        return MemReason_LDMRangeWrap;

    RecordA32DirectEligibleBlock(cpu, desc, region, true, writeback);
    return MemReason_DirectEligible;
}

A32MemHelperReason ClassifySTMHelperReason(ARM* cpu, const A32InstrDesc* desc)
{
    const u32 instr = desc->instr;
    const u32 rn = (instr >> 16) & 0xF;
    const u32 rlist = instr & 0xFFFF;
    const bool psrOrUserBank = (instr & (1u << 22)) != 0;

    if (!rlist)
        return MemReason_UnsupportedKind;
    if (psrOrUserBank)
        return MemReason_STMBankedUser;
    if (rn == 15 || (rlist & (1u << 15)) || (rlist & (1u << rn)))
        return MemReason_STMPCOrBase;

    const bool preinc = (instr & (1u << 24)) != 0;
    const bool increment = (instr & (1u << 23)) != 0;
    const u32 regCount = RegListCount(rlist);
    const u32 byteCount = regCount * 4;
    const s32 firstAddrDelta = increment
        ? (preinc ? 4 : 0)
        : -(s32)byteCount + (preinc ? 0 : 4);
    const u32 addr = cpu->R[rn] + (u32)firstAddrDelta;
    const int region = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(addr)
        : ARMJIT_Memory::ClassifyAddress7(addr);

    if (!RuntimeDirectSafeDataRegion(cpu, region, addr) ||
        (byteCount > 4 && !RuntimeDirectSafeDataRegion(cpu, region, addr + byteCount - 4)) ||
        RuntimeRangeOverlapsDTCM(cpu, region, addr, byteCount))
        return MemReason_STMUnsafeRegion;
    if (addr & 0x3)
        return MemReason_STMUnaligned;
    if (RuntimeRangeWraps(cpu, region, addr, byteCount))
        return MemReason_STMRangeWrap;
    if (RuntimeRangeCodeBitmapHit(cpu, region, addr, regCount))
        return MemReason_STMCodeWrite;

    const bool writeback = (instr & (1u << 21)) != 0;
    RecordA32DirectEligibleBlock(cpu, desc, region, false, writeback);
    return MemReason_DirectEligible;
}

A32MemHelperReason ClassifyMemHelperReason(ARM* cpu, const A32InstrDesc* desc)
{
    if (desc->thumb)
        return MemReason_Thumb;

    const u32 instr = desc->instr;
    if ((instr >> 28) == 0xF)
        return MemReason_CondNonAL;

    if ((instr >> 28) != Cond_AL &&
        !(IsArmWordTransferKind(desc->kind) || IsArmHalfTransferKind(desc->kind)))
        return MemReason_CondNonAL;

    if (desc->kind == ARMInstrInfo::ak_LDM)
        return ClassifyLDMHelperReason(cpu, desc);
    if (desc->kind == ARMInstrInfo::ak_STM)
        return ClassifySTMHelperReason(cpu, desc);
    if (IsArmWordTransferKind(desc->kind) || IsArmHalfTransferKind(desc->kind))
        return ClassifySingleMemHelperReason(cpu, desc);

    return MemReason_UnsupportedKind;
}

void RecordA32MemHelperReason(ARM* cpu, const A32InstrDesc* desc)
{
    const A32MemHelperReason reason = ClassifyMemHelperReason(cpu, desc);
    JitProfileCounters.MemReasons[reason]++;
}

const u32 DecodedMemMetaKindMask = 0xFFFF;
const u32 DecodedMemMetaSizeShift = 16;
const u32 DecodedMemMetaSizeMask = 0xFF;
const u32 DecodedMemMetaStore = 1u << 24;

u32 DecodedMemMeta(u32 kind, int size, bool store)
{
    return (kind & DecodedMemMetaKindMask) |
           ((u32)size << DecodedMemMetaSizeShift) |
           (store ? DecodedMemMetaStore : 0);
}

u32 DecodedMemKind(u32 meta)
{
    return meta & DecodedMemMetaKindMask;
}

int DecodedMemSize(u32 meta)
{
    return (int)((meta >> DecodedMemMetaSizeShift) & DecodedMemMetaSizeMask);
}

bool DecodedMemStore(u32 meta)
{
    return (meta & DecodedMemMetaStore) != 0;
}

void RecordA32DecodedIOBucket(ARM* cpu, u32 addr)
{
    if (!EnableA32JitProfiling)
        return;

    if (cpu->Num == 0)
    {
        if ((addr >= 0x04000000 && addr < 0x04000070) ||
            (addr >= 0x04001000 && addr < 0x04001070))
            JitProfileCounters.DecodedMemIOGPU2D++;
        else if (addr >= 0x04000320 && addr < 0x040006A4)
            JitProfileCounters.DecodedMemIOGPU3D++;
        else if (addr >= 0x040000B0 && addr < 0x040000F0)
            JitProfileCounters.DecodedMemIODMA++;
        else if (addr >= 0x04000100 && addr < 0x04000110)
            JitProfileCounters.DecodedMemIOTimer++;
        else if (addr >= 0x04000130 && addr < 0x04000138)
            JitProfileCounters.DecodedMemIOInput++;
        else if ((addr >= 0x04000180 && addr < 0x04000190) ||
                 addr == 0x04100000)
            JitProfileCounters.DecodedMemIOIPC++;
        else if ((addr >= 0x040001A0 && addr < 0x040001C0) ||
                 addr == 0x04100010)
            JitProfileCounters.DecodedMemIOCart++;
        else if (addr >= 0x04000280 && addr < 0x040002C0)
            JitProfileCounters.DecodedMemIODivSqrt++;
        else if ((addr >= 0x04000204 && addr < 0x04000218) ||
                 (addr >= 0x04000240 && addr < 0x04000250) ||
                 (addr >= 0x04000300 && addr < 0x04000308) ||
                 (addr >= 0x04004000 && addr < 0x04004014))
            JitProfileCounters.DecodedMemIOIRQPowerMem++;
        else
            JitProfileCounters.DecodedMemIOOther++;
    }
    else
    {
        if (addr >= 0x040000B0 && addr < 0x040000F0)
            JitProfileCounters.DecodedMemIODMA++;
        else if (addr >= 0x04000100 && addr < 0x04000110)
            JitProfileCounters.DecodedMemIOTimer++;
        else if (addr >= 0x04000130 && addr < 0x0400013C)
            JitProfileCounters.DecodedMemIOInput++;
        else if ((addr >= 0x04000180 && addr < 0x04000190) ||
                 addr == 0x04100000)
            JitProfileCounters.DecodedMemIOIPC++;
        else if ((addr >= 0x040001A0 && addr < 0x040001C4) ||
                 addr == 0x04100010)
            JitProfileCounters.DecodedMemIOCart++;
        else if ((addr >= 0x04000204 && addr < 0x04000218) ||
                 (addr >= 0x04000240 && addr < 0x04000242) ||
                 (addr >= 0x04000300 && addr < 0x0400030C))
            JitProfileCounters.DecodedMemIOIRQPowerMem++;
        else if (addr >= 0x04000400 && addr < 0x04000520)
            JitProfileCounters.DecodedMemIOSPU++;
        else
            JitProfileCounters.DecodedMemIOOther++;
    }
}

void RecordA32DecodedMemProfile(ARM* cpu, u32 meta, u32 addr)
{
    if (!EnableA32JitProfiling)
        return;

    const u32 kind = DecodedMemKind(meta);
    const int size = DecodedMemSize(meta);
    const bool store = DecodedMemStore(meta);
    const int region = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(addr)
        : ARMJIT_Memory::ClassifyAddress7(addr);

    A32MemHelperReason reason = MemReason_DirectEligible;
    if (!RuntimeDirectSafeDataRegion(cpu, region, addr) ||
        RuntimeRangeOverlapsDTCM(cpu, region, addr, (u32)size))
    {
        reason = MemReason_UnsafeRegion;
    }
    else if (size > 1 && (addr & (u32)(size - 1)))
    {
        reason = MemReason_Unaligned;
    }
    else if (store && RuntimeCodeBitmapHit(cpu, region, addr))
    {
        reason = MemReason_CodeWrite;
    }

    JitProfileCounters.MemReasons[reason]++;

    if (store)
    {
        if (size == 1) JitProfileCounters.DecodedMemWrite8++;
        else if (size == 2) JitProfileCounters.DecodedMemWrite16++;
        else JitProfileCounters.DecodedMemWrite32++;
    }
    else
    {
        if (size == 1) JitProfileCounters.DecodedMemRead8++;
        else if (size == 2) JitProfileCounters.DecodedMemRead16++;
        else JitProfileCounters.DecodedMemRead32++;
    }

    if (reason == MemReason_DirectEligible)
    {
        JitProfileCounters.DecodedMemDirectEligible++;
    }
    else if (reason == MemReason_UnsafeRegion)
    {
        if (cpu->Num == 0)
            JitProfileCounters.DecodedMemUnsafeARM9++;
        else
            JitProfileCounters.DecodedMemUnsafeARM7++;

        switch (region)
        {
        case ARMJIT_Memory::memregion_IO9:
            JitProfileCounters.DecodedMemUnsafeIO9++;
            RecordA32DecodedIOBucket(cpu, addr);
            break;
        case ARMJIT_Memory::memregion_IO7:
        case ARMJIT_Memory::memregion_Wifi:
            JitProfileCounters.DecodedMemUnsafeIO7++;
            RecordA32DecodedIOBucket(cpu, addr);
            break;
        case ARMJIT_Memory::memregion_VRAM:
            JitProfileCounters.DecodedMemUnsafeVRAM++;
            break;
        case ARMJIT_Memory::memregion_VWRAM:
            JitProfileCounters.DecodedMemUnsafeVWRAM++;
            break;
        case ARMJIT_Memory::memregion_BIOS9:
        case ARMJIT_Memory::memregion_BIOS7:
        case ARMJIT_Memory::memregion_BIOS9DSi:
        case ARMJIT_Memory::memregion_BIOS7DSi:
            JitProfileCounters.DecodedMemUnsafeBIOS++;
            break;
        default:
            JitProfileCounters.DecodedMemUnsafeOther++;
            break;
        }
    }

    A32InstrDesc desc;
    desc.instr = 0;
    desc.r15 = cpu->R[15];
    desc.codeCycles = (u32)cpu->CodeCycles;
    desc.kind = kind;
    desc.thumb = 0;
    RecordA32JitProfile(&desc, true);
}

void RecordA32DecodedMemCondFail(ARM* cpu, u32 meta)
{
    if (!EnableA32JitProfiling)
        return;

    JitProfileCounters.MemReasons[MemReason_CondNonAL]++;

    A32InstrDesc desc;
    desc.instr = 0;
    desc.r15 = cpu->R[15];
    desc.codeCycles = (u32)cpu->CodeCycles;
    desc.kind = DecodedMemKind(meta);
    desc.thumb = 0;
    RecordA32JitProfile(&desc, true);
}

u32 FastReadFrom(const u8* mem, u32 offset, int size)
{
    switch (size)
    {
    case 1: return *(u8*)&mem[offset];
    case 2: return *(u16*)&mem[offset];
    default: return *(u32*)&mem[offset];
    }
}

void FastWriteTo(u8* mem, u32 offset, int size, u32 val)
{
    switch (size)
    {
    case 1: *(u8*)&mem[offset] = (u8)val; break;
    case 2: *(u16*)&mem[offset] = (u16)val; break;
    default: *(u32*)&mem[offset] = val; break;
    }
}

void SetFastDataCycles9(ARMv5* cpu, u32 alignedAddr, int size, bool sequential)
{
    s32 cycles = 1;
    if (alignedAddr >= cpu->ITCMSize &&
        !(alignedAddr >= cpu->DTCMBase && alignedAddr < (cpu->DTCMBase + cpu->DTCMSize)))
    {
        const int slot = (size == 4) ? (sequential ? 3 : 2) : 1;
        cycles = cpu->MemTimings[alignedAddr >> 12][slot];
    }

    if (sequential)
        cpu->DataCycles += cycles;
    else
        cpu->DataCycles = cycles;
}

void SetFastDataCycles7(ARM* cpu, u32 alignedAddr, int size, bool sequential)
{
    const int slot = (size == 4) ? (sequential ? 3 : 2) : 0;
    const s32 cycles = NDS::ARM7MemTimings[alignedAddr >> 15][slot];

    if (sequential)
        cpu->DataCycles += cycles;
    else
        cpu->DataCycles = cycles;
}

u32 AlignDataAddr(u32 addr, int size)
{
    if (size == 4)
        return addr & ~3u;
    if (size == 2)
        return addr & ~1u;
    return addr;
}

bool TryFastDataRead(ARM* cpu, u32 addr, u32* val, int size, bool sequential)
{
    const u32 originalAddr = addr;
    addr = AlignDataAddr(addr, size);

    if (!sequential)
        cpu->DataRegion = originalAddr;

    if (cpu->Num == 0)
    {
        ARMv5* cpu9 = static_cast<ARMv5*>(cpu);

        if (addr < cpu9->ITCMSize)
        {
            *val = FastReadFrom(cpu9->ITCM, addr & (ITCMPhysicalSize - 1), size);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemRead++;
            return true;
        }

        if (addr >= cpu9->DTCMBase && addr < (cpu9->DTCMBase + cpu9->DTCMSize))
        {
            *val = FastReadFrom(cpu9->DTCM, (addr - cpu9->DTCMBase) & (DTCMPhysicalSize - 1), size);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemRead++;
            return true;
        }

        switch (addr & 0xFF000000)
        {
        case 0x02000000:
            *val = FastReadFrom(NDS::MainRAM, addr & NDS::MainRAMMask, size);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemRead++;
            return true;

        case 0x03000000:
            if (NDS::ConsoleType == 0 && NDS::SWRAM_ARM9.Mem)
            {
                *val = FastReadFrom(NDS::SWRAM_ARM9.Mem, addr & NDS::SWRAM_ARM9.Mask, size);
                SetFastDataCycles9(cpu9, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemRead++;
                return true;
            }
            break;
        }
    }
    else
    {
        switch (addr & 0xFF800000)
        {
        case 0x02000000:
        case 0x02800000:
            *val = FastReadFrom(NDS::MainRAM, addr & NDS::MainRAMMask, size);
            SetFastDataCycles7(cpu, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemRead++;
            return true;

        case 0x03000000:
            if (NDS::ConsoleType == 0)
            {
                if (NDS::SWRAM_ARM7.Mem)
                    *val = FastReadFrom(NDS::SWRAM_ARM7.Mem, addr & NDS::SWRAM_ARM7.Mask, size);
                else
                    *val = FastReadFrom(NDS::ARM7WRAM, addr & (NDS::ARM7WRAMSize - 1), size);
                SetFastDataCycles7(cpu, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemRead++;
                return true;
            }
            break;

        case 0x03800000:
            if (NDS::ConsoleType == 0)
            {
                *val = FastReadFrom(NDS::ARM7WRAM, addr & (NDS::ARM7WRAMSize - 1), size);
                SetFastDataCycles7(cpu, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemRead++;
                return true;
            }
            break;
        }
    }

    return false;
}

bool TryFastDataWrite(ARM* cpu, u32 addr, u32 val, int size, bool sequential)
{
    const u32 originalAddr = addr;
    addr = AlignDataAddr(addr, size);

    if (!sequential)
        cpu->DataRegion = originalAddr;

    if (cpu->Num == 0)
    {
        ARMv5* cpu9 = static_cast<ARMv5*>(cpu);

        if (addr < cpu9->ITCMSize)
        {
            FastWriteTo(cpu9->ITCM, addr & (ITCMPhysicalSize - 1), size, val);
            CheckAndInvalidate<0, ARMJIT_Memory::memregion_ITCM>(addr);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemWrite++;
            return true;
        }

        if (addr >= cpu9->DTCMBase && addr < (cpu9->DTCMBase + cpu9->DTCMSize))
        {
            FastWriteTo(cpu9->DTCM, (addr - cpu9->DTCMBase) & (DTCMPhysicalSize - 1), size, val);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemWrite++;
            return true;
        }

        switch (addr & 0xFF000000)
        {
        case 0x02000000:
            CheckAndInvalidate<0, ARMJIT_Memory::memregion_MainRAM>(addr);
            FastWriteTo(NDS::MainRAM, addr & NDS::MainRAMMask, size, val);
            SetFastDataCycles9(cpu9, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemWrite++;
            return true;

        case 0x03000000:
            if (NDS::ConsoleType == 0 && NDS::SWRAM_ARM9.Mem)
            {
                CheckAndInvalidate<0, ARMJIT_Memory::memregion_SharedWRAM>(addr);
                FastWriteTo(NDS::SWRAM_ARM9.Mem, addr & NDS::SWRAM_ARM9.Mask, size, val);
                SetFastDataCycles9(cpu9, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemWrite++;
                return true;
            }
            break;
        }
    }
    else
    {
        switch (addr & 0xFF800000)
        {
        case 0x02000000:
        case 0x02800000:
            CheckAndInvalidate<1, ARMJIT_Memory::memregion_MainRAM>(addr);
            FastWriteTo(NDS::MainRAM, addr & NDS::MainRAMMask, size, val);
            SetFastDataCycles7(cpu, addr, size, sequential);
            if (EnableA32JitProfiling)
                JitProfileCounters.FastMemWrite++;
            return true;

        case 0x03000000:
            if (NDS::ConsoleType == 0)
            {
                if (NDS::SWRAM_ARM7.Mem)
                {
                    CheckAndInvalidate<1, ARMJIT_Memory::memregion_SharedWRAM>(addr);
                    FastWriteTo(NDS::SWRAM_ARM7.Mem, addr & NDS::SWRAM_ARM7.Mask, size, val);
                }
                else
                {
                    CheckAndInvalidate<1, ARMJIT_Memory::memregion_WRAM7>(addr);
                    FastWriteTo(NDS::ARM7WRAM, addr & (NDS::ARM7WRAMSize - 1), size, val);
                }
                SetFastDataCycles7(cpu, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemWrite++;
                return true;
            }
            break;

        case 0x03800000:
            if (NDS::ConsoleType == 0)
            {
                CheckAndInvalidate<1, ARMJIT_Memory::memregion_WRAM7>(addr);
                FastWriteTo(NDS::ARM7WRAM, addr & (NDS::ARM7WRAMSize - 1), size, val);
                SetFastDataCycles7(cpu, addr, size, sequential);
                if (EnableA32JitProfiling)
                    JitProfileCounters.FastMemWrite++;
                return true;
            }
            break;
        }
    }

    return false;
}

bool IsARM9IRQMemIO(u32 addr)
{
    return (addr >= 0x04000204 && addr < 0x04000218) ||
           (addr >= 0x04000240 && addr < 0x04000250) ||
           (addr >= 0x04000300 && addr < 0x04000308) ||
           (addr >= 0x04004000 && addr < 0x04004014);
}

enum A32IOFastBucket : u8
{
    A32IO_Generic = 0,
    A32IO_GPU3D,
    A32IO_GPU3D_DISPCAP,
    A32IO_GPU3D_EDGE,
    A32IO_DMA,
    A32IO_Timer,
    A32IO_Input,
    A32IO_InputEdge,
    A32IO_IRQMem,
    A32IO_IRQMemEdge,
    A32IO_DivSqrt,
};

struct A32IOFastDispatchTable
{
    u8 Bucket[256];

    A32IOFastDispatchTable()
    {
        memset(Bucket, A32IO_Generic, sizeof(Bucket));

        // ARM9 IO hot ranges all sit in the first 4 KB page, so use 16-byte
        // sub-buckets there. This is M22C's table dispatch: one indexed load
        // replaces the previous GPU3D/DMA/timer/input/IRQ/divsqrt predicate walk.
        Bucket[0x06] = A32IO_GPU3D_DISPCAP; // 0x04000060, 16/32-bit only

        for (int i = 0x0B; i <= 0x0E; i++) Bucket[i] = A32IO_DMA;
        Bucket[0x10] = A32IO_Timer;
        Bucket[0x13] = A32IO_InputEdge;

        Bucket[0x20] = A32IO_IRQMemEdge;
        Bucket[0x21] = A32IO_IRQMemEdge;
        Bucket[0x24] = A32IO_IRQMem;
        Bucket[0x30] = A32IO_IRQMemEdge;

        for (int i = 0x28; i <= 0x2B; i++) Bucket[i] = A32IO_DivSqrt;

        for (int i = 0x32; i <= 0x69; i++) Bucket[i] = A32IO_GPU3D;
        Bucket[0x6A] = A32IO_GPU3D_EDGE;
    }
};

static const A32IOFastDispatchTable ARM9IOFastDispatch = {};

static inline A32IOFastBucket ARM9FastIOBucket(u32 addr, int size)
{
    if ((addr & 0xFFFFF000) == 0x04000000)
    {
        A32IOFastBucket bucket =
            (A32IOFastBucket)ARM9IOFastDispatch.Bucket[(addr >> 4) & 0xFF];
        switch (bucket)
        {
        case A32IO_GPU3D_DISPCAP:
            return (size > 1 && addr == 0x04000060) ? A32IO_GPU3D : A32IO_Generic;
        case A32IO_GPU3D_EDGE:
            return (addr < 0x040006A4) ? A32IO_GPU3D : A32IO_Generic;
        case A32IO_InputEdge:
            return (addr < 0x04000134) ? A32IO_Input : A32IO_Generic;
        case A32IO_IRQMemEdge:
            return IsARM9IRQMemIO(addr) ? A32IO_IRQMem : A32IO_Generic;
        default:
            return bucket;
        }
    }

    if (addr >= 0x04004000 && addr < 0x04004014)
        return A32IO_IRQMem;

    return A32IO_Generic;
}

static inline void CountDirectIORead(A32IOFastBucket bucket)
{
    if (!EnableA32JitProfiling)
        return;

    JitProfileCounters.DecodedMemDirectIORead++;
    switch (bucket)
    {
    case A32IO_GPU3D:   JitProfileCounters.DecodedMemDirectGPU3DRead++; break;
    case A32IO_DMA:     JitProfileCounters.DecodedMemDirectDMARead++; break;
    case A32IO_Timer:   JitProfileCounters.DecodedMemDirectTimerRead++; break;
    case A32IO_Input:   JitProfileCounters.DecodedMemDirectInputRead++; break;
    case A32IO_IRQMem:  JitProfileCounters.DecodedMemDirectIRQMemRead++; break;
    case A32IO_DivSqrt: JitProfileCounters.DecodedMemDirectDivSqrtRead++; break;
    default: break;
    }
}

static inline void CountDirectIOWrite(A32IOFastBucket bucket)
{
    if (!EnableA32JitProfiling)
        return;

    JitProfileCounters.DecodedMemDirectIOWrite++;
    switch (bucket)
    {
    case A32IO_GPU3D:   JitProfileCounters.DecodedMemDirectGPU3DWrite++; break;
    case A32IO_DMA:     JitProfileCounters.DecodedMemDirectDMAWrite++; break;
    case A32IO_Timer:   JitProfileCounters.DecodedMemDirectTimerWrite++; break;
    case A32IO_Input:   JitProfileCounters.DecodedMemDirectInputWrite++; break;
    case A32IO_IRQMem:  JitProfileCounters.DecodedMemDirectIRQMemWrite++; break;
    case A32IO_DivSqrt: JitProfileCounters.DecodedMemDirectDivSqrtWrite++; break;
    default: break;
    }
}

bool ReadIOReg8(u32 addr, u32 base, u32 reg, int bytes, u32* val)
{
    if (addr < base || addr >= base + (u32)bytes)
        return false;

    *val = (reg >> ((addr - base) * 8)) & 0xFF;
    return true;
}

bool ReadIOReg16(u32 addr, u32 base, u32 reg, u32* val)
{
    if (addr != base && addr != base + 2)
        return false;

    *val = (reg >> ((addr - base) * 8)) & 0xFFFF;
    return true;
}

bool WriteIOReg16(u32 addr, u32 base, u32 val, u32* reg)
{
    if (addr == base)
    {
        *reg = (*reg & 0xFFFF0000) | (u16)val;
        return true;
    }

    if (addr == base + 2)
    {
        *reg = (*reg & 0x0000FFFF) | ((u32)(u16)val << 16);
        return true;
    }

    return false;
}

bool TryDirectIRQMemRead(u32 addr, u32* val, int size)
{
    if (size == 1)
    {
        switch (addr)
        {
        case 0x04000208: *val = NDS::IME[0]; return true;
        case 0x04000240: *val = GPU::VRAMCNT[0]; return true;
        case 0x04000241: *val = GPU::VRAMCNT[1]; return true;
        case 0x04000242: *val = GPU::VRAMCNT[2]; return true;
        case 0x04000243: *val = GPU::VRAMCNT[3]; return true;
        case 0x04000244: *val = GPU::VRAMCNT[4]; return true;
        case 0x04000245: *val = GPU::VRAMCNT[5]; return true;
        case 0x04000246: *val = GPU::VRAMCNT[6]; return true;
        case 0x04000248: *val = GPU::VRAMCNT[7]; return true;
        case 0x04000249: *val = GPU::VRAMCNT[8]; return true;
        default: return false;
        }
    }

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000204: *val = NDS::ExMemCnt[0]; return true;
        case 0x04000208: *val = NDS::IME[0]; return true;
        case 0x04000210: *val = NDS::IE[0] & 0xFFFF; return true;
        case 0x04000212: *val = NDS::IE[0] >> 16; return true;
        case 0x04000240: *val = GPU::VRAMCNT[0] | (GPU::VRAMCNT[1] << 8); return true;
        case 0x04000242: *val = GPU::VRAMCNT[2] | (GPU::VRAMCNT[3] << 8); return true;
        case 0x04000244: *val = GPU::VRAMCNT[4] | (GPU::VRAMCNT[5] << 8); return true;
        case 0x04000248: *val = GPU::VRAMCNT[7] | (GPU::VRAMCNT[8] << 8); return true;
        case 0x04000304: *val = NDS::PowerControl9; return true;
        case 0x04004000:
        case 0x04004004:
        case 0x04004010: *val = 0; return true;
        default: return false;
        }
    }

    switch (addr)
    {
    case 0x04000208: *val = NDS::IME[0]; return true;
    case 0x04000210: *val = NDS::IE[0]; return true;
    case 0x04000214: *val = NDS::IF[0]; return true;
    case 0x04000240:
        *val = GPU::VRAMCNT[0] | (GPU::VRAMCNT[1] << 8) |
               (GPU::VRAMCNT[2] << 16) | (GPU::VRAMCNT[3] << 24);
        return true;
    case 0x04000248:
        *val = GPU::VRAMCNT[7] | (GPU::VRAMCNT[8] << 8);
        return true;
    case 0x04000304: *val = NDS::PowerControl9; return true;
    case 0x04004000:
    case 0x04004004:
    case 0x04004010: *val = 0; return true;
    default: return false;
    }
}

bool TryDirectIRQMemWrite(u32 addr, u32 val, int size)
{
    if (size == 1)
    {
        switch (addr)
        {
        case 0x04000208:
            NDS::IME[0] = val & 0x1;
            NDS::UpdateIRQ(0);
            return true;
        case 0x04000240: GPU::MapVRAM_AB(0, (u8)val); return true;
        case 0x04000241: GPU::MapVRAM_AB(1, (u8)val); return true;
        case 0x04000242: GPU::MapVRAM_CD(2, (u8)val); return true;
        case 0x04000243: GPU::MapVRAM_CD(3, (u8)val); return true;
        case 0x04000244: GPU::MapVRAM_E(4, (u8)val); return true;
        case 0x04000245: GPU::MapVRAM_FG(5, (u8)val); return true;
        case 0x04000246: GPU::MapVRAM_FG(6, (u8)val); return true;
        case 0x04000247: NDS::MapSharedWRAM((u8)val); return true;
        case 0x04000248: GPU::MapVRAM_H(7, (u8)val); return true;
        case 0x04000249: GPU::MapVRAM_I(8, (u8)val); return true;
        default: return false;
        }
    }

    if (size == 2)
    {
        const u16 hval = (u16)val;
        switch (addr)
        {
        case 0x04000208:
            NDS::IME[0] = hval & 0x1;
            NDS::UpdateIRQ(0);
            return true;
        case 0x04000210:
            NDS::IE[0] = (NDS::IE[0] & 0xFFFF0000) | hval;
            NDS::UpdateIRQ(0);
            return true;
        case 0x04000212:
            NDS::IE[0] = (NDS::IE[0] & 0x0000FFFF) | ((u32)hval << 16);
            NDS::UpdateIRQ(0);
            return true;
        case 0x04000240:
            GPU::MapVRAM_AB(0, hval & 0xFF);
            GPU::MapVRAM_AB(1, hval >> 8);
            return true;
        case 0x04000242:
            GPU::MapVRAM_CD(2, hval & 0xFF);
            GPU::MapVRAM_CD(3, hval >> 8);
            return true;
        case 0x04000244:
            GPU::MapVRAM_E(4, hval & 0xFF);
            GPU::MapVRAM_FG(5, hval >> 8);
            return true;
        case 0x04000246:
            GPU::MapVRAM_FG(6, hval & 0xFF);
            NDS::MapSharedWRAM(hval >> 8);
            return true;
        case 0x04000248:
            GPU::MapVRAM_H(7, hval & 0xFF);
            GPU::MapVRAM_I(8, hval >> 8);
            return true;
        case 0x04000304:
            NDS::PowerControl9 = val & 0x820F;
            GPU::SetPowerCnt(NDS::PowerControl9);
            return true;
        default: return false;
        }
    }

    switch (addr)
    {
    case 0x04000208:
        NDS::IME[0] = val & 0x1;
        NDS::UpdateIRQ(0);
        return true;
    case 0x04000210:
        NDS::IE[0] = val;
        NDS::UpdateIRQ(0);
        return true;
    case 0x04000214:
        NDS::IF[0] &= ~val;
        GPU3D::CheckFIFOIRQ();
        NDS::UpdateIRQ(0);
        return true;
    case 0x04000240:
        GPU::MapVRAM_AB(0, val & 0xFF);
        GPU::MapVRAM_AB(1, (val >> 8) & 0xFF);
        GPU::MapVRAM_CD(2, (val >> 16) & 0xFF);
        GPU::MapVRAM_CD(3, val >> 24);
        return true;
    case 0x04000244:
        GPU::MapVRAM_E(4, val & 0xFF);
        GPU::MapVRAM_FG(5, (val >> 8) & 0xFF);
        GPU::MapVRAM_FG(6, (val >> 16) & 0xFF);
        NDS::MapSharedWRAM(val >> 24);
        return true;
    case 0x04000248:
        GPU::MapVRAM_H(7, val & 0xFF);
        GPU::MapVRAM_I(8, (val >> 8) & 0xFF);
        return true;
    case 0x04000304:
        NDS::PowerControl9 = val & 0x820F;
        GPU::SetPowerCnt(NDS::PowerControl9);
        return true;
    default:
        return false;
    }
}

bool TryDirectDivSqrtRead(u32 addr, u32* val, int size)
{
    if (size == 1)
    {
        if (ReadIOReg8(addr, 0x04000280, NDS::DivCnt, 2, val)) return true;
        if (ReadIOReg8(addr, 0x04000290, NDS::DivNumerator[0], 4, val)) return true;
        if (ReadIOReg8(addr, 0x04000294, NDS::DivNumerator[1], 4, val)) return true;
        if (ReadIOReg8(addr, 0x04000298, NDS::DivDenominator[0], 4, val)) return true;
        if (ReadIOReg8(addr, 0x0400029C, NDS::DivDenominator[1], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002A0, NDS::DivQuotient[0], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002A4, NDS::DivQuotient[1], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002A8, NDS::DivRemainder[0], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002AC, NDS::DivRemainder[1], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002B0, NDS::SqrtCnt, 2, val)) return true;
        if (ReadIOReg8(addr, 0x040002B4, NDS::SqrtRes, 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002B8, NDS::SqrtVal[0], 4, val)) return true;
        if (ReadIOReg8(addr, 0x040002BC, NDS::SqrtVal[1], 4, val)) return true;
        return false;
    }

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000280: *val = NDS::DivCnt; return true;
        case 0x040002B0: *val = NDS::SqrtCnt; return true;
        default:
            if (ReadIOReg16(addr, 0x04000290, NDS::DivNumerator[0], val)) return true;
            if (ReadIOReg16(addr, 0x04000294, NDS::DivNumerator[1], val)) return true;
            if (ReadIOReg16(addr, 0x04000298, NDS::DivDenominator[0], val)) return true;
            if (ReadIOReg16(addr, 0x0400029C, NDS::DivDenominator[1], val)) return true;
            if (ReadIOReg16(addr, 0x040002A0, NDS::DivQuotient[0], val)) return true;
            if (ReadIOReg16(addr, 0x040002A4, NDS::DivQuotient[1], val)) return true;
            if (ReadIOReg16(addr, 0x040002A8, NDS::DivRemainder[0], val)) return true;
            if (ReadIOReg16(addr, 0x040002AC, NDS::DivRemainder[1], val)) return true;
            if (ReadIOReg16(addr, 0x040002B4, NDS::SqrtRes, val)) return true;
            if (ReadIOReg16(addr, 0x040002B8, NDS::SqrtVal[0], val)) return true;
            if (ReadIOReg16(addr, 0x040002BC, NDS::SqrtVal[1], val)) return true;
            return false;
        }
    }

    switch (addr)
    {
    case 0x04000280: *val = NDS::DivCnt; return true;
    case 0x04000290: *val = NDS::DivNumerator[0]; return true;
    case 0x04000294: *val = NDS::DivNumerator[1]; return true;
    case 0x04000298: *val = NDS::DivDenominator[0]; return true;
    case 0x0400029C: *val = NDS::DivDenominator[1]; return true;
    case 0x040002A0: *val = NDS::DivQuotient[0]; return true;
    case 0x040002A4: *val = NDS::DivQuotient[1]; return true;
    case 0x040002A8: *val = NDS::DivRemainder[0]; return true;
    case 0x040002AC: *val = NDS::DivRemainder[1]; return true;
    case 0x040002B0: *val = NDS::SqrtCnt; return true;
    case 0x040002B4: *val = NDS::SqrtRes; return true;
    case 0x040002B8: *val = NDS::SqrtVal[0]; return true;
    case 0x040002BC: *val = NDS::SqrtVal[1]; return true;
    default: return false;
    }
}

bool TryDirectDivSqrtWrite(u32 addr, u32 val, int size)
{
    if (size == 1)
        return false;

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000280:
            NDS::DivCnt = (u16)val;
            NDS::StartDiv();
            return true;
        case 0x040002B0:
            NDS::SqrtCnt = (u16)val;
            NDS::StartSqrt();
            return true;
        default:
            return false;
        }
    }

    switch (addr)
    {
    case 0x04000280:
        NDS::DivCnt = (u16)val;
        NDS::StartDiv();
        return true;
    case 0x04000290:
        NDS::DivNumerator[0] = val;
        NDS::StartDiv();
        return true;
    case 0x04000294:
        NDS::DivNumerator[1] = val;
        NDS::StartDiv();
        return true;
    case 0x04000298:
        NDS::DivDenominator[0] = val;
        NDS::StartDiv();
        return true;
    case 0x0400029C:
        NDS::DivDenominator[1] = val;
        NDS::StartDiv();
        return true;
    case 0x040002B0:
        NDS::SqrtCnt = (u16)val;
        NDS::StartSqrt();
        return true;
    case 0x040002B8:
        NDS::SqrtVal[0] = val;
        NDS::StartSqrt();
        return true;
    case 0x040002BC:
        NDS::SqrtVal[1] = val;
        NDS::StartSqrt();
        return true;
    default:
        return false;
    }
}

void WriteDMACnt16(int dma, bool high, u32 val)
{
    u32 cnt = NDS::DMAs[dma]->Cnt;
    if (high)
        cnt = (cnt & 0x0000FFFF) | ((u32)(u16)val << 16);
    else
        cnt = (cnt & 0xFFFF0000) | (u16)val;

    NDS::DMAs[dma]->WriteCnt(cnt);
}

bool TryDirectDMARead(u32 addr, u32* val, int size)
{
    if (size == 1)
        return false;

    if (size == 2)
    {
        switch (addr)
        {
        case 0x040000B8: *val = NDS::DMAs[0]->Cnt & 0xFFFF; return true;
        case 0x040000BA: *val = NDS::DMAs[0]->Cnt >> 16; return true;
        case 0x040000C4: *val = NDS::DMAs[1]->Cnt & 0xFFFF; return true;
        case 0x040000C6: *val = NDS::DMAs[1]->Cnt >> 16; return true;
        case 0x040000D0: *val = NDS::DMAs[2]->Cnt & 0xFFFF; return true;
        case 0x040000D2: *val = NDS::DMAs[2]->Cnt >> 16; return true;
        case 0x040000DC: *val = NDS::DMAs[3]->Cnt & 0xFFFF; return true;
        case 0x040000DE: *val = NDS::DMAs[3]->Cnt >> 16; return true;
        default:
            if (ReadIOReg16(addr, 0x040000E0, NDS::DMA9Fill[0], val)) return true;
            if (ReadIOReg16(addr, 0x040000E4, NDS::DMA9Fill[1], val)) return true;
            if (ReadIOReg16(addr, 0x040000E8, NDS::DMA9Fill[2], val)) return true;
            if (ReadIOReg16(addr, 0x040000EC, NDS::DMA9Fill[3], val)) return true;
            return false;
        }
    }

    switch (addr)
    {
    case 0x040000B0: *val = NDS::DMAs[0]->SrcAddr; return true;
    case 0x040000B4: *val = NDS::DMAs[0]->DstAddr; return true;
    case 0x040000B8: *val = NDS::DMAs[0]->Cnt; return true;
    case 0x040000BC: *val = NDS::DMAs[1]->SrcAddr; return true;
    case 0x040000C0: *val = NDS::DMAs[1]->DstAddr; return true;
    case 0x040000C4: *val = NDS::DMAs[1]->Cnt; return true;
    case 0x040000C8: *val = NDS::DMAs[2]->SrcAddr; return true;
    case 0x040000CC: *val = NDS::DMAs[2]->DstAddr; return true;
    case 0x040000D0: *val = NDS::DMAs[2]->Cnt; return true;
    case 0x040000D4: *val = NDS::DMAs[3]->SrcAddr; return true;
    case 0x040000D8: *val = NDS::DMAs[3]->DstAddr; return true;
    case 0x040000DC: *val = NDS::DMAs[3]->Cnt; return true;
    case 0x040000E0: *val = NDS::DMA9Fill[0]; return true;
    case 0x040000E4: *val = NDS::DMA9Fill[1]; return true;
    case 0x040000E8: *val = NDS::DMA9Fill[2]; return true;
    case 0x040000EC: *val = NDS::DMA9Fill[3]; return true;
    default: return false;
    }
}

bool TryDirectDMAWrite(u32 addr, u32 val, int size)
{
    if (size == 1)
        return false;

    if (size == 2)
    {
        switch (addr)
        {
        case 0x040000B8: WriteDMACnt16(0, false, val); return true;
        case 0x040000BA: WriteDMACnt16(0, true, val); return true;
        case 0x040000C4: WriteDMACnt16(1, false, val); return true;
        case 0x040000C6: WriteDMACnt16(1, true, val); return true;
        case 0x040000D0: WriteDMACnt16(2, false, val); return true;
        case 0x040000D2: WriteDMACnt16(2, true, val); return true;
        case 0x040000DC: WriteDMACnt16(3, false, val); return true;
        case 0x040000DE: WriteDMACnt16(3, true, val); return true;
        default:
            if (WriteIOReg16(addr, 0x040000E0, val, &NDS::DMA9Fill[0])) return true;
            if (WriteIOReg16(addr, 0x040000E4, val, &NDS::DMA9Fill[1])) return true;
            if (WriteIOReg16(addr, 0x040000E8, val, &NDS::DMA9Fill[2])) return true;
            if (WriteIOReg16(addr, 0x040000EC, val, &NDS::DMA9Fill[3])) return true;
            return false;
        }
    }

    switch (addr)
    {
    case 0x040000B0: NDS::DMAs[0]->SrcAddr = val; return true;
    case 0x040000B4: NDS::DMAs[0]->DstAddr = val; return true;
    case 0x040000B8: NDS::DMAs[0]->WriteCnt(val); return true;
    case 0x040000BC: NDS::DMAs[1]->SrcAddr = val; return true;
    case 0x040000C0: NDS::DMAs[1]->DstAddr = val; return true;
    case 0x040000C4: NDS::DMAs[1]->WriteCnt(val); return true;
    case 0x040000C8: NDS::DMAs[2]->SrcAddr = val; return true;
    case 0x040000CC: NDS::DMAs[2]->DstAddr = val; return true;
    case 0x040000D0: NDS::DMAs[2]->WriteCnt(val); return true;
    case 0x040000D4: NDS::DMAs[3]->SrcAddr = val; return true;
    case 0x040000D8: NDS::DMAs[3]->DstAddr = val; return true;
    case 0x040000DC: NDS::DMAs[3]->WriteCnt(val); return true;
    case 0x040000E0: NDS::DMA9Fill[0] = val; return true;
    case 0x040000E4: NDS::DMA9Fill[1] = val; return true;
    case 0x040000E8: NDS::DMA9Fill[2] = val; return true;
    case 0x040000EC: NDS::DMA9Fill[3] = val; return true;
    default: return false;
    }
}

bool TryDirectTimerRead(u32 addr, u32* val, int size)
{
    if (size == 1)
        return false;

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000100: *val = NDS::TimerGetCounter(0); return true;
        case 0x04000102: *val = NDS::Timers[0].Cnt; return true;
        case 0x04000104: *val = NDS::TimerGetCounter(1); return true;
        case 0x04000106: *val = NDS::Timers[1].Cnt; return true;
        case 0x04000108: *val = NDS::TimerGetCounter(2); return true;
        case 0x0400010A: *val = NDS::Timers[2].Cnt; return true;
        case 0x0400010C: *val = NDS::TimerGetCounter(3); return true;
        case 0x0400010E: *val = NDS::Timers[3].Cnt; return true;
        default: return false;
        }
    }

    switch (addr)
    {
    case 0x04000100: *val = NDS::TimerGetCounter(0) | (NDS::Timers[0].Cnt << 16); return true;
    case 0x04000104: *val = NDS::TimerGetCounter(1) | (NDS::Timers[1].Cnt << 16); return true;
    case 0x04000108: *val = NDS::TimerGetCounter(2) | (NDS::Timers[2].Cnt << 16); return true;
    case 0x0400010C: *val = NDS::TimerGetCounter(3) | (NDS::Timers[3].Cnt << 16); return true;
    default: return false;
    }
}

bool TryDirectTimerWrite(u32 addr, u32 val, int size)
{
    if (size == 1)
        return false;

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000100: NDS::Timers[0].Reload = (u16)val; return true;
        case 0x04000102: NDS::TimerStart(0, (u16)val); return true;
        case 0x04000104: NDS::Timers[1].Reload = (u16)val; return true;
        case 0x04000106: NDS::TimerStart(1, (u16)val); return true;
        case 0x04000108: NDS::Timers[2].Reload = (u16)val; return true;
        case 0x0400010A: NDS::TimerStart(2, (u16)val); return true;
        case 0x0400010C: NDS::Timers[3].Reload = (u16)val; return true;
        case 0x0400010E: NDS::TimerStart(3, (u16)val); return true;
        default: return false;
        }
    }

    switch (addr)
    {
    case 0x04000100:
        NDS::Timers[0].Reload = val & 0xFFFF;
        NDS::TimerStart(0, val >> 16);
        return true;
    case 0x04000104:
        NDS::Timers[1].Reload = val & 0xFFFF;
        NDS::TimerStart(1, val >> 16);
        return true;
    case 0x04000108:
        NDS::Timers[2].Reload = val & 0xFFFF;
        NDS::TimerStart(2, val >> 16);
        return true;
    case 0x0400010C:
        NDS::Timers[3].Reload = val & 0xFFFF;
        NDS::TimerStart(3, val >> 16);
        return true;
    default:
        return false;
    }
}

bool TryDirectInputRead(u32 addr, u32* val, int size)
{
    if (size == 1)
    {
        switch (addr)
        {
        case 0x04000130:
            NDS::LagFrameFlag = false;
            *val = NDS::KeyInput & 0xFF;
            return true;
        case 0x04000131:
            NDS::LagFrameFlag = false;
            *val = (NDS::KeyInput >> 8) & 0xFF;
            return true;
        case 0x04000132:
            *val = NDS::KeyCnt & 0xFF;
            return true;
        case 0x04000133:
            *val = NDS::KeyCnt >> 8;
            return true;
        default:
            return false;
        }
    }

    if (size == 2)
    {
        switch (addr)
        {
        case 0x04000130:
            NDS::LagFrameFlag = false;
            *val = NDS::KeyInput & 0xFFFF;
            return true;
        case 0x04000132:
            *val = NDS::KeyCnt;
            return true;
        default:
            return false;
        }
    }

    if (addr == 0x04000130)
    {
        NDS::LagFrameFlag = false;
        *val = (NDS::KeyInput & 0xFFFF) | (NDS::KeyCnt << 16);
        return true;
    }

    return false;
}

bool TryDirectInputWrite(u32 addr, u32 val, int size)
{
    if (size == 1)
    {
        switch (addr)
        {
        case 0x04000132:
            NDS::KeyCnt = (NDS::KeyCnt & 0xFF00) | (u8)val;
            return true;
        case 0x04000133:
            NDS::KeyCnt = (NDS::KeyCnt & 0x00FF) | ((u16)(u8)val << 8);
            return true;
        default:
            return false;
        }
    }

    if (size == 2)
    {
        if (addr == 0x04000132)
        {
            NDS::KeyCnt = (u16)val;
            return true;
        }
        return false;
    }

    if (addr == 0x04000130)
    {
        NDS::KeyCnt = val >> 16;
        return true;
    }

    return false;
}

bool TryDirectIORead(ARM* cpu, u32 addr, u32* val, int size)
{
    if (NDS::ConsoleType != 0)
        return false;

    const u32 originalAddr = addr;
    if (size == 4)
        addr &= ~3u;
    else if (size == 2)
        addr &= ~1u;

    if (cpu->Num == 0)
    {
        if ((addr & 0xFF000000) != 0x04000000)
            return false;

        ARMv5* cpu9 = static_cast<ARMv5*>(cpu);
        cpu->DataRegion = originalAddr;
        cpu->DataCycles = cpu9->MemTimings[addr >> 12][size == 4 ? 2 : 1];

        const A32IOFastBucket bucket = ARM9FastIOBucket(addr, size);
        switch (bucket)
        {
        case A32IO_GPU3D:
            if (size == 1)
                *val = GPU3D::Read8(addr);
            else if (size == 2)
                *val = GPU3D::Read16(addr);
            else
                *val = GPU3D::Read32(addr);
            CountDirectIORead(bucket);
            return true;

        case A32IO_DMA:
            if (TryDirectDMARead(addr, val, size))
            {
                CountDirectIORead(bucket);
                return true;
            }
            break;

        case A32IO_Timer:
            if (TryDirectTimerRead(addr, val, size))
            {
                CountDirectIORead(bucket);
                return true;
            }
            break;

        case A32IO_Input:
            if (TryDirectInputRead(addr, val, size))
            {
                CountDirectIORead(bucket);
                return true;
            }
            break;

        case A32IO_IRQMem:
            if (TryDirectIRQMemRead(addr, val, size))
            {
                CountDirectIORead(bucket);
                return true;
            }
            break;

        case A32IO_DivSqrt:
            if (TryDirectDivSqrtRead(addr, val, size))
            {
                CountDirectIORead(bucket);
                return true;
            }
            break;

        default:
            break;
        }

        if (size == 1)
            *val = NDS::ARM9IORead8(addr);
        else if (size == 2)
            *val = NDS::ARM9IORead16(addr);
        else
            *val = NDS::ARM9IORead32(addr);

        CountDirectIORead(A32IO_Generic);
        return true;
    }

    if ((addr & 0xFF800000) != 0x04000000)
        return false;

    cpu->DataRegion = originalAddr;
    cpu->DataCycles = NDS::ARM7MemTimings[addr >> 15][size == 4 ? 2 : 0];

    if (size == 1)
        *val = NDS::ARM7IORead8(addr);
    else if (size == 2)
        *val = NDS::ARM7IORead16(addr);
    else
        *val = NDS::ARM7IORead32(addr);

    CountDirectIORead(A32IO_Generic);
    return true;
}

bool TryDirectIOWrite(ARM* cpu, u32 addr, u32 val, int size)
{
    if (NDS::ConsoleType != 0)
        return false;

    const u32 originalAddr = addr;
    if (size == 4)
        addr &= ~3u;
    else if (size == 2)
        addr &= ~1u;

    if (cpu->Num == 0)
    {
        if ((addr & 0xFF000000) != 0x04000000)
            return false;

        ARMv5* cpu9 = static_cast<ARMv5*>(cpu);
        cpu->DataRegion = originalAddr;
        cpu->DataCycles = cpu9->MemTimings[addr >> 12][size == 4 ? 2 : 1];

        const A32IOFastBucket bucket = ARM9FastIOBucket(addr, size);
        switch (bucket)
        {
        case A32IO_GPU3D:
            if (size == 1)
                GPU3D::Write8(addr, (u8)val);
            else if (size == 2)
                GPU3D::Write16(addr, (u16)val);
            else
                GPU3D::Write32(addr, val);
            CountDirectIOWrite(bucket);
            return true;

        case A32IO_IRQMem:
            if (TryDirectIRQMemWrite(addr, val, size))
            {
                CountDirectIOWrite(bucket);
                return true;
            }
            break;

        case A32IO_DivSqrt:
            if (TryDirectDivSqrtWrite(addr, val, size))
            {
                CountDirectIOWrite(bucket);
                return true;
            }
            break;

        case A32IO_DMA:
            if (TryDirectDMAWrite(addr, val, size))
            {
                CountDirectIOWrite(bucket);
                return true;
            }
            break;

        case A32IO_Timer:
            if (TryDirectTimerWrite(addr, val, size))
            {
                CountDirectIOWrite(bucket);
                return true;
            }
            break;

        case A32IO_Input:
            if (TryDirectInputWrite(addr, val, size))
            {
                CountDirectIOWrite(bucket);
                return true;
            }
            break;

        default:
            break;
        }

        if (size == 1)
            NDS::ARM9IOWrite8(addr, (u8)val);
        else if (size == 2)
            NDS::ARM9IOWrite16(addr, (u16)val);
        else
            NDS::ARM9IOWrite32(addr, val);

        CountDirectIOWrite(A32IO_Generic);
        return true;
    }

    if ((addr & 0xFF800000) != 0x04000000)
        return false;

    cpu->DataRegion = originalAddr;
    cpu->DataCycles = NDS::ARM7MemTimings[addr >> 15][size == 4 ? 2 : 0];

    if (size == 1)
        NDS::ARM7IOWrite8(addr, (u8)val);
    else if (size == 2)
        NDS::ARM7IOWrite16(addr, (u16)val);
    else
        NDS::ARM7IOWrite32(addr, val);

    CountDirectIOWrite(A32IO_Generic);
    return true;
}

// FPS: in DS mode, every address >= 0x04000000 (except CP15-relocatable DTCM) is
// IO/palette/VRAM/OAM/cart — TryFastData* always fails there after scanning ITCM/
// DTCM/MainRAM/SWRAM. On the dominant decoded path (~710K/M heavy-scene events) that
// scan + function call is pure waste, so skip it and go straight to the direct IO
// handler / bus. DTCM (relocatable) and DSi NWRAM still take the full fast-data path.
static inline bool A32DecodedAddrIsUnsafeIO(ARM* cpu, u32 addr)
{
    if (NDS::ConsoleType != 0) return false;
    if (addr < 0x04000000)     return false;
    if (cpu->Num == 0)
    {
        ARMv5* cpu9 = static_cast<ARMv5*>(cpu);
        if (addr >= cpu9->DTCMBase && addr < (cpu9->DTCMBase + cpu9->DTCMSize))
            return false;
    }
    return true;
}

u32 RunA32DecodedMemRead(ARM* cpu, u32 addr, u32 meta)
{
    const int size = DecodedMemSize(meta);
    RecordA32DecodedMemProfile(cpu, meta, addr);

    const bool unsafeIO = A32DecodedAddrIsUnsafeIO(cpu, addr);
    if (unsafeIO)
        cpu->DataRegion = addr;   // TryFastData*/TryDirectIO* set this; preserve timing

    u32 val = 0;
    if ((unsafeIO || !TryFastDataRead(cpu, addr, &val, size, false)) &&
        !TryDirectIORead(cpu, addr, &val, size))
    {
        if (size == 1)
            cpu->DataRead8(addr, &val);
        else if (size == 2)
            cpu->DataRead16(addr, &val);
        else
            cpu->DataRead32(addr, &val);
    }

    if (size == 4)
        val = ROR(val, ((addr & 0x3) << 3));

    cpu->AddCycles_CDI();
    return val;
}

void RunA32DecodedMemWrite(ARM* cpu, u32 addr, u32 val, u32 meta)
{
    const int size = DecodedMemSize(meta);
    RecordA32DecodedMemProfile(cpu, meta, addr);

    const bool unsafeIO = A32DecodedAddrIsUnsafeIO(cpu, addr);
    if (unsafeIO)
        cpu->DataRegion = addr;   // TryFastData*/TryDirectIO* set this; preserve timing

    if ((unsafeIO || !TryFastDataWrite(cpu, addr, val, size, false)) &&
        !TryDirectIOWrite(cpu, addr, val, size))
    {
        if (size == 1)
            cpu->DataWrite8(addr, (u8)val);
        else if (size == 2)
            cpu->DataWrite16(addr, (u16)val);
        else
            cpu->DataWrite32(addr, val);
    }

    cpu->AddCycles_CD();
}

void RunArmMemInstr(ARM* cpu, const A32InstrDesc* desc)
{
    cpu->R[15] = desc->r15;
    cpu->CodeCycles = desc->codeCycles;

    if (EnableA32JitProfiling)
    {
        RecordA32MemHelperReason(cpu, desc);
        RecordA32JitProfile(desc, true);
    }

    const u32 instr = desc->instr;
    if (!cpu->CheckCondition(instr >> 28))
    {
        cpu->AddCycles_C();
        return;
    }

    bool load, byte, post, imm;
    if (ArmWordTransferDesc(desc->kind, load, byte, post, imm))
    {
        u32 rn = (instr >> 16) & 0xF;
        u32 rd = (instr >> 12) & 0xF;
        u32 offset = imm ? ArmImmOffset(instr) : ArmRegOffset(cpu, instr);
        u32 addr = post ? cpu->R[rn] : cpu->R[rn] + offset;

        if (load)
        {
            u32 val;
            if (byte)
            {
                if (!TryFastDataRead(cpu, addr, &val, 1, false))
                    cpu->DataRead8(addr, &val);
            }
            else
            {
                if (!TryFastDataRead(cpu, addr, &val, 4, false))
                    cpu->DataRead32(addr, &val);
                val = ROR(val, ((addr & 0x3) << 3));
            }

            if (post)
                cpu->R[rn] += offset;
            else if (instr & (1u << 21))
                cpu->R[rn] = addr;

            cpu->AddCycles_CDI();

            if (byte)
            {
                cpu->R[rd] = val;
            }
            else
            {
                if (rd == 15)
                {
                    if (cpu->Num == 1)
                        val &= ~0x1;
                    cpu->JumpTo(val);
                }
                else
                    cpu->R[rd] = val;
            }
        }
        else
        {
            if (byte)
            {
                if (!TryFastDataWrite(cpu, addr, cpu->R[rd], 1, false))
                    cpu->DataWrite8(addr, cpu->R[rd]);
            }
            else
            {
                if (!TryFastDataWrite(cpu, addr, cpu->R[rd], 4, false))
                    cpu->DataWrite32(addr, cpu->R[rd]);
            }

            if (post)
                cpu->R[rn] += offset;
            else if (instr & (1u << 21))
                cpu->R[rn] = addr;

            cpu->AddCycles_CD();
        }
        return;
    }

    bool half, sign;
    if (ArmHalfTransferDesc(desc->kind, load, half, sign, post, imm))
    {
        u32 rn = (instr >> 16) & 0xF;
        u32 rd = (instr >> 12) & 0xF;
        u32 offset = imm ? ((instr & 0xF) | ((instr >> 4) & 0xF0))
                         : cpu->R[instr & 0xF];
        if (!(instr & (1u << 23)))
            offset = (u32)-offset;
        u32 addr = post ? cpu->R[rn] : cpu->R[rn] + offset;

        if (load)
        {
            u32 val;
            if (half)
            {
                if (!TryFastDataRead(cpu, addr, &val, 2, false))
                    cpu->DataRead16(addr, &val);
            }
            else
            {
                if (!TryFastDataRead(cpu, addr, &val, 1, false))
                    cpu->DataRead8(addr, &val);
            }

            if (sign)
                val = half ? (u32)(s32)(s16)val : (u32)(s32)(s8)val;

            if (post)
                cpu->R[rn] += offset;
            else if (instr & (1u << 21))
                cpu->R[rn] = addr;

            cpu->R[rd] = val;
            cpu->AddCycles_CDI();
        }
        else
        {
            if (!TryFastDataWrite(cpu, addr, cpu->R[rd], 2, false))
                cpu->DataWrite16(addr, cpu->R[rd]);

            if (post)
                cpu->R[rn] += offset;
            else if (instr & (1u << 21))
                cpu->R[rn] = addr;

            cpu->AddCycles_CD();
        }
        return;
    }

    if (desc->kind == ARMInstrInfo::ak_LDM)
    {
        u32 baseid = (instr >> 16) & 0xF;
        u32 base = cpu->R[baseid];
        u32 wbbase = base;
        u32 preinc = instr & (1u << 24);
        bool first = true;

        if (!(instr & (1u << 23)))
        {
            for (int i = 0; i < 16; i++)
                if (instr & (1u << i))
                    base -= 4;

            if (instr & (1u << 21))
                wbbase = base;

            preinc = !preinc;
        }

        if ((instr & (1u << 22)) && !(instr & (1u << 15)))
            cpu->UpdateMode(cpu->CPSR, (cpu->CPSR & ~0x1F) | 0x10);

        for (int i = 0; i < 15; i++)
        {
            if (instr & (1u << i))
            {
                if (preinc) base += 4;
                if (first)
                {
                    if (!TryFastDataRead(cpu, base, &cpu->R[i], 4, false))
                        cpu->DataRead32(base, &cpu->R[i]);
                }
                else
                {
                    if (!TryFastDataRead(cpu, base, &cpu->R[i], 4, true))
                        cpu->DataRead32S(base, &cpu->R[i]);
                }
                first = false;
                if (!preinc) base += 4;
            }
        }

        if (instr & (1u << 15))
        {
            u32 pc;
            if (preinc) base += 4;
            if (first)
            {
                if (!TryFastDataRead(cpu, base, &pc, 4, false))
                    cpu->DataRead32(base, &pc);
            }
            else
            {
                if (!TryFastDataRead(cpu, base, &pc, 4, true))
                    cpu->DataRead32S(base, &pc);
            }
            if (!preinc) base += 4;

            if (cpu->Num == 1)
                pc &= ~0x1;

            cpu->JumpTo(pc, instr & (1u << 22));
        }

        if ((instr & (1u << 22)) && !(instr & (1u << 15)))
            cpu->UpdateMode((cpu->CPSR & ~0x1F) | 0x10, cpu->CPSR);

        if (instr & (1u << 21))
        {
            if (instr & (1u << 23))
                wbbase = base;

            if (instr & (1u << baseid))
            {
                if (cpu->Num == 0)
                {
                    u32 rlist = instr & 0xFFFF;
                    if ((!(rlist & ~(1u << baseid))) || (rlist & ~((2u << baseid) - 1)))
                        cpu->R[baseid] = wbbase;
                }
            }
            else
                cpu->R[baseid] = wbbase;
        }

        cpu->AddCycles_CDI();
        return;
    }

    if (desc->kind == ARMInstrInfo::ak_STM)
    {
        u32 baseid = (instr >> 16) & 0xF;
        u32 base = cpu->R[baseid];
        u32 oldbase = base;
        u32 preinc = instr & (1u << 24);
        bool first = true;

        if (!(instr & (1u << 23)))
        {
            for (u32 i = 0; i < 16; i++)
                if (instr & (1u << i))
                    base -= 4;

            if (instr & (1u << 21))
                cpu->R[baseid] = base;

            preinc = !preinc;
        }

        bool isbanked = false;
        if (instr & (1u << 22))
        {
            u32 mode = cpu->CPSR & 0x1F;
            if (mode == 0x11)
                isbanked = (baseid >= 8 && baseid < 15);
            else if (mode != 0x10 && mode != 0x1F)
                isbanked = (baseid >= 13 && baseid < 15);

            cpu->UpdateMode(cpu->CPSR, (cpu->CPSR & ~0x1F) | 0x10);
        }

        for (u32 i = 0; i < 16; i++)
        {
            if (instr & (1u << i))
            {
                if (preinc) base += 4;

                if (i == baseid && !isbanked)
                {
                    u32 val = ((cpu->Num == 0) || (!(instr & ((1u << i) - 1)))) ? oldbase : base;
                    if (first)
                    {
                        if (!TryFastDataWrite(cpu, base, val, 4, false))
                            cpu->DataWrite32(base, val);
                    }
                    else
                    {
                        if (!TryFastDataWrite(cpu, base, val, 4, true))
                            cpu->DataWrite32S(base, val);
                    }
                }
                else
                {
                    if (first)
                    {
                        if (!TryFastDataWrite(cpu, base, cpu->R[i], 4, false))
                            cpu->DataWrite32(base, cpu->R[i]);
                    }
                    else
                    {
                        if (!TryFastDataWrite(cpu, base, cpu->R[i], 4, true))
                            cpu->DataWrite32S(base, cpu->R[i]);
                    }
                }

                first = false;
                if (!preinc) base += 4;
            }
        }

        if (instr & (1u << 22))
            cpu->UpdateMode((cpu->CPSR & ~0x1F) | 0x10, cpu->CPSR);

        if ((instr & (1u << 23)) && (instr & (1u << 21)))
            cpu->R[baseid] = base;

        cpu->AddCycles_CD();
        return;
    }

    RunInterpreterInstr(cpu, desc);
}

void RunThumbMemInstr(ARM* cpu, const A32InstrDesc* desc)
{
    if (EnableA32JitProfiling)
        RecordA32JitProfile(desc, true);

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

// Register cache Phase 1: static analysis of a single ARM-mode instruction.
// Fills in which guest registers (0-14) are read and which are written.
// Conservative: over-marks written rather than under-marks, ensuring correctness.
// Called for every instruction in the block before code emission begins.
// Conservative guest-register usage analysis for the block-local register cache.
//
// SAFETY INVARIANT: `written` MUST be a *superset* of every GP register the
// instruction can modify.  A false negative here (a real write left out) makes
// CompileBlock pin a stale host copy of a register the guest actually changes —
// silent corruption.  A false positive only foregoes a caching opportunity.
// So every ambiguous encoding over-marks its destination-capable fields.
//
// This previously missed: MUL/MLA destination (Rd is bits[19:16], *not* [15:12]),
// MRS, SWP, halfword/signed transfers, and MRC — all of which appear constantly
// in boot/IRQ code.  Those misses only bit once blocks were allowed to exceed
// 32 instructions (the 32→128 JIT_MaxBlockSize bump), which is when a pinned
// register and a missed write to it started landing in the same block.
static void AnalyzeArmRegUsage(u32 op, u32& read, u32& written)
{
    const u32 grp   = (op >> 26) & 3;
    const u32 bit25 = (op >> 25) & 1;
    const u32 bit24 = (op >> 24) & 1;
    const u32 bit23 = (op >> 23) & 1;
    const u32 bit22 = (op >> 22) & 1;
    const u32 bit21 = (op >> 21) & 1;
    const u32 bit20 = (op >> 20) & 1;
    const u32 bit7  = (op >> 7) & 1;
    const u32 bit4  = (op >> 4) & 1;
    const u32 rn    = (op >> 16) & 0xF;
    const u32 rd    = (op >> 12) & 0xF;
    const u32 rm    = op & 0xF;
    const u32 rs    = (op >> 8) & 0xF;

    auto addR = [&](u32 r) { if (r < 15) read    |= (1u << r); };
    auto addW = [&](u32 r) { if (r < 15) written |= (1u << r); };

    if (grp == 0)
    {
        const bool mulOrSwap = bit4 && bit7 && (((op >> 5) & 3) == 0); // bits[7:4]==1001
        if (mulOrSwap && ((op >> 24) & 0xF) == 0)
        {
            // Multiply / multiply-long.  Destination is Rd=bits[19:16] (our `rn`);
            // bits[15:12] is the accumulate source (MLA) or RdLo (64-bit).
            const u32 mulGrp = (op >> 21) & 7;
            addR(rm); addR(rs);
            if (mulGrp >= 4) { addR(rn); addR(rd); addW(rn); addW(rd); } // UMULL/SMULL/..L
            else             { if (mulGrp & 1) addR(rd); addW(rn); }     // MUL/MLA → dest=rn
        }
        else if (mulOrSwap)
        {
            // SWP/SWPB (bits[27:24]!=0000, bits[7:4]==1001) and v6 LDREX/STREX:
            // base Rn, source Rm, destination Rd.
            addR(rn); addR(rm); addW(rd);
        }
        else if (bit4 && bit7)
        {
            // Halfword/signed/doubleword transfers.
            // bits[6:5] discriminate: 01=LDRH/STRH, 10=LDRD/STRD, 11=LDRSB/LDRSH
            const u32 sh = (op >> 5) & 3;
            addR(rn);
            if (!bit22) addR(rm);                 // bit22=0 → register offset
            if (sh == 2) // LDRD/STRD (ARMv5TE): operates on Rd AND Rd+1
            {
                if (bit20) { addW(rd); if ((rd+1) < 15) addW(rd+1); }  // LDRD writes both
                else       { addR(rd); if ((rd+1) < 15) addR(rd+1); }  // STRD reads both
            }
            else
            {
                if (bit20) addW(rd); else addR(rd);   // load writes Rd, store reads it
            }
            if (!bit24 || bit21) addW(rn);        // post-index / writeback
        }
        else if (bit24 && !bit23 && !bit20)
        {
            // Misc / control & DSP space (bits[24:23]==10, S=0): MRS/MSR/BX/BLX/CLZ/
            // QADD/QSUB/SMLA<xy>/SMUL<xy>/BKPT.  Destinations live in either bits[15:12]
            // (MRS/CLZ/QADD) or bits[19:16] (SMLA/SMUL) — over-mark both.  MSR/BX write
            // no GP reg, but marking their (SBO/unused) Rd/Rn fields is harmless.
            addR(rm); addR(rs);
            addW(rd); addW(rn);
        }
        else
        {
            // Standard data-processing (register / register-shifted-register operand).
            const u32 dpOp    = (op >> 21) & 0xF;
            const bool noWrite = dpOp >= 8 && dpOp <= 11; // TST/TEQ/CMP/CMN
            const bool noRn    = dpOp == 0xD || dpOp == 0xF; // MOV/MVN
            if (!noRn) addR(rn);
            if (!bit25) { addR(rm); if (bit4) addR(rs); }
            if (!noWrite) addW(rd);
        }
    }
    else if (grp == 1) // LDR/STR word/byte
    {
        addR(rn);
        if (bit25) addR(rm); // register offset (note: for LDR/STR bit25=1 → reg, 0 → imm)
        if (bit20) addW(rd); else addR(rd);
        if (!bit24 || bit21) addW(rn); // post-index or write-back
    }
    else if (grp == 2) // LDM/STM (bit25=0) or B/BL (bit25=1)
    {
        if (!bit25) // LDM/STM
        {
            addR(rn);
            if (bit21) addW(rn); // writeback
            const u32 regList = op & 0x7FFF; // bits 0-14 (exclude R15)
            if (bit20) { // LDM: list regs written
                u32 mask = regList;
                while (mask) { int r = __builtin_ctz(mask); addW(r); mask &= mask-1; }
            } else {     // STM: list regs read
                u32 mask = regList;
                while (mask) { int r = __builtin_ctz(mask); addR(r); mask &= mask-1; }
            }
        }
        else // B/BL: PC written (excluded anyway), BL also writes R14
        {
            if (bit24) addW(14); // BL writes LR
        }
    }
    else // grp == 3: coprocessor / SWI
    {
        // MRC (coproc → ARM reg, e.g. CP15 reads on ARM9) writes Rd=bits[15:12].
        // Encoding: bits[27:24]==1110, bit4==1, L=bit20==1.
        if (((op >> 24) & 0xF) == 0xE && bit4 && bit20)
            addW(rd);
    }
}

// ── Register allocator: template specializations ────────────────────────
template <>
const int RegisterCache<Compiler, int>::NativeRegAllocOrder[] =
{
    R5, R6, R7, R8, R9, R10, R11
};
template <>
const int RegisterCache<Compiler, int>::NativeRegsAvailable = 7;

void Compiler::LoadReg(int reg, int nativeReg)
{
    EmitLoadReg(nativeReg, R4, ArmMemberOffset(&ARM::R) + (u32)reg * 4);
}

void Compiler::SaveReg(int reg, int nativeReg)
{
    EmitStoreReg(nativeReg, R4, ArmMemberOffset(&ARM::R) + (u32)reg * 4);
}

void Compiler::FlushRegAllocForHelper()
{
    if (!m_useRegAlloc) return;
    if (m_hostFlagsDirty)
        EmitFlushDirtyFlags();
    RegAllocCache.Flush();
}

void Compiler::MarkHostFlagsDirty(u32 mask)
{
    m_hostFlagsDirty = true;
    m_dirtyFlagsMask = mask;
}

void Compiler::EmitFlushDirtyFlags()
{
    if (!m_hostFlagsDirty) return;
    EmitCopyHostFlags(m_dirtyFlagsMask);
    m_hostFlagsDirty = false;
    m_dirtyFlagsMask = 0;
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

    A32JIT_LOGI("melonDS JIT: armeabi-v7a/AArch32 JIT backend active (Thumb ALU/mul/address native + guarded ARM ALU/mul/branch native + ARM memory helpers with direct RAM/TCM/WRAM fast path; emitted safe RAM/TCM/WRAM ARM immediate/reg/post-index byte/half/word + safe-region conditional LDM/STM + PC-in-list JumpTo returns + LDR-to-PC JumpTo branch + S+PC restore returns + Thumb LDR_PCREL/SPREL direct load/store + ARM SP MainRAM/WRAM7 speculation + block-local pcrel-base+ALU-imm propagation + M17.9 decoded unsafe single-transfer helpers with direct DS/GPU3D/DMA/Timer/Input/IRQMem/DivSqrt IO + M22C 16-byte ARM9 IO dispatch table + direct-eligible shape profiling + wide runtime direct-region single-transfer chain + decoded fallback restoration + IO/VRAM high-addr fast-exit + ARM7 BIOS7 direct loads + M18 ARM9 self-link backward-loop tail + M18 immediate B/BL fast branch exits; banked/base-in-list multiple transfers stay on helpers; unsafe misc/carry ops fallback; %d-instr straight-line blocks; conditional+carry ALU + reg-shift-by-reg + MRS/MSR_f/MUL(arm7+9)/MLA(arm7+9)/SMULL/SMLAL/UMULL/condBX/CLZ/PC-imm/PC-imm-S native; A32 profiling %s with memory reasons + M17 decoded breakdown + direct-eligible shapes + runtime-direct counters; fastmem disabled)",
        Config::JIT_MaxBlockSize,
        EnableA32JitProfiling ? "on" : "off by default");
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

    // Worst-case A32 emission per guest instruction in this fork (conditional
    // memory op with the full wide runtime-direct guard chain + decoded fallback)
    // is ~190 bytes; 320 leaves margin. Scaling with instrsCount keeps larger
    // blocks (Config::JIT_MaxBlockSize up to 128) from overrunning the 16 MB code
    // buffer, and is strictly safer than the old fixed 8192 for full 32-instr blocks.
    const u32 bytesNeeded = (u32)instrsCount * 320 + 512;
    if (CodeSize - CodeOffset() < bytesNeeded)
    {
        A32JIT_LOGI("AArch32 JIT memory full, wrapping code buffer (used %u/%u bytes)...",
                    CodeOffset(), CodeSize);
        CodePtr = CodeStart;
        if (CodeSize < bytesNeeded)
        {
            A32JIT_LOGI("AArch32 JIT code buffer too small for block, full reset");
            ResetBlockCache();
        }
    }

    u8* start = CodePtr;
    int emitCount = instrsCount < MaxInstrsPerBlock ? instrsCount : MaxInstrsPerBlock;

    // Reset block-local constant-propagation state.
    memset(m_regKnown, 0, sizeof(m_regKnown));

    // Cycle batching (6.9): reset pending cycle accumulator for this block.
    m_pendingCycles = 0;

    // ── Register allocator (Phase 1) ────────────────────────────────────────
    // Uses ARMJIT_RegisterCache.h template to dynamically map guest regs R0-R14
    // onto host callee-saved R5-R11 across instruction boundaries.
    // LDRD/STRD AnalyzeArmRegUsage bug is now fixed (writes Rd AND Rd+1).
    memset(m_rcHostReg, -1, sizeof(m_rcHostReg));
    m_useRegAlloc = EnableRegAllocator;
    m_hostFlagsDirty = false;
    m_dirtyFlagsMask = 0;
    if (m_useRegAlloc)
        RegAllocCache = RegisterCache<Compiler, int>(this, instrs, emitCount);
    // instrStart: self-link loops back here to skip the cache-load preamble.
    u8* instrStart = CodePtr;

    // M18: track the last tail instruction for self-link candidate detection.
    const FetchedInstr* tailInstr = nullptr;
    u8* tailInstrCodePtr = nullptr;

    for (int i = 0; i < emitCount; i++)
    {
        const FetchedInstr& instr = instrs[i];
        bool forceExit = instr.Info.EndBlock || instr.Info.Branches() || instr.BranchFlags != 0;
        bool tailReturn = forceExit || i == (emitCount - 1);

        if (m_useRegAlloc)
            RegAllocCache.Prepare(thumb, i);

        // M18: save pointer to the start of the tail instruction's emitted code.
        // Flush accumulated cycles from all non-tail instructions here so they
        // land before tailInstrCodePtr and are NOT rewound by the self-link path.
        if (tailReturn)
        {
            EmitFlushPendingCycles();
            tailInstr = &instr;
            tailInstrCodePtr = CodePtr;
        }

        if (!TryEmitNative(instr, thumb, cpu, tailReturn))
        {
            FlushRegAllocForHelper();
            EmitInterpreterFallback(instr, thumb, tailReturn);
        }

        // Update block-local constant-propagation state (ARM mode only).
        if (!thumb)
        {
            const u32 op = instr.Instr;
            const u32 rd  = (op >> 12) & 0xF;
            const u32 rn  = (op >> 16) & 0xF;

            // Conservatively drop the known-constant for every register this
            // instruction may write (per the same superset AnalyzeArmRegUsage uses
            // for the register cache).  Catches encodings whose destination is not
            // bits[15:12] — notably MUL/MLA (dest bits[19:16]) and DSP multiplies —
            // which the rd-based invalidation below would otherwise leave holding a
            // stale value.  The propagation branches re-establish a fresh known
            // value below wherever one is actually provable.
            {
                u32 cpRead = 0, cpWritten = 0;
                AnalyzeArmRegUsage(op, cpRead, cpWritten);
                u32 cpMask = cpWritten;
                while (cpMask) { int r = __builtin_ctz(cpMask); m_regKnown[r] = false; cpMask &= cpMask - 1; }
            }

            if (instr.Info.Kind == ARMInstrInfo::ak_LDM)
            {
                // LDM writes every register in the list — clear them all.
                const u32 regList = op & 0xFFFF;
                for (int reg = 0; reg < 15; reg++)
                    if (regList & (1u << reg))
                        m_regKnown[reg] = false;
                // W bit (bit 21): base register is written back.
                if (((op >> 21) & 1) && rn < 15)
                    m_regKnown[rn] = false;
            }
            else if (IsArmWordTransferKind(instr.Info.Kind))
            {
                bool load, byte, post, imm;
                if (ArmWordTransferDesc(instr.Info.Kind, load, byte, post, imm))
                {
                    const bool writeback = (op >> 21) & 1;
                    if (load)
                    {
                        if (!byte && !post && imm && rd < 15 && rn == 15 && !writeback)
                        {
                            // pcBaseLiteralLoad: LDR Rd,[PC,#offset] loads the 32-bit word AT
                            // litAddr into Rd — the runtime value is *litAddr, not litAddr itself.
                            // Read the value now from host memory to get the exact runtime value.
                            const u32 litOffset = op & 0xFFF;
                            const bool litSub   = !((op >> 23) & 1);
                            const u32 litAddr   = (instr.Addr + 8) + (litSub ? -(int)litOffset : (int)litOffset);

                            const u8* hostBase = nullptr;
                            u32 hostMask = 0;
                            int litRegion = ARMJIT_Memory::memregion_Other;
                            if (cpu->Num == 0)
                            {
                                const ARMv5* arm9 = (ARMv5*)cpu;
                                if (litAddr < arm9->ITCMSize)
                                {
                                    hostBase = arm9->ITCM;
                                    hostMask = ITCMPhysicalSize - 1;
                                    litRegion = ARMJIT_Memory::memregion_ITCM;
                                }
                                else if ((litAddr & 0xFF000000) == 0x02000000)
                                {
                                    hostBase = NDS::MainRAM;
                                    hostMask = NDS::MainRAMMask;
                                    litRegion = ARMJIT_Memory::memregion_MainRAM;
                                }
                            }
                            else
                            {
                                const u32 seg = litAddr & 0xFF800000;
                                if (seg == 0x03800000)
                                {
                                    hostBase = NDS::ARM7WRAM;
                                    hostMask = NDS::ARM7WRAMSize - 1;
                                    litRegion = ARMJIT_Memory::memregion_WRAM7;
                                }
                                else if ((seg & 0xFF000000) == 0x03000000 && NDS::SWRAM_ARM7.Mem)
                                {
                                    hostBase = NDS::SWRAM_ARM7.Mem;
                                    hostMask = NDS::SWRAM_ARM7.Mask;
                                    litRegion = ARMJIT_Memory::memregion_SharedWRAM;
                                }
                            }

                            if (hostBase)
                            {
                                u32 litValue;
                                memcpy(&litValue, hostBase + (litAddr & hostMask), 4);
                                m_knownRegVals[rd] = litValue;
                                m_regKnown[rd]     = true;
                                if (EnableA32JitProfiling)
                                {
                                    JitProfileCounters.PcrelConstKnown++;
                                    switch (litRegion)
                                    {
                                    case ARMJIT_Memory::memregion_MainRAM:    JitProfileCounters.PcrelConstRegionMainRAM++; break;
                                    case ARMJIT_Memory::memregion_ITCM:       JitProfileCounters.PcrelConstRegionITCM++; break;
                                    case ARMJIT_Memory::memregion_WRAM7:      JitProfileCounters.PcrelConstRegionWRAM7++; break;
                                    case ARMJIT_Memory::memregion_SharedWRAM: JitProfileCounters.PcrelConstRegionShared++; break;
                                    default: break;
                                    }
                                }
                            }
                            else
                            {
                                m_regKnown[rd] = false;
                                if (EnableA32JitProfiling)
                                    JitProfileCounters.PcrelConstUnsafe++;
                            }
                        }
                        else if (rd < 15)
                        {
                            m_regKnown[rd] = false; // load overwrites rd with unknown value
                        }
                        // Post-indexed or writeback also modifies the base register.
                        if ((post || writeback) && rn < 15)
                            m_regKnown[rn] = false;
                    }
                    else
                    {
                        // Store: rd is the data source (not written); only base may change.
                        if ((post || writeback) && rn < 15)
                            m_regKnown[rn] = false;
                    }
                }
                else if (rd < 15)
                {
                    m_regKnown[rd] = false; // half/byte load/store — conservatively clobber rd
                }
            }
            else if (rd < 15)
            {
                // Try ALU immediate constant propagation for unconditional (cond=AL),
                // flag-silent (S=0), immediate-operand data-processing instructions.
                // Extends the M16 known-base chain through struct-offset / address
                // computations that follow a pcrel literal load in the same block.
                const bool aluImm = (op >> 28) == 0xE          // cond = AL
                    && ((op >> 26) & 3) == 0                    // data-processing group
                    && ((op >> 25) & 1)                         // immediate operand
                    && !((op >> 20) & 1);                       // S = 0
                bool propagated = false;
                if (aluImm)
                {
                    const u32 dpOp = (op >> 21) & 0xF;
                    const u32 imm8 = op & 0xFF;
                    const u32 rot  = ((op >> 8) & 0xF) * 2;
                    const u32 imm  = rot ? ((imm8 >> rot) | (imm8 << (32 - rot))) : imm8;
                    u32 result = 0;
                    switch (dpOp)
                    {
                    case 0xD: // MOV Rd, #imm
                        result = imm; propagated = true; break;
                    case 0x4: // ADD Rd, Rn, #imm
                        if (rn < 15 && m_regKnown[rn]) { result = m_knownRegVals[rn] + imm; propagated = true; } break;
                    case 0x2: // SUB Rd, Rn, #imm
                        if (rn < 15 && m_regKnown[rn]) { result = m_knownRegVals[rn] - imm; propagated = true; } break;
                    case 0xC: // ORR Rd, Rn, #imm
                        if (rn < 15 && m_regKnown[rn]) { result = m_knownRegVals[rn] | imm; propagated = true; } break;
                    case 0x0: // AND Rd, Rn, #imm
                        if (rn < 15 && m_regKnown[rn]) { result = m_knownRegVals[rn] & imm; propagated = true; } break;
                    case 0xE: // BIC Rd, Rn, #imm (bit-clear)
                        if (rn < 15 && m_regKnown[rn]) { result = m_knownRegVals[rn] & ~imm; propagated = true; } break;
                    default: break;
                    }
                    if (propagated)
                    {
                        m_knownRegVals[rd] = result;
                        m_regKnown[rd] = true;
                        if (EnableA32JitProfiling)
                            JitProfileCounters.AluImmPropKnown++;
                    }
                }
                if (!propagated)
                    m_regKnown[rd] = false;
            }
        }

        if (tailReturn)
            break;
    }

    // M18: if the tail is an unconditional B back to this block's start (ARM9 only),
    // replace the helper-call+ARM_Ret with an inline self-link tail that flushes
    // the cycle counter and branches directly back — saving the ARM_Dispatch/ARM_Ret
    // round-trip (~20-25 host cycles) on every tight-loop iteration.
    // Phase 1a: ARM-mode B.  Phase 1b: Thumb-mode B (same tail, different decode).
    if (tailInstr != nullptr && cpu->Num == 0)
    {
        bool doSelfLink = false;
        u32  branchCycles = 0;
        u32  target = 0xFFFFFFFF;

        if (!thumb)
        {
            const u32 op   = tailInstr->Instr;
            const u32 cond = op >> 28;
            if (tailInstr->Info.Kind == ARMInstrInfo::ak_B
                && cond == 0xE
                && (tailInstr->BranchFlags & branch_StaticTarget)
                && !(tailInstr->BranchFlags & branch_IdleBranch))
            {
                const s32 branchOffset = (s32)(op << 8) >> 6;
                target       = (tailInstr->Addr + 8) + (u32)branchOffset;
                branchCycles = 2 * tailInstr->CodeCycles;
                doSelfLink   = true;
            }
        }
        else
        {
            // Thumb B (tk_B): unconditional 11-bit offset branch.
            // target = (Addr+4) + SignExtend11(offset)<<1 — no Thumb bit.
            if (tailInstr->Info.Kind == ARMInstrInfo::tk_B
                && (tailInstr->BranchFlags & branch_StaticTarget)
                && !(tailInstr->BranchFlags & branch_IdleBranch))
            {
                const s32 offset = (s32)((tailInstr->Instr & 0x7FF) << 21) >> 20;
                target       = (tailInstr->Addr + 4) + (u32)offset;
                branchCycles = 2 * tailInstr->CodeCycles;
                doSelfLink   = true;
            }
        }

        if (doSelfLink && target == instrs[0].Addr && branchCycles <= 0xFF)
        {
            CodePtr = tailInstrCodePtr;
            EmitSelfLinkTail(cpu, instrStart, branchCycles);
            if (EnableA32JitProfiling)
                JitProfileCounters.SelfLinkEmitted++;
            g_M18SelfLinkEmitted++;
            if (g_M18SelfLinkEmitted == 1 || (g_M18SelfLinkEmitted & 0x7FF) == 0)
                A32JIT_LOGI("melonDS A32 JIT M18 self-link ACTIVE: emitted=%u "
                            "(latest block @ 0x%08X cyc=%u %s)",
                            g_M18SelfLinkEmitted, instrs[0].Addr, branchCycles,
                            thumb ? "Thumb" : "ARM");
        }
    }

    FlushIcache(start, CodePtr);
    return (JitBlockEntry)start;
}

bool Compiler::IsJITFault(u8* pc)
{
    return pc >= CodeStart && pc < (CodeStart + CodeSize);
}

u8* Compiler::RewriteMemAccess(u8* pc)
{
    // Phase 4: In fastmem mode, a faulting LDR/STR can be back-patched to a
    // helper BL. Full back-patching requires per-instruction patch-site tracking.
    // For now the guard-chain fallback handles the access via the fault handler.
    return pc;
}

bool Compiler::TryEmitNative(const FetchedInstr& instr, bool thumb, ARM* cpu, bool tailReturn)
{
    u8* start = CodePtr;

    if (thumb && TryEmitThumbNative(instr, cpu, tailReturn))
        return true;
    if (!thumb && TryEmitArmNative(instr, cpu, tailReturn))
        return true;

    CodePtr = start;
    return false;
}

// Emit native ARMv7 code for ARM-state instructions.
//
// Strategy:
//   Branches: unconditional immediate B/BL use the M18 fast branch emitter;
//   conditional/register branches stay on C helpers for exact CPSR/link semantics.
//   ALU (data-processing, cond=AL only): "instruction patching" — load guest registers
//   into host R0/R1, patch Rn/Rd/Rm fields, emit the ARM instruction verbatim on the host.
//
// Host register conventions (same as Thumb emitter):
//   R4  = ARM*  (callee-saved, loaded by ARM_Dispatch)
//   R0  = scratch / first ALU source (Rn)
//   R1  = scratch / second ALU source (Rm)
//   R2  = ALU result (Rd)
//   R3  = scratch (used in EmitCopyHostFlags)
//   R12 = scratch (used in EmitHelperCallWithDesc)
//
// Instruction-patch approach for ALU:
//   The guest ARM instruction already encodes the operation and shift amount.
//   We replace Rn→R0, Rd→R2, Rm→R1 in the opword and emit it directly.
//   The host CPU then executes the arithmetic and sets its own CPSR flags.
//   EmitCopyHostFlags propagates those flag bits into the guest CPSR struct.
//
// Carry-using ops (ADC/SBC/RSC) currently fall back to the interpreter
// because restoring guest carry into host CPSR is not safe enough yet.
bool Compiler::TryEmitArmDecodedMemoryHelper(const FetchedInstr& instr, ARM* cpu, bool tailReturn,
                                             bool load, bool wordTransfer, bool sign, bool post,
                                             bool imm, int size)
{
    const u32 op = instr.Instr;
    const u32 cond = op >> 28;
    const u32 rn = (op >> 16) & 0xF;
    const u32 rd = (op >> 12) & 0xF;
    const u32 rm = op & 0xF;
    const bool writeback = (op & (1u << 21)) != 0;
    const bool registerOffset = !imm;
    const bool pcBaseLiteralLoad = load && rn == 15 && rd != 15 &&
        imm && !post && !writeback;

    if (rd == 15)
        return false;
    if (rn == 15 && !pcBaseLiteralLoad)
        return false;
    if (registerOffset && rm == 15)
        return false;
    if (wordTransfer && registerOffset &&
        (((op >> 5) & 0x3) == Shift_ROR) &&
        (((op >> 7) & 0x1F) == 0))
    {
        return false;
    }

    const u32 offset = wordTransfer
        ? (op & 0xFFF)
        : ((op & 0xF) | ((op >> 4) & 0xF0));
    const bool subtractOffset = !(op & (1u << 23));
    const bool immediateZeroOffset = imm && offset == 0;
    const bool hasWriteback = post || writeback;
    const u32 meta = DecodedMemMeta(instr.Info.Kind, size, !load);

    auto emitOffsetIntoR1 = [&]()
    {
        if (imm)
        {
            EmitLoadImm(R1, offset);
            return;
        }

        EmitGuestLoad(R1, rm);
        if (wordTransfer)
        {
            const u32 shiftType = (op >> 5) & 0x3;
            const u32 shift = (op >> 7) & 0x1F;
            if (shift || shiftType != Shift_LSL)
                EmitMovRegShiftImm(R1, R1, shiftType, shift, false);
        }
    };

    auto emitOffsetApply = [&](int dst, int src)
    {
        if (dst != src)
            EmitMovRegShiftImm(dst, src, Shift_LSL, 0, false);
        if (immediateZeroOffset)
            return;

        emitOffsetIntoR1();
        if (subtractOffset)
            EmitSubReg(dst, dst, R1, false);
        else
            EmitAddReg(dst, dst, R1, false);
    };

    EmitStoreR15(instr.Addr + 8);
    EmitLoadImm(R1, (u32)instr.CodeCycles);
    EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::CodeCycles));

    u8* condFailBranch = nullptr;
    const bool conditional = cond != Cond_AL;
    if (conditional)
    {
        EmitLoadGuestCPSRFlags();
        condFailBranch = EmitBranchPlaceholder(cond ^ 1u);
    }

    if (pcBaseLiteralLoad)
        EmitLoadImm(R0, instr.Addr + 8);
    else
        EmitGuestLoad(R0, rn);

    const bool postNeedsWriteback = post && !immediateZeroOffset;
    if (post)
    {
        if (postNeedsWriteback)
            emitOffsetApply(R3, R0);
    }
    else if (!immediateZeroOffset)
    {
        emitOffsetApply(R0, R0);
    }

    if (load)
    {
        if (hasWriteback)
        {
            if (post)
            {
                if (postNeedsWriteback)
                    EmitGuestStore(rn, R3);
            }
            else
            {
                EmitGuestStore(rn, R0);
            }
        }

        EmitHelperCall2((const void*)&RunA32DecodedMemRead, R0, meta);

        if (sign)
        {
            if (size == 1)
            {
                EmitMovRegShiftImm(R0, R0, Shift_LSL, 24, false);
                EmitMovRegShiftImm(R0, R0, Shift_ASR, 24, false);
            }
            else if (size == 2)
            {
                EmitMovRegShiftImm(R0, R0, Shift_LSL, 16, false);
                EmitMovRegShiftImm(R0, R0, Shift_ASR, 16, false);
            }
        }

        EmitGuestStore(rd, R0);
    }
    else
    {
        EmitGuestLoad(R2, rd);

        if (hasWriteback)
        {
            if (post)
            {
                if (postNeedsWriteback)
                    EmitGuestStore(rn, R3);
            }
            else
            {
                EmitGuestStore(rn, R0);
            }
        }

        EmitHelperCall3((const void*)&RunA32DecodedMemWrite, R0, R2, meta);
    }

    u8* skipCondFail = nullptr;
    if (tailReturn)
        EmitReturn();
    else if (condFailBranch)
        skipCondFail = EmitBranchPlaceholder(Cond_AL);

    if (condFailBranch)
    {
        PatchBranch(condFailBranch, CodePtr);
        if (EnableA32JitProfiling)
            EmitHelperCall1((const void*)&RecordA32DecodedMemCondFail, meta);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
    }

    if (skipCondFail)
        PatchBranch(skipCondFail, CodePtr);

    return true;
}

bool Compiler::TryEmitArmNativeMemory(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    if (!EnableEmittedMainRAMLoadStore)
        return false;

    if (TryEmitArmNativeLDM(instr, cpu, tailReturn))
        return true;
    if (TryEmitArmNativeSTM(instr, cpu, tailReturn))
        return true;

    const u32 op = instr.Instr;
    const u32 cond = op >> 28;
    if (cond == 0xF)
        return false;

    bool load = false;
    bool byte = false;
    bool post = false;
    bool imm = false;
    bool sign = false;
    int size = 0;
    const bool wordTransfer = IsArmWordTransferKind(instr.Info.Kind);

    if (wordTransfer && ArmWordTransferDesc(instr.Info.Kind, load, byte, post, imm))
    {
        size = byte ? 1 : 4;
    }
    else
    {
        bool half = false;
        if (!ArmHalfTransferDesc(instr.Info.Kind, load, half, sign, post, imm))
            return false;
        size = half ? 2 : 1;
    }

    const u32 rn = (op >> 16) & 0xF;
    const u32 rd = (op >> 12) & 0xF;
    const u32 rm = op & 0xF;
    const bool writeback = (op & (1u << 21)) != 0;
    const bool registerOffset = !imm;
    const bool pcBaseLiteralLoad = load && rn == 15 && rd != 15 &&
        imm && !post && !writeback;
    // STR with Rd=PC (storing the current PC to memory) stays on the helper — uncommon.
    // LDR with Rd=PC mid-block is unsupported; tail-return LDR PC is handled via JumpTo.
    if (rd == 15 && (!load || !tailReturn))
        return false;
    if (rn == 15 && !pcBaseLiteralLoad)
        return false;
    if (registerOffset && rm == 15)
        return false;
    if (wordTransfer && registerOffset &&
        (((op >> 5) & 0x3) == Shift_ROR) &&
        (((op >> 7) & 0x1F) == 0))
    {
        // ROR #0 in ARM addressing mode 2 means RRX and consumes guest carry.
        return false;
    }

    // For SP-relative non-post-indexed accesses (rn==13, immediate offset):
    //   ARM9: speculate SP is in MainRAM (0x02XXXXXX) — typical for DS game stacks.
    //   ARM7: speculate SP is in WRAM7 (0x03800000) — ARM7 always uses WRAM7 for stack.
    // For accesses where Rn has a compile-time-known value from a block-local pcBaseLiteralLoad:
    //   Compute the effective address at compile time and determine the region directly.
    // The runtime guards in EmitDirectSafeRegionSetup catch any misprediction and fall
    // back to the helper — so correctness is never compromised.
    const bool spSpeculative = rn == 13 && !registerOffset && !post && !pcBaseLiteralLoad;
    bool isKnownBase = false;
    u32 dataRegionHint = instr.DataRegion;
    int expectedRegion;
    if (spSpeculative)
    {
        if (cpu->Num == 0)
        {
            expectedRegion = ARMJIT_Memory::memregion_MainRAM;
            dataRegionHint = 0x02000000;
        }
        else
        {
            expectedRegion = ARMJIT_Memory::memregion_WRAM7;
            dataRegionHint = 0x03800000;
        }
    }
    else if (!pcBaseLiteralLoad && !registerOffset && !post && rn < 15 && m_regKnown[rn])
    {
        // Constant propagation: Rn was loaded from a PC-relative literal earlier in
        // this block. Compute the effective address and classify the data region now.
        // Only applies to pre-indexed accesses (post-indexed modifies the base register).
        const u32 offset = wordTransfer
            ? (op & 0xFFF)
            : ((op & 0xF) | ((op >> 4) & 0xF0));
        const bool subtractOffset = !(op & (1u << 23));
        const u32 effectiveBase = m_knownRegVals[rn];
        const u32 effectiveAddr = subtractOffset ? effectiveBase - offset : effectiveBase + offset;
        dataRegionHint = effectiveAddr;
        expectedRegion = cpu->Num == 0
            ? ARMJIT_Memory::ClassifyAddress9(effectiveAddr)
            : ARMJIT_Memory::ClassifyAddress7(effectiveAddr);
        isKnownBase = true;
    }
    else
    {
        expectedRegion = cpu->Num == 0
            ? ARMJIT_Memory::ClassifyAddress9(instr.DataRegion)
            : ARMJIT_Memory::ClassifyAddress7(instr.DataRegion);
    }

    const bool primaryDirectSafe = IsDirectSafeDataRegion(cpu, expectedRegion, dataRegionHint);
    const bool runtimeDirectCandidate = EnableRuntimeDirectRegionChain && !pcBaseLiteralLoad;

    struct DirectRegionCandidate
    {
        int region;
        u32 hint;
        bool runtime;
    };

    DirectRegionCandidate candidates[8];
    int candidateCount = 0;

    auto addCandidate = [&](int region, u32 hint, bool runtime)
    {
        if (!IsDirectSafeDataRegion(cpu, region, hint))
            return;

        for (int i = 0; i < candidateCount; i++)
        {
            const bool hintMatters = region == ARMJIT_Memory::memregion_WRAM7;
            if (candidates[i].region == region &&
                (!hintMatters || candidates[i].hint == hint))
                return;
        }

        assert(candidateCount < (int)(sizeof(candidates) / sizeof(candidates[0])));
        candidates[candidateCount++] = { region, hint, runtime };
    };

    if (primaryDirectSafe)
        addCandidate(expectedRegion, dataRegionHint, false);

    if (runtimeDirectCandidate)
    {
        if (cpu->Num == 0)
        {
            // DTCM dominates the M17.6 tail, but keep the chain broad: stale
            // DataRegion hints can miss any runtime direct-safe RAM/TCM target.
            addCandidate(ARMJIT_Memory::memregion_DTCM, 0, true);
            addCandidate(ARMJIT_Memory::memregion_MainRAM, 0x02000000, true);
            addCandidate(ARMJIT_Memory::memregion_SharedWRAM, 0x03000000, true);
            addCandidate(ARMJIT_Memory::memregion_ITCM, 0, true);
        }
        else
        {
            addCandidate(ARMJIT_Memory::memregion_WRAM7, 0x03800000, true);
            addCandidate(ARMJIT_Memory::memregion_WRAM7, 0x03000000, true);
            addCandidate(ARMJIT_Memory::memregion_MainRAM, 0x02000000, true);
            addCandidate(ARMJIT_Memory::memregion_SharedWRAM, 0x03000000, true);
            // BIOS7 (0x00000000–0x00003FFF) is read-only ROM: safe for load fast paths.
            // Stores to BIOS7 cannot happen on real DS hardware and stay on the helper.
            if (load)
                addCandidate(ARMJIT_Memory::memregion_BIOS7, 0, true);
        }
    }

    if (!candidateCount)
    {
        return TryEmitArmDecodedMemoryHelper(instr, cpu, tailReturn,
                                            load, wordTransfer, sign, post, imm, size);
    }

    if (primaryDirectSafe && isKnownBase && EnableA32JitProfiling)
    {
        switch (expectedRegion)
        {
        case ARMJIT_Memory::memregion_MainRAM:    JitProfileCounters.KnownBaseRegionMainRAM++; break;
        case ARMJIT_Memory::memregion_ITCM:       JitProfileCounters.KnownBaseRegionITCM++; break;
        case ARMJIT_Memory::memregion_DTCM:       JitProfileCounters.KnownBaseRegionDTCM++; break;
        case ARMJIT_Memory::memregion_WRAM7:      JitProfileCounters.KnownBaseRegionWRAM7++; break;
        case ARMJIT_Memory::memregion_SharedWRAM: JitProfileCounters.KnownBaseRegionShared++; break;
        default: break;
        }
    }

    const u32 offset = wordTransfer
        ? (op & 0xFFF)
        : ((op & 0xF) | ((op >> 4) & 0xF0));
    const bool subtractOffset = !(op & (1u << 23));
    const bool immediateZeroOffset = imm && offset == 0;
    const bool hasWriteback = post || writeback;
    const u32 meta = DecodedMemMeta(instr.Info.Kind, size, !load);

    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + 8;
    desc.codeCycles = instr.CodeCycles;
    desc.kind = instr.Info.Kind;
    desc.thumb = 0;

    auto emitOffsetIntoR1 = [&]()
    {
        if (imm)
        {
            EmitLoadImm(R1, offset);
            return;
        }

        EmitGuestLoad(R1, rm);
        if (wordTransfer)
        {
            const u32 shiftType = (op >> 5) & 0x3;
            const u32 shift = (op >> 7) & 0x1F;
            if (shift || shiftType != Shift_LSL)
                EmitMovRegShiftImm(R1, R1, shiftType, shift, false);
        }
    };

    auto emitOffsetApply = [&](int dst, int src)
    {
        if (dst != src)
            EmitMovRegShiftImm(dst, src, Shift_LSL, 0, false);
        if (immediateZeroOffset)
            return;

        emitOffsetIntoR1();
        if (subtractOffset)
            EmitSubReg(dst, dst, R1, false);
        else
            EmitAddReg(dst, dst, R1, false);
    };

    auto emitRuntimeDirectCounters = [&](int region)
    {
        EmitIncrementCounter(&JitProfileCounters.RuntimeDirectSingleFastpath);
        if (load)
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectSingleRead);
        else
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectSingleWrite);

        switch (region)
        {
        case ARMJIT_Memory::memregion_MainRAM:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionMainRAM);
            break;
        case ARMJIT_Memory::memregion_ITCM:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionITCM);
            break;
        case ARMJIT_Memory::memregion_DTCM:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionDTCM);
            break;
        case ARMJIT_Memory::memregion_WRAM7:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionWRAM7);
            break;
        case ARMJIT_Memory::memregion_SharedWRAM:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionShared);
            break;
        case ARMJIT_Memory::memregion_BIOS7:
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectRegionBIOS7);
            break;
        default:
            break;
        }
    };

    EmitStoreR15(instr.Addr + 8);

    u8* condFailBranch = nullptr;
    const bool conditional = cond != Cond_AL;
    if (conditional)
    {
        EmitLoadGuestCPSRFlags();
        condFailBranch = EmitBranchPlaceholder(cond ^ 1u);
    }

    if (primaryDirectSafe && isKnownBase && EnableA32JitProfiling)
        EmitIncrementCounter(&JitProfileCounters.KnownBaseFastpath);

    if (pcBaseLiteralLoad)
        EmitLoadImm(R0, instr.Addr + 8);
    else
        EmitGuestLoad(R0, rn);
    if (post)
    {
        // Post-indexed transfers access the original base address. R2 keeps
        // the writeback address for loads; stores recompute it after writing.
        if (!immediateZeroOffset)
            emitOffsetApply(R2, R0);
    }
    else if (!immediateZeroOffset)
    {
        emitOffsetApply(R0, R0);
    }

    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::DataRegion));
    EmitLoadImm(R1, (u32)instr.DataCycles);
    EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::DataCycles));

    u8* helperBranches[16];
    int helperBranchCount = 0;
    if (size > 1)
    {
        EmitLoadImm(R3, size - 1);
        EmitAndReg(R3, R0, R3);
        EmitCmpImm(R3, 0);
        helperBranches[helperBranchCount++] = EmitBranchPlaceholder(Cond_NE);
    }

    // M17.9 Fast-exit: addresses >= 0x04000000 are never in any direct-safe RAM region.
    // All direct-safe regions (MainRAM/0x02, DTCM, ITCM, WRAM7/0x03, SharedWRAM/0x03)
    // live below 0x04000000. IO9, VRAM, OAM, palette, cart, and DSi-mapped regions
    // (0x04000000+) will always fail every runtime guard. Emit a single LSR+CMP+BHS
    // before the guard chain to skip ~30 cycles of failed guard checks for ~60% of
    // all helper-path cases (typically io9+vram = 600–700K/M events in heavy 3D scenes).
    // The decoded helper path at helperFallback is correct here: R0=access addr,
    // R2=writeback addr (post-indexed), DataRegion and DataCycles already stored.
    if (runtimeDirectCandidate)
    {
        EmitMovRegShiftImm(R3, R0, Shift_LSR, 26, false); // R3 = addr >> 26
        EmitCmpImm(R3, 0);                                 // any addr >= 0x04000000 gives R3 != 0
        helperBranches[helperBranchCount++] = EmitBranchPlaceholder(Cond_NE);
    }

    u8* successBranches[8];
    int successBranchCount = 0;
    bool hasRuntimeCandidate = false;

    auto emitCandidate = [&](const DirectRegionCandidate& candidate)
    {
        if (candidate.runtime)
            hasRuntimeCandidate = true;

        u8* regionFailBranches[16];
        int regionFailBranchCount = 0;

        if (load)
        {
            EmitDirectSafeRegionSetup(candidate.region, cpu, candidate.hint,
                                      R0, R12, R1, size,
                                      regionFailBranches, regionFailBranchCount);

            if (hasWriteback)
            {
                if (post)
                {
                    if (!immediateZeroOffset)
                        EmitGuestStore(rn, R2);
                }
                else
                {
                    EmitGuestStore(rn, R0);
                }
            }

            if (size == 1)
                EmitLoadByteRegOffset(R2, R1, R12);
            else if (size == 2)
                EmitLoadHalfRegOffset(R2, R1, R12);
            else
                EmitLoadRegOffset(R2, R1, R12);

            if (sign)
            {
                if (size == 1)
                {
                    EmitMovRegShiftImm(R2, R2, Shift_LSL, 24, false);
                    EmitMovRegShiftImm(R2, R2, Shift_ASR, 24, false);
                }
                else if (size == 2)
                {
                    EmitMovRegShiftImm(R2, R2, Shift_LSL, 16, false);
                    EmitMovRegShiftImm(R2, R2, Shift_ASR, 16, false);
                }
            }

            // LDR to PC is a branch: hand the loaded value to JumpTo() for exact
            // pipeline and T-bit semantics instead of raw-writing guest R[15].
            if (rd == 15)
                EmitHelperCallLoadedPC(R2, false);
            else
                EmitGuestStore(rd, R2);

            if (EnableA32JitProfiling)
            {
                EmitIncrementCounter(&JitProfileCounters.DirectMemRead);
                if (size != 4 || sign)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemNarrowRead);
                if (registerOffset)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemRegRead);
                if (post)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemPostRead);
                else if (writeback)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemWritebackRead);
                if (candidate.runtime)
                    emitRuntimeDirectCounters(candidate.region);
            }
        }
        else
        {
            EmitDirectSafeRegionSetup(candidate.region, cpu, candidate.hint,
                                      R0, R2, R1, size,
                                      regionFailBranches, regionFailBranchCount);

            // If this write hits a currently compiled 16-byte code block, delegate
            // to the helper so normal JIT invalidation runs before the memory write.
            EmitCodeBitmapGuard(candidate.region, R2,
                                regionFailBranches, regionFailBranchCount);
            EmitDirectSafeRegionBase(candidate.region, cpu, R1);

            EmitGuestLoad(R3, rd);

            if (!post && writeback)
                EmitGuestStore(rn, R0);

            if (size == 1)
                EmitStoreByteRegOffset(R3, R1, R2);
            else if (size == 2)
                EmitStoreHalfRegOffset(R3, R1, R2);
            else
                EmitStoreRegOffset(R3, R1, R2);

            if (post && !immediateZeroOffset)
            {
                emitOffsetApply(R3, R0);
                EmitGuestStore(rn, R3);
            }

            if (EnableA32JitProfiling)
            {
                EmitIncrementCounter(&JitProfileCounters.DirectMemWrite);
                if (size != 4)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemNarrowWrite);
                if (registerOffset)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemRegWrite);
                if (post)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemPostWrite);
                else if (writeback)
                    EmitIncrementCounter(&JitProfileCounters.DirectMemWritebackWrite);
                if (candidate.runtime)
                    emitRuntimeDirectCounters(candidate.region);
            }
        }

        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, load));

        if (tailReturn)
            EmitReturn();
        else
            successBranches[successBranchCount++] = EmitBranchPlaceholder(Cond_AL);

        u8* nextCandidate = CodePtr;
        for (int i = 0; i < regionFailBranchCount; i++)
            PatchBranch(regionFailBranches[i], nextCandidate);
    };

    for (int i = 0; i < candidateCount; i++)
        emitCandidate(candidates[i]);

    u8* helperFallback = CodePtr;
    for (int i = 0; i < helperBranchCount; i++)
        PatchBranch(helperBranches[i], helperFallback);

    if (primaryDirectSafe && isKnownBase && EnableA32JitProfiling)
        EmitIncrementCounter(&JitProfileCounters.KnownBaseGuardFail);
    if (hasRuntimeCandidate && EnableA32JitProfiling)
        EmitIncrementCounter(&JitProfileCounters.RuntimeDirectSingleGuardFail);

    if (load && rd == 15)
    {
        EmitHelperCallWithDesc((const void*)&RunArmMemInstr, &desc, sizeof(desc), tailReturn);
    }
    else
    {
        if (hasRuntimeCandidate && EnableA32JitProfiling)
            EmitIncrementCounter(&JitProfileCounters.RuntimeDirectSingleDecodedFallback);

        EmitLoadImm(R1, (u32)instr.CodeCycles);
        EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::CodeCycles));

        if (load)
        {
            if (hasWriteback)
            {
                if (post)
                {
                    if (!immediateZeroOffset)
                        EmitGuestStore(rn, R2);
                }
                else
                {
                    EmitGuestStore(rn, R0);
                }
            }

            EmitHelperCall2((const void*)&RunA32DecodedMemRead, R0, meta);

            if (sign)
            {
                if (size == 1)
                {
                    EmitMovRegShiftImm(R0, R0, Shift_LSL, 24, false);
                    EmitMovRegShiftImm(R0, R0, Shift_ASR, 24, false);
                }
                else if (size == 2)
                {
                    EmitMovRegShiftImm(R0, R0, Shift_LSL, 16, false);
                    EmitMovRegShiftImm(R0, R0, Shift_ASR, 16, false);
                }
            }

            EmitGuestStore(rd, R0);
        }
        else
        {
            EmitGuestLoad(R2, rd);

            if (hasWriteback)
            {
                if (post)
                {
                    if (!immediateZeroOffset)
                    {
                        emitOffsetApply(R3, R0);
                        EmitGuestStore(rn, R3);
                    }
                }
                else
                {
                    EmitGuestStore(rn, R0);
                }
            }

            EmitHelperCall3((const void*)&RunA32DecodedMemWrite, R0, R2, meta);
        }

        if (tailReturn)
            EmitReturn();
    }

    u8* helperCondFailSkip = nullptr;
    if (condFailBranch && !tailReturn)
        helperCondFailSkip = EmitBranchPlaceholder(Cond_AL);

    if (condFailBranch)
    {
        PatchBranch(condFailBranch, CodePtr);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
    }

    u8* done = CodePtr;
    for (int i = 0; i < successBranchCount; i++)
        PatchBranch(successBranches[i], done);
    if (helperCondFailSkip)
        PatchBranch(helperCondFailSkip, done);

    return true;
}

bool Compiler::TryEmitArmNativeLDM(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    const u32 op = instr.Instr;
    const u32 cond = op >> 28;
    // Unconditional (0xF) encodings are SIMD/PLD, not LDM — always reject.
    if (cond == 0xF || instr.Info.Kind != ARMInstrInfo::ak_LDM)
        return false;

    const u32 rn = (op >> 16) & 0xF;
    const u32 rlist = op & 0xFFFF;
    const bool hasPC = (rlist >> 15) & 1;
    const bool preinc = (op & (1u << 24)) != 0;
    const bool increment = (op & (1u << 23)) != 0;
    const bool psrOrUserBank = (op & (1u << 22)) != 0;
    const bool writeback = (op & (1u << 21)) != 0;

    // Banked/user register transfers without PC, empty lists, PC as base, and
    // writeback-while-loading-base are kept on the helper. PC-in-list returns
    // use JumpTo(); S+PC exception returns use JumpTo(..., true).
    if (!rlist || (psrOrUserBank && !hasPC) || rn == 15)
        return false;
    if (hasPC && !tailReturn)
        return false;
    const u32 nonPCrlist = rlist & 0x7FFF;
    if (writeback && (nonPCrlist & (1u << rn)))
        return false;

    const int expectedRegion = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(instr.DataRegion)
        : ARMJIT_Memory::ClassifyAddress7(instr.DataRegion);
    if (!IsDirectSafeDataRegion(cpu, expectedRegion, instr.DataRegion))
        return false;

    const u32 regCount = RegListCount(rlist);
    const u32 byteCount = regCount * 4;
    const s32 firstAddrDelta = increment
        ? (preinc ? 4 : 0)
        : -(s32)byteCount + (preinc ? 0 : 4);
    const s32 writebackDelta = increment ? (s32)byteCount : -(s32)byteCount;

    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + 8;
    desc.codeCycles = instr.CodeCycles;
    desc.kind = instr.Info.Kind;
    desc.thumb = 0;

    u8* fallbackBranches[64];
    int fallbackBranchCount = 0;

    EmitStoreR15(instr.Addr + 8);

    u8* condFailBranch = nullptr;
    if (cond != Cond_AL)
    {
        EmitLoadGuestCPSRFlags();
        condFailBranch = EmitBranchPlaceholder(cond ^ 1u);
    }

    EmitGuestLoad(R0, rn);
    if (firstAddrDelta)
    {
        EmitLoadImm(R1, AbsS32(firstAddrDelta));
        if (firstAddrDelta < 0)
            EmitSubReg(R0, R0, R1, false);
        else
            EmitAddReg(R0, R0, R1, false);
    }

    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::DataRegion));
    EmitLoadImm(R1, (u32)instr.DataCycles);
    EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::DataCycles));

    EmitDirectSafeRegionRangeSetup(expectedRegion, cpu, instr.DataRegion, R0, R0, R1, byteCount,
                                   fallbackBranches, fallbackBranchCount);
    EmitLoadImm(R3, 4);

    // Load all registers in the list. PC loads are branches, so keep the fast
    // memory path but hand the loaded value to JumpTo() for exact pipeline/state
    // semantics instead of storing a raw target into R[15].
    u32 remaining = regCount;
    for (u32 i = 0; i < 16; i++)
    {
        if (!(rlist & (1u << i)))
            continue;

        EmitLoadRegOffset(R2, R1, R0);
        if (i == 15)
            EmitHelperCallLoadedPC(R2, psrOrUserBank);
        else
            EmitGuestStore(i, R2);
        remaining--;
        if (remaining)
            EmitAddReg(R0, R0, R3, false);
    }

    if (writeback)
    {
        EmitGuestLoad(R2, rn);
        EmitLoadImm(R3, AbsS32(writebackDelta));
        if (writebackDelta < 0)
            EmitSubReg(R2, R2, R3, false);
        else
            EmitAddReg(R2, R2, R3, false);
        EmitGuestStore(rn, R2);
    }

    if (EnableA32JitProfiling)
        EmitIncrementCounter(&JitProfileCounters.DirectMemLDM);
    EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, true));

    u8* skipFallback = nullptr;
    if (tailReturn)
        EmitReturn();
    else
        skipFallback = EmitBranchPlaceholder(Cond_AL);

    u8* condFailSkip = nullptr;
    if (condFailBranch)
    {
        PatchBranch(condFailBranch, CodePtr);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        else
            condFailSkip = EmitBranchPlaceholder(Cond_AL);
    }

    u8* fallback = CodePtr;
    for (int i = 0; i < fallbackBranchCount; i++)
        PatchBranch(fallbackBranches[i], fallback);

    EmitHelperCallWithDesc((const void*)&RunArmMemInstr, &desc, sizeof(desc), tailReturn);

    if (skipFallback)
        PatchBranch(skipFallback, CodePtr);
    if (condFailSkip)
        PatchBranch(condFailSkip, CodePtr);

    return true;
}

bool Compiler::TryEmitArmNativeSTM(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    const u32 op = instr.Instr;
    const u32 cond = op >> 28;
    if (cond == 0xF || instr.Info.Kind != ARMInstrInfo::ak_STM)
        return false;

    const u32 rn = (op >> 16) & 0xF;
    const u32 rlist = op & 0xFFFF;
    const bool preinc = (op & (1u << 24)) != 0;
    const bool increment = (op & (1u << 23)) != 0;
    const bool psrOrUserBank = (op & (1u << 22)) != 0;
    const bool writeback = (op & (1u << 21)) != 0;

    // Banked/user transfers, empty lists, PC stores, and base-in-list STM
    // have enough ARM-specific edge cases to keep them on the helper path.
    if (!rlist || psrOrUserBank || rn == 15 || (rlist & (1u << 15)))
        return false;
    if (rlist & (1u << rn))
        return false;

    const int expectedRegion = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(instr.DataRegion)
        : ARMJIT_Memory::ClassifyAddress7(instr.DataRegion);
    if (!IsDirectSafeDataRegion(cpu, expectedRegion, instr.DataRegion))
        return false;

    const u32 regCount = RegListCount(rlist);
    const u32 byteCount = regCount * 4;
    const s32 firstAddrDelta = increment
        ? (preinc ? 4 : 0)
        : -(s32)byteCount + (preinc ? 0 : 4);
    const s32 writebackDelta = increment ? (s32)byteCount : -(s32)byteCount;

    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + 8;
    desc.codeCycles = instr.CodeCycles;
    desc.kind = instr.Info.Kind;
    desc.thumb = 0;

    u8* fallbackBranches[64];
    int fallbackBranchCount = 0;

    EmitStoreR15(instr.Addr + 8);

    u8* condFailBranch = nullptr;
    if (cond != Cond_AL)
    {
        EmitLoadGuestCPSRFlags();
        condFailBranch = EmitBranchPlaceholder(cond ^ 1u);
    }

    EmitGuestLoad(R0, rn);
    if (firstAddrDelta)
    {
        EmitLoadImm(R1, AbsS32(firstAddrDelta));
        if (firstAddrDelta < 0)
            EmitSubReg(R0, R0, R1, false);
        else
            EmitAddReg(R0, R0, R1, false);
    }

    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::DataRegion));
    EmitLoadImm(R1, (u32)instr.DataCycles);
    EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::DataCycles));

    EmitDirectSafeRegionRangeSetup(expectedRegion, cpu, instr.DataRegion, R0, R0, R1, byteCount,
                                   fallbackBranches, fallbackBranchCount);

    EmitMovRegShiftImm(R2, R0, Shift_LSL, 0, false);
    u32 remaining = regCount;
    for (u32 i = 0; i < 15; i++)
    {
        if (!(rlist & (1u << i)))
            continue;

        EmitCodeBitmapGuard(expectedRegion, R2, fallbackBranches, fallbackBranchCount);
        remaining--;
        if (remaining)
        {
            EmitLoadImm(R3, 4);
            EmitAddReg(R2, R2, R3, false);
        }
    }

    if (writeback)
    {
        EmitGuestLoad(R2, rn);
        EmitLoadImm(R3, AbsS32(writebackDelta));
        if (writebackDelta < 0)
            EmitSubReg(R2, R2, R3, false);
        else
            EmitAddReg(R2, R2, R3, false);
        EmitGuestStore(rn, R2);
    }

    EmitDirectSafeRegionBase(expectedRegion, cpu, R1);
    remaining = regCount;
    for (u32 i = 0; i < 15; i++)
    {
        if (!(rlist & (1u << i)))
            continue;

        EmitGuestLoad(R2, i);
        EmitStoreRegOffset(R2, R1, R0);

        remaining--;
        if (remaining)
        {
            EmitLoadImm(R3, 4);
            EmitAddReg(R0, R0, R3, false);
        }
    }

    if (EnableA32JitProfiling)
        EmitIncrementCounter(&JitProfileCounters.DirectMemSTM);
    EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, false));

    u8* skipFallback = nullptr;
    if (tailReturn)
        EmitReturn();
    else
        skipFallback = EmitBranchPlaceholder(Cond_AL);

    u8* condFailSkip = nullptr;
    if (condFailBranch)
    {
        PatchBranch(condFailBranch, CodePtr);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        else
            condFailSkip = EmitBranchPlaceholder(Cond_AL);
    }

    u8* fallback = CodePtr;
    for (int i = 0; i < fallbackBranchCount; i++)
        PatchBranch(fallbackBranches[i], fallback);

    EmitHelperCallWithDesc((const void*)&RunArmMemInstr, &desc, sizeof(desc), tailReturn);

    if (skipFallback)
        PatchBranch(skipFallback, CodePtr);
    if (condFailSkip)
        PatchBranch(condFailSkip, CodePtr);

    return true;
}

bool Compiler::TryEmitArmNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    const u32 op = instr.Instr;
    const u32 cond = op >> 28;

    // ---------- ARM-state branches (handle any condition) ----------
    switch (instr.Info.Kind)
    {
    case ARMInstrInfo::ak_B:
    {
        s32 offset = (s32)(op << 8) >> 6;
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 8;
        desc.codeCycles = instr.CodeCycles;
        desc.target = desc.r15 + offset;
        desc.aux = cond;
        if (cond == 0xE)
            EmitJumpToConst(cpu, desc.target, false, false, tailReturn);
        else
            EmitHelperCallWithDesc((const void*)&RunArmCondBranch,
                                   &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::ak_BL:
    {
        if (cond != 0xE)
            break;  // conditional BL: let interpreter handle
        s32 offset = (s32)(op << 8) >> 6;
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 8;
        desc.codeCycles = instr.CodeCycles;
        desc.target = desc.r15 + offset;
        desc.aux = instr.Addr + 4;  // LR = address of next instruction
        EmitLoadImm(R0, desc.aux);
        EmitGuestStore(14, R0);
        EmitJumpToConst(cpu, desc.target, false, false, tailReturn);
        return true;
    }

    case ARMInstrInfo::ak_BX:
    {
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 8;
        desc.codeCycles = instr.CodeCycles;
        desc.target = 0;
        desc.aux = op & 0xF;

        if (cond == 0xE)
        {
            EmitHelperCallWithDesc((const void*)&RunArmBX, &desc, sizeof(desc), tailReturn);
            return true;
        }

        // Conditional BX: BX is always a block-exiting branch (tailReturn guaranteed true).
        // Taken path: RunArmBX sets R15 to target+4 so dispatch lands at target.
        // Not-taken path: R15 = instr.Addr+8 → dispatch uses R15-4 = instr.Addr+4 (next instr).
        EmitStoreR15(instr.Addr + 8);
        EmitLoadGuestCPSRFlags();                              // seed host CPSR for condition eval
        u8* skipBranch = EmitBranchPlaceholder(cond ^ 1u);    // branch over taken path if cond fails
        EmitHelperCallWithDesc((const void*)&RunArmBX, &desc, sizeof(desc), true); // taken → exits JIT
        PatchBranch(skipBranch, CodePtr);                      // not-taken lands here
        EmitAddCycles(instr, false, cpu);
        EmitReturn();   // tailReturn always true for BX
        return true;
    }

    case ARMInstrInfo::ak_BLX_REG:
    {
        if (cond != 0xE || cpu->Num == 1)
            break;  // ARM7 has no BLX
        A32BranchDesc desc;
        desc.r15 = instr.Addr + 8;
        desc.codeCycles = instr.CodeCycles;
        desc.target = 0;
        desc.aux = op & 0xF;
        EmitHelperCallWithDesc((const void*)&RunArmBLXReg, &desc, sizeof(desc), tailReturn);
        return true;
    }

    case ARMInstrInfo::ak_Nop:
        EmitStoreR15(instr.Addr + 8);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        return true;

    case ARMInstrInfo::ak_MUL:
    {
        if (cond != 0xE)
            break;

        u32 rd = (op >> 16) & 0xF;
        u32 rs = (op >> 8) & 0xF;
        u32 rm = op & 0xF;
        bool setFlags = (op >> 20) & 1;
        if (rd == 15 || rs == 15 || rm == 15)
            break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R0, rm);
        EmitGuestLoad(R1, rs);
        const bool mulFlagsNeeded = setFlags && (bool)(instr.SetFlags & 0xF);
        EmitMulReg(R2, R0, R1, mulFlagsNeeded);
        EmitGuestStore(rd, R2);
        if (mulFlagsNeeded)
            MarkHostFlagsDirty(CPSR_NZMask);
        // ARM9: 1/3 cycles; ARM7TDMI: worst-case 1S+4I = 5 cycles
        EmitAddCyclesConst(cpu->Num == 0 ? (setFlags ? 3 : 1) : 5);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_MLA:
    {
        // MLA Rd, Rm, Rs, Rn: Rd = Rm * Rs + Rn  (cond=AL, ARM9 only)
        if (cond != 0xE) break;  // conditional MLA stays in interpreter
        u32 rd = (op >> 16) & 0xF;
        u32 rn = (op >> 12) & 0xF;  // accumulate register (Rn in encoding)
        u32 rs = (op >> 8) & 0xF;
        u32 rm = op & 0xF;
        bool setFlags = (op >> 20) & 1;
        if (rd == 15 || rn == 15 || rs == 15 || rm == 15) break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R0, rm);    // R0 = Rm
        EmitGuestLoad(R1, rs);    // R1 = Rs
        EmitGuestLoad(R2, rn);    // R2 = Rn (accumulate)
        // MLA R3, R0, R1, R2  (R3 = R0*R1 + R2)
        // Encoding: cccc 0000 001S Rd Rn Rs 1001 Rm  (Rd=3, Rn=2, Rs=1, Rm=0, cond=AL)
        const bool mlaFlagsNeeded = setFlags && (bool)(instr.SetFlags & 0xF);
        EmitU32(0xE0200090u | ((u32)mlaFlagsNeeded << 20) | (3u << 16) | (2u << 12) | (1u << 8) | 0u);
        EmitGuestStore(rd, R3);
        if (mlaFlagsNeeded)
            MarkHostFlagsDirty(CPSR_NZMask);
        // ARM9: 2/4 cycles; ARM7TDMI: 1S+(m+1)I, worst-case m=4 → 5/5 cycles
        EmitAddCyclesConst(cpu->Num == 0 ? (setFlags ? 4 : 2) : 5);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_SMLAL:
    {
        // SMLAL RdLo, RdHi, Rm, Rs: RdHi:RdLo += Rm * Rs  (signed 64-bit, cond=AL, ARM9)
        if (cond != 0xE || cpu->Num != 0) break;
        u32 rdHi = (op >> 16) & 0xF;
        u32 rdLo = (op >> 12) & 0xF;
        u32 rs   = (op >> 8)  & 0xF;
        u32 rm   = op & 0xF;
        bool setFlags = (op >> 20) & 1;
        if (rdHi == 15 || rdLo == 15 || rs == 15 || rm == 15) break;
        if (rdHi == rdLo) break;  // UNPREDICTABLE per ARM spec

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R0, rm);    // R0 = Rm
        EmitGuestLoad(R1, rs);    // R1 = Rs
        EmitGuestLoad(R2, rdLo);  // R2 = RdLo (accumulate low)
        EmitGuestLoad(R3, rdHi);  // R3 = RdHi (accumulate high)
        // SMLAL R2, R3, R0, R1 → R3:R2 += R0 * R1 (signed 64-bit)
        // Encoding: cccc 0000 111S RdHi RdLo Rs 1001 Rm
        // Base (S=0): 0xE0E00090  (bits [27:20] = 0000 1110; add S-bit via setFlags<<20)
        const bool smlalFlagsNeeded = setFlags && (bool)(instr.SetFlags & 0xF);
        EmitU32(0xE0E00090u | ((u32)smlalFlagsNeeded << 20) | (3u << 16) | (2u << 12) | (1u << 8) | 0u);
        EmitGuestStore(rdLo, R2);
        EmitGuestStore(rdHi, R3);  // stored before EmitCopyHostFlags overwrites R3 via MRS
        if (smlalFlagsNeeded)
            MarkHostFlagsDirty(CPSR_NZMask);
        EmitAddCyclesConst(setFlags ? 6 : 4);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_SMULL:
    {
        // SMULL RdLo, RdHi, Rm, Rs: RdHi:RdLo = Rm * Rs  (signed 64-bit, no accumulate)
        if (cond != 0xE || cpu->Num != 0) break;
        u32 rdHi = (op >> 16) & 0xF;
        u32 rdLo = (op >> 12) & 0xF;
        u32 rs   = (op >> 8)  & 0xF;
        u32 rm   = op & 0xF;
        bool setFlags = (op >> 20) & 1;
        if (rdHi == 15 || rdLo == 15 || rs == 15 || rm == 15) break;
        if (rdHi == rdLo) break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R0, rm);
        EmitGuestLoad(R1, rs);
        // SMULL R2, R3, R0, R1 → R3:R2 = R0 * R1 (signed 64-bit, no accumulate)
        // Encoding: cccc 0000 110S RdHi RdLo Rs 1001 Rm — base (S=0): 0xE0C00090
        const bool smullFlagsNeeded = setFlags && (bool)(instr.SetFlags & 0xF);
        EmitU32(0xE0C00090u | ((u32)smullFlagsNeeded << 20) | (3u << 16) | (2u << 12) | (1u << 8) | 0u);
        EmitGuestStore(rdLo, R2);
        EmitGuestStore(rdHi, R3);
        if (smullFlagsNeeded)
            MarkHostFlagsDirty(CPSR_NZMask);
        EmitAddCyclesConst(setFlags ? 5 : 3);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_UMULL:
    {
        // UMULL RdLo, RdHi, Rm, Rs: RdHi:RdLo = Rm * Rs  (unsigned 64-bit)
        if (cond != 0xE || cpu->Num != 0) break;
        u32 rdHi = (op >> 16) & 0xF;
        u32 rdLo = (op >> 12) & 0xF;
        u32 rs   = (op >> 8)  & 0xF;
        u32 rm   = op & 0xF;
        bool setFlags = (op >> 20) & 1;
        if (rdHi == 15 || rdLo == 15 || rs == 15 || rm == 15) break;
        if (rdHi == rdLo) break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R0, rm);
        EmitGuestLoad(R1, rs);
        // UMULL R2, R3, R0, R1 → R3:R2 = R0 * R1 (unsigned 64-bit)
        // Encoding: cccc 0000 100S RdHi RdLo Rs 1001 Rm — base (S=0): 0xE0800090
        const bool umullFlagsNeeded = setFlags && (bool)(instr.SetFlags & 0xF);
        EmitU32(0xE0800090u | ((u32)umullFlagsNeeded << 20) | (3u << 16) | (2u << 12) | (1u << 8) | 0u);
        EmitGuestStore(rdLo, R2);
        EmitGuestStore(rdHi, R3);
        if (umullFlagsNeeded)
            MarkHostFlagsDirty(CPSR_NZMask);
        EmitAddCyclesConst(setFlags ? 5 : 3);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_MRS:
    {
        // MRS Rd, CPSR — read CPSR into guest register (cond=AL, R=0/CPSR only)
        if (cond != 0xE) break;
        if ((op >> 22) & 1) break;  // R=1 means SPSR — let interpreter handle modes
        u32 rd = (op >> 12) & 0xF;
        if (rd == 15) break;

        EmitStoreR15(instr.Addr + 8);
        EmitLoadReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
        EmitGuestStore(rd, R0);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_MSR_REG:
    {
        // MSR CPSR_f, Rm — safe subset: flags-only write (field_mask=8), CPSR (R=0), cond=AL.
        // Writes Rm[31:28] into guest CPSR[31:28] (NZCV). No mode/interrupt change.
        if (cond != 0xE) break;
        if ((op >> 22) & 1) break;                  // R=1 → SPSR write, too complex
        if (((op >> 16) & 0xF) != 8) break;         // field_mask != flags-only → fall back
        u32 rm = op & 0xF;
        if (rm == 15) break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R1, rm);                                      // R1 = guest Rm
        EmitLoadReg(R0, R4, ArmMemberOffset(&ARM::CPSR));           // R0 = guest CPSR
        EmitU32(0xE3C004F0u);                                       // BIC R0, R0, #0xF0000000
        EmitU32(0xE20114F0u);                                       // AND R1, R1, #0xF0000000
        EmitOrrReg(R0, R0, R1);                                     // ORR R0, R0, R1
        EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::ak_CLZ:
    {
        // CLZ Rd, Rm — Count Leading Zeros. ARMv5+; safe for all ARM9 games.
        // cond=AL only: conditional CLZ is rare and complex to guard correctly.
        if (cond != 0xE) break;
        u32 rd = (op >> 12) & 0xF;
        u32 rm = op & 0xF;
        if (rd == 15 || rm == 15) break;

        EmitStoreR15(instr.Addr + 8);
        EmitGuestLoad(R1, rm);
        // CLZ R2, R1 — encoding: 0xE16F0F10 | (Rd<<12) | Rm  (host ARMv7 supports CLZ natively)
        EmitU32(0xE16F0F10u | (2u << 12) | 1u);
        EmitGuestStore(rd, R2);
        EmitAddCycles(instr, false, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    default:
        break;
    }

    if (IsArmWordTransferKind(instr.Info.Kind) ||
        IsArmHalfTransferKind(instr.Info.Kind) ||
        instr.Info.Kind == ARMInstrInfo::ak_LDM ||
        instr.Info.Kind == ARMInstrInfo::ak_STM)
    {
        if (TryEmitArmNativeMemory(instr, cpu, tailReturn))
            return true;

        A32InstrDesc desc;
        desc.instr = instr.Instr;
        desc.r15 = instr.Addr + 8;
        desc.codeCycles = instr.CodeCycles;
        desc.kind = instr.Info.Kind;
        desc.thumb = 0;

        EmitHelperCallWithDesc((const void*)&RunArmMemInstr, &desc, sizeof(desc), tailReturn);
        return true;
    }

    // ---------- ALU / data-processing ----------
    // cond=0xF is UNPREDICTABLE/reserved for standard DP instructions on ARMv5.
    if (cond == 0xF)
        return false;
    // For conditional instructions (cond != AL) we load guest flags into host CPSR
    // so native condition codes work. EmitCopyHostFlags is always safe: if the
    // condition was NOT taken the host CPSR still holds the original guest flags, so
    // re-writing them is a no-op.
    bool isConditional = (cond != 0xE);

    // Only emit decoded ARM ALU/test instructions. The raw data-processing
    // bit pattern also covers system/misc instructions such as MRS/MSR, and
    // executing those guest opwords directly on Android can raise SIGILL.
    if (!IsPatchableArmAluKind(instr.Info.Kind))
        return false;

    // Only data-processing instructions: bits 27-26 must be 00.
    if ((op >> 26) & 3)
        return false;

    bool isImmForm = (op >> 25) & 1;
    // bit 4=1 with bit 7=1 → multiply or misc: fall back to interpreter.
    // bit 4=1 with bit 7=0 → register-shifted-by-register: emit natively using R3 for Rs.
    bool isRegShiftByReg = false;
    if (!isImmForm && ((op >> 4) & 1))
    {
        if ((op >> 7) & 1)
            return false;  // multiply / extended DSP / misc — interpreter handles these
        isRegShiftByReg = true;
    }

    u32 Rd_guest = (op >> 12) & 0xF;
    u32 Rn_guest = (op >> 16) & 0xF;
    u32 Rm_guest = op & 0xF;
    bool setFlags = (op >> 20) & 1;
    u32 armOpcode = (op >> 21) & 0xF;

    // Categorise the instruction:
    //   MOV (0xD) / MVN (0xF): Rn is SBZ — only Rm (or immediate) feeds the output.
    //   TST (0x8) / TEQ (0x9) / CMP (0xA) / CMN (0xB): Rd is SBZ — result discarded,
    //     only flags are written; these always have S=1 in the encoding.
    //   ADC (0x5) / SBC (0x6) / RSC (0x7): consume guest carry, so they fall back.
    //   Everything else: standard two-operand ALU with Rn and Rd.
    bool isMovLike  = (armOpcode == 0xD || armOpcode == 0xF);
    bool isTestLike = (armOpcode >= 0x8 && armOpcode <= 0xB);
    bool usesCarry  = (armOpcode == 0x5 || armOpcode == 0x6 || armOpcode == 0x7);
    bool isLogical  = (armOpcode == 0x0 || armOpcode == 0x1 ||
                       armOpcode == 0x8 || armOpcode == 0x9 ||
                       armOpcode == 0xC || armOpcode == 0xD ||
                       armOpcode == 0xE || armOpcode == 0xF);

    // ADC/SBC/RSC consume the carry flag from guest CPSR. We emit these natively by
    // pre-loading guest CPSR flags into the host CPSR (same mechanism as conditional
    // instructions), so the host ADC/SBC instruction reads the correct carry bit.
    // RRX (ROR #0) also consumes carry the same way — it is not a usesCarry op
    // (armOpcode 0xD/0xF with shift form) so the guard below handles it separately.

    // ROR #0 is RRX, which consumes the input carry as part of operand2.
    // Fall back: the RRX encoding is only valid for reg-shifted-by-imm forms.
    if (!isImmForm && !isRegShiftByReg &&
        (((op >> 5) & 0x3) == Shift_ROR) && (((op >> 7) & 0x1F) == 0))
        return false;

    // Guard PC in register operand positions.
    if (!isTestLike && Rd_guest == 15)
        return false;   // writing PC triggers a branch — let interpreter handle
    if (!isMovLike && !isTestLike && Rn_guest == 15)
    {
        // EmitStoreR15(instr.Addr+8) runs before any guest-register loads, so at runtime
        // cpu->R[15] == instr.Addr+8. EmitGuestLoad(R0, 15) returns the correct pipeline PC.
        // For setFlags=1: flags reflect (PC+8) op #imm, computed by the host ALU at runtime —
        // no compile-time reasoning needed. Register forms (Rm=PC) still fall back.
        if (!isImmForm)
            return false;
        // Fall through: both setFlags=0 and setFlags=1 immediate forms are handled correctly.
    }
    if (!isImmForm && Rm_guest == 15)
        return false;
    if (isRegShiftByReg && ((op >> 8) & 0xF) == 15)
        return false;   // PC as shift register — unusual, let interpreter handle

    EmitStoreR15(instr.Addr + 8);

    // Seed host CPSR flags from guest CPSR for two reasons:
    //   • Conditional instructions: host condition codes must reflect guest flags.
    //   • Carry-consuming ops (ADC/SBC/RSC): host carry must reflect guest carry.
    if (isConditional || usesCarry)
        EmitLoadGuestCPSRFlags();

    if (isMovLike)
    {
        // MOV / MVN: output = f(Rm) or f(#imm), Rn not used.
        if (!isImmForm)
        {
            EmitGuestLoad(R1, Rm_guest);
            if (isRegShiftByReg)
            {
                // Also need the shift-count register Rs → R3.
                EmitGuestLoad(R3, (op >> 8) & 0xF);
                // Patch: Rd→R2, Rm→R1, Rs→R3  (cond bits kept from op)
                u32 patched = (op & ~0x0000FF0Fu) | (2u << 12) | (3u << 8) | 1u;
                EmitU32(patched);
            }
            else
            {
                // Patch: Rd→R2, Rm→R1  (Rn is SBZ=0, cond bits kept from op)
                u32 patched = (op & ~0x0000F00Fu) | (2u << 12) | 1u;
                EmitU32(patched);
            }
        }
        else
        {
            // Patch: Rd→R2  (no Rm field; Rn is SBZ=0, cond bits kept from op)
            u32 patched = (op & ~0x0000F000u) | (2u << 12);
            EmitU32(patched);
        }
        if (isConditional)
            EmitCondGuestStore(Rd_guest, R2, cond);
        else
            EmitGuestStore(Rd_guest, R2);
    }
    else if (isTestLike)
    {
        // TST / TEQ / CMP / CMN: output discarded, only CPSR flags matter.
        EmitGuestLoad(R0, Rn_guest);
        if (!isImmForm)
        {
            EmitGuestLoad(R1, Rm_guest);
            if (isRegShiftByReg)
            {
                EmitGuestLoad(R3, (op >> 8) & 0xF);
                // Patch: Rn→R0, Rd→0(SBZ), Rm→R1, Rs→R3  (cond bits kept)
                u32 patched = (op & ~0x000FFF0Fu) | (0u << 16) | (0u << 12) | (3u << 8) | 1u;
                EmitU32(patched);
            }
            else
            {
                // Patch: Rn→R0, Rd→0 (SBZ, already 0000), Rm→R1
                u32 patched = (op & ~0x000FF00Fu) | (0u << 16) | (0u << 12) | 1u;
                EmitU32(patched);
            }
        }
        else
        {
            // Patch: Rn→R0, Rd→0 (SBZ)
            u32 patched = (op & ~0x000FF000u) | (0u << 16) | (0u << 12);
            EmitU32(patched);
        }
        // No result to store; flags are handled below (test ops always set flags).
    }
    else
    {
        // Normal two-operand ALU: output = Rn op Rm (or #imm), write to Rd.
        EmitGuestLoad(R0, Rn_guest);
        if (!isImmForm)
        {
            EmitGuestLoad(R1, Rm_guest);
            if (isRegShiftByReg)
            {
                EmitGuestLoad(R3, (op >> 8) & 0xF);
                // Patch: Rn→R0, Rd→R2, Rm→R1, Rs→R3  (cond bits kept)
                u32 patched = (op & ~0x000FFF0Fu) | (0u << 16) | (2u << 12) | (3u << 8) | 1u;
                EmitU32(patched);
            }
            else
            {
                // Patch: Rn→R0, Rd→R2, Rm→R1
                u32 patched = (op & ~0x000FF00Fu) | (0u << 16) | (2u << 12) | 1u;
                EmitU32(patched);
            }
        }
        else
        {
            // Patch: Rn→R0, Rd→R2  (immediate encoding in bits 11-0 unchanged)
            u32 patched = (op & ~0x000FF000u) | (0u << 16) | (2u << 12);
            EmitU32(patched);
        }
        if (isConditional)
            EmitCondGuestStore(Rd_guest, R2, cond);
        else
            EmitGuestStore(Rd_guest, R2);
    }

    // Copy host CPSR flag bits (set by the emitted instruction) to guest CPSR.
    // Test ops always have S=1; all others only when the S bit is set.
    // For conditional instructions this is still correct: if the condition was not
    // taken, host CPSR still holds the original guest flags (loaded above), so
    // EmitCopyHostFlags is a no-op in that case.
    //
    // Dead-flag elimination (6.2): if FloodFillSetFlags determined that no later
    // instruction in this block reads these flag bits (SetFlags == 0) AND this is
    // not a tail instruction (tail flags may be consumed by the following block),
    // skip the five-instruction MRS/BIC/AND/ORR/STR sequence entirely.
    // Dead-flag elimination (JIT doc §6.2): FloodFillSetFlags propagates backwards
    // from block-exit branch instructions and marks each instruction's SetFlags bits
    // non-zero if a later instruction reads those flags.  Skip the five-instruction
    // MRS/BIC/AND/ORR/STR flag-copy sequence whenever the liveness analysis says no
    // later instruction will read these flags — whether at the tail OR in mid-block.
    // The original code only applied this elimination at the tail; mid-block
    // instructions always emitted the flag copy, wasting 5 host instructions per
    // dead S-bit op in every loop body.
    if ((setFlags || isTestLike) && (instr.SetFlags & 0xF))
    {
        // For reg-shift-by-reg the shift amount is not known at compile time, so
        // a logical operation ALWAYS potentially updates the carry flag.
        bool operand2UpdatesCarry = isRegShiftByReg || ArmOperand2UpdatesCarry(op, isImmForm);
        u32 flagMask = isLogical
            ? (operand2UpdatesCarry ? CPSR_NZCMask : CPSR_NZMask)
            : CPSR_NZCVMask;
        MarkHostFlagsDirty(flagMask);
    }

    EmitAddCycles(instr, false, cpu);
    if (tailReturn)
        EmitReturn();
    return true;
}

// M28: native emission for Thumb register-base immediate-offset single transfers.
// addr = R[rn] + offset (offset is always added, >= 0). rd/rn are low regs (R0-R7).
// Models the ARM single-transfer direct path: a runtime direct-region candidate
// chain (each guarded, DTCM-overlap-safe via EmitDirectSafeRegionSetup) with a
// fallback to RunThumbMemInstr for any address that isn't in a direct-safe region
// (IO/VRAM/misaligned/unmapped). Correctness can never regress: a guard miss just
// takes the same helper the instruction used before.
bool Compiler::TryEmitThumbMemImmDirect(const FetchedInstr& instr, ARM* cpu, bool tailReturn,
                                        int rd, int rn, u32 offset, int size, bool load)
{
    if (!EnableEmittedMainRAMLoadStore || !EnableThumbMemImmDirect)
        return false;

    struct ThumbMemCandidate { int region; u32 hint; };
    ThumbMemCandidate candidates[6];
    int candidateCount = 0;
    auto addCandidate = [&](int region, u32 hint)
    {
        if (!IsDirectSafeDataRegion(cpu, region, hint))
            return;
        for (int i = 0; i < candidateCount; i++)
        {
            const bool hintMatters = region == ARMJIT_Memory::memregion_WRAM7;
            if (candidates[i].region == region &&
                (!hintMatters || candidates[i].hint == hint))
                return;
        }
        candidates[candidateCount++] = { region, hint };
    };

    if (cpu->Num == 0)
    {
        addCandidate(ARMJIT_Memory::memregion_DTCM, 0);
        addCandidate(ARMJIT_Memory::memregion_MainRAM, 0x02000000);
        addCandidate(ARMJIT_Memory::memregion_SharedWRAM, 0x03000000);
        addCandidate(ARMJIT_Memory::memregion_ITCM, 0);
    }
    else
    {
        addCandidate(ARMJIT_Memory::memregion_WRAM7, 0x03800000);
        addCandidate(ARMJIT_Memory::memregion_WRAM7, 0x03000000);
        addCandidate(ARMJIT_Memory::memregion_MainRAM, 0x02000000);
        addCandidate(ARMJIT_Memory::memregion_SharedWRAM, 0x03000000);
    }

    if (!candidateCount)
        return false;

    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + 4;
    desc.codeCycles = instr.CodeCycles;
    desc.kind = instr.Info.Kind;
    desc.thumb = 1;

    EmitStoreR15(instr.Addr + 4);

    // R0 = effective address = R[rn] + offset
    EmitGuestLoad(R0, rn);
    if (offset)
    {
        EmitLoadImm(R1, offset);
        EmitAddReg(R0, R0, R1, false);
    }

    u8* helperBranches[8];
    int helperBranchCount = 0;

    // Alignment guard (half/word): misaligned accesses take the helper, which
    // applies the hardware rotate (ROR) / forced alignment semantics.
    if (size > 1)
    {
        EmitLoadImm(R3, (u32)(size - 1));
        EmitAndReg(R3, R0, R3);
        EmitCmpImm(R3, 0);
        helperBranches[helperBranchCount++] = EmitBranchPlaceholder(Cond_NE);
    }

    // M17.9 fast-exit: any addr >= 0x04000000 (IO/VRAM/etc.) fails every direct
    // guard — skip the chain straight to the helper.
    EmitMovRegShiftImm(R3, R0, Shift_LSR, 26, false);
    EmitCmpImm(R3, 0);
    helperBranches[helperBranchCount++] = EmitBranchPlaceholder(Cond_NE);

    u8* successBranches[8];
    int successBranchCount = 0;

    for (int c = 0; c < candidateCount; c++)
    {
        u8* regionFail[16];
        int regionFailCount = 0;

        if (load)
        {
            EmitDirectSafeRegionSetup(candidates[c].region, cpu, candidates[c].hint,
                                      R0, R12, R1, size, regionFail, regionFailCount);
            if (size == 1)
                EmitLoadByteRegOffset(R2, R1, R12);
            else if (size == 2)
                EmitLoadHalfRegOffset(R2, R1, R12);
            else
                EmitLoadRegOffset(R2, R1, R12);
            EmitGuestStore(rd, R2);
        }
        else
        {
            EmitDirectSafeRegionSetup(candidates[c].region, cpu, candidates[c].hint,
                                      R0, R2, R1, size, regionFail, regionFailCount);
            // A store into a 16-byte chunk that currently holds compiled code must
            // run JIT invalidation first → take the helper.
            EmitCodeBitmapGuard(candidates[c].region, R2, regionFail, regionFailCount);
            EmitDirectSafeRegionBase(candidates[c].region, cpu, R1);
            EmitGuestLoad(R3, rd);
            if (size == 1)
                EmitStoreByteRegOffset(R3, R1, R2);
            else if (size == 2)
                EmitStoreHalfRegOffset(R3, R1, R2);
            else
                EmitStoreRegOffset(R3, R1, R2);
        }

        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, load));

        if (tailReturn)
            EmitReturn();
        else
            successBranches[successBranchCount++] = EmitBranchPlaceholder(Cond_AL);

        u8* nextCandidate = CodePtr;
        for (int i = 0; i < regionFailCount; i++)
            PatchBranch(regionFail[i], nextCandidate);
    }

    u8* helperFallback = CodePtr;
    for (int i = 0; i < helperBranchCount; i++)
        PatchBranch(helperBranches[i], helperFallback);

    EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);

    u8* done = CodePtr;
    for (int i = 0; i < successBranchCount; i++)
        PatchBranch(successBranches[i], done);

    return true;
}

// M28: native emission for Thumb block transfers (PUSH/POP/STMIA/LDMIA).
// Mirrors the ARM LDM/STM direct paths: classify the base region from the
// compile-time DataRegion, guard the whole [base,base+N*4) range (DTCM-overlap
// safe), then unroll the per-register loads/stores. Any guard miss (region
// mismatch, misalignment, base-in-list, ARM7 POP-PC) falls back to the proven
// RunThumbMemInstr helper, so correctness can never regress.
bool Compiler::TryEmitThumbBlockTransfer(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    if (!EnableEmittedMainRAMLoadStore || !EnableThumbBlockTransfer)
        return false;

    const u32 op = instr.Instr;
    const u16 kind = instr.Info.Kind;

    bool load;
    bool decBefore;     // PUSH uses STMDB (decrement-before, full-descending)
    u32 rn;
    u32 rlist;
    switch (kind)
    {
    case ARMInstrInfo::tk_PUSH:
        load = false; decBefore = true;  rn = 13;
        rlist = (op & 0xFF) | ((op & 0x100) ? (1u << 14) : 0u); // +LR
        break;
    case ARMInstrInfo::tk_POP:
        load = true;  decBefore = false; rn = 13;
        rlist = (op & 0xFF) | ((op & 0x100) ? (1u << 15) : 0u); // +PC
        break;
    case ARMInstrInfo::tk_STMIA:
        load = false; decBefore = false; rn = (op >> 8) & 0x7;
        rlist = op & 0xFF;
        break;
    case ARMInstrInfo::tk_LDMIA:
        load = true;  decBefore = false; rn = (op >> 8) & 0x7;
        rlist = op & 0xFF;
        break;
    default:
        return false;
    }

    if (!rlist)
        return false;

    const bool hasPC = (rlist >> 15) & 1;
    if (hasPC)
    {
        // PC load is a block-exiting return. Native only on ARM9 (RunArmLoadedPC
        // matches the Thumb-POP interworking there); ARM7 needs pc|=1 → helper.
        if (cpu->Num != 0 || !tailReturn)
            return false;
    }

    // STMIA/LDMIA with the base register inside the list have writeback corner
    // cases (store-old-vs-new, suppressed writeback on load) — keep on the helper.
    if ((kind == ARMInstrInfo::tk_STMIA || kind == ARMInstrInfo::tk_LDMIA) &&
        (rlist & (1u << rn)))
        return false;

    const int expectedRegion = cpu->Num == 0
        ? ARMJIT_Memory::ClassifyAddress9(instr.DataRegion)
        : ARMJIT_Memory::ClassifyAddress7(instr.DataRegion);
    if (!IsDirectSafeDataRegion(cpu, expectedRegion, instr.DataRegion))
        return false;

    const u32 regCount = RegListCount(rlist);
    const u32 byteCount = regCount * 4;
    const s32 firstAddrDelta = decBefore ? -(s32)byteCount : 0;
    const s32 writebackDelta = decBefore ? -(s32)byteCount : (s32)byteCount;

    A32InstrDesc desc;
    desc.instr = instr.Instr;
    desc.r15 = instr.Addr + 4;
    desc.codeCycles = instr.CodeCycles;
    desc.kind = kind;
    desc.thumb = 1;

    u8* fallbackBranches[64];
    int fallbackBranchCount = 0;

    EmitStoreR15(instr.Addr + 4);

    EmitGuestLoad(R0, rn);
    if (firstAddrDelta)
    {
        EmitLoadImm(R1, AbsS32(firstAddrDelta));
        EmitSubReg(R0, R0, R1, false);   // firstAddrDelta is always negative here
    }

    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::DataRegion));
    EmitLoadImm(R1, (u32)instr.DataCycles);
    EmitStoreReg(R1, R4, ArmMemberOffset(&ARM::DataCycles));

    EmitDirectSafeRegionRangeSetup(expectedRegion, cpu, instr.DataRegion, R0, R0, R1, byteCount,
                                   fallbackBranches, fallbackBranchCount);

    if (load)
    {
        // R0 = running offset, R1 = base ptr.
        EmitLoadImm(R3, 4);
        u32 remaining = regCount;
        for (u32 i = 0; i < 16; i++)
        {
            if (!(rlist & (1u << i)))
                continue;
            EmitLoadRegOffset(R2, R1, R0);
            if (i == 15)
                EmitHelperCallLoadedPC(R2, false); // ARM9-only (guarded above)
            else
                EmitGuestStore(i, R2);
            remaining--;
            if (remaining)
                EmitAddReg(R0, R0, R3, false);
        }

        // Writeback (POP/LDMIA always writes the base back here; base never in list).
        EmitGuestLoad(R2, rn);
        EmitLoadImm(R3, AbsS32(writebackDelta));
        if (writebackDelta < 0)
            EmitSubReg(R2, R2, R3, false);
        else
            EmitAddReg(R2, R2, R3, false);
        EmitGuestStore(rn, R2);

        if (EnableA32JitProfiling)
            EmitIncrementCounter(&JitProfileCounters.DirectMemLDM);
        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, true));
    }
    else
    {
        // Store: first guard every target slot against the code bitmap (so a write
        // into compiled code triggers helper-side invalidation before any store),
        // then writeback, then the actual stores.
        EmitMovRegShiftImm(R2, R0, Shift_LSL, 0, false);
        u32 remaining = regCount;
        for (u32 i = 0; i < 15; i++)
        {
            if (!(rlist & (1u << i)))
                continue;
            EmitCodeBitmapGuard(expectedRegion, R2, fallbackBranches, fallbackBranchCount);
            remaining--;
            if (remaining)
            {
                EmitLoadImm(R3, 4);
                EmitAddReg(R2, R2, R3, false);
            }
        }

        EmitGuestLoad(R2, rn);
        EmitLoadImm(R3, AbsS32(writebackDelta));
        if (writebackDelta < 0)
            EmitSubReg(R2, R2, R3, false);
        else
            EmitAddReg(R2, R2, R3, false);
        EmitGuestStore(rn, R2);

        EmitDirectSafeRegionBase(expectedRegion, cpu, R1);
        remaining = regCount;
        for (u32 i = 0; i < 15; i++)
        {
            if (!(rlist & (1u << i)))
                continue;
            EmitGuestLoad(R2, i);
            EmitStoreRegOffset(R2, R1, R0);
            remaining--;
            if (remaining)
            {
                EmitLoadImm(R3, 4);
                EmitAddReg(R0, R0, R3, false);
            }
        }

        if (EnableA32JitProfiling)
            EmitIncrementCounter(&JitProfileCounters.DirectMemSTM);
        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, false));
    }

    u8* skipFallback = nullptr;
    if (tailReturn)
        EmitReturn();
    else
        skipFallback = EmitBranchPlaceholder(Cond_AL);

    u8* fallback = CodePtr;
    for (int i = 0; i < fallbackBranchCount; i++)
        PatchBranch(fallbackBranches[i], fallback);

    EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);

    if (skipFallback)
        PatchBranch(skipFallback, CodePtr);

    return true;
}

bool Compiler::TryEmitThumbNative(const FetchedInstr& instr, ARM* cpu, bool tailReturn)
{
    const u32 op = instr.Instr;
    // Dead-flag elimination: FloodFillSetFlags marks SetFlags non-zero if any later
    // instruction in this block reads the flags this instruction produces.  All Thumb
    // data-processing instructions always have S=1 in their encoding, but the five-
    // instruction flag-copy sequence (EmitCopyHostFlags/EmitCopyNZFromReg) is only
    // useful when flags will actually be consumed.  tf=false means skip the copy.
    const bool tf = (instr.SetFlags & 0xF) != 0;

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
        if (tf) MarkHostFlagsDirty(shift ? CPSR_NZCMask : CPSR_NZMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) EmitCopyNZFromReg(R2);
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
        if (tf) EmitCopyNZFromReg(R2);
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
        if (tf) EmitCopyNZFromReg(R2);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
        if (tf) EmitCopyNZFromReg(R2);
        EmitAddCycles(instr, true, cpu);
        if (tailReturn)
            EmitReturn();
        return true;
    }

    case ARMInstrInfo::tk_MUL_REG:
    {
        if (cpu->Num != 0)
            return false;

        u32 dst = op & 0x7;
        u32 src = (op >> 3) & 0x7;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, dst);
        EmitGuestLoad(R1, src);
        EmitMulReg(R2, R0, R1, true);
        EmitGuestStore(dst, R2);
        if (tf) MarkHostFlagsDirty(CPSR_NZMask);
        EmitAddCyclesConst(3);
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
        if (tf) EmitCopyNZFromReg(R2);
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
        if (tf) EmitCopyNZFromReg(R2);
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
        if (tf) MarkHostFlagsDirty(CPSR_NZCVMask);
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
    {
        // The literal address is fully determined at compile time:
        //   litAddr = (PC & ~2) + imm8*4 = ((instr.Addr+4) & ~2) + (op&0xFF)<<2
        // If the literal pool is in a safe region (ITCM/MainRAM for ARM9 code),
        // emit a direct fast-path load instead of calling RunThumbMemInstr.
        const u32 rd = (op >> 8) & 0x7;
        const u32 litAddr = ((instr.Addr + 4) & ~2u) + ((op & 0xFF) << 2);

        A32InstrDesc desc;
        desc.instr = instr.Instr;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.kind = instr.Info.Kind;
        desc.thumb = 1;

        const int region = cpu->Num == 0
            ? ARMJIT_Memory::ClassifyAddress9(litAddr)
            : ARMJIT_Memory::ClassifyAddress7(litAddr);

        if (!IsDirectSafeDataRegion(cpu, region, litAddr))
        {
            EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);
            return true;
        }

        u8* fallbackBranches[8];
        int fallbackBranchCount = 0;

        EmitStoreR15(instr.Addr + 4);
        EmitLoadImm(R0, litAddr);
        EmitDirectSafeRegionSetup(region, cpu, litAddr, R0, R12, R1, 4,
                                  fallbackBranches, fallbackBranchCount);
        EmitLoadRegOffset(R2, R1, R12);
        EmitGuestStore(rd, R2);
        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, true));

        u8* skipFallback = nullptr;
        if (tailReturn)
            EmitReturn();
        else
            skipFallback = EmitBranchPlaceholder(Cond_AL);

        u8* fallback = CodePtr;
        for (int i = 0; i < fallbackBranchCount; i++)
            PatchBranch(fallbackBranches[i], fallback);

        EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);

        if (skipFallback)
            PatchBranch(skipFallback, CodePtr);

        return true;
    }

    case ARMInstrInfo::tk_LDR_SPREL:
    {
        // LDR Rd, [SP, #imm8*4] — word load from SP + (op&0xFF)<<2.
        // ARM9: speculate SP is in MainRAM (actual DS game stacks live at 0x02XXXXXX).
        // ARM7: speculate SP is in WRAM7 (0x03800000).
        // Runtime guard falls through to RunThumbMemInstr if SP is elsewhere.
        const u32 rd = (op >> 8) & 0x7;
        const u32 imm = (op & 0xFF) << 2;

        A32InstrDesc desc;
        desc.instr = instr.Instr;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.kind = instr.Info.Kind;
        desc.thumb = 1;

        const int region = cpu->Num == 0
            ? ARMJIT_Memory::memregion_MainRAM
            : ARMJIT_Memory::memregion_WRAM7;
        const u32 regionHint = cpu->Num == 0 ? 0x02000000u : 0x03800000u;

        if (!IsDirectSafeDataRegion(cpu, region, regionHint))
        {
            EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);
            return true;
        }

        u8* fallbackBranches[8];
        int fallbackBranchCount = 0;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, 13);
        if (imm)
        {
            EmitLoadImm(R1, imm);
            EmitAddReg(R0, R0, R1, false);
        }

        EmitDirectSafeRegionSetup(region, cpu, regionHint, R0, R12, R1, 4,
                                  fallbackBranches, fallbackBranchCount);
        EmitLoadRegOffset(R2, R1, R12);
        EmitGuestStore(rd, R2);
        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, true));

        u8* skipFallback = nullptr;
        if (tailReturn)
            EmitReturn();
        else
            skipFallback = EmitBranchPlaceholder(Cond_AL);

        u8* fallback = CodePtr;
        for (int i = 0; i < fallbackBranchCount; i++)
            PatchBranch(fallbackBranches[i], fallback);

        EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);

        if (skipFallback)
            PatchBranch(skipFallback, CodePtr);

        return true;
    }

    case ARMInstrInfo::tk_STR_SPREL:
    {
        // STR Rd, [SP, #imm8*4] — word store to SP + (op&0xFF)<<2.
        // Same SP speculation as LDR_SPREL: MainRAM for ARM9, WRAM7 for ARM7.
        const u32 rd = (op >> 8) & 0x7;
        const u32 imm = (op & 0xFF) << 2;

        A32InstrDesc desc;
        desc.instr = instr.Instr;
        desc.r15 = instr.Addr + 4;
        desc.codeCycles = instr.CodeCycles;
        desc.kind = instr.Info.Kind;
        desc.thumb = 1;

        const int region = cpu->Num == 0
            ? ARMJIT_Memory::memregion_MainRAM
            : ARMJIT_Memory::memregion_WRAM7;
        const u32 regionHint = cpu->Num == 0 ? 0x02000000u : 0x03800000u;

        if (!IsDirectSafeDataRegion(cpu, region, regionHint))
        {
            EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);
            return true;
        }

        u8* fallbackBranches[8];
        int fallbackBranchCount = 0;

        EmitStoreR15(instr.Addr + 4);
        EmitGuestLoad(R0, 13);
        if (imm)
        {
            EmitLoadImm(R1, imm);
            EmitAddReg(R0, R0, R1, false);
        }

        EmitDirectSafeRegionSetup(region, cpu, regionHint, R0, R2, R1, 4,
                                  fallbackBranches, fallbackBranchCount);
        EmitCodeBitmapGuard(region, R2, fallbackBranches, fallbackBranchCount);
        EmitDirectSafeRegionBase(region, cpu, R1);

        EmitGuestLoad(R3, rd);
        EmitStoreRegOffset(R3, R1, R2);
        EmitAddCyclesConst(ArmDirectMemoryCycles(instr, cpu, false));

        u8* skipFallback = nullptr;
        if (tailReturn)
            EmitReturn();
        else
            skipFallback = EmitBranchPlaceholder(Cond_AL);

        u8* fallback = CodePtr;
        for (int i = 0; i < fallbackBranchCount; i++)
            PatchBranch(fallbackBranches[i], fallback);

        EmitHelperCallWithDesc((const void*)&RunThumbMemInstr, &desc, sizeof(desc), tailReturn);

        if (skipFallback)
            PatchBranch(skipFallback, CodePtr);

        return true;
    }

    // M28: register-base immediate-offset single transfers. rd=bits[2:0],
    // rn=bits[5:3]; offset scaling matches the Thumb encoding (word imm5*4,
    // half imm5*2, byte imm5). Try the native direct path; on any setup miss
    // (no safe candidate) fall through to the RunThumbMemInstr helper below.
    case ARMInstrInfo::tk_STR_IMM:
    case ARMInstrInfo::tk_LDR_IMM:
    case ARMInstrInfo::tk_STRB_IMM:
    case ARMInstrInfo::tk_LDRB_IMM:
    case ARMInstrInfo::tk_STRH_IMM:
    case ARMInstrInfo::tk_LDRH_IMM:
    {
        const int rd = op & 0x7;
        const int rn = (op >> 3) & 0x7;
        int size; bool load; u32 offset;
        switch (instr.Info.Kind)
        {
        case ARMInstrInfo::tk_LDR_IMM:  size = 4; load = true;  offset = (op >> 4) & 0x7C; break;
        case ARMInstrInfo::tk_STR_IMM:  size = 4; load = false; offset = (op >> 4) & 0x7C; break;
        case ARMInstrInfo::tk_LDRH_IMM: size = 2; load = true;  offset = (op >> 5) & 0x3E; break;
        case ARMInstrInfo::tk_STRH_IMM: size = 2; load = false; offset = (op >> 5) & 0x3E; break;
        case ARMInstrInfo::tk_LDRB_IMM: size = 1; load = true;  offset = (op >> 6) & 0x1F; break;
        default: /* tk_STRB_IMM */      size = 1; load = false; offset = (op >> 6) & 0x1F; break;
        }
        if (TryEmitThumbMemImmDirect(instr, cpu, tailReturn, rd, rn, offset, size, load))
            return true;
    }
    // fallthrough to the helper path
    [[fallthrough]];

    // M28: block transfers — try the native path; on any miss fall through.
    case ARMInstrInfo::tk_PUSH:
    case ARMInstrInfo::tk_POP:
    case ARMInstrInfo::tk_STMIA:
    case ARMInstrInfo::tk_LDMIA:
        if (TryEmitThumbBlockTransfer(instr, cpu, tailReturn))
            return true;
        [[fallthrough]];

    case ARMInstrInfo::tk_STR_REG:
    case ARMInstrInfo::tk_STRB_REG:
    case ARMInstrInfo::tk_LDR_REG:
    case ARMInstrInfo::tk_LDRB_REG:
    {
        if (EnableThumbMemImmDirect)
        {
            const u32 rd = op & 0x7;
            const u32 rn = (op >> 3) & 0x7;
            const u32 rm = (op >> 6) & 0x7;
            bool isLoad = (instr.Info.Kind == ARMInstrInfo::tk_LDR_REG ||
                           instr.Info.Kind == ARMInstrInfo::tk_LDRB_REG);
            int size = (instr.Info.Kind == ARMInstrInfo::tk_LDRB_REG ||
                        instr.Info.Kind == ARMInstrInfo::tk_STRB_REG) ? 1 : 4;

            EmitStoreR15(instr.Addr + 4);
            EmitGuestLoad(R0, rn);
            EmitGuestLoad(R1, rm);
            EmitAddReg(R0, R0, R1, false);
            if (TryEmitThumbMemImmDirect(instr, cpu, tailReturn, rd, rn, 0, size, isLoad))
                return true;
        }
        goto thumb_reg_mem_fallback;
    }

    case ARMInstrInfo::tk_STRH_REG:
    case ARMInstrInfo::tk_LDRSB_REG:
    case ARMInstrInfo::tk_LDRH_REG:
    case ARMInstrInfo::tk_LDRSH_REG:
    thumb_reg_mem_fallback:
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
        const s32 offset = (s32)(op << 24) >> 23;
        const u32 cond = (op >> 8) & 0xF;
        const u32 r15 = instr.Addr + 4;
        const u32 target = r15 + (u32)offset;

        if (cond < 0xE)
        {
            EmitStoreR15(r15);
            EmitLoadGuestCPSRFlags();
            EmitAddCycles(instr, true, cpu);

            u8* condBranch = EmitBranchPlaceholder(cond);
            EmitReturn();

            PatchBranch(condBranch, CodePtr);
            EmitJumpToConst(cpu, target, true, true, tailReturn);
            return true;
        }

        A32BranchDesc desc;
        desc.r15 = r15;
        desc.codeCycles = instr.CodeCycles;
        desc.target = target + 1;
        desc.aux = cond;

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

        EmitJumpToConst(cpu, desc.target, true, true, tailReturn);
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
    FlushRegAllocForHelper();
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

void Compiler::EmitHelperCallLoadedPC(int pcReg, bool restoreCPSR)
{
    FlushRegAllocForHelper();
    EmitU32(0xE1A00004); // mov r0, r4
    if (pcReg != R1)
        EmitMovRegShiftImm(R1, pcReg, Shift_LSL, 0, false);
    EmitU32(0xE59FC004); // ldr ip, [pc, #4]
    EmitU32(0xE12FFF3C); // blx ip
    EmitU32(0xEA000000); // b after helper literal
    EmitU32((u32)(uintptr_t)(restoreCPSR ? &RunArmLoadedPCRestoreCPSR : &RunArmLoadedPC));
}

void Compiler::EmitHelperCall1(const void* helper, u32 arg1Imm)
{
    FlushRegAllocForHelper();
    EmitLoadImm(R1, arg1Imm);
    EmitU32(0xE1A00004); // mov r0, r4
    EmitU32(0xE59FC004); // ldr ip, [pc, #4]
    EmitU32(0xE12FFF3C); // blx ip
    EmitU32(0xEA000000); // b after helper literal
    EmitU32((u32)(uintptr_t)helper);
}

void Compiler::EmitHelperCall2(const void* helper, int arg1Reg, u32 arg2Imm)
{
    FlushRegAllocForHelper();
    if (arg1Reg != R1)
        EmitMovRegShiftImm(R1, arg1Reg, Shift_LSL, 0, false);
    EmitLoadImm(R2, arg2Imm);
    EmitU32(0xE1A00004); // mov r0, r4
    EmitU32(0xE59FC004); // ldr ip, [pc, #4]
    EmitU32(0xE12FFF3C); // blx ip
    EmitU32(0xEA000000); // b after helper literal
    EmitU32((u32)(uintptr_t)helper);
}

void Compiler::EmitHelperCall3(const void* helper, int arg1Reg, int arg2Reg, u32 arg3Imm)
{
    FlushRegAllocForHelper();
    if (arg2Reg != R2)
        EmitMovRegShiftImm(R2, arg2Reg, Shift_LSL, 0, false);
    if (arg1Reg != R1)
        EmitMovRegShiftImm(R1, arg1Reg, Shift_LSL, 0, false);
    EmitLoadImm(R3, arg3Imm);
    EmitU32(0xE1A00004); // mov r0, r4
    EmitU32(0xE59FC004); // ldr ip, [pc, #4]
    EmitU32(0xE12FFF3C); // blx ip
    EmitU32(0xEA000000); // b after helper literal
    EmitU32((u32)(uintptr_t)helper);
}

void Compiler::EmitReturn()
{
    FlushRegAllocForHelper();
    EmitFlushPendingCycles();
    EmitU32(0xE59FC000); // ldr ip, [pc]
    EmitU32(0xE12FFF1C); // bx ip
    EmitU32((u32)(uintptr_t)&ARM_Ret);
}

void Compiler::EmitFlushPendingCycles()
{
    if (!m_pendingCycles)
        return;
    EmitLoadReg(R0, R4, ArmMemberOffset(&ARM::Cycles));
    EmitLoadImm(R1, m_pendingCycles);
    EmitAddReg(R0, R0, R1, false);
    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::Cycles));
    m_pendingCycles = 0;
}

u8* Compiler::EmitBranchPlaceholder(u32 cond)
{
    u8* branch = CodePtr;
    EmitU32((cond << 28) | 0x0A000000);
    return branch;
}

void Compiler::PatchBranch(u8* branch, u8* target)
{
    const s32 delta = (s32)(target - (branch + 8));
    assert((delta & 0x3) == 0);

    const s32 imm = delta >> 2;
    assert(imm >= -0x800000 && imm <= 0x7FFFFF);

    u32 word;
    memcpy(&word, branch, sizeof(word));
    word = (word & 0xFF000000) | ((u32)imm & 0x00FFFFFF);
    memcpy(branch, &word, sizeof(word));
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

void Compiler::EmitLoadRegOffset(int dst, int base, int offsetReg)
{
    EmitU32(0xE7900000
        | ((u32)base << 16)
        | ((u32)dst << 12)
        | (u32)offsetReg);
}

void Compiler::EmitStoreRegOffset(int src, int base, int offsetReg)
{
    EmitU32(0xE7800000
        | ((u32)base << 16)
        | ((u32)src << 12)
        | (u32)offsetReg);
}

void Compiler::EmitLoadByteRegOffset(int dst, int base, int offsetReg)
{
    EmitU32(0xE7D00000
        | ((u32)base << 16)
        | ((u32)dst << 12)
        | (u32)offsetReg);
}

void Compiler::EmitStoreByteRegOffset(int src, int base, int offsetReg)
{
    EmitU32(0xE7C00000
        | ((u32)base << 16)
        | ((u32)src << 12)
        | (u32)offsetReg);
}

void Compiler::EmitLoadHalfRegOffset(int dst, int base, int offsetReg)
{
    EmitU32(0xE19000B0
        | ((u32)base << 16)
        | ((u32)dst << 12)
        | (u32)offsetReg);
}

void Compiler::EmitStoreHalfRegOffset(int src, int base, int offsetReg)
{
    EmitU32(0xE18000B0
        | ((u32)base << 16)
        | ((u32)src << 12)
        | (u32)offsetReg);
}

bool Compiler::IsDirectSafeDataRegion(ARM* cpu, int region, u32 compileAddr)
{
    switch (region)
    {
    case ARMJIT_Memory::memregion_ITCM:
    case ARMJIT_Memory::memregion_DTCM:
        return cpu->Num == 0;
    case ARMJIT_Memory::memregion_MainRAM:
        return true;
    case ARMJIT_Memory::memregion_SharedWRAM:
        return NDS::ConsoleType == 0 &&
            (cpu->Num == 0 ? NDS::SWRAM_ARM9.Mem != nullptr : NDS::SWRAM_ARM7.Mem != nullptr);
    case ARMJIT_Memory::memregion_WRAM7:
        return cpu->Num == 1 && NDS::ConsoleType == 0 &&
            ((compileAddr & 0xFF800000) == 0x03000000 ||
             (compileAddr & 0xFF800000) == 0x03800000);
    case ARMJIT_Memory::memregion_BIOS7:
        // BIOS7 is read-only ROM (NDS::ARM7BIOS, 16 KB). Direct data reads are safe;
        // stores never reach BIOS7 on real hardware and must still go via the helper.
        // Non-DSi only: DSi remaps BIOS7 to a different region (memregion_BIOS7DSi).
        return cpu->Num == 1 && NDS::ConsoleType == 0;
    default:
        return false;
    }
}

void Compiler::EmitDirectSafeRegionAddressGuard(int region, ARM* cpu, u32 compileAddr, int addrReg,
                                                u8** fallbackBranches, int& fallbackBranchCount,
                                                bool checkDTCM)
{
    switch (region)
    {
    case ARMJIT_Memory::memregion_ITCM:
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::ITCMSize));
        EmitCmpReg(addrReg, R3);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_HS);
        break;

    case ARMJIT_Memory::memregion_DTCM:
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMSize));
        EmitCmpImm(R3, 0);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_EQ);

        EmitLoadReg(R12, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMBase));
        EmitCmpReg(addrReg, R12);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_LO);

        EmitAddReg(R3, R12, R3, false);
        EmitCmpReg(addrReg, R3);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_HS);
        break;

    case ARMJIT_Memory::memregion_MainRAM:
        EmitMovRegShiftImm(R3, addrReg, Shift_LSR, 24, false);
        EmitCmpImm(R3, 0x02);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);
        break;

    case ARMJIT_Memory::memregion_SharedWRAM:
        if (cpu->Num == 0)
        {
            EmitMovRegShiftImm(R3, addrReg, Shift_LSR, 24, false);
            EmitCmpImm(R3, 0x03);
        }
        else
        {
            EmitMovRegShiftImm(R3, addrReg, Shift_LSR, 23, false);
            EmitCmpImm(R3, 0x06);
        }
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);

        {
            const NDS::MemRegion* mapped = cpu->Num == 0 ? &NDS::SWRAM_ARM9 : &NDS::SWRAM_ARM7;
            EmitLoadImm(R3, (u32)(uintptr_t)&mapped->Mem);
            EmitLoadReg(R3, R3, 0);
            EmitLoadImm(R12, (u32)(uintptr_t)mapped->Mem);
            EmitCmpReg(R3, R12);
            fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);

            EmitLoadImm(R3, (u32)(uintptr_t)&mapped->Mask);
            EmitLoadReg(R3, R3, 0);
            EmitLoadImm(R12, mapped->Mask);
            EmitCmpReg(R3, R12);
            fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);
        }
        break;

    case ARMJIT_Memory::memregion_WRAM7:
        EmitMovRegShiftImm(R3, addrReg, Shift_LSR, 23, false);
        EmitCmpImm(R3, (compileAddr >> 23) & 0xFF);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);

        if ((compileAddr & 0xFF800000) == 0x03000000)
        {
            EmitLoadImm(R3, (u32)(uintptr_t)&NDS::SWRAM_ARM7.Mem);
            EmitLoadReg(R3, R3, 0);
            EmitCmpImm(R3, 0);
            fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);
        }
        break;

    case ARMJIT_Memory::memregion_BIOS7:
        // BIOS7 occupies 0x00000000–0x00003FFF (16 KB). Guard: addr < 0x4000.
        // Use LSR #14: any address inside the 16 KB window gives R3 == 0.
        EmitMovRegShiftImm(R3, addrReg, Shift_LSR, 14, false);
        EmitCmpImm(R3, 0);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);
        break;

    default:
        assert(false && "unsupported A32 direct memory region");
        break;
    }

    if (checkDTCM &&
        cpu->Num == 0 &&
        region != ARMJIT_Memory::memregion_ITCM &&
        region != ARMJIT_Memory::memregion_DTCM)
    {
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMSize));
        EmitCmpImm(R3, 0);
        u8* noDTCM = EmitBranchPlaceholder(Cond_EQ);

        EmitLoadReg(R12, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMBase));
        EmitCmpReg(addrReg, R12);
        u8* belowDTCM = EmitBranchPlaceholder(Cond_LO);

        EmitAddReg(R3, R12, R3, false);
        EmitCmpReg(addrReg, R3);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_LO);

        u8* afterDTCMCheck = CodePtr;
        PatchBranch(noDTCM, afterDTCMCheck);
        PatchBranch(belowDTCM, afterDTCMCheck);
    }
}

void Compiler::EmitDirectSafeRegionBase(int region, ARM* cpu, int baseReg)
{
    switch (region)
    {
    case ARMJIT_Memory::memregion_ITCM:
        EmitLoadImm(R3, CpuMemberOffset<ARMv5>(&ARMv5::ITCM));
        EmitAddReg(baseReg, R4, R3, false);
        break;

    case ARMJIT_Memory::memregion_DTCM:
        // DTCM lives after the 32 KB ITCM array in ARMv5, so its member offset
        // is far beyond ARM's 12-bit immediate LDR range.
        EmitLoadImm(R3, CpuMemberOffset<ARMv5>(&ARMv5::DTCM));
        EmitLoadRegOffset(baseReg, R4, R3);
        break;

    case ARMJIT_Memory::memregion_MainRAM:
        EmitLoadImm(baseReg, (u32)(uintptr_t)NDS::MainRAM);
        break;

    case ARMJIT_Memory::memregion_SharedWRAM:
        EmitLoadImm(baseReg, (u32)(uintptr_t)NDS::SharedWRAM);
        break;

    case ARMJIT_Memory::memregion_WRAM7:
        EmitLoadImm(baseReg, (u32)(uintptr_t)NDS::ARM7WRAM);
        break;

    case ARMJIT_Memory::memregion_BIOS7:
        EmitLoadImm(baseReg, (u32)(uintptr_t)NDS::ARM7BIOS);
        break;

    default:
        assert(false && "unsupported A32 direct memory region");
        break;
    }

    (void)cpu;
}

void Compiler::EmitDirectSafeRegionSetup(int region, ARM* cpu, u32 compileAddr, int addrReg, int offsetReg,
                                         int baseReg, int size, u8** fallbackBranches, int& fallbackBranchCount)
{
    // checkDTCM=true is REQUIRED for correctness here. On ARM9 the SP-relative
    // speculation (and stale DataRegion hints) classify stack/flag accesses as
    // MainRAM, but the MainRAM guard only tests (addr>>24)==0x02. Nintendo-SDK
    // titles (e.g. NSMB) map DTCM — which holds the stack and IRQ flags — at
    // 0x027C0000, which ALSO satisfies (addr>>24)==0x02. Without the DTCM-overlap
    // exclusion the MainRAM candidate wrongly swallows DTCM accesses and routes
    // them into MainRAM[addr & mask], so an IRQ flag write/read lands in the wrong
    // array and the game hangs in a wait loop at full speed. The LDM/STM range
    // setup already does this inline; single transfers must match.
    EmitDirectSafeRegionAddressGuard(region, cpu, compileAddr, addrReg,
                                     fallbackBranches, fallbackBranchCount,
                                     /*checkDTCM=*/true);

    if (region == ARMJIT_Memory::memregion_DTCM)
    {
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMBase));
        EmitSubReg(offsetReg, addrReg, R3, false);
    }
    else if (offsetReg != addrReg)
    {
        EmitMovRegShiftImm(offsetReg, addrReg, Shift_LSL, 0, false);
    }

    const u32 offsetMask = DirectSafeRegionMask(cpu, region) & ~(u32)(size - 1);
    EmitLoadImm(R3, offsetMask);
    EmitAndReg(offsetReg, offsetReg, R3);

    if (region == ARMJIT_Memory::memregion_SharedWRAM)
    {
        const u32 memOffset = DirectSharedWRAMOffset(cpu);
        if (memOffset)
        {
            EmitLoadImm(R3, memOffset);
            EmitAddReg(offsetReg, offsetReg, R3, false);
        }
    }

    EmitDirectSafeRegionBase(region, cpu, baseReg);
}

void Compiler::EmitDirectSafeRegionRangeSetup(int region, ARM* cpu, u32 compileAddr, int addrReg,
                                              int offsetReg, int baseReg, u32 byteCount,
                                              u8** fallbackBranches, int& fallbackBranchCount)
{
    assert(byteCount >= 4);

    const u32 lastByteOffset = byteCount - 4;

    EmitDirectSafeRegionAddressGuard(region, cpu, compileAddr, addrReg,
                                     fallbackBranches, fallbackBranchCount, false);

    if (lastByteOffset)
    {
        EmitLoadImm(R2, lastByteOffset);
        EmitAddReg(R2, addrReg, R2, false);
        EmitDirectSafeRegionAddressGuard(region, cpu, compileAddr, R2,
                                         fallbackBranches, fallbackBranchCount, false);
    }

    if (cpu->Num == 0 &&
        region != ARMJIT_Memory::memregion_ITCM &&
        region != ARMJIT_Memory::memregion_DTCM)
    {
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMSize));
        EmitCmpImm(R3, 0);
        u8* noDTCM = EmitBranchPlaceholder(Cond_EQ);

        EmitLoadReg(R1, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMBase));
        EmitLoadImm(R2, byteCount);
        EmitAddReg(R2, addrReg, R2, false); // exclusive end address

        EmitCmpReg(addrReg, R1);
        u8* startBelowDTCM = EmitBranchPlaceholder(Cond_LO);

        EmitAddReg(R3, R1, R3, false); // DTCM end
        EmitCmpReg(addrReg, R3);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_LO);
        u8* afterStartInDTCMCheck = EmitBranchPlaceholder(Cond_AL);

        u8* startBelowTarget = CodePtr;
        PatchBranch(startBelowDTCM, startBelowTarget);
        EmitCmpReg(R1, R2);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_LO);

        u8* afterDTCMCheck = CodePtr;
        PatchBranch(noDTCM, afterDTCMCheck);
        PatchBranch(afterStartInDTCMCheck, afterDTCMCheck);
    }

    EmitLoadImm(R3, 3);
    EmitAndReg(R3, addrReg, R3);
    EmitCmpImm(R3, 0);
    fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);

    if (region == ARMJIT_Memory::memregion_DTCM)
    {
        EmitLoadReg(R3, R4, CpuMemberOffset<ARMv5>(&ARMv5::DTCMBase));
        EmitSubReg(offsetReg, addrReg, R3, false);
    }
    else if (offsetReg != addrReg)
    {
        EmitMovRegShiftImm(offsetReg, addrReg, Shift_LSL, 0, false);
    }

    const u32 offsetMask = DirectSafeRegionMask(cpu, region) & ~3u;
    EmitLoadImm(R3, offsetMask);
    EmitAndReg(offsetReg, offsetReg, R3);

    if (lastByteOffset)
    {
        assert(lastByteOffset <= offsetMask);
        EmitLoadImm(R3, offsetMask - lastByteOffset);
        EmitCmpReg(R3, offsetReg);
        fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_LO);
    }

    if (region == ARMJIT_Memory::memregion_SharedWRAM)
    {
        const u32 memOffset = DirectSharedWRAMOffset(cpu);
        if (memOffset)
        {
            EmitLoadImm(R3, memOffset);
            EmitAddReg(offsetReg, offsetReg, R3, false);
        }
    }

    EmitDirectSafeRegionBase(region, cpu, baseReg);
}

void Compiler::EmitCodeBitmapGuard(int region, int offsetReg, u8** fallbackBranches, int& fallbackBranchCount)
{
    if (!CodeMemRegions[region])
        return;

    EmitMovRegShiftImm(R12, offsetReg, Shift_LSR, 9, false);
    EmitLoadImm(R3, sizeof(AddressRange));
    EmitMulReg(R12, R12, R3, false);
    EmitLoadImm(R3, (u32)(uintptr_t)CodeMemRegions[region]);
    EmitAddReg(R12, R3, R12, false);
    EmitLoadReg(R12, R12, AddressRangeCodeOffset());

    EmitMovRegShiftImm(R1, offsetReg, Shift_LSR, 4, false);
    EmitLoadImm(R3, 31);
    EmitAndReg(R1, R1, R3);
    EmitLoadImm(R3, 1);
    EmitMovRegShiftReg(R3, R3, Shift_LSL, R1, false);
    EmitAndReg(R12, R12, R3);
    EmitCmpImm(R12, 0);
    fallbackBranches[fallbackBranchCount++] = EmitBranchPlaceholder(Cond_NE);
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

void Compiler::EmitMulReg(int dst, int lhs, int rhs, bool setFlags)
{
    EmitU32(0xE0000090
        | (setFlags ? (1u << 20) : 0)
        | ((u32)dst << 16)
        | ((u32)rhs << 8)
        | (u32)lhs);
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

void Compiler::EmitMovRegShiftReg(int dst, int src, int shiftType, int shiftReg, bool setFlags)
{
    EmitU32(0xE1A00010
        | (setFlags ? (1u << 20) : 0)
        | ((u32)dst << 12)
        | ((u32)shiftReg << 8)
        | ((u32)shiftType << 5)
        | (u32)src);
}

void Compiler::EmitCmpImm(int lhs, u32 imm)
{
    assert(imm <= 0xFF);
    EmitU32(0xE3500000 | ((u32)lhs << 16) | imm);
}

void Compiler::EmitCmpReg(int lhs, int rhs)
{
    EmitU32(0xE1500000 | ((u32)lhs << 16) | (u32)rhs);
}

void Compiler::EmitMRS(int dst)
{
    EmitU32(0xE10F0000 | ((u32)dst << 12)); // mrs dst, cpsr
}

// Load guest CPSR flags (NZCV) into host CPSR so native condition codes work.
// Uses R3 as scratch. Must be called before emitting any conditional instruction.
void Compiler::EmitLoadGuestCPSRFlags()
{
    EmitFlushDirtyFlags();
    EmitLoadReg(R3, R4, ArmMemberOffset(&ARM::CPSR));  // R3 = guest CPSR
    EmitU32(0xE128F003);  // MSR CPSR_f, R3  — copies R3[31:24] (NZCV) into host CPSR
}

void Compiler::EmitGuestLoad(int dst, int guestReg)
{
    if (m_useRegAlloc && guestReg >= 0 && guestReg < 15)
    {
        int mapped = RegAllocCache.Mapping[guestReg];
        if (mapped != -1)
        {
            if (dst != mapped)
                EmitMovRegShiftImm(dst, mapped, Shift_LSL, 0, false);
            return;
        }
    }
    if (guestReg >= 0 && guestReg < 16 && m_rcHostReg[guestReg] != -1)
    {
        EmitMovRegShiftImm(dst, m_rcHostReg[guestReg], Shift_LSL, 0, false);
        return;
    }
    EmitLoadReg(dst, R4, ArmMemberOffset(&ARM::R) + guestReg * 4);
}

void Compiler::EmitGuestStore(int guestReg, int src)
{
    if (m_useRegAlloc && guestReg >= 0 && guestReg < 15)
    {
        int mapped = RegAllocCache.Mapping[guestReg];
        if (mapped != -1)
        {
            if (src != mapped)
                EmitMovRegShiftImm(mapped, src, Shift_LSL, 0, false);
            RegAllocCache.DirtyRegs |= (1 << guestReg);
            return;
        }
    }
    EmitStoreReg(src, R4, ArmMemberOffset(&ARM::R) + guestReg * 4);
}

// Conditional guest-register store: STR{cond} src, [R4, #guest_Rx_offset].
void Compiler::EmitCondGuestStore(int guestReg, int src, u32 cond)
{
    if (m_useRegAlloc && guestReg >= 0 && guestReg < 15)
    {
        int mapped = RegAllocCache.Mapping[guestReg];
        if (mapped != -1)
        {
            if (src != mapped)
                EmitU32(0x01A00000u | ((u32)cond << 28) | ((u32)mapped << 12) | (u32)src);
            RegAllocCache.DirtyRegs |= (1 << guestReg);
            return;
        }
    }
    u32 offset = ArmMemberOffset(&ARM::R) + (u32)guestReg * 4;
    assert(offset <= 0xFFF);
    EmitU32(0x05800000u | ((u32)R4 << 16) | ((u32)src << 12) | offset | ((u32)cond << 28));
}

void Compiler::EmitStoreR15(u32 r15)
{
    EmitLoadImm(R0, r15);
    EmitGuestStore(15, R0);
}

void Compiler::EmitCopyNZFromReg(int reg)
{
    EmitCmpImm(reg, 0);
    MarkHostFlagsDirty(CPSR_NZMask);
}

void Compiler::EmitCopyHostFlags(u32 mask)
{
    EmitMRS(R3);
    EmitLoadReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
    EmitLoadImm(R1, ~mask);
    EmitAndReg(R0, R0, R1);
    EmitLoadImm(R1, mask);
    EmitAndReg(R3, R3, R1);
    EmitOrrReg(R0, R0, R3);
    EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
}

void Compiler::EmitAddCycles(const FetchedInstr& instr, bool thumb, ARM* cpu)
{
    u32 r15 = instr.Addr + (thumb ? 4 : 8);
    u32 cycles = 0;

    if (cpu->Num == 0)
        cycles = (r15 & 0x2) ? 0 : instr.CodeCycles;
    else
        cycles = NDS::ARM7MemTimings[instr.CodeCycles][thumb ? 1 : 3];

    m_pendingCycles += cycles;
}

void Compiler::EmitAddCyclesConst(u32 cycles)
{
    m_pendingCycles += cycles;
}

void Compiler::EmitJumpToConst(ARM* cpu, u32 target, bool sourceThumb,
                               bool thumbTarget, bool tailReturn)
{
    FlushRegAllocForHelper();
    u32 newPC = 0;
    u32 cycles = 0;

    if (cpu->Num == 0)
    {
        u32 regionCodeCycles = 0;
        cycles = ARM9JumpCycles((ARMv5*)cpu, target, thumbTarget,
                                newPC, regionCodeCycles);
        EmitLoadImm(R0, regionCodeCycles);
        EmitStoreReg(R0, R4, CpuMemberOffset<ARMv5>(&ARMv5::RegionCodeCycles));
    }
    else
    {
        u32 codeRegion = 0;
        u32 codeCycles = 0;
        cycles = ARM7JumpCycles(target, thumbTarget, newPC,
                                codeRegion, codeCycles);
        EmitLoadImm(R0, codeRegion);
        EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::CodeRegion));
        EmitLoadImm(R0, codeCycles);
        EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::CodeCycles));
    }

    if (sourceThumb != thumbTarget)
    {
        EmitLoadReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
        if (thumbTarget)
        {
            EmitLoadImm(R1, 0x20);
            EmitOrrReg(R0, R0, R1);
        }
        else
        {
            EmitLoadImm(R1, ~0x20u);
            EmitAndReg(R0, R0, R1);
        }
        EmitStoreReg(R0, R4, ArmMemberOffset(&ARM::CPSR));
    }

    EmitLoadImm(R0, newPC);
    EmitGuestStore(15, R0);
    EmitAddCyclesConst(cycles);

    g_M18FastBranchEmitted++;
    if (g_M18FastBranchEmitted == 1 || (g_M18FastBranchEmitted & 0xFFF) == 0)
        A32JIT_LOGI("melonDS A32 JIT M18 fast branch ACTIVE: emitted=%u "
                    "(latest target=0x%08X pc=0x%08X cyc=%u %s)",
                    g_M18FastBranchEmitted, target, newPC, cycles,
                    thumbTarget ? "Thumb" : "ARM");

    if (tailReturn)
        EmitReturn();
}

void Compiler::EmitIncrementCounter(u32* counter)
{
    EmitLoadImm(R12, (u32)(uintptr_t)counter);
    EmitLoadReg(R3, R12, 0);
    EmitLoadImm(R1, 1);
    EmitAddReg(R3, R3, R1, false);
    EmitStoreReg(R3, R12, 0);
}

u32 Compiler::CodeOffset() const
{
    return (u32)(CodePtr - CodeStart);
}

void Compiler::FlushIcache(u8* start, u8* end)
{
    __builtin___clear_cache((char*)start, (char*)end);
}

void Compiler::EmitSelfLinkTail(ARM* cpu, u8* blockStart, u32 branchCycles)
{
    FlushRegAllocForHelper();
    // M18 self-link tail (ARM9, ARM/Thumb guest self-loops).
    // Replaces EmitHelperCallWithDesc+ARM_Ret for an unconditional B targeting
    // the current block's first instruction.  Saves the ARM_Dispatch/ARM_Ret
    // round-trip (~20-25 host cycles) on every tight-loop iteration.
    //
    // r4 = ARM* cpu throughout (callee-saved in ARM_Dispatch frame).
    // Does NOT call RunArmBranch; accounts for pipeline-refill cycles inline.
    //
    // Fixed 104-byte layout (26 words):
    //  [+0]   LDR r0,[r4,#StopExec]     check cpu->StopExecution
    //  [+4]   CMP r0,#0
    //  [+8]   BNE .exit               → [+84]
    //  [+12]  LDR r0,[r4,#Cycles]       r0 = cpu->Cycles
    //  [+16]  ADD r0,r0,#branchCycles   r0 += 2×CodeCycles (pipeline refill)
    //  [+20]  MOV r1,#0
    //  [+24]  STR r1,[r4,#Cycles]       cpu->Cycles = 0
    //  [+28]  LDR r1,[pc,#60]         → [+96] &NDS::ARM9Timestamp
    //  [+32]  LDR r2,[r1,#0]            r2 = ts_lo
    //  [+36]  LDR r3,[r1,#4]            r3 = ts_hi
    //  [+40]  ADDS r2,r2,r0             ts_lo += cycles
    //  [+44]  ADC  r3,r3,#0             ts_hi += carry
    //  [+48]  STR r2,[r1,#0]
    //  [+52]  STR r3,[r1,#4]
    //  [+56]  LDR r1,[pc,#36]         → [+100] &NDS::ARM9Target
    //  [+60]  LDR r12,[r1,#0]           r12 = tgt_lo
    //  [+64]  LDR r1,[r1,#4]            r1  = tgt_hi
    //  [+68]  CMP r3,r1                 compare ts_hi vs tgt_hi
    //  [+72]  CMPEQ r2,r12              compare ts_lo (if hi equal)
    //  [+76]  BHS .exit               → [+84]
    //  [+80]  B blockStart              direct loop-back
    //  [+84]  LDR ip,[pc]             → [+92] &ARM_Ret
    //  [+88]  BX ip
    //  [+92]  .word &ARM_Ret
    //  [+96]  .word &NDS::ARM9Timestamp
    // [+100]  .word &NDS::ARM9Target

    u8* tailStart = CodePtr;
    const u32 stopExecOff = ArmMemberOffset(&ARM::StopExecution);
    const u32 cyclesOff   = ArmMemberOffset(&ARM::Cycles);

    EmitLoadReg(R0, R4, stopExecOff);                          // [+0]
    EmitU32(0xE3500000);                                       // [+4]  CMP r0, #0
    u8* stopBranch = EmitBranchPlaceholder(Cond_NE);           // [+8]  BNE .exit
    EmitLoadReg(R0, R4, cyclesOff);                            // [+12]
    EmitU32(0xE2800000 | branchCycles);                        // [+16] ADD r0, r0, #branchCycles
    EmitU32(0xE3A01000);                                       // [+20] MOV r1, #0
    EmitStoreReg(R1, R4, cyclesOff);                           // [+24]
    EmitU32(0xE59F103C);                                       // [+28] LDR r1,[pc,#60] → [+96]
    EmitU32(LdrImm(R2, R1, 0));                                // [+32]
    EmitU32(LdrImm(R3, R1, 4));                                // [+36]
    EmitU32(0xE0B22000);                                       // [+40] ADDS r2,r2,r0
    EmitU32(0xE2A33000);                                       // [+44] ADC  r3,r3,#0
    EmitU32(StrImm(R2, R1, 0));                                // [+48]
    EmitU32(StrImm(R3, R1, 4));                                // [+52]
    EmitU32(0xE59F1024);                                       // [+56] LDR r1,[pc,#36] → [+100]
    EmitU32(LdrImm(R12, R1, 0));                               // [+60]
    EmitU32(LdrImm(R1, R1, 4));                                // [+64]
    EmitCmpReg(R3, R1);                                        // [+68] CMP r3, r1
    EmitU32(0x0152000C);                                       // [+72] CMPEQ r2, r12
    u8* deadlineBranch = EmitBranchPlaceholder(Cond_HS);       // [+76] BHS .exit
    // [+80] B blockStart — direct backward branch into current block
    {
        const s32 loopDelta = (s32)(blockStart - (CodePtr + 8));
        assert((loopDelta & 3) == 0);
        const s32 loopImm = loopDelta >> 2;
        assert(loopImm >= -(1 << 23) && loopImm < (1 << 23));
        EmitU32(0xEA000000 | ((u32)loopImm & 0x00FFFFFF));
    }
    // .exit at [+84]
    u8* exitPoint = CodePtr;
    PatchBranch(stopBranch, exitPoint);
    PatchBranch(deadlineBranch, exitPoint);
    EmitU32(0xE59FC000);                                       // [+84] LDR ip,[pc] → [+92]
    EmitU32(0xE12FFF1C);                                       // [+88] BX ip
    EmitU32((u32)(uintptr_t)&ARM_Ret);                        // [+92]
    EmitU32((u32)(uintptr_t)&NDS::ARM9Timestamp);             // [+96]
    EmitU32((u32)(uintptr_t)&NDS::ARM9Target);                // [+100]

    assert(CodePtr == tailStart + 104);
}

}
