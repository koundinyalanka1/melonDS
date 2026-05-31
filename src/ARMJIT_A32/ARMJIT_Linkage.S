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

.syntax unified
.text
.arm

.p2align 4,,15
.global ARM_Dispatch
.type ARM_Dispatch, %function
ARM_Dispatch:
    push {r4-r11, r12, lr}
    mov r4, r0
    bx r1

.p2align 4,,15
.global ARM_Ret
.type ARM_Ret, %function
ARM_Ret:
    pop {r4-r11, r12, lr}
    bx lr

.p2align 4,,15
.global ARM_RestoreContext
.type ARM_RestoreContext, %function
ARM_RestoreContext:
    mov sp, r0
    pop {r0-r12, lr}
    pop {pc}
