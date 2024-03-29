;*****************************************************************************
;* x86-optimized Float DSP functions
;******************************************************************************
;*****************************************************************************
;* Author: Peter Barfuss <pbarfuss@uwaterloo.ca>
;*
;* Permission to use, copy, modify, and/or distribute this software for any
;* purpose with or without fee is hereby granted, provided that the above
;* copyright notice and this permission notice appear in all copies.
;*
;* THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
;* WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
;* MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
;* ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
;* WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
;* ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
;* OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
;*****************************************************************************

%ifdef ARCH_X86_64
    %ifidn __OUTPUT_FORMAT__,win64
        %define WIN64
    %else
        %define UNIX64
    %endif
    %define gprsize 8
%else
    %define gprsize 4
; x86_32 doesn't require PIC.
; Some distros prefer shared objects to be PIC, but nothing breaks if
; the code contains a few textrels, so we'll skip that complexity.
    %undef PIC
%endif

%ifidn __OUTPUT_FORMAT__,elf
%define BINFMT_IS_ELF
%elifidn __OUTPUT_FORMAT__,elf32
%define BINFMT_IS_ELF
%elifidn __OUTPUT_FORMAT__,elf64
%define BINFMT_IS_ELF
%endif

%ifidn __OUTPUT_FORMAT__,win32
    %define mangle(x) _ %+ x
%else
    %define mangle(x) x
%endif

%ifdef WIN64
    %define PIC
%endif
%ifdef PIC
    default rel
%endif

%ifdef WIN64 ; Windows x64 ;=================================================

%define r0 rcx
%define r1 rdx
%define r2 r8
%define r3 r9
%define r4 rdi
%define r5 rsi

%macro LOAD_IF_USED 2 ; reg_id, number_of_args
    %if %1 < %2
        mov r%1, [rbp + stack_offset + %1*8]
    %endif
%endmacro

%macro PROLOGUE 2-4+ 0 ; #args, #regs, #xmm_regs, arg_names...
    %assign regs_used %2
    %if regs_used > 4
        push r4
        push r5
        %assign stack_offset stack_offset+16
    %endif
    LOAD_IF_USED 4, %1
    LOAD_IF_USED 5, %1
%endmacro

%macro RET 0
    %if regs_used > 4
        pop r5
        pop r4
    %endif
    pop rbp
    ret
%endmacro

%elifdef ARCH_X86_64 ; *nix x64 ;=============================================

%define r0 rdi
%define r1 rsi
%define r2 rdx
%define r3 rcx
%define r4 r8
%define r5 r9
%define r6 r10

%macro PROLOGUE 2-4+ ; #args, #regs, #xmm_regs, arg_names...
%endmacro

%macro RET 0
    pop rbp
    ret
%endmacro

%else ; X86_32 ;==============================================================

%define r0 ecx
%define r1 edx
%define r2 ebx
%define r3 esi
%define r4 edi
%define r5 eax
%define rbp ebp
%define rsp esp

%macro PUSH_IF_USED 1 ; reg_id
    %if %1 < regs_used
        push r%1
        %assign stack_offset stack_offset+4
    %endif
%endmacro

%macro POP_IF_USED 1 ; reg_id
    %if %1 < regs_used
        pop r%1
    %endif
%endmacro

%macro LOAD_IF_USED 2 ; reg_id, number_of_args
    %if %1 < %2
        mov r%1, [esp + stack_offset + 8 + %1*4]
    %endif
%endmacro

%macro PROLOGUE 2-4+ ; #args, #regs, #xmm_regs, arg_names...
    %assign regs_used %2
    PUSH_IF_USED 2
    PUSH_IF_USED 3
    PUSH_IF_USED 4
    PUSH_IF_USED 5
    LOAD_IF_USED 0, %1
    LOAD_IF_USED 1, %1
    LOAD_IF_USED 2, %1
    LOAD_IF_USED 3, %1
    LOAD_IF_USED 4, %1
    LOAD_IF_USED 5, %1
%endmacro

%macro RET 0
    POP_IF_USED 5
    POP_IF_USED 4
    POP_IF_USED 3
    POP_IF_USED 2
	pop rbp
    ret
%endmacro

%endif ;======================================================================

;=============================================================================
; arch-independent part
;=============================================================================

; Symbol prefix for C linkage
%macro cglobal 1-2+ "" ; name, [PROLOGUE args]
    cglobal_internal %1 %+ SUFFIX, %2
%endmacro
%macro cglobal_internal 1-2+
    %xdefine %1 mangle(%1)
    global %1
%ifdef BINFMT_IS_ELF
    [type %1 function]
%endif
    align 16 
    %1:
%ifdef WIN64
    ;%assign stack_offset 8
    %assign stack_offset 0
%else
    %assign stack_offset 0
%endif
	push rbp
	mov rbp, rsp
    PROLOGUE %2
%endmacro
%macro cendfunc 1
    cendfunc_internal %1 %+ SUFFIX
%endmacro
%macro cendfunc_internal 1
%ifdef BINFMT_IS_ELF
%1 %+ size EQU $-%1
[size %1 %1 %+ size]
%endif
%endmacro

; This is needed for ELF, otherwise the GNU linker assumes the stack is
; executable by default.
%ifdef BINFMT_IS_ELF
SECTION .note.GNU-stack noalloc noexec nowrite progbits
%endif

%macro CAT_XDEFINE 3
    %xdefine %1%2 %3
%endmacro

%macro CAT_UNDEF 2
    %undef %1%2
%endmacro

; base-4 constants for shuffles
%assign i 0
%rep 256
    %assign j ((i>>6)&3)*1000 + ((i>>4)&3)*100 + ((i>>2)&3)*10 + (i&3)
    %if j < 10
        CAT_XDEFINE q000, j, i
    %elif j < 100
        CAT_XDEFINE q00, j, i
    %elif j < 1000
        CAT_XDEFINE q0, j, i
    %else
        CAT_XDEFINE q, j, i
    %endif
%assign i i+1
%endrep
%undef i
%undef j

section .rodata align=16

pdw_80000000:    times 4 dd 0x80000000
pdw_7fffffff:    times 4 dd 0x7fffffff
ps_p1m1p1m1: dd 0x0, 0x80000000, 0x0, 0x80000000
ps_p1p1p1m1: dd 0x0, 0x0, 0x0, 0x80000000

align 16
ps_25: times 4 dd 0.25
ps_5: times 4 dd 0.5

section .text align=16

%xdefine SUFFIX _sse

;-----------------------------------------------------------------------------
; void vector_fmul(float *dst, const float *src0, const float *src1, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul, 4,4,2
    lea     r3, [r3*4 - 2*0x10]
.loop:
    movaps  xmm0, [r1 + r3]
    movaps  xmm1, [r1 + r3 + 0x10]
    mulps   xmm0, [r2 + r3]
    mulps   xmm1, [r2 + r3 + 0x10]
    movaps  [r0 + r3], xmm0
    movaps  [r0 + r3 + 0x10], xmm1

    sub     r3, 0x20
    jge     .loop
    RET
cendfunc vector_fmul

;------------------------------------------------------------------------------
; void vector_fmac_scalar(float *dst, const float *src, float mul, uint32_t len)
;------------------------------------------------------------------------------

%ifdef UNIX64
cglobal vector_fmac_scalar, 3,3,3
%define len r2
%else
cglobal vector_fmac_scalar, 4,4,3
%define len r3 
%endif
%ifdef ARCH_X86_64
%ifdef WIN64
    movaps xmm0, xmm2
%endif
%else
    movss  xmm0, [esp + stack_offset + 0x10] 
%endif
    shufps xmm0, xmm0, 0
    lea     len, [len*4-2*0x10]
.loop:
    movaps   xmm1, [r1+len]
    movaps   xmm2, [r1+len+0x10]
    mulps    xmm1, xmm0
    mulps    xmm2, xmm0
    addps    xmm1, [r0+len]
    addps    xmm2, [r0+len+0x10]
    movaps  [r0+len      ], xmm1
    movaps  [r0+len+0x10], xmm2
    sub     len, 0x20
    jge    .loop
    RET
%undef len
cendfunc vector_fmac_scalar

;------------------------------------------------------------------------------
; void vector_fmul_scalar(float *dst, const float *src, float mul, uint32_t len)
;------------------------------------------------------------------------------

%ifdef UNIX64
cglobal vector_fmul_scalar, 3,3,2
%define len r2
%else
cglobal vector_fmul_scalar, 4,4,3
%define len r3
%ifdef WIN64
    movaps      xmm0, xmm2
%else
    movss       xmm0, [esp + stack_offset + 0x10] 
%endif
%endif
    shufps      xmm0, xmm0, 0
    lea          len, [len*4-0x10]
.loop:
    movaps      xmm1, [r1+len]
    mulps       xmm1, xmm0
    movaps  [r0+len], xmm1
    sub     len, 0x10
    jge     .loop
    RET
%undef len
cendfunc vector_fmul_scalar

;-----------------------------------------------------------------------------
; void vector_fmul_copy(float *dst, const float *src, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul_copy, 3,4,2
    lea       r2, [r2*4 - 0x40]
.loop:
    movaps  xmm0, [r1 + r2]
    movaps  xmm1, [r1 + r2 + 0x10]
    movaps  xmm2, [r1 + r2 + 0x20]
    movaps  xmm3, [r1 + r2 + 0x30]
    movaps    [r0 + r2], xmm0
    movaps    [r0 + r2 + 0x10], xmm1
    movaps    [r0 + r2 + 0x20], xmm2
    movaps    [r0 + r2 + 0x30], xmm3

    sub       r2, 0x40
    jge     .loop
    RET
cendfunc vector_fmul_copy

;-----------------------------------------------------------------------------
; void vector_fmul_add(float *dst, const float *src0, const float *src1,
;                      const float *src2, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul_add, 5,6,2
    shl      r4, 2
    xor      r5, r5
.loop:
    movaps  xmm0, [r1 + r5]
    movaps  xmm1, [r1 + r5 + 0x10]
    movaps  xmm2, [r1 + r5 + 0x20]
    movaps  xmm3, [r1 + r5 + 0x30]
    mulps   xmm0, [r2 + r5]
    mulps   xmm1, [r2 + r5 + 0x10]
    mulps   xmm2, [r2 + r5 + 0x20]
    mulps   xmm3, [r2 + r5 + 0x30]
    addps   xmm0, [r3 + r5]
    addps   xmm1, [r3 + r5 + 0x10]
    addps   xmm2, [r3 + r5 + 0x20]
    addps   xmm3, [r3 + r5 + 0x30]
    movaps    [r0 + r5], xmm0
    movaps    [r0 + r5 + 0x10], xmm1
    movaps    [r0 + r5 + 0x20], xmm2
    movaps    [r0 + r5 + 0x30], xmm3

    add      r5, 0x40
    cmp      r5, r4 
    jl .loop
    RET
cendfunc vector_fmul_add

;-----------------------------------------------------------------------------
; void vector_fmul_reverse(float *dst, const float *src0, const float *src1, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul_reverse, 4,4,2
    lea       r3, [r3*4 - 2*0x10]
.loop:
    movaps  xmm0, [r2]
    movaps  xmm1, [r2 + 0x10]
    shufps  xmm0, xmm0, 0x1b
    shufps  xmm1, xmm1, 0x1b
    mulps   xmm0, [r1 + r3 + 0x10]
    mulps   xmm1, [r1 + r3]
    movaps  [r0 + r3 + 0x10], xmm0
    movaps  [r0 + r3], xmm1
    add     r2, 2*0x10
    sub     r3,  2*0x10
    jge     .loop
    RET
cendfunc vector_fmul_reverse

;-----------------------------------------------------------------------------
; void vector_fmul_window(float *dst, const float *src0, const float *src1,
;                         const float *win, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul_window, 5,6,6
%ifdef WIN64
    movsxd     rdi, edi
%endif
    shl       r4, 0x2
    mov       r5, r4
    neg       r5 
    add       r0, r4
    add       r1, r4
    add       r3, r4
    sub       r4, 0x10
.loop:
    movaps xmm1, [r3+r4]
    movaps xmm0, [r3+r5]
    movaps xmm5, [r2+r4]
    movaps xmm4, [r1+r5]
    shufps xmm1, xmm1, 0x1b
    shufps xmm5, xmm5, 0x1b
    movaps xmm2, xmm0
    movaps xmm3, xmm1
    mulps  xmm2, xmm4
    mulps  xmm3, xmm5
    mulps  xmm1, xmm4
    mulps  xmm0, xmm5
    addps  xmm2, xmm3
    subps  xmm1, xmm0
    shufps xmm2, xmm2, 0x1b
    movaps [r0+r5], xmm1
    movaps [r0+r4], xmm2
    sub r4, 0x10
    add r5, 0x10
    jl .loop
    RET
cendfunc vector_fmul_window

;-----------------------------------------------------------------------------
; void butterflies_float(float *src0, float *src1, uint32_t len);
;-----------------------------------------------------------------------------
cglobal butterflies_float, 3,3,3
    test        r2, r2 
    jz .end
    shl         r2, 2
    lea         r0, [r0 + r2]
    lea         r1, [r1 + r2]
    neg         r2 
.loop:
    movaps    xmm0, [r0 + r2]
    movaps    xmm1, [r1 + r2]
    movaps    xmm2, xmm0
    subps     xmm2, xmm1
    addps     xmm0, xmm1
    movaps      [r1 + r2], xmm2
    movaps      [r0 + r2], xmm0
    add         r2, 0x10
    jl .loop
.end:
    RET
cendfunc butterflies_float

;-----------------------------------------------------------------------------
; float scalarproduct_float(float *v1, float *v2, uint32_t len)
;-----------------------------------------------------------------------------
cglobal scalarproduct_float, 3,3,3
    shl        r2, 2
    xor        r3, r3
    xorps    xmm0, xmm0
.loop:
    movups   xmm1, [r0+r3]
    movups   xmm2, [r1+r3]
    mulps    xmm1, xmm2
    addps    xmm0, xmm1
    add        r3, 0x10
    cmp        r3, r2 
    jl .loop
    movaps   xmm1, xmm0
    shufps   xmm1, xmm0, 0x1b
    addps    xmm1, xmm0
    movhlps  xmm0, xmm1
    addss    xmm0, xmm1
%ifndef ARCH_X86_64
    movss     [esp + stack_offset + 8],  xmm0
    fld dword [esp + stack_offset + 8] 
%endif
    RET
cendfunc scalarproduct_float

;-----------------------------------------------------------------------------
; float calc_power_spectrum(float *psd, FFTComplex *vin, uint32_t len)
;-----------------------------------------------------------------------------
cglobal calc_power_spectrum, 3,3,3
    shl      r2, 1
    xor      r3, r3
.loop:
    movups   xmm2, [r1+2*r3] ; [r0 i0 r1 i1]
    movups   xmm5, [r1+2*r3+0x10] ; [r2 i2 r3 i3]
    movaps   xmm3, xmm2
    shufps   xmm2, xmm5, q2020 ; {r0,r1,r2,r3}
    shufps   xmm3, xmm5, q3131 ; {i0,i1,i2,i3}
    mulps    xmm2, xmm2        ; {r0^2, r1^2, r2^r, r3^2}
    mulps    xmm3, xmm3        ; {i0^2, i1^2, i2^2, i3^2}
    addps    xmm3, xmm2        ; {r0^2 + i0^2, r1^2 + i1^2, r2^2 + i2^2, r3^2 + i3^2}
    movups   [r0+r3], xmm3 
    add      r3, 0x10
    cmp      r3, r2
    jl .loop

    RET
cendfunc calc_power_spectrum

;-----------------------------------------------------
;void vector_clipf(float *dst, const float *src, float min, float max, uint32_t len)
;-----------------------------------------------------
%ifdef WIN64
cglobal vector_clipf, 5,6,6
    SWAP 0, 2
    SWAP 1, 3
%else
cglobal vector_clipf, 3,4,6
%ifdef ARCH_X86_32
    movss   xmm0, [esp + stack_offset + 0x10] 
    movss   xmm1, [esp + stack_offset + 0x14] 
    movss     r2, [esp + stack_offset + 0x18] 
%endif
    shl r2, 2
    xor r3, r3
%endif

    shufps xmm0, xmm0, 0x0
    shufps xmm1, xmm1, 0x0
.loop:
    movaps    xmm2,  [r1+r2     ]
    movaps    xmm3,  [r1+r2+0x10]
    movaps    xmm4,  [r1+r2+0x20]
    movaps    xmm5,  [r1+r2+0x30]
    maxps     xmm2, xmm0
    maxps     xmm3, xmm0
    maxps     xmm4, xmm0
    maxps     xmm5, xmm0
    minps     xmm2, xmm1
    minps     xmm3, xmm1
    minps     xmm4, xmm1
    minps     xmm5, xmm1
    movaps    [r0+r2     ], xmm2
    movaps    [r0+r2+0x10], xmm3
    movaps    [r0+r2+0x20], xmm4
    movaps    [r0+r2+0x30], xmm5
    add        r3, 0x40
    cmp        r3, r2 
    jl .loop
    RET
cendfunc vector_clipf

;-----------------------------------------------------------------------------
; void vector_fmul_cf(FFTComplex *dst, const FFTComplex *src0, const float *src1, uint32_t len)
;-----------------------------------------------------------------------------
cglobal vector_fmul_cf, 4,4,2, dst, src0, src1, len
    shl      r3, 2
    xor      r4, r4
.loop:
    movaps   xmm1, [r1+2*r4] ; [r0 i0 r1 i1]
    movaps   xmm2, [r1+2*r4+0x10] ; [r2 i2 r3 i3]
    movaps   xmm0, [r2+r4]
    movaps   xmm3, xmm1
    shufps   xmm1, xmm2, q3131 ; {r0,r1,r2,r3}
    shufps   xmm3, xmm2, q2020 ; {i0,i1,i2,i3}
    mulps    xmm1, xmm0
    mulps    xmm3, xmm0
    movaps   xmm2, xmm3
    unpcklps xmm2, xmm1
    unpckhps xmm3, xmm1
    movaps   [r0+2*r4], xmm2 ; [r0 i0 r1 i1]
    movaps   [r0+2*r4+0x10], xmm3 ; [r2 i2 r3 i3]

    add      r4, 0x10
    cmp      r4, r3
    jl .loop
    RET
cendfunc vector_fmul_cf

;-----------------------------------------------------------------------------
; void sbr_sum64x5(float *z)
;-----------------------------------------------------------------------------
cglobal sbr_sum64x5, 1,2,4
    xor     r1, r1
.loop:
    movaps  xmm0, [r0+r1+   0]
    addps   xmm0, [r0+r1+ 256]
    addps   xmm0, [r0+r1+ 512]
    addps   xmm0, [r0+r1+ 768]
    addps   xmm0, [r0+r1+1024]
    movaps  [r0+r1], xmm0
    add     r1, 0x10 
    cmp     r1, 1024
    jne  .loop
    RET
cendfunc sbr_sum64x5

;-----------------------------------------------------------------------------
; void sbrenc_sum128x5(float *z)
;-----------------------------------------------------------------------------
cglobal sbrenc_sum128x5, 1,2,4
    xor     r1, r1
.loop:
    movaps  xmm0, [r0+r1+   0]
    addps   xmm0, [r0+r1+ 512]
    addps   xmm0, [r0+r1+1024]
    addps   xmm0, [r0+r1+1536]
    addps   xmm0, [r0+r1+2048]
    movaps  [r0+r1], xmm0
    add     r1, 0x10 
    cmp     r1, 2048
    jne  .loop
    RET
cendfunc sbrenc_sum128x5

;-----------------------------------------------------------------------------
; void sbr_qmf_pre_shuffle(float *z)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_pre_shuffle, 1,4,6
    mov      r3, 0x60
    xor      r1, r1
.loop:
    movups   xmm0, [r0 + r1 + 0x84]
    movups   xmm2, [r0 + r1 + 0x94]
    movups   xmm1, [r0 + r3 + 0x14]
    movups   xmm3, [r0 + r3 + 0x04]

    xorps    xmm2, [pdw_80000000]
    xorps    xmm0, [pdw_80000000]
    shufps   xmm2, xmm2, 0x1b
    shufps   xmm0, xmm0, 0x1b
    movaps   xmm5, xmm2
    unpcklps xmm2, xmm3
    unpckhps xmm5, xmm3
    movaps   xmm4, xmm0
    unpcklps xmm0, xmm1
    unpckhps xmm4, xmm1
    movaps  [r0 + 2*r3 + 0x100], xmm2
    movaps  [r0 + 2*r3 + 0x110], xmm5
    movaps  [r0 + 2*r3 + 0x120], xmm0
    movaps  [r0 + 2*r3 + 0x130], xmm4
    add       r1, 0x20
    sub       r3, 0x20
    jge      .loop
    movaps   xmm2, [r0]
    movlps  [r0 + 0x100], xmm2
    RET
cendfunc sbr_qmf_pre_shuffle

;-----------------------------------------------------------------------------
; void sbr_sum64x5(float *z)
;-----------------------------------------------------------------------------
cglobal sbr_ldqmf_pre_shuffle, 1,4,6
    xor     r1, r1
    mov     r2, 0x70
.loop:
    movaps  xmm0, [r0+r2+0x40] ; z[47-k]
    movaps  xmm1, [r0+r1+0xc0] ; z[k+48]
    movaps  xmm2, [r0+r2     ] ; z[31-k]
    movaps  xmm3, [r0+r1+   0] ; z[k]
    shufps  xmm0, xmm0, 0x1b
    shufps  xmm2, xmm2, 0x1b
    movaps  xmm4, xmm1
    movaps  xmm5, xmm3
    addps   xmm1, xmm0 ; z[47-k] + z[k+48]
    subps   xmm0, xmm4 ; z[47-k] - z[k+48]
    subps   xmm3, xmm2 ; z[31-k] - z[k   ]
    addps   xmm2, xmm5 ; z[31-k] + z[k   ]
    xorps   xmm2, [pdw_80000000]
    movaps  [r0+r1+0x100], xmm1
    movaps  [r0+r1+0x140], xmm3
    movaps  [r0+r1+0x180], xmm0
    movaps  [r0+r1+0x1c0], xmm2
    sub     r2, 0x10
    add     r1, 0x10
    cmp     r1, 0x100
    jne  .loop
    RET
cendfunc sbr_ldqmf_pre_shuffle

;-----------------------------------------------------------------------------
; float sbr_qmf_post_shuffle_sse(FFTComplex *z)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_post_shuffle, 2,3,4
    lea              r2, [r1 + (64-4)*4]
    xor r3, r3
.loop:
    movaps          xmm0, [r1+r3]
    movaps          xmm1, [r2]
    shufps          xmm1, xmm1, 0x1b ; [0 1 2 3] -> [3 2 1 0]
    movaps          xmm2, xmm0
    unpcklps        xmm2, xmm1 ; [0.0 1.3 0.1 1.2]
    unpckhps        xmm0, xmm1 ; [0.2 1.1 0.3 1.0]
    movaps     [r0 + 2*r3 +    0], xmm2
    movaps     [r0 + 2*r3 + 0x10], xmm0
    sub               r2, 0x10
    add               r3, 0x10
    cmp               r3, 256 
    jl             .loop
    RET
cendfunc sbr_qmf_post_shuffle

;-----------------------------------------------------------------------------
; float sbr_qmf_post_shuffle_avx(FFTComplex *z)
;-----------------------------------------------------------------------------
cglobal_internal sbr_qmf_post_shuffle_avx, 2,3,4
    lea              r2, [r1 + (64-8)*4]
    xor r3, r3
.loop:
    vmovups         ymm0, [r1+r3]
    vmovups         ymm1, [r2]
    vperm2f128      ymm1, ymm1, ymm1, 0x01 ; [0 1 2 3 4 5 6 7] -> [4 5 6 7 0 1 2 3]
    vshufps         ymm1, ymm1, ymm1, 0x1b ; [4 5 6 7 0 1 2 3] -> [7 6 5 4 3 2 1 0]
    vunpckhps       ymm2, ymm0, ymm1
    vunpcklps       ymm0, ymm0, ymm1
    vextractf128    [r0 + 2*r3       ], ymm0, 0
    vextractf128    [r0 + 2*r3 + 0x10], ymm2, 0
    vextractf128    [r0 + 2*r3 + 0x20], ymm0, 1
    vextractf128    [r0 + 2*r3 + 0x30], ymm2, 1
    sub               r2, 0x20
    add               r3, 0x20
    cmp               r3, 256
    jl             .loop
    vzeroupper
    RET
cendfunc_internal sbr_qmf_post_shuffle_avx

;-----------------------------------------------------------------------------
; void sbr_qmf_deint_bfly(float *v, const float *src0, const float *src1)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_deint_bfly, 3,5,8
%ifdef WIN64
 sub rsp, 2*16+16
 movaps [rsp + 0x20], xmm7
 movaps [rsp + 0x10], xmm6
%endif
    mov               r4, 64*4-32
    lea               r3, [r0 + 64*4]
.loop:
    movaps            xmm0, [r1+r4]
    movaps            xmm4, [r1+r4+0x10]
    movaps            xmm1, [r2]
    movaps            xmm5, [r2+0x10]
%ifdef ARCH_X86_64
    pshufd            xmm2, xmm0, 0x1b
    pshufd            xmm3, xmm1, 0x1b
    pshufd            xmm6, xmm4, 0x1b
    pshufd            xmm7, xmm5, 0x1b
%else
    movaps            xmm2, xmm0
    movaps            xmm3, xmm1
    shufps            xmm2, xmm2, 0x1b
    shufps            xmm3, xmm3, 0x1b
    movaps            xmm6, xmm4
    movaps            xmm7, xmm5
    shufps            xmm6, xmm6, 0x1b
    shufps            xmm7, xmm7, 0x1b
%endif
    subps             xmm5, xmm2
    addps             xmm7, xmm0
    subps             xmm1, xmm6
    addps             xmm3, xmm4
    movaps            [r3], xmm1
    movaps       [r3+0x10], xmm5
    movaps         [r0+r4], xmm7
    movaps    [r0+r4+0x10], xmm3 
    add                r2, 0x20
    add                 r3, 0x20
    sub                 r4, 0x20
    jge            .loop
%ifdef WIN64
 movaps xmm7, [rsp + 0x20]
 movaps xmm6, [rsp + 0x10]
 add rsp, 2*16+16
%endif
    RET
cendfunc sbr_qmf_deint_bfly

;-----------------------------------------------------------------------------
; void sbr_qmf_deint_neg(float *v, const float *src)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_deint_neg, 2,4,4
    mov        r2, -128
    mov        r3, 0x70
    add        r1, 0x100
    movaps   xmm3, [pdw_80000000]
.loop:
    movaps   xmm0, [r1 + 2*r2]
    movaps   xmm1, [r1 + 2*r2 + 0x10]
    movaps   xmm2, xmm0
    shufps   xmm2, xmm1, q2020
    shufps   xmm1, xmm0, q1313
    xorps    xmm2, xmm3
    movaps   [r0 + r3], xmm1
    movaps   [r0 + r2 + 0x100], xmm2
    sub        r3, 0x10 
    add        r2, 0x10 
    jl      .loop
    RET
cendfunc sbr_qmf_deint_neg

;-----------------------------------------------------------------------------
; void sbrenc_qmf_deint_bfly(float *v, const float *src0, const float *src1)
;-----------------------------------------------------------------------------
cglobal sbrenc_qmf_deint_bfly, 2,5,8
%ifdef WIN64
 sub rsp, 2*16+16
 movaps [rsp + 0x20], xmm7
 movaps [rsp + 0x10], xmm6
%endif
    mov                 r4, 64*4-32
    lea                 r3, [r0 + 64*4]
.loop:
    movaps            xmm0, [r1+r4]
    movaps            xmm4, [r1+r4+0x10]
    movaps            xmm1, [r2]
    movaps            xmm5, [r2+0x10]
%ifdef ARCH_X86_64
    pshufd            xmm2, xmm0, 0x1b
    pshufd            xmm3, xmm1, 0x1b
    pshufd            xmm6, xmm4, 0x1b
    pshufd            xmm7, xmm5, 0x1b
%else
    movaps            xmm2, xmm0
    movaps            xmm3, xmm1
    shufps            xmm2, xmm2, 0x1b
    shufps            xmm3, xmm3, 0x1b
    movaps            xmm6, xmm4
    movaps            xmm7, xmm5
    shufps            xmm6, xmm6, 0x1b
    shufps            xmm7, xmm7, 0x1b
%endif
    addps             xmm5, xmm2
    subps             xmm0, xmm7
    addps             xmm1, xmm6
    subps             xmm4, xmm3
    movaps            [r3], xmm1
    movaps       [r3+0x10], xmm5
    movaps         [r0+r4], xmm0
    movaps    [r0+r4+0x10], xmm4 
    add                 r2, 0x20
    add                 r3, 0x20
    sub                 r4, 0x20
    jge            .loop
%ifdef WIN64
 movaps xmm7, [rsp + 0x20]
 movaps xmm6, [rsp + 0x10]
 add rsp, 2*16+16
%endif
    RET
cendfunc sbrenc_qmf_deint_bfly

%macro ACSTEP 3 ;xmm0, xmm1                                                       
    movaps  xmm3, %1
    movaps  xmm4, %1
    mulps   xmm3, %2
    mulps   xmm4, %3
    mulps   %1, %1
    addps   xmm6, xmm3 ; r01r += x[i].re * x[i+1].re, x[i].im * x[i+1].im; r01i += x[i].re * x[i+1].im, x[i].im * x[i+1].re;
    addps   xmm5, xmm4 ; r02r += x[i].re * x[i+2].re, x[i].im * x[i+2].im; r02i += x[i].re * x[i+2].im, x[i].im * x[i+2].re;
    addps   xmm7, %1   ; r11r += x[i].re * x[i].re,   x[i].im * x[i].im;
    movlhps %2, %2
%endmacro

cglobal sbr_autocorrelate, 3,4,8
    shl r2, 3
    movlps  xmm7, [r0+8 ]
    movlhps xmm7, xmm7
    movaps  xmm6, xmm7
    movaps  xmm5, xmm7

    add r0, 16
    movlps  xmm1, [r0  ]
    shufps  xmm1, xmm1, q0110
    movlps  xmm2, [r0+8]
    shufps  xmm2, xmm2, q0110
    mulps   xmm6, xmm1 ; r01r = x[1].re * x[2].re, x[1].im * x[2].im; r01i = x[1].re * x[2].im, x[1].im * x[2].re
    mulps   xmm5, xmm2 ; r02r = x[1].re * x[3].re, x[1].im * x[3].im; r02i = x[1].re * x[3].im, x[1].im * x[3].re
    mulps   xmm7, xmm7 ; r11r = x[1].re * x[1].re, x[1].im * x[1].im;
    shufps  xmm1, xmm1, q1010
    mov   r3, 2*8

align 16
.loop:
    movlps  xmm0, [r0+r3   ]
    shufps  xmm0, xmm0, q0110
    ACSTEP  xmm1, xmm2, xmm0

    movlps  xmm1, [r0+r3+8]
    shufps  xmm1, xmm1, q0110
    ACSTEP  xmm2, xmm0, xmm1

    movlps  xmm2, [r0+r3+16]
    shufps  xmm2, xmm2, q0110
    ACSTEP  xmm0, xmm1, xmm2

    add     r3, 24
    cmp     r3, r2
    jl .loop

    xorps   xmm5, [ps_p1p1p1m1]
    xorps   xmm6, [ps_p1p1p1m1]

    movaps  xmm2, xmm5
    movaps  xmm0, xmm6
    shufps  xmm2, xmm2, q0301
    shufps  xmm0, xmm0, q0301
    addps   xmm5, xmm2
    addps   xmm6, xmm0

    shufps  xmm6, xmm5, q2020
    xorps   xmm6, [ps_p1m1p1m1]
    movaps  [r1     ], xmm6

    movss   xmm2, xmm7
    shufps  xmm7, xmm7, q0001
    addss   xmm7, xmm2

    movss   [r1+0x10], xmm7
    RET
cendfunc sbr_autocorrelate

;-----------------------------------------------------------------------------
; void sbr_hf_g_filt(FFTComplex *Y, FFTComplex *X_high[40],
;                    const float *g_filt, size_t m_max, size_t ixh)
;-----------------------------------------------------------------------------
%define STEP  40*4*2
cglobal sbr_hf_g_filt, 5, 5, 5
    lea         r1, [r1 + 8*r4] ; offset by ixh elements into X_high
    mov         r4, r3
    and         r3, 0xFE
    lea         r2, [r2 + r3*4]
    lea         r0, [r0 + r3*8]
    neg         r3
    jz          .loop1
.loop2:
    movlps      xmm0, [r2 + 4*r3]
    movlps      xmm2, [r1 + 0*STEP]
    movhps      xmm2, [r1 + 1*STEP]
    unpcklps    xmm0, xmm0
    mulps       xmm0, xmm2
    movups      [r0 + 8*r3], xmm0
    add         r1, 2*STEP
    add         r3, 2 
    jnz         .loop2
    and         r4, 1 ; number of single element loops
    jz          .end
.loop1: 
    ; element 0 and 1 can be computed at the same time
    movss       xmm0, [r2]
    movlps      xmm2, [r1]
    unpcklps    xmm0, xmm0
    mulps       xmm0, xmm2
    movlps    [r0], xmm0
.end:
    RET
cendfunc sbr_hf_g_filt

;-----------------------------------------------------------------------------
; void sbr_hf_gen(FFTComplex *X_high, FFTComplex *X_low,
;                 float alpha[4], unsigned int start, unsigned int end)
;-----------------------------------------------------------------------------
cglobal sbr_hf_gen, 5,5,8
%ifdef WIN64
 sub rsp, 2*16+16
 movaps [rsp + 0x20], xmm7
 movaps [rsp + 0x10], xmm6
%endif
    movaps     xmm2, [r2] ; (a0[0] a0[1])*bw    = (a[2] a[3])*bw    = (a2 a3)
    movhlps    xmm1, xmm2     ; (a1[0] a1[1])*bw*bw = (a[0] a[1])*bw*bw = (a0 a1)
    movaps     xmm3, xmm1 ; (a2 a3)
    movaps     xmm4, xmm2 ; (a0 a1)
    shufps     xmm3, xmm3, 0x55 ; (-a3 a3 -a3 a3)
    shufps     xmm4, xmm4, 0x55 ; (-a1 a1 -a1 a1)
    shufps     xmm1, xmm1, 0x00 ; (a2 a2 a2 a2)
    shufps     xmm2, xmm2, 0x00 ; (a0 a0 a0 a0)
    xorps      xmm3, [ps_p1m1p1m1]
    xorps      xmm4, [ps_p1m1p1m1]

    shl          r3, 3
    shl          r4, 3
    lea         r1, [r1 - 2*2*4]

    movaps      xmm0, [r1 + r3]
.loop2:
    movups      xmm7, [r1 + r3 + 8]        ; BbCc
    movaps      xmm5, xmm7
    movaps      xmm6, xmm0
    shufps      xmm0, xmm0, 0xb1                   ; aAbB
    shufps      xmm7, xmm7, 0xb1                   ; bBcC
    mulps       xmm0, xmm4
    mulps       xmm6, xmm2
    mulps       xmm7, xmm3
    mulps       xmm5, xmm1
    addps       xmm7, xmm0
    addps       xmm7, xmm5
    addps       xmm7, xmm6
    movaps      xmm0, [r1 + r3 +16]        ; CcDd
    addps       xmm7, xmm0
    movaps  [r0 + r3], xmm7
    add           r3, 16
    cmp           r3, r4 
    jl .loop2
%ifdef WIN64
 movaps xmm7, [rsp + 0x20]
 movaps xmm6, [rsp + 0x10]
 add rsp, 2*16+16
%endif
    RET
cendfunc sbr_hf_gen

;-----------------------------------------------------------------------------
; void aacenc_calc_expspec(float *expspec, const float *mdct_spectrum, uint32_t len)
;-----------------------------------------------------------------------------
cglobal aacenc_calc_expspec, 3,4,2
    shl       r2, 2
    xor       r3, r3
    movaps  xmm4, [pdw_7fffffff]
    movaps  xmm5, [pdw_80000000]
.loop:
    movaps  xmm0, [r1 + r3]
    andps   xmm0, xmm4    

    rsqrtps xmm2, xmm0
    mulps   xmm2, xmm0
    rsqrtps xmm2, xmm2

    movaps  xmm3, xmm2
    mulps   xmm3, xmm3 ; t = y*y
    mulps   xmm3, xmm3 ; t = y*y*y*y
    mulps   xmm3, xmm2 ; t = y*y*y*y*y
    mulps   xmm3, xmm0 ; t = x*y*y*y*y*y
    xorps   xmm3, xmm5 ; t = -x*y*y*y*y*y
    addps   xmm3, xmm2 ; t = y - (x*y*y*y*y*y)
    mulps   xmm3, [ps_25] ; t = 0.25f*(y - (x*y*y*y*y*y))
    addps   xmm2, xmm3 ; y = y + 0.25f*(y - (x*y*y*y*y*y))
    mulps   xmm0, xmm2 ; y = y * x

    movaps  [r0 + r3], xmm0 
    add      r3, 0x10
    cmp      r3, r2 
    jl .loop
    RET
cendfunc aacenc_calc_expspec 

;-----------------------------------------------------------------------------
; void aac_update_ltp(float *dst, const float *buf, const float *win, uint32_t len)
;-----------------------------------------------------------------------------
cglobal aac_update_ltp, 4,4,2
    lea       r3, [r3*8 - 2*0x10]
    xor       r4, r4
.loop:
    movaps  xmm0, [r1 + r3 + 0x10]
    movaps  xmm1, [r1 + r3]
    movaps  xmm2, [r2 + r4]
    movaps  xmm3, [r2 + r4 + 0x10]
    movaps  xmm4, [r2 + r3 + 0x10]
    movaps  xmm5, [r2 + r3]
    shufps  xmm2, xmm2, 0x1b
    shufps  xmm3, xmm3, 0x1b
    mulps   xmm2, xmm0
    mulps   xmm3, xmm1
    mulps   xmm4, xmm0
    mulps   xmm5, xmm1                                                                                                                                  
    shufps  xmm4, xmm4, 0x1b
    shufps  xmm5, xmm5, 0x1b
    movaps  [r0 + r4 + 0x10], xmm5
    movaps  [r0 + r4], xmm4
    movaps  [r0 + r3], xmm3
    movaps  [r0 + r3 + 0x10], xmm2
    add     r4, 2*0x10
    sub     r3, 2*0x10
    jge     .loop
    RET
cendfunc aac_update_ltp

;-----------------------------------------------------------------------------
; void vorbis_inverse_coupling(float *mag, float *ang, unsigned int blocksize)
;-----------------------------------------------------------------------------
cglobal vorbis_inverse_coupling, 3, 4, 6
    movaps             xmm5, [pdw_80000000]
    xor                  r3, r3 
align 16
.loop:
    movaps             xmm0, [r0+r3*4]
    movaps             xmm1, [r1+r3*4]
    xorps              xmm2, xmm2
    xorps              xmm3, xmm3
    cmpleps            xmm2, xmm0     ; m <= 0.0
    cmpleps            xmm3, xmm1     ; a <= 0.0
    andps              xmm2, xmm5     ; keep only the sign bit
    xorps              xmm1, xmm2
    movaps             xmm4, xmm3
    andps              xmm3, xmm1
    andnps             xmm4, xmm1
    addps              xmm3, xmm0     ; a = m + ((a < 0) & (a ^ sign(m)))
    subps              xmm0, xmm4     ; m = m + ((a > 0) & (a ^ sign(m)))
    movaps        [r1+r3*4], xmm3
    movaps        [r0+r3*4], xmm0
    add                  r3, 4
    cmp                  r3, r2 
    jl .loop
    RET
cendfunc vorbis_inverse_coupling

;-----------------------------------------------------------------------------
; void sbr_qmf_synthesis_window_sse(float *out, float *v, float *sbr_qmf_window)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_synthesis_window, 3,4,2
    xor      r3, r3
.loop:
     movaps  xmm0, [r1 + 2*r3 + 12*64]
     movaps  xmm1, [r1 + 2*r3 + 28*64]
     movaps  xmm2, [r1 + 2*r3 + 44*64]
     movaps  xmm3, [r1 + 2*r3 + 60*64]
     mulps   xmm0, [r2 + 2*r3 +  4*64]
     mulps   xmm1, [r2 + 2*r3 + 12*64]
     mulps   xmm2, [r2 + 2*r3 + 20*64]
     mulps   xmm3, [r2 + 2*r3 + 28*64]
     addps   xmm2, xmm3
     addps   xmm0, xmm1
     addps   xmm0, xmm2

     movaps  [r0 + 2*r3], xmm0
     add      r3, 0x08
     cmp      r3, 0x100 
     jl .loop

    xor      r3, r3
.loop2:
     movaps   xmm0, [r0 + r3]
     movaps   xmm1, [r0 + r3 +  4*64]
     movaps   xmm4, [r1 + r3]
     mulps    xmm4, [r2 + r3]
     movaps   xmm5, [r1 + r3 + 76*64]
     mulps    xmm5, [r2 + r3 + 36*64]
     addps    xmm4, xmm5
     addps    xmm0, xmm4
     addps    xmm0, xmm1
     movaps   [r0 + r3], xmm0
     add      r3, 0x10
     cmp      r3, 0x100 
     jl .loop2
     RET
cendfunc sbr_qmf_synthesis_window

;-----------------------------------------------------------------------------
; void sbr_qmf_synthesis_window_ds_sse(float *out, float *v, float *sbr_qmf_window)
;-----------------------------------------------------------------------------
cglobal sbr_qmf_synthesis_window_ds, 3,4,2
    xor      r3, r3
.loop:
     movaps  xmm0, [r1 + 2*r3 + 12*32]
     movaps  xmm1, [r1 + 2*r3 + 28*32]
     movaps  xmm2, [r1 + 2*r3 + 44*32]
     movaps  xmm3, [r1 + 2*r3 + 60*32]
     mulps   xmm0, [r2 + 2*r3 +  4*32]
     mulps   xmm1, [r2 + 2*r3 + 12*32]
     mulps   xmm2, [r2 + 2*r3 + 20*32]
     mulps   xmm3, [r2 + 2*r3 + 28*32]
     addps   xmm2, xmm3
     addps   xmm0, xmm1
     addps   xmm0, xmm2

     movaps  [r0 + 2*r3], xmm0
     add      r3, 0x08
     cmp      r3, 0x80
     jl .loop

    xor      r3, r3
.loop2:
     movaps   xmm0, [r0 + r3]
     movaps   xmm1, [r0 + r3 +  4*32]
     movaps   xmm4, [r1 + r3]
     mulps    xmm4, [r2 + r3]
     movaps   xmm5, [r1 + r3 + 76*32]
     mulps    xmm5, [r2 + r3 + 36*32]
     addps    xmm4, xmm5
     addps    xmm0, xmm4
     addps    xmm0, xmm1
     movaps   [r0 + r3], xmm0
     add      r3, 0x10
     cmp      r3, 0x80
     jl .loop2
     RET
cendfunc sbr_qmf_synthesis_window_ds

;------------------------------------------------------------------------------
; void conv_fltp_to_flt_2ch(float *dst, float **src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_fltp_to_flt_2ch, 3,5,3
    mov        r4, [r1+gprsize]
    mov        r1, [r1]
    shl        r2, 2
    xor        r3, r3
.loop:
    movaps   xmm0, [r1+r3]
    movaps   xmm1, [r4+r3]
    movaps   xmm2, xmm0
    unpckhps xmm2, xmm1
    unpcklps xmm0, xmm1
    movaps  [r0+2*r3     ], xmm0
    movaps  [r0+2*r3+0x10], xmm2
    add        r3, 0x10
    cmp        r3, r2 
    jl .loop
    RET
cendfunc conv_fltp_to_flt_2ch

;------------------------------------------------------------------------------
; void conv_flt_to_fltp_2ch(float **dst, float *src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_flt_to_fltp_2ch, 3,5,3
    mov          r4, [r0+gprsize]
    mov          r0, [r0        ]
    shl          r2, 2
    xor          r3, r3
.loop:
    movaps     xmm0, [r1+2*r3     ]
    movaps     xmm1, [r1+2*r3+0x10]
    movaps     xmm2, xmm0
    shufps     xmm2, xmm1, 0xdd
    shufps     xmm0, xmm1, 0x88
    movaps  [r0+r3], xmm0
    movaps  [r4+r3], xmm2
    add          r3, 0x10
    cmp          r3, r2 
    jl .loop
    RET
cendfunc conv_flt_to_fltp_2ch

%undef SUFFIX
%xdefine SUFFIX _mmx2

;------------------------------------------------------------------------------
; void ff_conv_s16_to_flt(float *dst, const int16_t *src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_s16_to_flt_mmx2, 3,3,3
    shl        r2, 1
    xor        r3, r3

    movaps   xmm3, [pf_s16_inv_scale]
.loop:
    movq      mm2, [r1+r3] ; mm2 =  0,  1,  2,  3
    pxor      mm0, mm0
    pxor      mm1, mm1
    punpcklwd mm0, mm2     ; mm0 =      0,      1
    punpckhwd mm1, mm2     ; mm1 =      2,      3
    cvtpi2ps xmm0, mm0
    cvtpi2ps xmm1, mm1
    movlhps  xmm0, xmm1
    mulps    xmm0, xmm3
    movaps  [r0+2*r3], xmm0
    add        r3, 0x08
    cmp        r3, r2
    jl .loop
    emms
    RET
cendfunc conv_s16_to_flt_mmx2

;------------------------------------------------------------------------------                                                                                           
; void ff_conv_flt_to_s16(int16_t *dst, const float *src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_flt_to_s16_mmx2, 3,3,5, dst, src, len
    shl        r2, 1
    xor        r3, r3

    movaps   xmm3, [pf_s16_scale]
.loop:
    movups   xmm0, [r1+2*r3]
    mulps    xmm0, xmm3
    movhlps  xmm1, xmm0
    cvtps2pi  mm0, xmm0
    cvtps2pi  mm1, xmm1
    packssdw  mm0, mm1
    movq  [r0+r3], mm0
    add        r3, 0x08
    cmp        r3, r2 
    jl .loop
    emms
    RET
cendfunc conv_flt_to_s16_mmx2

%undef SUFFIX
%xdefine SUFFIX _sse2

;------------------------------------------------------------------------------
; void conv_s16p_to_s16_2ch(int16_t *dst, int16_t **src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_s16p_to_s16_2ch, 3,5,4
    mov        r4, [r1+gprsize]
    mov        r1, [r1]
    shl        r2, 1
    xor        r3, r3
.loop:
    movaps     xmm0, [r1+r3]   ; xmm0 = 0, 2, 4, 6, 8, 10, 12, 14
    movaps     xmm1, [r4+r3]   ; xmm1 = 1, 3, 5, 7, 9, 11, 13, 15
    movaps     xmm2, xmm0
    punpcklwd  xmm0, xmm1      ; xmm0 = 0, 1, 2, 3, 4, 5, 6, 7
    punpckhwd  xmm2, xmm1      ; xmm0 = 0, 1, 2, 3, 4, 5, 6, 7
    movaps  [r0+2*r3     ], xmm0
    movaps  [r0+2*r3+0x10], xmm2
    add        r3, 0x10
    cmp        r3, r2 
    jl .loop
    RET
cendfunc conv_s16p_to_s16_2ch

;------------------------------------------------------------------------------
; void conv_s16_to_s16p_2ch(int16_t **dst, int16_t *src, unsigned int len);
;------------------------------------------------------------------------------

cglobal conv_s16_to_s16p_2ch, 3,5,3
    mov          r4, [r0+gprsize]
    mov          r0, [r0        ]
    shl          r2, 1
    xor          r3, r3
.loop:
    movaps     xmm0, [r1+2*r3]       ; xmm0 = 0,  1,  2,  3,  4,  5,  6,  7
    movaps     xmm1, [r1+2*r3+0x10]  ; xmm1 = 8,  9, 10, 11, 12, 13, 14, 15
    pshuflw    xmm0, xmm0, 0xd8      ; xmm0 = 0,  2,  1,  3,  4,  5,  6,  7 
    pshufhw    xmm0, xmm0, 0xd8      ; xmm0 = 0,  2,  1,  3,  4,  6,  5,  7
    pshuflw    xmm1, xmm1, 0xd8      ; xmm1 = 8, 10,  9, 11, 12, 13, 14, 15 
    pshufhw    xmm1, xmm1, 0xd8      ; xmm1 = 8, 10,  9, 11, 12, 14, 13, 15
    movaps     xmm2, xmm0
    shufps     xmm0, xmm1, 0x88
    shufps     xmm2, xmm1, 0xdd
    movaps  [r0+r3], xmm0
    movaps  [r4+r3], xmm2 
    add        r3, 0x10
    cmp        r3, r2 
    jl .loop
    RET
cendfunc conv_s16_to_s16p_2ch

;----------------------------------------------------------------------------------
;void ff_flac_decorrelate_[lrm]s_sse2(int16_T *out, int32_t **in, unsigned long len, int shift);
;----------------------------------------------------------------------------------
cglobal flac_decorrelate_mono, 2, 4, 4
%ifndef ARCH_X86_64
    mov      r2, r2m
%endif
    shl      r2, 1
    mov      r1, [r1]
    xor      r4, r4

.loop:
    movups    xmm0, [r1 + 2*r4]
    movups    xmm1, [r1 + 2*r4 + 0x10]
    packssdw  xmm0, xmm0
    packssdw  xmm1, xmm1
    punpcklwd xmm0, xmm1
    movups [r0 + r4], xmm0
    add r4, 0x10
    cmp r4, r2
    jle .loop
    RET
cendfunc flac_decorrelate_mono

cglobal flac_decorrelate_indep2, 2, 4, 4
%ifndef ARCH_X86_64
    mov      r2, r2m
%endif
    shl      r2, 2
    mov      r3, [r1 + gprsize]
    mov      r1, [r1]
    xor      r4, r4

.loop:
    movups    xmm0, [r1 + r4]
    movups    xmm1, [r3 + r4]
    packssdw  xmm0, xmm0 
    packssdw  xmm1, xmm1
    punpcklwd xmm0, xmm1
    movups [r0 + r4], xmm0
    add r4, 0x10
    cmp r4, r2
    jle .loop
    RET
cendfunc flac_decorrelate_indep2

cglobal flac_decorrelate_ls, 2, 4, 4, out, in0, len, in1
%ifndef ARCH_X86_64
    mov       r2, r2m
%endif
    shl       r2, 2
    mov       r3, [r1 + gprsize]
    mov       r1, [r1 ]
    xor       r4, r4

.loop:
    movups    xmm0, [r1 + r4]
    movups    xmm1, [r3 + r4]
    movups    xmm2, xmm0
    psubd     xmm2, xmm1
    packssdw  xmm0, xmm0
    packssdw  xmm1, xmm1
    punpcklwd xmm0, xmm1
    movups [r0 + r4], xmm0
    add r4, 0x10
    cmp r4, r2 
    jle .loop
    RET
cendfunc flac_decorrelate_ls

cglobal flac_decorrelate_rs, 2, 4, 4
%ifndef ARCH_X86_64
    mov      r2, r2m
%endif
    shl      r2, 2
    mov      r3, [r1 + gprsize]
    mov      r1, [r1]
    xor      r4, r4

.loop:
    movups    xmm0, [r1 + r4]
    movups    xmm1, [r3 + r4]
    movups    xmm2, xmm0
    paddd     xmm2, xmm1
    packssdw  xmm2, xmm2
    packssdw  xmm1, xmm1
    punpcklwd xmm2, xmm1
    movups [r0 + r4], xmm2
    add r4, 0x10
    cmp r4, r2 
    jle .loop
    RET
cendfunc flac_decorrelate_rs

cglobal flac_decorrelate_ms, 2, 4, 4
%ifndef ARCH_X86_64
    mov      r2, r2m
%endif
    shl      r2, 2
    mov      r3, [r1 + gprsize]
    mov      r1, [r1]
    xor      r4, r4

.loop:
    movups     xmm0, [r1 + r4]
    movups     xmm1, [r3 + r4]
    movaps     xmm2, xmm1
    psrad      xmm2, 1
    psubd      xmm0, xmm2
    movaps     xmm2, xmm0
    paddd      xmm2, xmm1
    packssdw   xmm2, xmm2
    packssdw   xmm0, xmm0
    punpcklwd  xmm2, xmm0
    movups [r0 + r4], xmm2
    add r4, 0x10
    cmp r4, r2
    jle .loop
    RET
cendfunc flac_decorrelate_ms

cglobal tta_decorrelate_ms, 2, 4, 4
%ifndef ARCH_X86_64
    mov      r2, r2m
%endif
    shl      r2, 2
    mov      r3, [r1 + gprsize]
    mov      r1, [r1]
    xor      r4, r4

.loop:
    movups     xmm0, [r1 + r4]
    movups     xmm1, [r3 + r4]
    movaps     xmm2, xmm1
    psrld      xmm2, 31
    paddd      xmm2, xmm1
    psrad      xmm2, 1
    psubd      xmm0, xmm2
    movaps     xmm2, xmm0
    paddd      xmm2, xmm1
    packssdw   xmm2, xmm2
    packssdw   xmm0, xmm0
    punpcklwd  xmm2, xmm0
    movups [r0 + r4], xmm2
    add r4, 0x10
    cmp r4, r2
    jle .loop
    RET
cendfunc tta_decorrelate_ms

;cglobal alac_decorrelate_stereo, 2, 5, 8, buf0, len, shift, weight, buf1
;    movd    m6, shiftm
;    movd    m7, weightm
;    shufps  m7, m7, 0x00
;    shl     lend, 2
;    mov     buf1q, [buf0q + gprsize]
;    mov     buf0q, [buf0q]
;    xor     r4, r4
;
;.loop:
;    movaps     m0, [buf0q + r4]
;    movaps     m2, [buf1q + r4]
;    pmulld     m4, m2, m7
;    psrad      m4, m6
;    psubd      m0, m4
;    paddd      m2, m0
;    packssdw   m2, m2
;    packssdw   m0, m0
;    punpcklwd  m2, m0
;    movups [outq + r4], m2
;    add   r4, 0x10 
;    jl .loop
;    RET
;cendfunc alac_decorrelate_stereo

%ifdef FDSP_DLL
%ifidn __OUTPUT_FORMAT__,win64
%define NEED_DLLEXPORT_TABLE
%elifidn __OUTPUT_FORMAT__,win32
%define NEED_DLLEXPORT_TABLE
%elifidn __OUTPUT_FORMAT__,win
%define NEED_DLLEXPORT_TABLE
%endif
%endif

%ifdef NEED_DLLEXPORT_TABLE
section .drectve align=1 noexecute
db " -export:vector_fmul_sse"
db " -export:vector_fmul_scalar_sse"
db " -export:vector_fmac_scalar_sse"
db " -export:vector_fmul_copy_sse"
db " -export:vector_fmul_reverse_sse"
db " -export:vector_fmul_window_sse"
db " -export:vector_fmul_add_sse"
db " -export:scalarproduct_float_sse"
db " -export:butterflies_float_sse"
db " -export:sbr_sum64x5_sse"
db " -export:sbr_qmf_pre_shuffle_sse"
db " -export:sbr_qmf_post_shuffle_sse"
db " -export:sbr_qmf_deint_bfly_sse"
db " -export:sbr_hf_g_filt_sse"
db " -export:sbr_hf_gen_sse"
db " -export:sbr_qmf_synthesis_window_sse"
db " -export:sbrenc_sum128x5_sse"
db " -export:sbrenc_qmf_deint_bfly_sse"
db " -export:aacenc_calc_expspec_sse"
db " -export:vorbis_inverse_coupling_sse"
db " -export:conv_fltp_to_flt_2ch_sse"
db " -export:conv_flt_to_fltp_2ch_sse"
db " -export:conv_s16p_to_s16_2ch_sse"
db " -export:conv_s16_to_s16p_2ch_sse"
%endif

