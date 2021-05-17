/*
 * LOONGARCH LASX optimizations for libjpeg-turbo
 *
 * Copyright (C) 2021 Loongson Technology Corporation Limited
 * All rights reserved.
 * Contributed by Jin Bo (jinbo@loongson.cn)
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define JPEG_INTERNALS
#include "../jinclude.h"
#include "../jpeglib.h"
#include "../jsimd.h"
#include "jmacros_lasx.h"
#include <limits.h>

/* Expanded entropy encoder object for Huffman encoding.
 *
 * The savable_state subrecord contains fields that change within an MCU,
 * but must not be updated permanently until we complete the MCU.
 */

typedef struct {
  size_t put_buffer;            /* current bit-accumulation buffer */
  int put_bits;                 /* # of bits now in it */
  int last_dc_val[MAX_COMPS_IN_SCAN]; /* last DC coef for each component */
} savable_state;


/* Working state while writing an MCU.
 * This struct contains all the fields that are needed by subroutines.
 */

typedef struct {
  JOCTET *next_output_byte;     /* => next byte to write in buffer */
  size_t free_in_buffer;        /* # of byte spaces remaining in buffer */
  savable_state cur;            /* Current bit buffer & DC state */
  j_compress_ptr cinfo;         /* dump_buffer needs access to this */
} working_state;

#define EMIT_BYTE() { \
  JOCTET c; \
  put_bits -= 8; \
  c = (JOCTET)GETJOCTET(put_buffer >> put_bits); \
  *buffer++ = c; \
  if (c == 0xFF)  /* need to stuff a zero byte? */ \
    *buffer++ = 0; \
}

#define PUT_BITS(code, size) { \
  put_bits += size; \
  put_buffer = (put_buffer << size) | code; \
}

#define CHECKBUF47() { \
  if (put_bits > 47) { \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
  } \
}

#define CHECKBUF31() { \
  if (put_bits > 31) { \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
    EMIT_BYTE() \
  } \
}

#define EMIT_BITS(code, size) { \
  CHECKBUF47() \
  PUT_BITS(code, size) \
}

#define EMIT_CODE(code, size) { \
  temp2 &= (((int) 1)<<nbits) - 1; \
  CHECKBUF31() \
  PUT_BITS(code, size) \
  PUT_BITS(temp2, nbits) \
}

#define GET_VAL_2(_idx, _out1, _out2) \
{ \
  int _tmp_idx; \
  v16u16 _t0 = (v16u16)abs0; \
  v16u16 _t1 = (v16u16)abs1; \
  v16u16 _t2 = (v16u16)abs2; \
  v16u16 _t3 = (v16u16)abs3; \
  v16i16 _t4 = (v16i16)tmp4; \
  v16i16 _t5 = (v16i16)tmp5; \
  v16i16 _t6 = (v16i16)tmp6; \
  v16i16 _t7 = (v16i16)tmp7; \
  \
  if (_idx < 16) { \
    _out1 = _t0[_idx];\
    _out2 = _t4[_idx];\
  } else if (_idx < 32) { \
    _tmp_idx = _idx % 16; \
    _out1 = _t1[_tmp_idx]; \
    _out2 = _t5[_tmp_idx]; \
  } else if (_idx < 48) { \
    _tmp_idx = _idx % 32; \
    _out1 = _t2[_tmp_idx]; \
    _out2 = _t6[_tmp_idx]; \
  } else { \
    _tmp_idx = _idx % 48; \
    _out1 = _t3[_tmp_idx]; \
    _out2 = _t7[_tmp_idx]; \
  } \
}

#define BUILTIN_CTZ_D(_src, _dst) { \
  __asm__ volatile ( \
    "ctz.d %0, %1"   \
    : "=r"(_dst)     \
    : "r"(_src)      \
    : "memory"       \
 );                  \
}

#define BUILTIN_CLZ_W(_src, _dst) { \
  __asm__ volatile ( \
    "clz.w %0, %1"   \
    : "=r"(_dst)     \
    : "r"(_src)      \
    : "memory"       \
 );                  \
 _dst = 32 - _dst;   \
}

GLOBAL(JOCTET*)
jsimd_huff_encode_one_block_lasx(void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val,
                                 c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
  int temp, temp2, temp3;
  int nbits;
  int r, k, code, size;
  size_t put_buffer;
  int put_bits;
  uint64_t t0, t1, t2, t3;
  int code_0xf0 = actbl->ehufco[0xf0];
  int size_0xf0 = actbl->ehufsi[0xf0];
  working_state *state_ptr;

  __m256i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m256i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m256i abs0, abs1, abs2, abs3;
  __m256i zero = {0};

  state_ptr = (working_state*)state;

  put_buffer = state_ptr->cur.put_buffer;
  put_bits = state_ptr->cur.put_bits;

  /* Encode the DC coefficient difference per section F.1.2.1 */
  temp = temp2 = block[0] - last_dc_val;

  /* This is a well-known technique for obtaining the absolute value without a
  * branch.  It is derived from an assembly language technique presented in
  * "How to Optimize for the Pentium Processors", Copyright (c) 1996, 1997 by
  * Agner Fog.
  */
  temp3 = temp >> (CHAR_BIT * sizeof(int) - 1);
  temp ^= temp3;
  temp -= temp3;

  /* For a negative input, want temp2 = bitwise complement of abs(input) */
  /* This code assumes we are on a two's complement machine */
  temp2 += temp3;

  /* Find the number of bits needed for the magnitude of the coefficient */
  BUILTIN_CLZ_W(temp, nbits);

  /* Emit the Huffman-coded symbol for the number of bits */
  code = dctbl->ehufco[nbits];
  size = dctbl->ehufsi[nbits];
  EMIT_BITS(code, size)

  /* Mask off any extra bits in code */
  temp2 &= (((int) 1)<<nbits) - 1;

  /* Emit that number of bits of the value, if positive, */
  /* or the complement of its magnitude, if negative. */
  EMIT_BITS(temp2, nbits)

  /* rearrange block[i] data */
  {
    __m256i mask0 = {0x0008000300020001,0x0000000000000009,0,0};
    __m256i mask1 = {0x0009000300080000,0x0004000A00020001,0,0};
    __m256i mask2 = {0x0000000A00050004,0,0x0009000800040003,0};
    __m256i mask3 = {0x0002000700080006,0x0005000100000004,0,0};
    __m256i mask4 = {0x0000000A00090003,0,0x0009000800030002,0};
    __m256i mask5 = {0x0006000100040000,0x0005000200070008,0,0};
    __m256i mask6 = {0x000D000C00070006,0,0x0000000C00060005,0};
    __m256i mask7 = {0x0001000000040002,0x000B000600030005,0,0};
    __m256i mask8 = {0x0000000A00090004,0,0x0009000800030002,0};
    __m256i mask9 = {0x0007000600010004,0x000D000000050002,0,0};
    __m256i maskA = {0x00000000000F000E,0,0x00000000000E0007,0};
    __m256i maskB = {0x00000000000B0005,0,0x00000000000A0004,0};
    __m256i maskC = {0x0005000100040000,0x000D0009000C0008,0,0};
    __m256i maskD = {0x000D000C00070006,0,0x0000000B00060005,0};
    __m256i maskE = {0x0000000400020006,0x000300050001000F,0,0};
    __m256i maskF = {0x000F000200050004,0x0000000700060003,0,0};

    /* vec0 xx 01 02 03 04 05 06 07  08 09 10 11 12 13 14 15
     * vec2 16 17 18 19 20 21 22 23  24 25 26 27 28 29 30 31
     * vec4 32 33 34 35 36 37 38 39  40 41 42 43 44 45 46 47
     * vec6 48 49 50 51 52 53 54 55  56 57 58 59 60 61 62 63
     */
    LASX_LD_4(block, 16, vec0, vec2, vec4, vec6);

    /* vec1 08 09 10 11 12 13 14 15 ...
     * vec3 24 25 26 27 28 29 30 31 ...
     * vec5 40 41 42 43 44 45 46 47 ...
     */
    vec1 = __lasx_xvpermi_q(zero, vec0, 0x31);
    vec3 = __lasx_xvpermi_q(zero, vec2, 0x31);
    vec5 = __lasx_xvpermi_q(zero, vec4, 0x31);

    /* 01 08 16 09 02 03 10 17 xx xx xx xx xx xx xx xxt */
    tmp0 = __lasx_xvshuf_h(mask0, vec2, vec0);
    tmp0 = __lasx_xvshuf_h(mask1, vec1, tmp0);

    /* 24 32 25 18 11 04 05 12 xx xx xx xx xx xx xx xx */
    tmp1 = __lasx_xvshuf_h(mask2, vec2, vec0);
    tmp1 = __lasx_xvpermi_d(tmp1, 0xd8);
    tmp1 = __lasx_xvshuf_h(mask3, vec4, tmp1);

    /* 19 26 33 40 48 41 34 27 xx xx xx xx xx xx xx xx */
    tmp2 = __lasx_xvshuf_h(mask4, vec4, vec2);
    tmp2 = __lasx_xvpermi_d(tmp2, 0xd8);
    tmp2 = __lasx_xvshuf_h(mask5, vec6, tmp2);

    /* 20 13 06 07 14 21 28 35 xx xx xx xx xx xx xx xx */
    tmp3 = __lasx_xvshuf_h(mask6, vec2, vec0);
    tmp3 = __lasx_xvpermi_d(tmp3, 0xd8);
    tmp3 = __lasx_xvshuf_h(mask7, vec4, tmp3);

    /* 42 49 56 57 50 43 36 29 xx xx xx xx xx xx xx xx */
    tmp4 = __lasx_xvshuf_h(mask8, vec6, vec4);
    tmp4 = __lasx_xvpermi_d(tmp4, 0xd8);
    tmp4 = __lasx_xvshuf_h(mask9, vec3, tmp4);

    /* 22 15 23 30 37 44 51 58 xx xx xx xx xx xx xx xx */
    tmp5 = __lasx_xvshuf_h(maskA, vec2, vec0);
    tmp6 = __lasx_xvshuf_h(maskB, vec6, vec4);
    tmp5 = __lasx_xvpermi_d(tmp5, 0xd8);
    tmp6 = __lasx_xvpermi_d(tmp6, 0xd8);
    tmp5 = __lasx_xvshuf_h(maskC, tmp6, tmp5);

    /* 60 61 54 47 55 62 63 xx xx xx xx xx xx xx xx xx */
    tmp6 = __lasx_xvshuf_h(maskD, vec6, vec4);
    tmp6 = __lasx_xvpermi_d(tmp6, 0xd8);
    tmp6 = __lasx_xvshuf_h(maskE, vec3, tmp6);

    /* 60 61 54 47 55 62 63 xx xx xx xx xx xx xx xx xx */
    tmp7 = __lasx_xvpermi_d(vec6, 0x8d);
    tmp7 = __lasx_xvshuf_h(maskF, vec5, tmp7);

    /*
     * 01 08 16 09 02 03 10 17 24 32 25 18 11 04 05 12
     * 19 26 33 40 48 41 34 27 20 13 06 07 14 21 28 35
     * 42 49 56 57 50 43 36 29 22 15 23 30 37 44 51 58
     * 59 52 45 38 31 39 46 53 60 61 54 47 55 62 63 xx
     */
    tmp0 = __lasx_xvpermi_q(tmp1, tmp0, 0x20);
    tmp1 = __lasx_xvpermi_q(tmp3, tmp2, 0x20);
    tmp2 = __lasx_xvpermi_q(tmp5, tmp4, 0x20);
    tmp3 = __lasx_xvpermi_q(tmp7, tmp6, 0x20);
  }

  // get sign
  tmp4 = __lasx_xvslt_h(tmp0, zero);
  tmp5 = __lasx_xvslt_h(tmp1, zero);
  tmp6 = __lasx_xvslt_h(tmp2, zero);
  tmp7 = __lasx_xvslt_h(tmp3, zero);

  // get abs
  abs0 = __lasx_xvabsd_h(tmp0, zero);
  abs1 = __lasx_xvabsd_h(tmp1, zero);
  abs2 = __lasx_xvabsd_h(tmp2, zero);
  abs3 = __lasx_xvabsd_h(tmp3, zero);

  // temp2
  tmp4 = __lasx_xvxor_v(abs0, tmp4);
  tmp5 = __lasx_xvxor_v(abs1, tmp5);
  tmp6 = __lasx_xvxor_v(abs2, tmp6);
  tmp7 = __lasx_xvxor_v(abs3, tmp7);

  tmp0 = __lasx_xvseq_h(abs0, zero);
  tmp1 = __lasx_xvseq_h(abs1, zero);
  tmp2 = __lasx_xvseq_h(abs2, zero);
  tmp3 = __lasx_xvseq_h(abs3, zero);

  LASX_PCKEV_B(zero, tmp0, tmp0);
  LASX_PCKEV_B(zero, tmp1, tmp1);
  LASX_PCKEV_B(zero, tmp2, tmp2);
  LASX_PCKEV_B(zero, tmp3, tmp3);

  vec7 = __lasx_xvmskgez_b(tmp0);
  t0   = __lasx_xvpickve2gr_d(vec7, 0);
  vec7 = __lasx_xvmskgez_b(tmp1);
  t1   = __lasx_xvpickve2gr_d(vec7, 0);
  vec7 = __lasx_xvmskgez_b(tmp2);
  t2   = __lasx_xvpickve2gr_d(vec7, 0);
  vec7 = __lasx_xvmskgez_b(tmp3);
  t3   = __lasx_xvpickve2gr_d(vec7, 0);

  t0 = t0 | t1 << 16 | t2 << 32 | t3 << 48;
  t0 &= 0x7FFFFFFFFFFFFFFF;

  k = 0;
  BUILTIN_CTZ_D(t0, r);
  while (r < 64) {
    k += r;
    t0 >>= r;
    GET_VAL_2(k, temp3, temp2);
    BUILTIN_CLZ_W(temp3, nbits)
    while (r > 15) {
      EMIT_BITS(code_0xf0, size_0xf0)
      r -= 16;
    }
    temp3 = (r << 4) + nbits;
    code = actbl->ehufco[temp3];
    size = actbl->ehufsi[temp3];
    EMIT_CODE(code, size)
    k++;
    t0 >>= 1;
    BUILTIN_CTZ_D(t0, r);
  }
  r = 63 - k;
  /* If the last coef(s) were zero, emit an end-of-block code */
  if (r > 0) {
    code = actbl->ehufco[0];
    size = actbl->ehufsi[0];
    EMIT_BITS(code, size)
  }

  state_ptr->cur.put_buffer = put_buffer;
  state_ptr->cur.put_bits = put_bits;

  return buffer;
}
