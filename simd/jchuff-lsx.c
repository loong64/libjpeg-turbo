/*
 * LOONGARCH LSX optimizations for libjpeg-turbo
 *
 * Copyright (C) 2023 Loongson Technology Corporation Limited
 * All rights reserved.
 * Contributed by Song Ding (songding@loongson.cn)
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
#include "jmacros_lsx.h"
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

#define EMIT_BYTE() {                               \
  JOCTET c;                                         \
  put_bits -= 8;                                    \
  c = (JOCTET)GETJOCTET(put_buffer >> put_bits);    \
  *buffer++ = c;                                    \
  if (c == 0xFF)  /* need to stuff a zero byte? */  \
    *buffer++ = 0;                                  \
}

#define PUT_BITS(code, size) {              \
  put_bits += size;                         \
  put_buffer = (put_buffer << size) | code; \
}

#define CHECKBUF47() { \
  if (put_bits > 47) { \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
  }                    \
}

#define CHECKBUF31() { \
  if (put_bits > 31) { \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
    EMIT_BYTE()        \
  }                    \
}

#define EMIT_BITS(code, size) { \
  CHECKBUF47()                  \
  PUT_BITS(code, size)          \
}

#define EMIT_CODE(code, size) {    \
  temp2 &= (((int) 1)<<nbits) - 1; \
  CHECKBUF31()                     \
  PUT_BITS(code, size)             \
  PUT_BITS(temp2, nbits)           \
}

#define GET_VAL_2(_idx, _out1, _out2) \
{                                     \
  int _tmp_idx;                       \
  v8u16 _t0 = (v8u16)abs0;            \
  v8u16 _t1 = (v8u16)abs1;            \
  v8u16 _t2 = (v8u16)abs2;            \
  v8u16 _t3 = (v8u16)abs3;            \
  v8u16 _t4 = (v8u16)abs4;            \
  v8u16 _t5 = (v8u16)abs5;            \
  v8u16 _t6 = (v8u16)abs6;            \
  v8u16 _t7 = (v8u16)abs7;            \
  v8i16 _t8 = (v8i16)tmp8;            \
  v8i16 _t9 = (v8i16)tmp9;            \
  v8i16 _t10 = (v8i16)tmp10;          \
  v8i16 _t11 = (v8i16)tmp11;          \
  v8i16 _t12 = (v8i16)tmp12;          \
  v8i16 _t13 = (v8i16)tmp13;          \
  v8i16 _t14 = (v8i16)tmp14;          \
  v8i16 _t15 = (v8i16)tmp15;          \
  \
  if (_idx < 8) {                     \
    _out1 = _t0[_idx];                \
    _out2 = _t8[_idx];                \
  } else if (_idx < 16) {             \
    _tmp_idx = _idx % 8;              \
    _out1 = _t1[_tmp_idx];            \
    _out2 = _t9[_tmp_idx];            \
  } else if (_idx < 24) {             \
    _tmp_idx = _idx % 16;             \
    _out1 = _t2[_tmp_idx];            \
    _out2 = _t10[_tmp_idx];           \
  } else if (_idx < 32) {             \
    _tmp_idx = _idx % 24;             \
    _out1 = _t3[_tmp_idx];            \
    _out2 = _t11[_tmp_idx];           \
  } else if (_idx < 40) {             \
    _tmp_idx = _idx % 32;             \
    _out1 = _t4[_tmp_idx];            \
    _out2 = _t12[_tmp_idx];           \
  } else if (_idx < 48) {             \
    _tmp_idx = _idx % 40;             \
    _out1 = _t5[_tmp_idx];            \
    _out2 = _t13[_tmp_idx];           \
  } else if (_idx < 56) {             \
    _tmp_idx = _idx % 48;             \
    _out1 = _t6[_tmp_idx];            \
    _out2 = _t14[_tmp_idx];           \
  } else {                            \
    _tmp_idx = _idx % 56;             \
    _out1 = _t7[_tmp_idx];            \
    _out2 = _t15[_tmp_idx];           \
  }                                   \
}

#define BUILTIN_CTZ_D(_src, _dst) { \
  __asm__ volatile (                \
    "ctz.d %0, %1"                  \
    : "=r"(_dst)                    \
    : "r"(_src)                     \
    : "memory"                      \
 );                                 \
}

#define BUILTIN_CLZ_W(_src, _dst) { \
  __asm__ volatile (                \
    "clz.w %0, %1"                  \
    : "=r"(_dst)                    \
    : "r"(_src)                     \
    : "memory"                      \
 );                                 \
 _dst = 32 - _dst;                  \
}

GLOBAL(JOCTET*)
jsimd_huff_encode_one_block_lsx(void *state, JOCTET *buffer, JCOEFPTR block, int last_dc_val,
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

  __m128i vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;
  __m128i tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
  __m128i tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14, tmp15;
  __m128i abs0, abs1, abs2, abs3, abs4, abs5, abs6, abs7;
  __m128i tmpx, tmpy;
  __m128i zero = {0};

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

    /* vec0 xx 01 02 03 04 05 06 07
     * vec1 08 09 10 11 12 13 14 15
     * vec2 16 17 18 19 20 21 22 23
     * vec3 24 25 26 27 28 29 30 31
     * vec4 32 33 34 35 36 37 38 39
     * vec5 40 41 42 43 44 45 46 47
     * vec6 48 49 50 51 52 53 54 55
     * vec7 56 57 58 59 60 61 62 63
     */
    LSX_LD_8(block, 8, vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7);

    v8i16 mask0 = {1, 8, 0, 9, 2, 3, 10, 0};
    v8i16 mask1 = {0, 1, 8, 3, 4, 5, 6, 9};
    /* 01 08 16 09 02 03 10 17 */
    tmp0 = __lsx_vshuf_h((__m128i)mask0, vec1, vec0);
    tmp0 = __lsx_vshuf_h((__m128i)mask1, vec2, tmp0);

    /* 24 32 25 18 11 04 05 12 */
    v8i16 mask2 = {0, 0, 0, 0, 11, 4, 5, 12};
    v8i16 mask3 = {0, 8, 9, 2, 0, 0, 0, 0};
    v8i16 mask4 = {9, 8, 10, 11, 4, 5, 6, 7};
    tmpx = __lsx_vshuf_h((__m128i)mask2, vec1, vec0); /* xx xx xx xx 11 04 05 12 */
    tmp1 = __lsx_vshuf_h((__m128i)mask3, vec3, vec2); /* xx 24 25 18 xx xx xx xx */
    tmp1 = __lsx_vextrins_h(tmp1, vec4, 0);           /* 32 24 25 18 xx xx xx xx */
    tmp1 = __lsx_vshuf_h((__m128i)mask4, tmp1, tmpx); /* 24 32 25 18 11 04 05 12 */

    /* 19 26 33 40 48 41 34 27 */
    v8i16 mask5 = {3, 6, 9, 12, 0, 13, 10, 7};
    tmpx = vec3;
    tmpy = vec5;
    tmpx = __lsx_vpermi_w(tmpx, vec2, 0x44);          /* 16 17 18 19 24 25 26 27 */
    tmpy = __lsx_vpermi_w(tmpy, vec4, 0x44);          /* 32 33 34 35 40 41 42 43 */
    tmp2 = __lsx_vshuf_h((__m128i)mask5, tmpy, tmpx); /* 19 26 33 40 xx 41 34 27 */
    tmp2 = __lsx_vextrins_h(tmp2, vec6, 0x40);        /* 19 26 33 40 48 41 34 27 */

    /* 20 13 06 07 14 21 28 35 */
    v8i16 mask6 = {8, 5, 2, 3, 6, 9, 12, 0};
    tmpx = vec1;
    tmpy = vec3;
    tmpx = __lsx_vpermi_w(tmpx, vec0, 0xEE);          /* 04 05 06 07 12 13 14 15 */
    tmpy = __lsx_vpermi_w(tmpy, vec2, 0xEE);          /* 20 21 22 23 28 29 30 31 */
    tmp3 = __lsx_vshuf_h((__m128i)mask6, tmpy, tmpx); /* 20 13 06 07 14 21 28 xx */
    tmp3  = __lsx_vextrins_h(tmp3, vec4, 0x73);       /* 20 13 06 07 14 21 28 35 */

    /* 42 49 56 57 50 43 36 29 */
    v8i16 mask7 = {0, 1, 8, 9, 2, 0, 0, 0};
    v8i16 mask8 = {2, 9, 10, 11, 12, 3, 6, 0};
    tmpx = vec5;
    tmpx = __lsx_vextrins_h(tmpx, vec4, 0x64);        /* 40 41 42 43 44 45 36 47 */
    tmpy = __lsx_vshuf_h((__m128i)mask7, vec7, vec6); /* xx 49 56 57 50 xx xx xx */
    tmp4 = __lsx_vshuf_h((__m128i)mask8, tmpy, tmpx); /* 42 49 56 57 50 43 36 xx */
    tmp4 = __lsx_vextrins_h(tmp4, vec3, 0x75);        /* 42 49 56 57 50 43 36 29 */

    /* 22 15 23 30 37 44 51 58 */
    v8i16 mask9 = {14, 7, 15, 0, 0, 0, 0, 0};
    tmp5 = __lsx_vshuf_h((__m128i)mask9, vec2, vec1); /* 22 15 23 xx xx xx xx xx */
    tmp5 = __lsx_vextrins_h(tmp5, vec3, 0x36);        /* 22 15 23 30 xx xx xx xx */
    tmp5 = __lsx_vextrins_h(tmp5, vec4, 0x45);        /* 22 15 23 30 37 xx xx xx */
    tmp5 = __lsx_vextrins_h(tmp5, vec5, 0x54);        /* 22 15 23 30 37 44 xx xx */
    tmp5 = __lsx_vextrins_h(tmp5, vec6, 0x63);        /* 22 15 23 30 37 44 51 xx */
    tmp5 = __lsx_vextrins_h(tmp5, vec7, 0x72);        /* 22 15 23 30 37 44 51 58 */

    /* 59 52 45 38 31 39 46 53 */
    v8i16 maskA = {0, 0, 13, 6, 0, 7, 14, 0};
    v8i16 maskB = {11, 4, 0, 0, 0, 0, 0, 5};
    v8i16 maskC = {8, 9, 2, 3, 0, 5, 6, 15};
    tmpx = __lsx_vshuf_h((__m128i)maskA, vec5, vec4); /* xx xx 45 38 xx 39 46 xx */
    tmpy = __lsx_vshuf_h((__m128i)maskB, vec7, vec6); /* 59 52 xx xx xx xx xx 53 */
    tmp6 = __lsx_vshuf_h((__m128i)maskC, tmpy, tmpx); /* 59 52 45 38 xx 39 46 53 */
    tmp6 = __lsx_vextrins_h(tmp6, vec3, 0x47);        /* 59 52 45 38 31 39 46 53 */

    /* 60 61 54 47 55 62 63 63 */
    v8i16 maskD = {12, 13, 6, 0, 7, 14, 15, 15};
    tmp7 = __lsx_vshuf_h((__m128i)maskD, vec7, vec6); /* 60 61 54 x 55 62 63 63 */
    tmp7 = __lsx_vextrins_h(tmp7, vec5, 0x37);        /* 60 61 54 47 55 62 63 63 */
  }

  // get sign
  tmp8 = __lsx_vslt_h(tmp0, zero);
  tmp9 = __lsx_vslt_h(tmp1, zero);
  tmp10 = __lsx_vslt_h(tmp2, zero);
  tmp11 = __lsx_vslt_h(tmp3, zero);
  tmp12 = __lsx_vslt_h(tmp4, zero);
  tmp13 = __lsx_vslt_h(tmp5, zero);
  tmp14 = __lsx_vslt_h(tmp6, zero);
  tmp15 = __lsx_vslt_h(tmp7, zero);

  //get abs
  abs0 = __lsx_vabsd_h(tmp0, zero);
  abs1 = __lsx_vabsd_h(tmp1, zero);
  abs2 = __lsx_vabsd_h(tmp2, zero);
  abs3 = __lsx_vabsd_h(tmp3, zero);
  abs4 = __lsx_vabsd_h(tmp4, zero);
  abs5 = __lsx_vabsd_h(tmp5, zero);
  abs6 = __lsx_vabsd_h(tmp6, zero);
  abs7 = __lsx_vabsd_h(tmp7, zero);

  // temp
  tmp8 = __lsx_vxor_v(abs0, tmp8);
  tmp9 = __lsx_vxor_v(abs1, tmp9);
  tmp10 = __lsx_vxor_v(abs2, tmp10);
  tmp11 = __lsx_vxor_v(abs3, tmp11);
  tmp12 = __lsx_vxor_v(abs4, tmp12);
  tmp13 = __lsx_vxor_v(abs5, tmp13);
  tmp14 = __lsx_vxor_v(abs6, tmp14);
  tmp15 = __lsx_vxor_v(abs7, tmp15);

  tmp0 = __lsx_vseq_h(tmp0, zero);
  tmp1 = __lsx_vseq_h(tmp1, zero);
  tmp2 = __lsx_vseq_h(tmp2, zero);
  tmp3 = __lsx_vseq_h(tmp3, zero);
  tmp4 = __lsx_vseq_h(tmp4, zero);
  tmp5 = __lsx_vseq_h(tmp5, zero);
  tmp6 = __lsx_vseq_h(tmp6, zero);
  tmp7 = __lsx_vseq_h(tmp7, zero);

  tmp0 = __lsx_vsat_b(tmp0, 7);
  tmp1 = __lsx_vsat_b(tmp1, 7);
  tmp2 = __lsx_vsat_b(tmp2, 7);
  tmp3 = __lsx_vsat_b(tmp3, 7);
  tmp4 = __lsx_vsat_b(tmp4, 7);
  tmp5 = __lsx_vsat_b(tmp5, 7);
  tmp6 = __lsx_vsat_b(tmp6, 7);
  tmp7 = __lsx_vsat_b(tmp7, 7);

  tmp0 = __lsx_vpickev_b(tmp1, tmp0);
  tmp2 = __lsx_vpickev_b(tmp3, tmp2);
  tmp4 = __lsx_vpickev_b(tmp5, tmp4);
  tmp6 = __lsx_vpickev_b(tmp7, tmp6);

  vec7 = __lsx_vmskgez_b(tmp0);
  t0   = __lsx_vpickve2gr_d(vec7, 0);
  vec7 = __lsx_vmskgez_b(tmp2);
  t1   = __lsx_vpickve2gr_d(vec7, 0);
  vec7 = __lsx_vmskgez_b(tmp4);
  t2   = __lsx_vpickve2gr_d(vec7, 0);
  vec7 = __lsx_vmskgez_b(tmp6);
  t3   = __lsx_vpickve2gr_d(vec7, 0);

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
