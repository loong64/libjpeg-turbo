/*
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2011, 2014, 2016, 2018, 2020, 2022, 2025,
 *           D. R. Commander.
 * Copyright (C) 2013-2014, MIPS Technologies, Inc., California.
 * Copyright (C) 2015, 2018, Matthieu Darbois.
 * Copyright (C) 2021, 2023, Loongson Technology Corporation Limited, BeiJing.
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
 *
 * This file contains the interface between the "normal" portions
 * of the library and the SIMD implementations when running on a
 * 64-bit LOONGARCH architecture.
 */

#include "../jsimdint.h"

#include <sys/auxv.h>
#include <asm/hwcap.h>

HIDDEN unsigned int
jpeg_simd_cpu_support(void)
{
  unsigned int simd_support = 0;
  uint64_t hwcaps = getauxval(AT_HWCAP);

  if (hwcaps & HWCAP_LOONGARCH_LSX)
    simd_support |= JSIMD_LSX;
  if (hwcaps & HWCAP_LOONGARCH_LASX)
    simd_support |= JSIMD_LASX;

  return simd_support;
}
