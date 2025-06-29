<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Third-Party Software Attributions

This project uses the following third-party libraries. Each library is open-source and licensed under the terms indicated below.

## glibc - 2.35

- **Repository URL**: https://sourceware.org/git/?p=glibc.git
- **License URL**: https://sourceware.org/git?p=glibc.git;a=blob;f=LICENSES
- **License name**: GNU Lesser General Public License v2
### License Text:

```
This file contains the copying permission notices for various files in the
   2 GNU C Library distribution that have copyright owners other than the Free
   3 Software Foundation.  These notices all require that a copy of the notice
   4 be included in the accompanying documentation and be distributed with
   5 binary distributions of the code, so be sure to include this file along
   6 with any binary distributions derived from the GNU C Library.
   7
   8 \f
   9 All code incorporated from 4.4 BSD is distributed under the following
  10 license:
  11
  12 Copyright (C) 1991 Regents of the University of California.
  13 All rights reserved.
  14
  15 Redistribution and use in source and binary forms, with or without
  16 modification, are permitted provided that the following conditions
  17 are met:
  18
  19 1. Redistributions of source code must retain the above copyright
  20    notice, this list of conditions and the following disclaimer.
  21 2. Redistributions in binary form must reproduce the above copyright
  22    notice, this list of conditions and the following disclaimer in the
  23    documentation and/or other materials provided with the distribution.
  24 3. [This condition was removed.]
  25 4. Neither the name of the University nor the names of its contributors
  26    may be used to endorse or promote products derived from this software
  27    without specific prior written permission.
  28
  29 THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
  30 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  31 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  32 ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
  33 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  34 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
  35 OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
  36 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  37 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
  38 OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
  39 SUCH DAMAGE.
  40 \f
  41 The DNS resolver code, taken from BIND 4.9.5, is copyrighted by UC
  42 Berkeley, by Digital Equipment Corporation and by Internet Software
  43 Consortium.  The DEC portions are under the following license:
  44
  45 Portions Copyright (C) 1993 by Digital Equipment Corporation.
  46
  47 Permission to use, copy, modify, and distribute this software for any
  48 purpose with or without fee is hereby granted, provided that the above
  49 copyright notice and this permission notice appear in all copies, and
  50 that the name of Digital Equipment Corporation not be used in
  51 advertising or publicity pertaining to distribution of the document or
  52 software without specific, written prior permission.
  53
  54 THE SOFTWARE IS PROVIDED ``AS IS'' AND DIGITAL EQUIPMENT CORP.
  55 DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
  56 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.  IN NO EVENT SHALL
  57 DIGITAL EQUIPMENT CORPORATION BE LIABLE FOR ANY SPECIAL, DIRECT,
  58 INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
  59 FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
  60 NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
  61 WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
  62 \f
  63 The ISC portions are under the following license:
  64
  65 Portions Copyright (c) 1996-1999 by Internet Software Consortium.
  66
  67 Permission to use, copy, modify, and distribute this software for any
  68 purpose with or without fee is hereby granted, provided that the above
  69 copyright notice and this permission notice appear in all copies.
  70
  71 THE SOFTWARE IS PROVIDED "AS IS" AND INTERNET SOFTWARE CONSORTIUM DISCLAIMS
  72 ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES
  73 OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL INTERNET SOFTWARE
  74 CONSORTIUM BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL
  75 DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
  76 PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  77 ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
  78 SOFTWARE.
  79 \f
  80 The Sun RPC support (from rpcsrc-4.0) is covered by the following
  81 license:
  82
  83 Copyright (c) 2010, Oracle America, Inc.
  84
  85 Redistribution and use in source and binary forms, with or without
  86 modification, are permitted provided that the following conditions are
  87 met:
  88
  89     * Redistributions of source code must retain the above copyright
  90       notice, this list of conditions and the following disclaimer.
  91     * Redistributions in binary form must reproduce the above
  92       copyright notice, this list of conditions and the following
  93       disclaimer in the documentation and/or other materials
  94       provided with the distribution.
  95     * Neither the name of the "Oracle America, Inc." nor the names of its
  96       contributors may be used to endorse or promote products derived
  97       from this software without specific prior written permission.
  98
  99   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 100   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 101   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 102   FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 103   COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 104   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 105   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 106   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 107   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 108   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 109   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 110   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 111
 112 \f
 113 The following CMU license covers some of the support code for Mach,
 114 derived from Mach 3.0:
 115
 116 Mach Operating System
 117 Copyright (C) 1991,1990,1989 Carnegie Mellon University
 118 All Rights Reserved.
 119
 120 Permission to use, copy, modify and distribute this software and its
 121 documentation is hereby granted, provided that both the copyright
 122 notice and this permission notice appear in all copies of the
 123 software, derivative works or modified versions, and any portions
 124 thereof, and that both notices appear in supporting documentation.
 125
 126 CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS ``AS IS''
 127 CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 128 ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 129
 130 Carnegie Mellon requests users of this software to return to
 131
 132  Software Distribution Coordinator
 133  School of Computer Science
 134  Carnegie Mellon University
 135  Pittsburgh PA 15213-3890
 136
 137 or Software.Distribution@CS.CMU.EDU any improvements or
 138 extensions that they make and grant Carnegie Mellon the rights to
 139 redistribute these changes.
 140 \f
 141 The file if_ppp.h is under the following CMU license:
 142
 143  Redistribution and use in source and binary forms, with or without
 144  modification, are permitted provided that the following conditions
 145  are met:
 146  1. Redistributions of source code must retain the above copyright
 147     notice, this list of conditions and the following disclaimer.
 148  2. Redistributions in binary form must reproduce the above copyright
 149     notice, this list of conditions and the following disclaimer in the
 150     documentation and/or other materials provided with the distribution.
 151  3. Neither the name of the University nor the names of its contributors
 152     may be used to endorse or promote products derived from this software
 153     without specific prior written permission.
 154
 155  THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY AND
 156  CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,
 157  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 158  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 159  IN NO EVENT SHALL THE UNIVERSITY OR CONTRIBUTORS BE LIABLE FOR ANY
 160  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 161  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 162  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 163  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 164  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 165  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
 166  IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 167 \f
 168 The files nss/getnameinfo.c and nss/getaddrinfo.c are copyright (C) by
 169 Craig Metz and are distributed under the following license:
 170
 171 /* The Inner Net License, Version 2.00
 172
 173   The author(s) grant permission for redistribution and use in source and
 174 binary forms, with or without modification, of the software and documentation
 175 provided that the following conditions are met:
 176
 177 0. If you receive a version of the software that is specifically labelled
 178    as not being for redistribution (check the version message and/or README),
 179    you are not permitted to redistribute that version of the software in any
 180    way or form.
 181 1. All terms of the all other applicable copyrights and licenses must be
 182    followed.
 183 2. Redistributions of source code must retain the authors' copyright
 184    notice(s), this list of conditions, and the following disclaimer.
 185 3. Redistributions in binary form must reproduce the authors' copyright
 186    notice(s), this list of conditions, and the following disclaimer in the
 187    documentation and/or other materials provided with the distribution.
 188 4. [The copyright holder has authorized the removal of this clause.]
 189 5. Neither the name(s) of the author(s) nor the names of its contributors
 190    may be used to endorse or promote products derived from this software
 191    without specific prior written permission.
 192
 193 THIS SOFTWARE IS PROVIDED BY ITS AUTHORS AND CONTRIBUTORS ``AS IS'' AND ANY
 194 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 195 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 196 DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY
 197 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 198 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 199 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 200 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 201 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 202 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 203
 204   If these license terms cause you a real problem, contact the author.  */
 205 \f
 206 The file sunrpc/des_impl.c is copyright Eric Young:
 207
 208 Copyright (C) 1992 Eric Young
 209 Collected from libdes and modified for SECURE RPC by Martin Kuck 1994
 210 This file is distributed under the terms of the GNU Lesser General
 211 Public License, version 2.1 or later - see the file COPYING.LIB for details.
 212 If you did not receive a copy of the license with this program, please
 213 see <https://www.gnu.org/licenses/> to obtain a copy.
 214 \f
 215 The file inet/rcmd.c is under a UCB copyright and the following:
 216
 217 Copyright (C) 1998 WIDE Project.
 218 All rights reserved.
 219
 220 Redistribution and use in source and binary forms, with or without
 221 modification, are permitted provided that the following conditions
 222 are met:
 223 1. Redistributions of source code must retain the above copyright
 224    notice, this list of conditions and the following disclaimer.
 225 2. Redistributions in binary form must reproduce the above copyright
 226    notice, this list of conditions and the following disclaimer in the
 227    documentation and/or other materials provided with the distribution.
 228 3. Neither the name of the project nor the names of its contributors
 229    may be used to endorse or promote products derived from this software
 230    without specific prior written permission.
 231
 232 THIS SOFTWARE IS PROVIDED BY THE PROJECT AND CONTRIBUTORS ``AS IS'' AND
 233 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 234 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 235 ARE DISCLAIMED.  IN NO EVENT SHALL THE PROJECT OR CONTRIBUTORS BE LIABLE
 236 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 237 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 238 OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 239 HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 240 LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 241 OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 242 SUCH DAMAGE.
 243 \f
 244 The file posix/runtests.c is copyright Tom Lord:
 245
 246 Copyright 1995 by Tom Lord
 247
 248                         All Rights Reserved
 249
 250 Permission to use, copy, modify, and distribute this software and its
 251 documentation for any purpose and without fee is hereby granted,
 252 provided that the above copyright notice appear in all copies and that
 253 both that copyright notice and this permission notice appear in
 254 supporting documentation, and that the name of the copyright holder not be
 255 used in advertising or publicity pertaining to distribution of the
 256 software without specific, written prior permission.
 257
 258 Tom Lord DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
 259 INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO
 260 EVENT SHALL TOM LORD BE LIABLE FOR ANY SPECIAL, INDIRECT OR
 261 CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
 262 USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 263 OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 264 PERFORMANCE OF THIS SOFTWARE.
 265 \f
 266 The posix/rxspencer tests are copyright Henry Spencer:
 267
 268 Copyright 1992, 1993, 1994, 1997 Henry Spencer.  All rights reserved.
 269 This software is not subject to any license of the American Telephone
 270 and Telegraph Company or of the Regents of the University of California.
 271
 272 Permission is granted to anyone to use this software for any purpose on
 273 any computer system, and to alter it and redistribute it, subject
 274 to the following restrictions:
 275
 276 1. The author is not responsible for the consequences of use of this
 277    software, no matter how awful, even if they arise from flaws in it.
 278
 279 2. The origin of this software must not be misrepresented, either by
 280    explicit claim or by omission.  Since few users ever read sources,
 281    credits must appear in the documentation.
 282
 283 3. Altered versions must be plainly marked as such, and must not be
 284    misrepresented as being the original software.  Since few users
 285    ever read sources, credits must appear in the documentation.
 286
 287 4. This notice may not be removed or altered.
 288 \f
 289 The file posix/PCRE.tests is copyright University of Cambridge:
 290
 291 Copyright (c) 1997-2003 University of Cambridge
 292
 293 Permission is granted to anyone to use this software for any purpose on any
 294 computer system, and to redistribute it freely, subject to the following
 295 restrictions:
 296
 297 1. This software is distributed in the hope that it will be useful,
 298    but WITHOUT ANY WARRANTY; without even the implied warranty of
 299    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 300
 301 2. The origin of this software must not be misrepresented, either by
 302    explicit claim or by omission. In practice, this means that if you use
 303    PCRE in software that you distribute to others, commercially or
 304    otherwise, you must put a sentence like this
 305
 306      Regular expression support is provided by the PCRE library package,
 307      which is open source software, written by Philip Hazel, and copyright
 308      by the University of Cambridge, England.
 309
 310    somewhere reasonably visible in your documentation and in any relevant
 311    files or online help data or similar. A reference to the ftp site for
 312    the source, that is, to
 313
 314      ftp://ftp.csx.cam.ac.uk/pub/software/programming/pcre/
 315
 316    should also be given in the documentation. However, this condition is not
 317    intended to apply to whole chains of software. If package A includes PCRE,
 318    it must acknowledge it, but if package B is software that includes package
 319    A, the condition is not imposed on package B (unless it uses PCRE
 320    independently).
 321
 322 3. Altered versions must be plainly marked as such, and must not be
 323    misrepresented as being the original software.
 324
 325 4. If PCRE is embedded in any software that is released under the GNU
 326   General Purpose Licence (GPL), or Lesser General Purpose Licence (LGPL),
 327   then the terms of that licence shall supersede any condition above with
 328   which it is incompatible.
 329 \f
 330 Files from Sun fdlibm are copyright Sun Microsystems, Inc.:
 331
 332 Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 333
 334 Developed at SunPro, a Sun Microsystems, Inc. business.
 335 Permission to use, copy, modify, and distribute this
 336 software is freely granted, provided that this notice
 337 is preserved.
 338 \f
 339 Various long double libm functions are copyright Stephen L. Moshier:
 340
 341 Copyright 2001 by Stephen L. Moshier <moshier@na-net.ornl.gov>
 342
 343  This library is free software; you can redistribute it and/or
 344  modify it under the terms of the GNU Lesser General Public
 345  License as published by the Free Software Foundation; either
 346  version 2.1 of the License, or (at your option) any later version.
 347
 348  This library is distributed in the hope that it will be useful,
 349  but WITHOUT ANY WARRANTY; without even the implied warranty of
 350  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 351  Lesser General Public License for more details.
 352
 353  You should have received a copy of the GNU Lesser General Public
 354  License along with this library; if not, see
 355  <https://www.gnu.org/licenses/>.  */
 356 \f
 357 Copyright (c) 1995 IBM Corporation
 358
 359 Permission is hereby granted, free of charge, to any person obtaining
 360 a copy of this software and associated documentation files (the
 361 'Software'), to deal in the Software without restriction, including
 362 without limitation the rights to use, copy, modify, merge, publish,
 363 distribute, sublicense, and/or sell copies of the Software, and to
 364 permit persons to whom the Software is furnished to do so, subject to
 365 the following conditions:
 366
 367 The above copyright notice and this permission notice shall be
 368 included in all copies or substantial portions of the Software.
 369
 370 THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
 371 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 372 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 373 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 374 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 375 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 376 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 377 \f
 378 Various files in sysdeps/ieee754/flt-32, taken from the CORE-MATH project
 379 <https://core-math.gitlabpages.inria.fr/>, are distributed under the
 380 following license:
 381
 382 Copyright (c) 2022-2024 Alexei Sibidanov. Paul Zimmermann.
 383
 384 Permission is hereby granted, free of charge, to any person obtaining a copy
 385 of this software and associated documentation files (the "Software"), to deal
 386 in the Software without restriction, including without limitation the rights
 387 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 388 copies of the Software, and to permit persons to whom the Software is
 389 furnished to do so, subject to the following conditions:
 390
 391 The above copyright notice and this permission notice shall be included in all
 392 copies or substantial portions of the Software.
 393
 394 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 395 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 396 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 397 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 398 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 399 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 400 SOFTWARE.
```
## libgcc-s1 - 12.1.0

- **Repository URL**: https://gcc.gnu.org/git/gitweb.cgi?p=gcc.git
- **License URL**: https://gcc.gnu.org/git/?p=gcc.git;a=blob;f=COPYING.RUNTIME
- **License name**: GNU General Public License (v3 or later) with version 3.1 of the GCC Runtime Library Exception
### License Text:

```
  1 GCC RUNTIME LIBRARY EXCEPTION
   2
   3 Version 3.1, 31 March 2009
   4
   5 Copyright (C) 2009 Free Software Foundation, Inc. <http://fsf.org/>
   6
   7 Everyone is permitted to copy and distribute verbatim copies of this
   8 license document, but changing it is not allowed.
   9
  10 This GCC Runtime Library Exception ("Exception") is an additional
  11 permission under section 7 of the GNU General Public License, version
  12 3 ("GPLv3"). It applies to a given file (the "Runtime Library") that
  13 bears a notice placed by the copyright holder of the file stating that
  14 the file is governed by GPLv3 along with this Exception.
  15
  16 When you use GCC to compile a program, GCC may combine portions of
  17 certain GCC header files and runtime libraries with the compiled
  18 program. The purpose of this Exception is to allow compilation of
  19 non-GPL (including proprietary) programs to use, in this way, the
  20 header files and runtime libraries covered by this Exception.
  21
  22 0. Definitions.
  23
  24 A file is an "Independent Module" if it either requires the Runtime
  25 Library for execution after a Compilation Process, or makes use of an
  26 interface provided by the Runtime Library, but is not otherwise based
  27 on the Runtime Library.
  28
  29 "GCC" means a version of the GNU Compiler Collection, with or without
  30 modifications, governed by version 3 (or a specified later version) of
  31 the GNU General Public License (GPL) with the option of using any
  32 subsequent versions published by the FSF.
  33
  34 "GPL-compatible Software" is software whose conditions of propagation,
  35 modification and use would permit combination with GCC in accord with
  36 the license of GCC.
  37
  38 "Target Code" refers to output from any compiler for a real or virtual
  39 target processor architecture, in executable form or suitable for
  40 input to an assembler, loader, linker and/or execution
  41 phase. Notwithstanding that, Target Code does not include data in any
  42 format that is used as a compiler intermediate representation, or used
  43 for producing a compiler intermediate representation.
  44
  45 The "Compilation Process" transforms code entirely represented in
  46 non-intermediate languages designed for human-written code, and/or in
  47 Java Virtual Machine byte code, into Target Code. Thus, for example,
  48 use of source code generators and preprocessors need not be considered
  49 part of the Compilation Process, since the Compilation Process can be
  50 understood as starting with the output of the generators or
  51 preprocessors.
  52
  53 A Compilation Process is "Eligible" if it is done using GCC, alone or
  54 with other GPL-compatible software, or if it is done without using any
  55 work based on GCC. For example, using non-GPL-compatible Software to
  56 optimize any GCC intermediate representations would not qualify as an
  57 Eligible Compilation Process.
  58
  59 1. Grant of Additional Permission.
  60
  61 You have permission to propagate a work of Target Code formed by
  62 combining the Runtime Library with Independent Modules, even if such
  63 propagation would otherwise violate the terms of GPLv3, provided that
  64 all Target Code was generated by Eligible Compilation Processes. You
  65 may then convey such a combination under terms of your choice,
  66 consistent with the licensing of the Independent Modules.
  67
  68 2. No Weakening of GCC Copyleft.
  69
  70 The availability of this Exception does not imply any general
  71 presumption that third-party software is unaffected by the copyleft
  72 requirements of the license of GCC.
  73
  ```
## libstdc++ - 12.1.0

- **Repository URL**: https://gcc.gnu.org/git/gitweb.cgi?p=gcc.git
- **License URL**: https://gcc.gnu.org/git/?p=gcc.git;a=blob;f=COPYING.RUNTIME
- **License name**: GNU General Public License (v3 or later) with version 3.1 of the GCC Runtime Library Exception
### License Text:

```
  1 GCC RUNTIME LIBRARY EXCEPTION
   2
   3 Version 3.1, 31 March 2009
   4
   5 Copyright (C) 2009 Free Software Foundation, Inc. <http://fsf.org/>
   6
   7 Everyone is permitted to copy and distribute verbatim copies of this
   8 license document, but changing it is not allowed.
   9
  10 This GCC Runtime Library Exception ("Exception") is an additional
  11 permission under section 7 of the GNU General Public License, version
  12 3 ("GPLv3"). It applies to a given file (the "Runtime Library") that
  13 bears a notice placed by the copyright holder of the file stating that
  14 the file is governed by GPLv3 along with this Exception.
  15
  16 When you use GCC to compile a program, GCC may combine portions of
  17 certain GCC header files and runtime libraries with the compiled
  18 program. The purpose of this Exception is to allow compilation of
  19 non-GPL (including proprietary) programs to use, in this way, the
  20 header files and runtime libraries covered by this Exception.
  21
  22 0. Definitions.
  23
  24 A file is an "Independent Module" if it either requires the Runtime
  25 Library for execution after a Compilation Process, or makes use of an
  26 interface provided by the Runtime Library, but is not otherwise based
  27 on the Runtime Library.
  28
  29 "GCC" means a version of the GNU Compiler Collection, with or without
  30 modifications, governed by version 3 (or a specified later version) of
  31 the GNU General Public License (GPL) with the option of using any
  32 subsequent versions published by the FSF.
  33
  34 "GPL-compatible Software" is software whose conditions of propagation,
  35 modification and use would permit combination with GCC in accord with
  36 the license of GCC.
  37
  38 "Target Code" refers to output from any compiler for a real or virtual
  39 target processor architecture, in executable form or suitable for
  40 input to an assembler, loader, linker and/or execution
  41 phase. Notwithstanding that, Target Code does not include data in any
  42 format that is used as a compiler intermediate representation, or used
  43 for producing a compiler intermediate representation.
  44
  45 The "Compilation Process" transforms code entirely represented in
  46 non-intermediate languages designed for human-written code, and/or in
  47 Java Virtual Machine byte code, into Target Code. Thus, for example,
  48 use of source code generators and preprocessors need not be considered
  49 part of the Compilation Process, since the Compilation Process can be
  50 understood as starting with the output of the generators or
  51 preprocessors.
  52
  53 A Compilation Process is "Eligible" if it is done using GCC, alone or
  54 with other GPL-compatible software, or if it is done without using any
  55 work based on GCC. For example, using non-GPL-compatible Software to
  56 optimize any GCC intermediate representations would not qualify as an
  57 Eligible Compilation Process.
  58
  59 1. Grant of Additional Permission.
  60
  61 You have permission to propagate a work of Target Code formed by
  62 combining the Runtime Library with Independent Modules, even if such
  63 propagation would otherwise violate the terms of GPLv3, provided that
  64 all Target Code was generated by Eligible Compilation Processes. You
  65 may then convey such a combination under terms of your choice,
  66 consistent with the licensing of the Independent Modules.
  67
  68 2. No Weakening of GCC Copyleft.
  69
  70 The availability of this Exception does not imply any general
  71 presumption that third-party software is unaffected by the copyleft
  72 requirements of the license of GCC.
  73
  ```
## ucx - 1.19.0

- **Repository URL**: https://github.com/openucx/ucx/
- **License URL**: https://github.com/openucx/ucx/blob/master/LICENSE
- **License name**: BSD3
### License Text:

```
Copyright (c) 2014-2015      UT-Battelle, LLC. All rights reserved.
Copyright (c) 2014-2020      NVIDIA corporation & affiliates. All rights reserved.
Copyright (C) 2014-2015      The University of Houston System. All rights reserved.
Copyright (C) 2015           The University of Tennessee and The University
                             of Tennessee Research Foundation. All rights reserved.
Copyright (C) 2016-2020      ARM Ltd. All rights reserved.
Copyright (c) 2016           Los Alamos National Security, LLC. All rights reserved.
Copyright (C) 2016-2020      Advanced Micro Devices, Inc.  All rights reserved.
Copyright (C) 2019           UChicago Argonne, LLC.  All rights reserved.
Copyright (C) 2020           Huawei Technologies Co., Ltd. All rights reserved.
Copyright (C) 2016-2020      Stony Brook University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## etcd-cpp-apiv3 - 0.15.4

- **Repository URL**: https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3
- **License URL**: https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3/blob/master/LICENSE.txt
- **License name**: BSD-3-Clause license
### License Text:

```
Copyright (c) 2015, Nokia
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions
and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## cuda - 12.8

- **Repository URL**: https://developer.nvidia.com/cuda-12-8-0-download-archive
- **License URL**: https://docs.nvidia.com/cuda/eula/index.html
- **License name**: NVIDIA Proprietary license
### License Text:

```
End User License Agreement

NVIDIA Software License Agreement and CUDA Supplement to Software License Agreement.

The CUDA Toolkit End User License Agreement applies to the NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA Display Driver, NVIDIA Nsight tools (Visual Studio Edition), and the associated documentation on CUDA APIs, programming model and development tools. If you do not agree with the terms and conditions of the license agreement, then do not download or use the software.

Last updated: January 7, 2025

Preface

The Software License Agreement in Chapter 1 and the Supplement in Chapter 2 contain license terms and conditions that govern the use of NVIDIA CUDA toolkit. By accepting this agreement, you agree to comply with all the terms and conditions applicable to the product(s) included herein.

NVIDIA Driver

Description

This package contains the operating system driver and fundamental system software components for NVIDIA GPUs.

NVIDIA CUDA Toolkit

Description

The NVIDIA CUDA Toolkit provides command-line and graphical tools for building, debugging and optimizing the performance of applications accelerated by NVIDIA GPUs, runtime and math libraries, and documentation including programming guides, user manuals, and API references.

Default Install Location of CUDA Toolkit

Windows platform:

%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v#.#
Linux platform:

/usr/local/cuda-#.#
Mac platform:

/Developer/NVIDIA/CUDA-#.#
NVIDIA CUDA Samples

Description

CUDA Samples are now located in https://github.com/nvidia/cuda-samples, which includes instructions for obtaining, building, and running the samples. They are no longer included in the CUDA toolkit.

NVIDIA Nsight Visual Studio Edition (Windows only)

Description

NVIDIA Nsight Development Platform, Visual Studio Edition is a development environment integrated into Microsoft Visual Studio that provides tools for debugging, profiling, analyzing and optimizing your GPU computing and graphics applications.

Default Install Location of Nsight Visual Studio Edition

Windows platform:

%ProgramFiles(x86)%\NVIDIA Corporation\Nsight Visual Studio Edition #.#
1. License Agreement for NVIDIA Software Development Kits
Important Notice—Read before downloading, installing, copying or using the licensed software:

This license agreement, including exhibits attached (“Agreement”) is a legal agreement between you and NVIDIA Corporation (“NVIDIA”) and governs your use of a NVIDIA software development kit (“SDK”).

Each SDK has its own set of software and materials, but here is a description of the types of items that may be included in a SDK: source code, header files, APIs, data sets and assets (examples include images, textures, models, scenes, videos, native API input/output files), binary software, sample code, libraries, utility programs, programming code and documentation.

This Agreement can be accepted only by an adult of legal age of majority in the country in which the SDK is used.

If you are entering into this Agreement on behalf of a company or other legal entity, you represent that you have the legal authority to bind the entity to this Agreement, in which case “you” will mean the entity you represent.

If you don’t have the required age or authority to accept this Agreement, or if you don’t accept all the terms and conditions of this Agreement, do not download, install or use the SDK.

You agree to use the SDK only for purposes that are permitted by (a) this Agreement, and (b) any applicable law, regulation or generally accepted practices or guidelines in the relevant jurisdictions.

1.1. License
1.1.1. License Grant
Subject to the terms of this Agreement, NVIDIA hereby grants you a non-exclusive, non-transferable license, without the right to sublicense (except as expressly provided in this Agreement) to:

Install and use the SDK,

Modify and create derivative works of sample source code delivered in the SDK, and

Distribute those portions of the SDK that are identified in this Agreement as distributable, as incorporated in object code format into a software application that meets the distribution requirements indicated in this Agreement.

1.1.2. Distribution Requirements
These are the distribution requirements for you to exercise the distribution grant:

Your application must have material additional functionality, beyond the included portions of the SDK.

The distributable portions of the SDK shall only be accessed by your application.

The following notice shall be included in modifications and derivative works of sample source code distributed: “This software contains source code provided by NVIDIA Corporation.”

Unless a developer tool is identified in this Agreement as distributable, it is delivered for your internal use only.

The terms under which you distribute your application must be consistent with the terms of this Agreement, including (without limitation) terms relating to the license grant and license restrictions and protection of NVIDIA’s intellectual property rights. Additionally, you agree that you will protect the privacy, security and legal rights of your application users.

You agree to notify NVIDIA in writing of any known or suspected distribution or use of the SDK not in compliance with the requirements of this Agreement, and to enforce the terms of your agreements with respect to distributed SDK.

1.1.3. Authorized Users
You may allow employees and contractors of your entity or of your subsidiary(ies) to access and use the SDK from your secure network to perform work on your behalf.

If you are an academic institution you may allow users enrolled or employed by the academic institution to access and use the SDK from your secure network.

You are responsible for the compliance with the terms of this Agreement by your authorized users. If you become aware that your authorized users didn’t follow the terms of this Agreement, you agree to take reasonable steps to resolve the non-compliance and prevent new occurrences.

1.1.4. Pre-Release SDK
The SDK versions identified as alpha, beta, preview or otherwise as pre-release, may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, accessibility, availability, and reliability standards relative to commercial versions of NVIDIA software and materials. Use of a pre-release SDK may result in unexpected results, loss of data, project delays or other unpredictable damage or loss.

You may use a pre-release SDK at your own risk, understanding that pre-release SDKs are not intended for use in production or business-critical systems.

NVIDIA may choose not to make available a commercial version of any pre-release SDK. NVIDIA may also choose to abandon development and terminate the availability of a pre-release SDK at any time without liability.

1.1.5. Updates
NVIDIA may, at its option, make available patches, workarounds or other updates to this SDK. Unless the updates are provided with their separate governing terms, they are deemed part of the SDK licensed to you as provided in this Agreement. You agree that the form and content of the SDK that NVIDIA provides may change without prior notice to you. While NVIDIA generally maintains compatibility between versions, NVIDIA may in some cases make changes that introduce incompatibilities in future versions of the SDK.

1.1.6. Components Under Other Licenses
The SDK may come bundled with, or otherwise include or be distributed with, NVIDIA or third-party components with separate legal notices or terms as may be described in proprietary notices accompanying the SDK. If and to the extent there is a conflict between the terms in this Agreement and the license terms associated with the component, the license terms associated with the components control only to the extent necessary to resolve the conflict.

Subject to the other terms of this Agreement, you may use the SDK to develop and test applications released under Open Source Initiative (OSI) approved open source software licenses.

1.1.7. Reservation of Rights
NVIDIA reserves all rights, title, and interest in and to the SDK, not expressly granted to you under this Agreement.

1.2. Limitations
The following license limitations apply to your use of the SDK:

You may not reverse engineer, decompile or disassemble, or remove copyright or other proprietary notices from any portion of the SDK or copies of the SDK.

Except as expressly provided in this Agreement, you may not copy, sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SDK. For clarity, you may not distribute or sublicense the SDK as a stand-alone product.

Unless you have an agreement with NVIDIA for this purpose, you may not indicate that an application created with the SDK is sponsored or endorsed by NVIDIA.

You may not bypass, disable, or circumvent any encryption, security, digital rights management or authentication mechanism in the SDK.

You may not use the SDK in any manner that would cause it to become subject to an open source software license. As examples, licenses that require as a condition of use, modification, and/or distribution that the SDK be:

Disclosed or distributed in source code form;

Licensed for the purpose of making derivative works; or

Redistributable at no charge.

You acknowledge that the SDK as delivered is not tested or certified by NVIDIA for use in connection with the design, construction, maintenance, and/or operation of any system where the use or failure of such system could result in a situation that threatens the safety of human life or results in catastrophic damages (each, a “Critical Application”). Examples of Critical Applications include use in avionics, navigation, autonomous vehicle applications, ai solutions for automotive products, military, medical, life support or other life critical applications. NVIDIA shall not be liable to you or any third party, in whole or in part, for any claims or damages arising from such uses. You are solely responsible for ensuring that any product or service developed with the SDK as a whole includes sufficient features to comply with all applicable legal and regulatory standards and requirements.

You agree to defend, indemnify and hold harmless NVIDIA and its affiliates, and their respective employees, contractors, agents, officers and directors, from and against any and all claims, damages, obligations, losses, liabilities, costs or debt, fines, restitutions and expenses (including but not limited to attorney’s fees and costs incident to establishing the right of indemnification) arising out of or related to products or services that use the SDK in or for Critical Applications, and for use of the SDK outside of the scope of this Agreement or not in compliance with its terms.

You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform.

1.3. Ownership
NVIDIA or its licensors hold all rights, title and interest in and to the SDK and its modifications and derivative works, including their respective intellectual property rights, subject to your rights under Section 1.3.2. This SDK may include software and materials from NVIDIA’s licensors, and these licensors are intended third party beneficiaries that may enforce this Agreement with respect to their intellectual property rights.

You hold all rights, title and interest in and to your applications and your derivative works of the sample source code delivered in the SDK, including their respective intellectual property rights, subject to NVIDIA’s rights under Section 1.3.1.

You may, but don’t have to, provide to NVIDIA suggestions, feature requests or other feedback regarding the SDK, including possible enhancements or modifications to the SDK. For any feedback that you voluntarily provide, you hereby grant NVIDIA and its affiliates a perpetual, non-exclusive, worldwide, irrevocable license to use, reproduce, modify, license, sublicense (through multiple tiers of sublicensees), and distribute (through multiple tiers of distributors) it without the payment of any royalties or fees to you. NVIDIA will use feedback at its choice. NVIDIA is constantly looking for ways to improve its products, so you may send feedback to NVIDIA through the developer portal at https://developer.nvidia.com.

1.4. No Warranties
THE SDK IS PROVIDED BY NVIDIA “AS IS” AND “WITH ALL FAULTS.” TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES EXPRESSLY DISCLAIM ALL WARRANTIES OF ANY KIND OR NATURE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, OR THE ABSENCE OF ANY DEFECTS THEREIN, WHETHER LATENT OR PATENT. NO WARRANTY IS MADE ON THE BASIS OF TRADE USAGE, COURSE OF DEALING OR COURSE OF TRADE.

1.5. Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES SHALL NOT BE LIABLE FOR ANY (I) SPECIAL, INCIDENTAL, PUNITIVE OR CONSEQUENTIAL DAMAGES, OR (II) DAMAGES FOR (A) ANY LOST PROFITS, LOSS OF USE, LOSS OF DATA OR LOSS OF GOODWILL, OR (B) THE COSTS OF PROCURING SUBSTITUTE PRODUCTS, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT OR THE USE OR PERFORMANCE OF THE SDK, WHETHER SUCH LIABILITY ARISES FROM ANY CLAIM BASED UPON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER CAUSE OF ACTION OR THEORY OF LIABILITY. IN NO EVENT WILL NVIDIA’S AND ITS AFFILIATES TOTAL CUMULATIVE LIABILITY UNDER OR ARISING OUT OF THIS AGREEMENT EXCEED US$10.00. THE NATURE OF THE LIABILITY OR THE NUMBER OF CLAIMS OR SUITS SHALL NOT ENLARGE OR EXTEND THIS LIMIT.

These exclusions and limitations of liability shall apply regardless if NVIDIA or its affiliates have been advised of the possibility of such damages, and regardless of whether a remedy fails its essential purpose. These exclusions and limitations of liability form an essential basis of the bargain between the parties, and, absent any of these exclusions or limitations of liability, the provisions of this Agreement, including, without limitation, the economic terms, would be substantially different.

1.6. Termination
This Agreement will continue to apply until terminated by either you or NVIDIA as described below.

If you want to terminate this Agreement, you may do so by stopping to use the SDK.

NVIDIA may, at any time, terminate this Agreement if:

(i) you fail to comply with any term of this Agreement and the non-compliance is not fixed within thirty (30) days following notice from NVIDIA (or immediately if you violate NVIDIA’s intellectual property rights);

(ii) you commence or participate in any legal proceeding against NVIDIA with respect to the SDK; or

(iii) NVIDIA decides to no longer provide the SDK in a country or, in NVIDIA’s sole discretion, the continued use of it is no longer commercially viable.

Upon any termination of this Agreement, you agree to promptly discontinue use of the SDK and destroy all copies in your possession or control. Your prior distributions in accordance with this Agreement are not affected by the termination of this Agreement. Upon written request, you will certify in writing that you have complied with your commitments under this section. Upon any termination of this Agreement all provisions survive except for the license grant provisions.

1.7. General
If you wish to assign this Agreement or your rights and obligations, including by merger, consolidation, dissolution or operation of law, contact NVIDIA to ask for permission. Any attempted assignment not approved by NVIDIA in writing shall be void and of no effect. NVIDIA may assign, delegate or transfer this Agreement and its rights and obligations, and if to a non-affiliate you will be notified.

You agree to cooperate with NVIDIA and provide reasonably requested information to verify your compliance with this Agreement.

This Agreement will be governed in all respects by the laws of the United States and of the State of Delaware, without regard to the conflicts of laws principles. The United Nations Convention on Contracts for the International Sale of Goods is specifically disclaimed. You agree to all terms of this Agreement in the English language.

The state or federal courts residing in Santa Clara County, California shall have exclusive jurisdiction over any dispute or claim arising out of this Agreement. Notwithstanding this, you agree that NVIDIA shall still be allowed to apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

If any court of competent jurisdiction determines that any provision of this Agreement is illegal, invalid or unenforceable, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law and the remaining provisions will remain in full force and effect. Unless otherwise specified, remedies are cumulative.

Each party acknowledges and agrees that the other is an independent contractor in the performance of this Agreement.

The SDK has been developed entirely at private expense and is “commercial items” consisting of “commercial computer software” and “commercial computer software documentation” provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions in this Agreement pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (c)(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2788 San Tomas Expressway, Santa Clara, CA 95051.

The SDK is subject to United States export laws and regulations. You agree that you will not ship, transfer or export the SDK into any country, or use the SDK in any manner, prohibited by the United States Bureau of Industry and Security or economic sanctions regulations administered by the U.S. Department of Treasury’s Office of Foreign Assets Control (OFAC), or any applicable export laws, restrictions or regulations. These laws include restrictions on destinations, end users and end use. By accepting this Agreement, you confirm that you are not located in a country currently embargoed by the U.S. or otherwise prohibited from receiving the SDK under U.S. law.

Any notice delivered by NVIDIA to you under this Agreement will be delivered via mail, email or fax. You agree that any notices that NVIDIA sends you electronically will satisfy any legal communication requirements. Please direct your legal notices or other correspondence to NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, California 95051, United States of America, Attention: Legal Department.

This Agreement and any exhibits incorporated into this Agreement constitute the entire agreement of the parties with respect to the subject matter of this Agreement and supersede all prior negotiations or documentation exchanged between the parties relating to this SDK license. Any additional and/or conflicting terms on documents issued by you are null, void, and invalid. Any amendment or waiver under this Agreement shall be in writing and signed by representatives of both parties.

2. CUDA Toolkit Supplement to Software License Agreement for NVIDIA Software Development Kits
The terms in this supplement govern your use of the NVIDIA CUDA Toolkit SDK under the terms of your license agreement (“Agreement”) as modified by this supplement. Capitalized terms used but not defined below have the meaning assigned to them in the Agreement.

This supplement is an exhibit to the Agreement and is incorporated as an integral part of the Agreement. In the event of conflict between the terms in this supplement and the terms in the Agreement, the terms in this supplement govern.

2.1. License Scope
The SDK is licensed for you to develop applications only for use in systems with NVIDIA GPUs.

2.2. Distribution
The portions of the SDK that are distributable under the Agreement are listed in Attachment A.

2.3. Operating Systems
Those portions of the SDK designed exclusively for use on the Linux or FreeBSD operating systems, or other operating systems derived from the source code to these operating systems, may be copied and redistributed for use in accordance with this Agreement, provided that the object code files are not modified in any way (except for unzipping of compressed files).

2.4. Audio and Video Encoders and Decoders
You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale your products or services that include or incorporate any third-party software and content relating to audio and/or video encoders and decoders from, including but not limited to, Microsoft, Thomson, Fraunhofer IIS, Sisvel S.p.A., MPEG-LA, and Coding Technologies. NVIDIA does not grant to you under this Agreement any necessary patent or other rights with respect to any audio and/or video encoders and decoders.

2.5. Licensing
If the distribution terms in this Agreement are not suitable for your organization, or for any questions regarding this Agreement, please contact NVIDIA at nvidia-compute-license-questions@nvidia.com.

2.6. Attachment A
The following CUDA Toolkit files may be distributed with applications developed by you, including certain variations of these files that have version number or architecture specific information embedded in the file name - as an example only, for release version 9.0 of the 64-bit Windows software, the file cudart64_90.dll is redistributable.

Component

CUDA Runtime

Windows

cudart.dll, cudart_static.lib, cudadevrt.lib

Mac OSX

libcudart.dylib, libcudart_static.a, libcudadevrt.a

Linux

libcudart.so, libcudart_static.a, libcudadevrt.a

Android

libcudart.so, libcudart_static.a, libcudadevrt.a

Component

CUDA FFT Library

Windows

cufft.dll, cufftw.dll, cufft.lib, cufftw.lib

Mac OSX

libcufft.dylib, libcufft_static.a, libcufftw.dylib, libcufftw_static.a

Linux

libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a

Android

libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a

Component

CUDA BLAS Library

Windows

cublas.dll, cublasLt.dll

Mac OSX

libcublas.dylib, libcublasLt.dylib, libcublas_static.a, libcublasLt_static.a

Linux

libcublas.so, libcublasLt.so, libcublas_static.a, libcublasLt_static.a

Android

libcublas.so, libcublasLt.so, libcublas_static.a, libcublasLt_static.a

Component

NVIDIA “Drop-in” BLAS Library

Windows

nvblas.dll

Mac OSX

libnvblas.dylib

Linux

libnvblas.so

Component

CUDA Sparse Matrix Library

Windows

cusparse.dll, cusparse.lib

Mac OSX

libcusparse.dylib, libcusparse_static.a

Linux

libcusparse.so, libcusparse_static.a

Android

libcusparse.so, libcusparse_static.a

Component

CUDA Linear Solver Library

Windows

cusolver.dll, cusolver.lib

Mac OSX

libcusolver.dylib, libcusolver_static.a

Linux

libcusolver.so, libcusolver_static.a

Android

libcusolver.so, libcusolver_static.a

Component

CUDA Random Number Generation Library

Windows

curand.dll, curand.lib

Mac OSX

libcurand.dylib, libcurand_static.a

Linux

libcurand.so, libcurand_static.a

Android

libcurand.so, libcurand_static.a

Component

NVIDIA Performance Primitives Library

Windows

nppc.dll, nppc.lib, nppial.dll, nppial.lib, nppicc.dll, nppicc.lib, nppicom.dll, nppicom.lib, nppidei.dll, nppidei.lib, nppif.dll, nppif.lib, nppig.dll, nppig.lib, nppim.dll, nppim.lib, nppist.dll, nppist.lib, nppisu.dll, nppisu.lib, nppitc.dll, nppitc.lib, npps.dll, npps.lib

Mac OSX

libnppc.dylib, libnppc_static.a, libnppial.dylib, libnppial_static.a, libnppicc.dylib, libnppicc_static.a, libnppicom.dylib, libnppicom_static.a, libnppidei.dylib, libnppidei_static.a, libnppif.dylib, libnppif_static.a, libnppig.dylib, libnppig_static.a, libnppim.dylib, libnppisu_static.a, libnppitc.dylib, libnppitc_static.a, libnpps.dylib, libnpps_static.a

Linux

libnppc.so, libnppc_static.a, libnppial.so, libnppial_static.a, libnppicc.so, libnppicc_static.a, libnppicom.so, libnppicom_static.a, libnppidei.so, libnppidei_static.a, libnppif.so, libnppif_static.a libnppig.so, libnppig_static.a, libnppim.so, libnppim_static.a, libnppist.so, libnppist_static.a, libnppisu.so, libnppisu_static.a, libnppitc.so libnppitc_static.a, libnpps.so, libnpps_static.a

Android

libnppc.so, libnppc_static.a, libnppial.so, libnppial_static.a, libnppicc.so, libnppicc_static.a, libnppicom.so, libnppicom_static.a, libnppidei.so, libnppidei_static.a, libnppif.so, libnppif_static.a libnppig.so, libnppig_static.a, libnppim.so, libnppim_static.a, libnppist.so, libnppist_static.a, libnppisu.so, libnppisu_static.a, libnppitc.so libnppitc_static.a, libnpps.so, libnpps_static.a

Component

NVIDIA JPEG Library

Windows

nvjpeg.lib, nvjpeg.dll

Linux

libnvjpeg.so, libnvjpeg_static.a

Component

Internal common library required for statically linking to cuBLAS, cuSPARSE, cuFFT, cuRAND, nvJPEG and NPP

Mac OSX

libculibos.a

Linux

libculibos.a

Component

NVIDIA Runtime Compilation Library and Header

All

nvrtc.h

Windows

nvrtc.dll, nvrtc-builtins.dll

Mac OSX

libnvrtc.dylib, libnvrtc-builtins.dylib

Linux

libnvrtc.so, libnvrtc-builtins.so, libnvrtc_static.a, libnvrtc-builtins_static.a

Component

NVIDIA Optimizing Compiler Library

Windows

nvvm.dll

Mac OSX

libnvvm.dylib

Linux

libnvvm.so

Component

NVIDIA JIT Linking Library

Windows

libnvJitLink.dll, libnvJitLink.lib

Linux

libnvJitLink.so, libnvJitLink_static.a

Component

NVIDIA Common Device Math Functions Library

Windows

libdevice.10.bc

Mac OSX

libdevice.10.bc

Linux

libdevice.10.bc

Component

CUDA Occupancy Calculation Header Library

All

cuda_occupancy.h

Component

CUDA Floating Point Type Headers

All

cuda_fp16.h, cuda_fp16.hpp, cuda_bf16.h, cuda_bf16.hpp, cuda_fp8.h, cuda_fp8.hpp, cuda_fp6.h, cuda_fp6.hpp, cuda_fp4.h, cuda_fp4.hpp

Component

CUDA Headers for Runtime Compilation

All

crt/host_defines.h, cuComplex.h, cuda_awbarrier_helpers.h, cuda_awbarrier_primitives.h, cuda_wbarrier.h, cuda_pipeline_helpers.h, cuda_pipeline_primitives.h, cuda_pipeline.h, cuda_runtime_api.h, cuda.h, cuda/std/tuple, cuda/std/type_traits, cuda/std/type_traits, cuda/std/utility, device_types.h, vector_functions.h, vector_types.h

Component

CUDA Profiling Tools Interface (CUPTI) Library

Windows

cupti.dll

Mac OSX

libcupti.dylib

Linux

libcupti.so

Component

NVIDIA Tools Extension Library

Windows

nvToolsExt.dll, nvToolsExt.lib

Mac OSX

libnvToolsExt.dylib

Linux

libnvToolsExt.so

Component

NVIDIA CUDA Driver Libraries

Linux

libcuda.so, libnvidia-ptxjitcompiler.so, libnvptxcompiler_static.a

Component

NVIDIA CUDA File IO Libraries and Header

All

cufile.h

Linux

libcufile.so, libcufile_rdma.so, libcufile_static.a, libcufile_rdma_static.a

In addition to the rights above, for parties that are developing software intended solely for use on Jetson development kits or Jetson modules, and running Linux for Tegra software, the following shall apply:

The SDK may be distributed in its entirety, as provided by NVIDIA, and without separation of its components, for you and/or your licensees to create software development kits for use only on the Jetson platform and running Linux for Tegra software.

2.7. Attachment B
Additional Licensing Obligations

The following third party components included in the SOFTWARE are licensed to Licensee pursuant to the following terms and conditions:

Licensee’s use of the GDB third party component is subject to the terms and conditions of GNU GPL v3:

This product includes copyrighted third-party software licensed
under the terms of the GNU General Public License v3 ("GPL v3").
All third-party software packages are copyright by their respective
authors. GPL v3 terms and conditions are hereby incorporated into
the Agreement by this reference: http://www.gnu.org/licenses/gpl.txt
Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses. To obtain source code for software provided under licenses that require redistribution of source code, including the GNU General Public License (GPL) and GNU Lesser General Public License (LGPL), contact oss-requests@nvidia.com. This offer is valid for a period of three (3) years from the date of the distribution of this product by NVIDIA CORPORATION.

Component          License
CUDA-GDB           GPL v3
Licensee represents and warrants that any and all third party licensing and/or royalty payment obligations in connection with Licensee’s use of the H.264 video codecs are solely the responsibility of Licensee.

Licensee’s use of the Thrust library is subject to the terms and conditions of the Apache License Version 2.0. All third-party software packages are copyright by their respective authors. Apache License Version 2.0 terms and conditions are hereby incorporated into the Agreement by this reference. http://www.apache.org/licenses/LICENSE-2.0.html

In addition, Licensee acknowledges the following notice: Thrust includes source code from the Boost Iterator, Tuple, System, and Random Number libraries.

Boost Software License - Version 1.0 - August 17th, 2003
. . . .

Permission is hereby granted, free of charge, to any person or
organization obtaining a copy of the software and accompanying
documentation covered by this license (the "Software") to use,
reproduce, display, distribute, execute, and transmit the Software,
and to prepare derivative works of the Software, and to permit
third-parties to whom the Software is furnished to do so, all
subject to the following:

The copyright notices in the Software and this entire statement,
including the above license grant, this restriction and the following
disclaimer, must be included in all copies of the Software, in whole
or in part, and all derivative works of the Software, unless such
copies or derivative works are solely in the form of machine-executable
object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
Licensee’s use of the LLVM third party component is subject to the following terms and conditions:

  ========================================================================================================
  All LLVM after 8.0 are distributed under Apache-2.0 with LLVM-exception license, an OSI-approved license
  ========================================================================================================

  Apache License
  Version 2.0, January 2004
  http://www.apache.org/licenses/

  TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

  1. Definitions.

  "License" shall mean the terms and conditions for use, reproduction,
  and distribution as defined by Sections 1 through 9 of this document.

  "Licensor" shall mean the copyright owner or entity authorized by
  the copyright owner that is granting the License.

  "Legal Entity" shall mean the union of the acting entity and all
  other entities that control, are controlled by, or are under common
  control with that entity. For the purposes of this definition,
  "control" means (i) the power, direct or indirect, to cause the
  direction or management of such entity, whether by contract or
  otherwise, or (ii) ownership of fifty percent (50%) or more of the
  outstanding shares, or (iii) beneficial ownership of such entity.

  "You" (or "Your") shall mean an individual or Legal Entity
  exercising permissions granted by this License.

  "Source" form shall mean the preferred form for making modifications,
  including but not limited to software source code, documentation
  source, and configuration files.

  "Object" form shall mean any form resulting from mechanical
  transformation or translation of a Source form, including but
  not limited to compiled object code, generated documentation,
  and conversions to other media types.

  "Work" shall mean the work of authorship, whether in Source or
  Object form, made available under the License, as indicated by a
  copyright notice that is included in or attached to the work
  (an example is provided in the Appendix below).

  "Derivative Works" shall mean any work, whether in Source or Object
  form, that is based on (or derived from) the Work and for which the
  editorial revisions, annotations, elaborations, or other modifications
  represent, as a whole, an original work of authorship. For the purposes
  of this License, Derivative Works shall not include works that remain
  separable from, or merely link (or bind by name) to the interfaces of,
  the Work and Derivative Works thereof.

  "Contribution" shall mean any work of authorship, including
  the original version of the Work and any modifications or additions
  to that Work or Derivative Works thereof, that is intentionally
  submitted to Licensor for inclusion in the Work by the copyright owner
  or by an individual or Legal Entity authorized to submit on behalf of
  the copyright owner. For the purposes of this definition, "submitted"
  means any form of electronic, verbal, or written communication sent
  to the Licensor or its representatives, including but not limited to
  communication on electronic mailing lists, source code control systems,
  and issue tracking systems that are managed by, or on behalf of, the
  Licensor for the purpose of discussing and improving the Work, but
  excluding communication that is conspicuously marked or otherwise
  designated in writing by the copyright owner as "Not a Contribution."

  "Contributor" shall mean Licensor and any individual or Legal Entity
  on behalf of whom a Contribution has been received by Licensor and
  subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
  this License, each Contributor hereby grants to You a perpetual,
  worldwide, non-exclusive, no-charge, royalty-free, irrevocable
  copyright license to reproduce, prepare Derivative Works of,
  publicly display, publicly perform, sublicense, and distribute the
  Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
  this License, each Contributor hereby grants to You a perpetual,
  worldwide, non-exclusive, no-charge, royalty-free, irrevocable
  (except as stated in this section) patent license to make, have made,
  use, offer to sell, sell, import, and otherwise transfer the Work,
  where such license applies only to those patent claims licensable
  by such Contributor that are necessarily infringed by their
  Contribution(s) alone or by combination of their Contribution(s)
  with the Work to which such Contribution(s) was submitted. If You
  institute patent litigation against any entity (including a
  cross-claim or counterclaim in a lawsuit) alleging that the Work
  or a Contribution incorporated within the Work constitutes direct
  or contributory patent infringement, then any patent licenses
  granted to You under this License for that Work shall terminate
  as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
  Work or Derivative Works thereof in any medium, with or without
  modifications, and in Source or Object form, provided that You
  meet the following conditions:

  (a) You must give any other recipients of the Work or
      Derivative Works a copy of this License; and

  (b) You must cause any modified files to carry prominent notices
      stating that You changed the files; and

  (c) You must retain, in the Source form of any Derivative Works
      that You distribute, all copyright, patent, trademark, and
      attribution notices from the Source form of the Work,
      excluding those notices that do not pertain to any part of
      the Derivative Works; and
  (d) If the Work includes a "NOTICE" text file as part of its
      distribution, then any Derivative Works that You distribute must
      include a readable copy of the attribution notices contained
      within such NOTICE file, excluding those notices that do not
      pertain to any part of the Derivative Works, in at least one
      of the following places: within a NOTICE text file distributed
      as part of the Derivative Works; within the Source form or
      documentation, if provided along with the Derivative Works; or,
      within a display generated by the Derivative Works, if and
      wherever such third-party notices normally appear. The contents
      of the NOTICE file are for informational purposes only and
      do not modify the License. You may add Your own attribution
      notices within Derivative Works that You distribute, alongside
      or as an addendum to the NOTICE text from the Work, provided
      that such additional attribution notices cannot be construed
      as modifying the License.

  You may add Your own copyright statement to Your modifications and
  may provide additional or different license terms and conditions
  for use, reproduction, or distribution of Your modifications, or
  for any such Derivative Works as a whole, provided Your use,
  reproduction, and distribution of the Work otherwise complies with
  the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
  any Contribution intentionally submitted for inclusion in the Work
  by You to the Licensor shall be under the terms and conditions of
  this License, without any additional terms or conditions.
  Notwithstanding the above, nothing herein shall supersede or modify
  the terms of any separate license agreement you may have executed
  with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
  names, trademarks, service marks, or product names of the Licensor,
  except as required for reasonable and customary use in describing the
  origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
  agreed to in writing, Licensor provides the Work (and each
  Contributor provides its Contributions) on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied, including, without limitation, any warranties or conditions
  of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
  PARTICULAR PURPOSE. You are solely responsible for determining the
  appropriateness of using or redistributing the Work and assume any
  risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
  whether in tort (including negligence), contract, or otherwise,
  unless required by applicable law (such as deliberate and grossly
  negligent acts) or agreed to in writing, shall any Contributor be
  liable to You for damages, including any direct, indirect, special,
  incidental, or consequential damages of any character arising as a
  result of this License or out of the use or inability to use the
  Work (including but not limited to damages for loss of goodwill,
  work stoppage, computer failure or malfunction, or any and all
  other commercial damages or losses), even if such Contributor
  has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
  the Work or Derivative Works thereof, You may choose to offer,
  and charge a fee for, acceptance of support, warranty, indemnity,
  or other liability obligations and/or rights consistent with this
  License. However, in accepting such obligations, You may act only
  on Your own behalf and on Your sole responsibility, not on behalf
  of any other Contributor, and only if You agree to indemnify,
  defend, and hold each Contributor harmless for any liability
  incurred by, or claims asserted against, such Contributor by reason
  of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

  To apply the Apache License to your work, attach the following
  boilerplate notice, with the fields enclosed by brackets "[]"
  replaced with your own identifying information. (Don't include
  the brackets!)  The text should be enclosed in the appropriate
  comment syntax for the file format. We also recommend that a
  file or class name and description of purpose be included on the
  same "printed page" as the copyright notice for easier
  identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------
LLVM Exceptions to the Apache 2.0 License
-----------------------------------------

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
========================================================
Software from third parties included in the LLVM Project
========================================================

The LLVM Project contains third party software which is under different license
terms. All such code will be identified clearly using at least one of two
mechanisms:
1) It will be in a separate directory tree with its own `LICENSE.txt` or
`LICENSE` file at the top containing the specific license and restrictions
which apply to that software, or
2) It will contain specific license and restriction terms at the top of every
file.
==================================================================================================
LLVM releases prior to LLVM 8.0 was licensed under this University of Illinois Open Source License
==================================================================================================

University of Illinois/NCSA
Open Source License

Copyright (c) 2003-2019 University of Illinois at Urbana-Champaign.
All rights reserved.

Developed by:

    LLVM Team

    University of Illinois at Urbana-Champaign

    http://llvm.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal with
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimers in the
      documentation and/or other materials provided with the distribution.

   * Neither the names of the LLVM Team, University of Illinois at
     Urbana-Champaign, nor the names of its contributors may be used to
     endorse or promote products derived from this Software without specific
     prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
Licensee’s use of the PCRE third party component is subject to the following terms and conditions:

------------
PCRE LICENCE
------------
PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.
Release 8 of PCRE is distributed under the terms of the "BSD" licence, as
specified below. The documentation for PCRE, supplied in the "doc"
directory, is distributed under the same terms as the software itself. The
basic library functions are written in C and are freestanding. Also
included in the distribution is a set of C++ wrapper functions, and a just-
in-time compiler that can be used to optimize pattern matching. These are
both optional features that can be omitted when the library is built.

THE BASIC LIBRARY FUNCTIONS
---------------------------
Written by:       Philip Hazel
Email local part: ph10
Email domain:     cam.ac.uk
University of Cambridge Computing Service,
Cambridge, England.
Copyright (c) 1997-2012 University of Cambridge
All rights reserved.

PCRE JUST-IN-TIME COMPILATION SUPPORT
-------------------------------------
Written by:       Zoltan Herczeg
Email local part: hzmester
Emain domain:     freemail.hu
Copyright(c) 2010-2012 Zoltan Herczeg
All rights reserved.

STACK-LESS JUST-IN-TIME COMPILER
--------------------------------
Written by:       Zoltan Herczeg
Email local part: hzmester
Emain domain:     freemail.hu
Copyright(c) 2009-2012 Zoltan Herczeg
All rights reserved.

THE C++ WRAPPER FUNCTIONS
-------------------------
Contributed by:   Google Inc.
Copyright (c) 2007-2012, Google Inc.
All rights reserved.
THE "BSD" LICENCE
-----------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

  * Neither the name of the University of Cambridge nor the name of Google
    Inc. nor the names of their contributors may be used to endorse or
    promote products derived from this software without specific prior
    written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Vasily Volkov and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2007-2009, Regents of the University of California

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the University of California, Berkeley nor
      the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Davide Barbieri and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2008-2009 Davide Barbieri @ University of Rome Tor Vergata.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * The name of the author may not be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were derived from code developed by the University of Tennessee and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2010 The University of Tennessee.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer listed in this license in the documentation and/or
      other materials provided with the distribution.
    * Neither the name of the copyright holders nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Jonathan Hogg and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2012, The Science and Technology Facilities Council (STFC).

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the STFC nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE STFC BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Ahmad M. Abdelfattah, David Keyes, and Hatem Ltaief, and are subject to the Apache License, Version 2.0, as follows:

-- (C) Copyright 2013 King Abdullah University of Science and Technology
 Authors:
 Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
 David Keyes (david.keyes@kaust.edu.sa)
 Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

 Redistribution  and  use  in  source and binary forms, with or without
 modification,  are  permitted  provided  that the following conditions
 are met:

 * Redistributions  of  source  code  must  retain  the above copyright
   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * Redistributions  in  binary  form must reproduce the above copyright
   notice,  this list of conditions and the following disclaimer in the
   documentation  and/or other materials provided with the distribution.
 * Neither  the  name of the King Abdullah University of Science and
   Technology nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior
   written permission.

 THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
Some of the cuSPARSE library routines were written by or derived from code written by Li-Wen Chang and are subject to the NCSA Open Source License as follows:

Copyright (c) 2012, University of Illinois.

All rights reserved.

Developed by: IMPACT Group, University of Illinois, http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal with the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimers in the documentation and/or other materials provided
      with the distribution.
    * Neither the names of IMPACT Group, University of Illinois, nor
      the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior
      written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
Some of the cuRAND library routines were written by or derived from code written by Mutsuo Saito and Makoto Matsumoto and are subject to the following license:

Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
University. All rights reserved.

Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
University and University of Tokyo.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the Hiroshima University nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuRAND library routines were derived from code developed by D. E. Shaw Research and are subject to the following license:

Copyright 2010-2011, D. E. Shaw Research.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions, and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions, and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of D. E. Shaw Research nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the Math library routines were written by or derived from code developed by Norbert Juffa and are subject to the following license:

Copyright (c) 2015-2017, Norbert Juffa
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Licensee’s use of the lz4 third party component is subject to the following terms and conditions:

Copyright (C) 2011-2013, Yann Collet.
BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The NPP library uses code from the Boost Math Toolkit, and is subject to the following license:

Boost Software License - Version 1.0 - August 17th, 2003
. . . .

Permission is hereby granted, free of charge, to any person or
organization obtaining a copy of the software and accompanying
documentation covered by this license (the "Software") to use,
reproduce, display, distribute, execute, and transmit the Software,
and to prepare derivative works of the Software, and to permit
third-parties to whom the Software is furnished to do so, all
subject to the following:

The copyright notices in the Software and this entire statement,
including the above license grant, this restriction and the following
disclaimer, must be included in all copies of the Software, in whole
or in part, and all derivative works of the Software, unless such
copies or derivative works are solely in the form of machine-executable
object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
Portions of the Nsight Eclipse Edition is subject to the following license:

The Eclipse Foundation makes available all content in this plug-in
("Content"). Unless otherwise indicated below, the Content is provided
to you under the terms and conditions of the Eclipse Public License
Version 1.0 ("EPL"). A copy of the EPL is available at http://
www.eclipse.org/legal/epl-v10.html. For purposes of the EPL, "Program"
will mean the Content.

If you did not receive this Content directly from the Eclipse
Foundation, the Content is being redistributed by another party
("Redistributor") and different terms and conditions may apply to your
use of any object code in the Content. Check the Redistributor's
license that was provided with the Content. If no such license exists,
contact the Redistributor. Unless otherwise indicated below, the terms
and conditions of the EPL still apply to any source code in the
Content and such source code may be obtained at http://www.eclipse.org.
Some of the cuBLAS library routines uses code from OpenAI, which is subject to the following license:

License URL
https://github.com/openai/openai-gemm/blob/master/LICENSE

License Text
The MIT License

Copyright (c) 2016 OpenAI (http://openai.com), 2016 Google Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
Licensee’s use of the Visual Studio Setup Configuration Samples is subject to the following license:

The MIT License (MIT)
Copyright (C) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Licensee’s use of linmath.h header for CPU functions for GL vector/matrix operations from lunarG is subject to the Apache License Version 2.0.

The DX12-CUDA sample uses the d3dx12.h header, which is subject to the MIT license .

Components of the driver and compiler used for binary management, including nvFatBin, nvcc, and cuobjdump, use the Zstandard library which is subject to the following license:

BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name Facebook, nor Meta, nor the names of its contributors may
      be used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

## cudart - 12.8

- **Repository URL**: https://developer.nvidia.com/cuda-12-8-0-download-archive
- **License URL**: https://docs.nvidia.com/cuda/eula/index.html
- **License name**: NVIDIA Proprietary license
### License Text:

```
End User License Agreement

NVIDIA Software License Agreement and CUDA Supplement to Software License Agreement.

The CUDA Toolkit End User License Agreement applies to the NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA Display Driver, NVIDIA Nsight tools (Visual Studio Edition), and the associated documentation on CUDA APIs, programming model and development tools. If you do not agree with the terms and conditions of the license agreement, then do not download or use the software.

Last updated: January 7, 2025

Preface

The Software License Agreement in Chapter 1 and the Supplement in Chapter 2 contain license terms and conditions that govern the use of NVIDIA CUDA toolkit. By accepting this agreement, you agree to comply with all the terms and conditions applicable to the product(s) included herein.

NVIDIA Driver

Description

This package contains the operating system driver and fundamental system software components for NVIDIA GPUs.

NVIDIA CUDA Toolkit

Description

The NVIDIA CUDA Toolkit provides command-line and graphical tools for building, debugging and optimizing the performance of applications accelerated by NVIDIA GPUs, runtime and math libraries, and documentation including programming guides, user manuals, and API references.

Default Install Location of CUDA Toolkit

Windows platform:

%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v#.#
Linux platform:

/usr/local/cuda-#.#
Mac platform:

/Developer/NVIDIA/CUDA-#.#
NVIDIA CUDA Samples

Description

CUDA Samples are now located in https://github.com/nvidia/cuda-samples, which includes instructions for obtaining, building, and running the samples. They are no longer included in the CUDA toolkit.

NVIDIA Nsight Visual Studio Edition (Windows only)

Description

NVIDIA Nsight Development Platform, Visual Studio Edition is a development environment integrated into Microsoft Visual Studio that provides tools for debugging, profiling, analyzing and optimizing your GPU computing and graphics applications.

Default Install Location of Nsight Visual Studio Edition

Windows platform:

%ProgramFiles(x86)%\NVIDIA Corporation\Nsight Visual Studio Edition #.#
1. License Agreement for NVIDIA Software Development Kits
Important Notice—Read before downloading, installing, copying or using the licensed software:

This license agreement, including exhibits attached (“Agreement”) is a legal agreement between you and NVIDIA Corporation (“NVIDIA”) and governs your use of a NVIDIA software development kit (“SDK”).

Each SDK has its own set of software and materials, but here is a description of the types of items that may be included in a SDK: source code, header files, APIs, data sets and assets (examples include images, textures, models, scenes, videos, native API input/output files), binary software, sample code, libraries, utility programs, programming code and documentation.

This Agreement can be accepted only by an adult of legal age of majority in the country in which the SDK is used.

If you are entering into this Agreement on behalf of a company or other legal entity, you represent that you have the legal authority to bind the entity to this Agreement, in which case “you” will mean the entity you represent.

If you don’t have the required age or authority to accept this Agreement, or if you don’t accept all the terms and conditions of this Agreement, do not download, install or use the SDK.

You agree to use the SDK only for purposes that are permitted by (a) this Agreement, and (b) any applicable law, regulation or generally accepted practices or guidelines in the relevant jurisdictions.

1.1. License
1.1.1. License Grant
Subject to the terms of this Agreement, NVIDIA hereby grants you a non-exclusive, non-transferable license, without the right to sublicense (except as expressly provided in this Agreement) to:

Install and use the SDK,

Modify and create derivative works of sample source code delivered in the SDK, and

Distribute those portions of the SDK that are identified in this Agreement as distributable, as incorporated in object code format into a software application that meets the distribution requirements indicated in this Agreement.

1.1.2. Distribution Requirements
These are the distribution requirements for you to exercise the distribution grant:

Your application must have material additional functionality, beyond the included portions of the SDK.

The distributable portions of the SDK shall only be accessed by your application.

The following notice shall be included in modifications and derivative works of sample source code distributed: “This software contains source code provided by NVIDIA Corporation.”

Unless a developer tool is identified in this Agreement as distributable, it is delivered for your internal use only.

The terms under which you distribute your application must be consistent with the terms of this Agreement, including (without limitation) terms relating to the license grant and license restrictions and protection of NVIDIA’s intellectual property rights. Additionally, you agree that you will protect the privacy, security and legal rights of your application users.

You agree to notify NVIDIA in writing of any known or suspected distribution or use of the SDK not in compliance with the requirements of this Agreement, and to enforce the terms of your agreements with respect to distributed SDK.

1.1.3. Authorized Users
You may allow employees and contractors of your entity or of your subsidiary(ies) to access and use the SDK from your secure network to perform work on your behalf.

If you are an academic institution you may allow users enrolled or employed by the academic institution to access and use the SDK from your secure network.

You are responsible for the compliance with the terms of this Agreement by your authorized users. If you become aware that your authorized users didn’t follow the terms of this Agreement, you agree to take reasonable steps to resolve the non-compliance and prevent new occurrences.

1.1.4. Pre-Release SDK
The SDK versions identified as alpha, beta, preview or otherwise as pre-release, may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, accessibility, availability, and reliability standards relative to commercial versions of NVIDIA software and materials. Use of a pre-release SDK may result in unexpected results, loss of data, project delays or other unpredictable damage or loss.

You may use a pre-release SDK at your own risk, understanding that pre-release SDKs are not intended for use in production or business-critical systems.

NVIDIA may choose not to make available a commercial version of any pre-release SDK. NVIDIA may also choose to abandon development and terminate the availability of a pre-release SDK at any time without liability.

1.1.5. Updates
NVIDIA may, at its option, make available patches, workarounds or other updates to this SDK. Unless the updates are provided with their separate governing terms, they are deemed part of the SDK licensed to you as provided in this Agreement. You agree that the form and content of the SDK that NVIDIA provides may change without prior notice to you. While NVIDIA generally maintains compatibility between versions, NVIDIA may in some cases make changes that introduce incompatibilities in future versions of the SDK.

1.1.6. Components Under Other Licenses
The SDK may come bundled with, or otherwise include or be distributed with, NVIDIA or third-party components with separate legal notices or terms as may be described in proprietary notices accompanying the SDK. If and to the extent there is a conflict between the terms in this Agreement and the license terms associated with the component, the license terms associated with the components control only to the extent necessary to resolve the conflict.

Subject to the other terms of this Agreement, you may use the SDK to develop and test applications released under Open Source Initiative (OSI) approved open source software licenses.

1.1.7. Reservation of Rights
NVIDIA reserves all rights, title, and interest in and to the SDK, not expressly granted to you under this Agreement.

1.2. Limitations
The following license limitations apply to your use of the SDK:

You may not reverse engineer, decompile or disassemble, or remove copyright or other proprietary notices from any portion of the SDK or copies of the SDK.

Except as expressly provided in this Agreement, you may not copy, sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SDK. For clarity, you may not distribute or sublicense the SDK as a stand-alone product.

Unless you have an agreement with NVIDIA for this purpose, you may not indicate that an application created with the SDK is sponsored or endorsed by NVIDIA.

You may not bypass, disable, or circumvent any encryption, security, digital rights management or authentication mechanism in the SDK.

You may not use the SDK in any manner that would cause it to become subject to an open source software license. As examples, licenses that require as a condition of use, modification, and/or distribution that the SDK be:

Disclosed or distributed in source code form;

Licensed for the purpose of making derivative works; or

Redistributable at no charge.

You acknowledge that the SDK as delivered is not tested or certified by NVIDIA for use in connection with the design, construction, maintenance, and/or operation of any system where the use or failure of such system could result in a situation that threatens the safety of human life or results in catastrophic damages (each, a “Critical Application”). Examples of Critical Applications include use in avionics, navigation, autonomous vehicle applications, ai solutions for automotive products, military, medical, life support or other life critical applications. NVIDIA shall not be liable to you or any third party, in whole or in part, for any claims or damages arising from such uses. You are solely responsible for ensuring that any product or service developed with the SDK as a whole includes sufficient features to comply with all applicable legal and regulatory standards and requirements.

You agree to defend, indemnify and hold harmless NVIDIA and its affiliates, and their respective employees, contractors, agents, officers and directors, from and against any and all claims, damages, obligations, losses, liabilities, costs or debt, fines, restitutions and expenses (including but not limited to attorney’s fees and costs incident to establishing the right of indemnification) arising out of or related to products or services that use the SDK in or for Critical Applications, and for use of the SDK outside of the scope of this Agreement or not in compliance with its terms.

You may not reverse engineer, decompile or disassemble any portion of the output generated using SDK elements for the purpose of translating such output artifacts to target a non-NVIDIA platform.

1.3. Ownership
NVIDIA or its licensors hold all rights, title and interest in and to the SDK and its modifications and derivative works, including their respective intellectual property rights, subject to your rights under Section 1.3.2. This SDK may include software and materials from NVIDIA’s licensors, and these licensors are intended third party beneficiaries that may enforce this Agreement with respect to their intellectual property rights.

You hold all rights, title and interest in and to your applications and your derivative works of the sample source code delivered in the SDK, including their respective intellectual property rights, subject to NVIDIA’s rights under Section 1.3.1.

You may, but don’t have to, provide to NVIDIA suggestions, feature requests or other feedback regarding the SDK, including possible enhancements or modifications to the SDK. For any feedback that you voluntarily provide, you hereby grant NVIDIA and its affiliates a perpetual, non-exclusive, worldwide, irrevocable license to use, reproduce, modify, license, sublicense (through multiple tiers of sublicensees), and distribute (through multiple tiers of distributors) it without the payment of any royalties or fees to you. NVIDIA will use feedback at its choice. NVIDIA is constantly looking for ways to improve its products, so you may send feedback to NVIDIA through the developer portal at https://developer.nvidia.com.

1.4. No Warranties
THE SDK IS PROVIDED BY NVIDIA “AS IS” AND “WITH ALL FAULTS.” TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES EXPRESSLY DISCLAIM ALL WARRANTIES OF ANY KIND OR NATURE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, OR THE ABSENCE OF ANY DEFECTS THEREIN, WHETHER LATENT OR PATENT. NO WARRANTY IS MADE ON THE BASIS OF TRADE USAGE, COURSE OF DEALING OR COURSE OF TRADE.

1.5. Limitation of Liability
TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES SHALL NOT BE LIABLE FOR ANY (I) SPECIAL, INCIDENTAL, PUNITIVE OR CONSEQUENTIAL DAMAGES, OR (II) DAMAGES FOR (A) ANY LOST PROFITS, LOSS OF USE, LOSS OF DATA OR LOSS OF GOODWILL, OR (B) THE COSTS OF PROCURING SUBSTITUTE PRODUCTS, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT OR THE USE OR PERFORMANCE OF THE SDK, WHETHER SUCH LIABILITY ARISES FROM ANY CLAIM BASED UPON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER CAUSE OF ACTION OR THEORY OF LIABILITY. IN NO EVENT WILL NVIDIA’S AND ITS AFFILIATES TOTAL CUMULATIVE LIABILITY UNDER OR ARISING OUT OF THIS AGREEMENT EXCEED US$10.00. THE NATURE OF THE LIABILITY OR THE NUMBER OF CLAIMS OR SUITS SHALL NOT ENLARGE OR EXTEND THIS LIMIT.

These exclusions and limitations of liability shall apply regardless if NVIDIA or its affiliates have been advised of the possibility of such damages, and regardless of whether a remedy fails its essential purpose. These exclusions and limitations of liability form an essential basis of the bargain between the parties, and, absent any of these exclusions or limitations of liability, the provisions of this Agreement, including, without limitation, the economic terms, would be substantially different.

1.6. Termination
This Agreement will continue to apply until terminated by either you or NVIDIA as described below.

If you want to terminate this Agreement, you may do so by stopping to use the SDK.

NVIDIA may, at any time, terminate this Agreement if:

(i) you fail to comply with any term of this Agreement and the non-compliance is not fixed within thirty (30) days following notice from NVIDIA (or immediately if you violate NVIDIA’s intellectual property rights);

(ii) you commence or participate in any legal proceeding against NVIDIA with respect to the SDK; or

(iii) NVIDIA decides to no longer provide the SDK in a country or, in NVIDIA’s sole discretion, the continued use of it is no longer commercially viable.

Upon any termination of this Agreement, you agree to promptly discontinue use of the SDK and destroy all copies in your possession or control. Your prior distributions in accordance with this Agreement are not affected by the termination of this Agreement. Upon written request, you will certify in writing that you have complied with your commitments under this section. Upon any termination of this Agreement all provisions survive except for the license grant provisions.

1.7. General
If you wish to assign this Agreement or your rights and obligations, including by merger, consolidation, dissolution or operation of law, contact NVIDIA to ask for permission. Any attempted assignment not approved by NVIDIA in writing shall be void and of no effect. NVIDIA may assign, delegate or transfer this Agreement and its rights and obligations, and if to a non-affiliate you will be notified.

You agree to cooperate with NVIDIA and provide reasonably requested information to verify your compliance with this Agreement.

This Agreement will be governed in all respects by the laws of the United States and of the State of Delaware, without regard to the conflicts of laws principles. The United Nations Convention on Contracts for the International Sale of Goods is specifically disclaimed. You agree to all terms of this Agreement in the English language.

The state or federal courts residing in Santa Clara County, California shall have exclusive jurisdiction over any dispute or claim arising out of this Agreement. Notwithstanding this, you agree that NVIDIA shall still be allowed to apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

If any court of competent jurisdiction determines that any provision of this Agreement is illegal, invalid or unenforceable, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law and the remaining provisions will remain in full force and effect. Unless otherwise specified, remedies are cumulative.

Each party acknowledges and agrees that the other is an independent contractor in the performance of this Agreement.

The SDK has been developed entirely at private expense and is “commercial items” consisting of “commercial computer software” and “commercial computer software documentation” provided with RESTRICTED RIGHTS. Use, duplication or disclosure by the U.S. Government or a U.S. Government subcontractor is subject to the restrictions in this Agreement pursuant to DFARS 227.7202-3(a) or as set forth in subparagraphs (c)(1) and (2) of the Commercial Computer Software - Restricted Rights clause at FAR 52.227-19, as applicable. Contractor/manufacturer is NVIDIA, 2788 San Tomas Expressway, Santa Clara, CA 95051.

The SDK is subject to United States export laws and regulations. You agree that you will not ship, transfer or export the SDK into any country, or use the SDK in any manner, prohibited by the United States Bureau of Industry and Security or economic sanctions regulations administered by the U.S. Department of Treasury’s Office of Foreign Assets Control (OFAC), or any applicable export laws, restrictions or regulations. These laws include restrictions on destinations, end users and end use. By accepting this Agreement, you confirm that you are not located in a country currently embargoed by the U.S. or otherwise prohibited from receiving the SDK under U.S. law.

Any notice delivered by NVIDIA to you under this Agreement will be delivered via mail, email or fax. You agree that any notices that NVIDIA sends you electronically will satisfy any legal communication requirements. Please direct your legal notices or other correspondence to NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, California 95051, United States of America, Attention: Legal Department.

This Agreement and any exhibits incorporated into this Agreement constitute the entire agreement of the parties with respect to the subject matter of this Agreement and supersede all prior negotiations or documentation exchanged between the parties relating to this SDK license. Any additional and/or conflicting terms on documents issued by you are null, void, and invalid. Any amendment or waiver under this Agreement shall be in writing and signed by representatives of both parties.

2. CUDA Toolkit Supplement to Software License Agreement for NVIDIA Software Development Kits
The terms in this supplement govern your use of the NVIDIA CUDA Toolkit SDK under the terms of your license agreement (“Agreement”) as modified by this supplement. Capitalized terms used but not defined below have the meaning assigned to them in the Agreement.

This supplement is an exhibit to the Agreement and is incorporated as an integral part of the Agreement. In the event of conflict between the terms in this supplement and the terms in the Agreement, the terms in this supplement govern.

2.1. License Scope
The SDK is licensed for you to develop applications only for use in systems with NVIDIA GPUs.

2.2. Distribution
The portions of the SDK that are distributable under the Agreement are listed in Attachment A.

2.3. Operating Systems
Those portions of the SDK designed exclusively for use on the Linux or FreeBSD operating systems, or other operating systems derived from the source code to these operating systems, may be copied and redistributed for use in accordance with this Agreement, provided that the object code files are not modified in any way (except for unzipping of compressed files).

2.4. Audio and Video Encoders and Decoders
You acknowledge and agree that it is your sole responsibility to obtain any additional third-party licenses required to make, have made, use, have used, sell, import, and offer for sale your products or services that include or incorporate any third-party software and content relating to audio and/or video encoders and decoders from, including but not limited to, Microsoft, Thomson, Fraunhofer IIS, Sisvel S.p.A., MPEG-LA, and Coding Technologies. NVIDIA does not grant to you under this Agreement any necessary patent or other rights with respect to any audio and/or video encoders and decoders.

2.5. Licensing
If the distribution terms in this Agreement are not suitable for your organization, or for any questions regarding this Agreement, please contact NVIDIA at nvidia-compute-license-questions@nvidia.com.

2.6. Attachment A
The following CUDA Toolkit files may be distributed with applications developed by you, including certain variations of these files that have version number or architecture specific information embedded in the file name - as an example only, for release version 9.0 of the 64-bit Windows software, the file cudart64_90.dll is redistributable.

Component

CUDA Runtime

Windows

cudart.dll, cudart_static.lib, cudadevrt.lib

Mac OSX

libcudart.dylib, libcudart_static.a, libcudadevrt.a

Linux

libcudart.so, libcudart_static.a, libcudadevrt.a

Android

libcudart.so, libcudart_static.a, libcudadevrt.a

Component

CUDA FFT Library

Windows

cufft.dll, cufftw.dll, cufft.lib, cufftw.lib

Mac OSX

libcufft.dylib, libcufft_static.a, libcufftw.dylib, libcufftw_static.a

Linux

libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a

Android

libcufft.so, libcufft_static.a, libcufftw.so, libcufftw_static.a

Component

CUDA BLAS Library

Windows

cublas.dll, cublasLt.dll

Mac OSX

libcublas.dylib, libcublasLt.dylib, libcublas_static.a, libcublasLt_static.a

Linux

libcublas.so, libcublasLt.so, libcublas_static.a, libcublasLt_static.a

Android

libcublas.so, libcublasLt.so, libcublas_static.a, libcublasLt_static.a

Component

NVIDIA “Drop-in” BLAS Library

Windows

nvblas.dll

Mac OSX

libnvblas.dylib

Linux

libnvblas.so

Component

CUDA Sparse Matrix Library

Windows

cusparse.dll, cusparse.lib

Mac OSX

libcusparse.dylib, libcusparse_static.a

Linux

libcusparse.so, libcusparse_static.a

Android

libcusparse.so, libcusparse_static.a

Component

CUDA Linear Solver Library

Windows

cusolver.dll, cusolver.lib

Mac OSX

libcusolver.dylib, libcusolver_static.a

Linux

libcusolver.so, libcusolver_static.a

Android

libcusolver.so, libcusolver_static.a

Component

CUDA Random Number Generation Library

Windows

curand.dll, curand.lib

Mac OSX

libcurand.dylib, libcurand_static.a

Linux

libcurand.so, libcurand_static.a

Android

libcurand.so, libcurand_static.a

Component

NVIDIA Performance Primitives Library

Windows

nppc.dll, nppc.lib, nppial.dll, nppial.lib, nppicc.dll, nppicc.lib, nppicom.dll, nppicom.lib, nppidei.dll, nppidei.lib, nppif.dll, nppif.lib, nppig.dll, nppig.lib, nppim.dll, nppim.lib, nppist.dll, nppist.lib, nppisu.dll, nppisu.lib, nppitc.dll, nppitc.lib, npps.dll, npps.lib

Mac OSX

libnppc.dylib, libnppc_static.a, libnppial.dylib, libnppial_static.a, libnppicc.dylib, libnppicc_static.a, libnppicom.dylib, libnppicom_static.a, libnppidei.dylib, libnppidei_static.a, libnppif.dylib, libnppif_static.a, libnppig.dylib, libnppig_static.a, libnppim.dylib, libnppisu_static.a, libnppitc.dylib, libnppitc_static.a, libnpps.dylib, libnpps_static.a

Linux

libnppc.so, libnppc_static.a, libnppial.so, libnppial_static.a, libnppicc.so, libnppicc_static.a, libnppicom.so, libnppicom_static.a, libnppidei.so, libnppidei_static.a, libnppif.so, libnppif_static.a libnppig.so, libnppig_static.a, libnppim.so, libnppim_static.a, libnppist.so, libnppist_static.a, libnppisu.so, libnppisu_static.a, libnppitc.so libnppitc_static.a, libnpps.so, libnpps_static.a

Android

libnppc.so, libnppc_static.a, libnppial.so, libnppial_static.a, libnppicc.so, libnppicc_static.a, libnppicom.so, libnppicom_static.a, libnppidei.so, libnppidei_static.a, libnppif.so, libnppif_static.a libnppig.so, libnppig_static.a, libnppim.so, libnppim_static.a, libnppist.so, libnppist_static.a, libnppisu.so, libnppisu_static.a, libnppitc.so libnppitc_static.a, libnpps.so, libnpps_static.a

Component

NVIDIA JPEG Library

Windows

nvjpeg.lib, nvjpeg.dll

Linux

libnvjpeg.so, libnvjpeg_static.a

Component

Internal common library required for statically linking to cuBLAS, cuSPARSE, cuFFT, cuRAND, nvJPEG and NPP

Mac OSX

libculibos.a

Linux

libculibos.a

Component

NVIDIA Runtime Compilation Library and Header

All

nvrtc.h

Windows

nvrtc.dll, nvrtc-builtins.dll

Mac OSX

libnvrtc.dylib, libnvrtc-builtins.dylib

Linux

libnvrtc.so, libnvrtc-builtins.so, libnvrtc_static.a, libnvrtc-builtins_static.a

Component

NVIDIA Optimizing Compiler Library

Windows

nvvm.dll

Mac OSX

libnvvm.dylib

Linux

libnvvm.so

Component

NVIDIA JIT Linking Library

Windows

libnvJitLink.dll, libnvJitLink.lib

Linux

libnvJitLink.so, libnvJitLink_static.a

Component

NVIDIA Common Device Math Functions Library

Windows

libdevice.10.bc

Mac OSX

libdevice.10.bc

Linux

libdevice.10.bc

Component

CUDA Occupancy Calculation Header Library

All

cuda_occupancy.h

Component

CUDA Floating Point Type Headers

All

cuda_fp16.h, cuda_fp16.hpp, cuda_bf16.h, cuda_bf16.hpp, cuda_fp8.h, cuda_fp8.hpp, cuda_fp6.h, cuda_fp6.hpp, cuda_fp4.h, cuda_fp4.hpp

Component

CUDA Headers for Runtime Compilation

All

crt/host_defines.h, cuComplex.h, cuda_awbarrier_helpers.h, cuda_awbarrier_primitives.h, cuda_wbarrier.h, cuda_pipeline_helpers.h, cuda_pipeline_primitives.h, cuda_pipeline.h, cuda_runtime_api.h, cuda.h, cuda/std/tuple, cuda/std/type_traits, cuda/std/type_traits, cuda/std/utility, device_types.h, vector_functions.h, vector_types.h

Component

CUDA Profiling Tools Interface (CUPTI) Library

Windows

cupti.dll

Mac OSX

libcupti.dylib

Linux

libcupti.so

Component

NVIDIA Tools Extension Library

Windows

nvToolsExt.dll, nvToolsExt.lib

Mac OSX

libnvToolsExt.dylib

Linux

libnvToolsExt.so

Component

NVIDIA CUDA Driver Libraries

Linux

libcuda.so, libnvidia-ptxjitcompiler.so, libnvptxcompiler_static.a

Component

NVIDIA CUDA File IO Libraries and Header

All

cufile.h

Linux

libcufile.so, libcufile_rdma.so, libcufile_static.a, libcufile_rdma_static.a

In addition to the rights above, for parties that are developing software intended solely for use on Jetson development kits or Jetson modules, and running Linux for Tegra software, the following shall apply:

The SDK may be distributed in its entirety, as provided by NVIDIA, and without separation of its components, for you and/or your licensees to create software development kits for use only on the Jetson platform and running Linux for Tegra software.

2.7. Attachment B
Additional Licensing Obligations

The following third party components included in the SOFTWARE are licensed to Licensee pursuant to the following terms and conditions:

Licensee’s use of the GDB third party component is subject to the terms and conditions of GNU GPL v3:

This product includes copyrighted third-party software licensed
under the terms of the GNU General Public License v3 ("GPL v3").
All third-party software packages are copyright by their respective
authors. GPL v3 terms and conditions are hereby incorporated into
the Agreement by this reference: http://www.gnu.org/licenses/gpl.txt
Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses. To obtain source code for software provided under licenses that require redistribution of source code, including the GNU General Public License (GPL) and GNU Lesser General Public License (LGPL), contact oss-requests@nvidia.com. This offer is valid for a period of three (3) years from the date of the distribution of this product by NVIDIA CORPORATION.

Component          License
CUDA-GDB           GPL v3
Licensee represents and warrants that any and all third party licensing and/or royalty payment obligations in connection with Licensee’s use of the H.264 video codecs are solely the responsibility of Licensee.

Licensee’s use of the Thrust library is subject to the terms and conditions of the Apache License Version 2.0. All third-party software packages are copyright by their respective authors. Apache License Version 2.0 terms and conditions are hereby incorporated into the Agreement by this reference. http://www.apache.org/licenses/LICENSE-2.0.html

In addition, Licensee acknowledges the following notice: Thrust includes source code from the Boost Iterator, Tuple, System, and Random Number libraries.

Boost Software License - Version 1.0 - August 17th, 2003
. . . .

Permission is hereby granted, free of charge, to any person or
organization obtaining a copy of the software and accompanying
documentation covered by this license (the "Software") to use,
reproduce, display, distribute, execute, and transmit the Software,
and to prepare derivative works of the Software, and to permit
third-parties to whom the Software is furnished to do so, all
subject to the following:

The copyright notices in the Software and this entire statement,
including the above license grant, this restriction and the following
disclaimer, must be included in all copies of the Software, in whole
or in part, and all derivative works of the Software, unless such
copies or derivative works are solely in the form of machine-executable
object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
Licensee’s use of the LLVM third party component is subject to the following terms and conditions:

  ========================================================================================================
  All LLVM after 8.0 are distributed under Apache-2.0 with LLVM-exception license, an OSI-approved license
  ========================================================================================================

  Apache License
  Version 2.0, January 2004
  http://www.apache.org/licenses/

  TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

  1. Definitions.

  "License" shall mean the terms and conditions for use, reproduction,
  and distribution as defined by Sections 1 through 9 of this document.

  "Licensor" shall mean the copyright owner or entity authorized by
  the copyright owner that is granting the License.

  "Legal Entity" shall mean the union of the acting entity and all
  other entities that control, are controlled by, or are under common
  control with that entity. For the purposes of this definition,
  "control" means (i) the power, direct or indirect, to cause the
  direction or management of such entity, whether by contract or
  otherwise, or (ii) ownership of fifty percent (50%) or more of the
  outstanding shares, or (iii) beneficial ownership of such entity.

  "You" (or "Your") shall mean an individual or Legal Entity
  exercising permissions granted by this License.

  "Source" form shall mean the preferred form for making modifications,
  including but not limited to software source code, documentation
  source, and configuration files.

  "Object" form shall mean any form resulting from mechanical
  transformation or translation of a Source form, including but
  not limited to compiled object code, generated documentation,
  and conversions to other media types.

  "Work" shall mean the work of authorship, whether in Source or
  Object form, made available under the License, as indicated by a
  copyright notice that is included in or attached to the work
  (an example is provided in the Appendix below).

  "Derivative Works" shall mean any work, whether in Source or Object
  form, that is based on (or derived from) the Work and for which the
  editorial revisions, annotations, elaborations, or other modifications
  represent, as a whole, an original work of authorship. For the purposes
  of this License, Derivative Works shall not include works that remain
  separable from, or merely link (or bind by name) to the interfaces of,
  the Work and Derivative Works thereof.

  "Contribution" shall mean any work of authorship, including
  the original version of the Work and any modifications or additions
  to that Work or Derivative Works thereof, that is intentionally
  submitted to Licensor for inclusion in the Work by the copyright owner
  or by an individual or Legal Entity authorized to submit on behalf of
  the copyright owner. For the purposes of this definition, "submitted"
  means any form of electronic, verbal, or written communication sent
  to the Licensor or its representatives, including but not limited to
  communication on electronic mailing lists, source code control systems,
  and issue tracking systems that are managed by, or on behalf of, the
  Licensor for the purpose of discussing and improving the Work, but
  excluding communication that is conspicuously marked or otherwise
  designated in writing by the copyright owner as "Not a Contribution."

  "Contributor" shall mean Licensor and any individual or Legal Entity
  on behalf of whom a Contribution has been received by Licensor and
  subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
  this License, each Contributor hereby grants to You a perpetual,
  worldwide, non-exclusive, no-charge, royalty-free, irrevocable
  copyright license to reproduce, prepare Derivative Works of,
  publicly display, publicly perform, sublicense, and distribute the
  Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
  this License, each Contributor hereby grants to You a perpetual,
  worldwide, non-exclusive, no-charge, royalty-free, irrevocable
  (except as stated in this section) patent license to make, have made,
  use, offer to sell, sell, import, and otherwise transfer the Work,
  where such license applies only to those patent claims licensable
  by such Contributor that are necessarily infringed by their
  Contribution(s) alone or by combination of their Contribution(s)
  with the Work to which such Contribution(s) was submitted. If You
  institute patent litigation against any entity (including a
  cross-claim or counterclaim in a lawsuit) alleging that the Work
  or a Contribution incorporated within the Work constitutes direct
  or contributory patent infringement, then any patent licenses
  granted to You under this License for that Work shall terminate
  as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
  Work or Derivative Works thereof in any medium, with or without
  modifications, and in Source or Object form, provided that You
  meet the following conditions:

  (a) You must give any other recipients of the Work or
      Derivative Works a copy of this License; and

  (b) You must cause any modified files to carry prominent notices
      stating that You changed the files; and

  (c) You must retain, in the Source form of any Derivative Works
      that You distribute, all copyright, patent, trademark, and
      attribution notices from the Source form of the Work,
      excluding those notices that do not pertain to any part of
      the Derivative Works; and
  (d) If the Work includes a "NOTICE" text file as part of its
      distribution, then any Derivative Works that You distribute must
      include a readable copy of the attribution notices contained
      within such NOTICE file, excluding those notices that do not
      pertain to any part of the Derivative Works, in at least one
      of the following places: within a NOTICE text file distributed
      as part of the Derivative Works; within the Source form or
      documentation, if provided along with the Derivative Works; or,
      within a display generated by the Derivative Works, if and
      wherever such third-party notices normally appear. The contents
      of the NOTICE file are for informational purposes only and
      do not modify the License. You may add Your own attribution
      notices within Derivative Works that You distribute, alongside
      or as an addendum to the NOTICE text from the Work, provided
      that such additional attribution notices cannot be construed
      as modifying the License.

  You may add Your own copyright statement to Your modifications and
  may provide additional or different license terms and conditions
  for use, reproduction, or distribution of Your modifications, or
  for any such Derivative Works as a whole, provided Your use,
  reproduction, and distribution of the Work otherwise complies with
  the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
  any Contribution intentionally submitted for inclusion in the Work
  by You to the Licensor shall be under the terms and conditions of
  this License, without any additional terms or conditions.
  Notwithstanding the above, nothing herein shall supersede or modify
  the terms of any separate license agreement you may have executed
  with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
  names, trademarks, service marks, or product names of the Licensor,
  except as required for reasonable and customary use in describing the
  origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
  agreed to in writing, Licensor provides the Work (and each
  Contributor provides its Contributions) on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
  implied, including, without limitation, any warranties or conditions
  of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
  PARTICULAR PURPOSE. You are solely responsible for determining the
  appropriateness of using or redistributing the Work and assume any
  risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
  whether in tort (including negligence), contract, or otherwise,
  unless required by applicable law (such as deliberate and grossly
  negligent acts) or agreed to in writing, shall any Contributor be
  liable to You for damages, including any direct, indirect, special,
  incidental, or consequential damages of any character arising as a
  result of this License or out of the use or inability to use the
  Work (including but not limited to damages for loss of goodwill,
  work stoppage, computer failure or malfunction, or any and all
  other commercial damages or losses), even if such Contributor
  has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
  the Work or Derivative Works thereof, You may choose to offer,
  and charge a fee for, acceptance of support, warranty, indemnity,
  or other liability obligations and/or rights consistent with this
  License. However, in accepting such obligations, You may act only
  on Your own behalf and on Your sole responsibility, not on behalf
  of any other Contributor, and only if You agree to indemnify,
  defend, and hold each Contributor harmless for any liability
  incurred by, or claims asserted against, such Contributor by reason
  of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

  To apply the Apache License to your work, attach the following
  boilerplate notice, with the fields enclosed by brackets "[]"
  replaced with your own identifying information. (Don't include
  the brackets!)  The text should be enclosed in the appropriate
  comment syntax for the file format. We also recommend that a
  file or class name and description of purpose be included on the
  same "printed page" as the copyright notice for easier
  identification within third-party archives.

Copyright [yyyy] [name of copyright owner]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-----------------------------------------
LLVM Exceptions to the Apache 2.0 License
-----------------------------------------

As an exception, if, as a result of your compiling your source code, portions
of this Software are embedded into an Object form of such source code, you
may redistribute such embedded portions in such Object form without complying
with the conditions of Sections 4(a), 4(b) and 4(d) of the License.

In addition, if you combine or link compiled forms of this Software with
software that is licensed under the GPLv2 ("Combined Software") and if a
court of competent jurisdiction determines that the patent provision (Section
3), the indemnity provision (Section 9) or other Section of the License
conflicts with the conditions of the GPLv2, you may retroactively and
prospectively choose to deem waived or otherwise exclude such Section(s) of
the License, but only in their entirety and only with respect to the Combined
Software.
========================================================
Software from third parties included in the LLVM Project
========================================================

The LLVM Project contains third party software which is under different license
terms. All such code will be identified clearly using at least one of two
mechanisms:
1) It will be in a separate directory tree with its own `LICENSE.txt` or
`LICENSE` file at the top containing the specific license and restrictions
which apply to that software, or
2) It will contain specific license and restriction terms at the top of every
file.
==================================================================================================
LLVM releases prior to LLVM 8.0 was licensed under this University of Illinois Open Source License
==================================================================================================

University of Illinois/NCSA
Open Source License

Copyright (c) 2003-2019 University of Illinois at Urbana-Champaign.
All rights reserved.

Developed by:

    LLVM Team

    University of Illinois at Urbana-Champaign

    http://llvm.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal with
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimers in the
      documentation and/or other materials provided with the distribution.

   * Neither the names of the LLVM Team, University of Illinois at
     Urbana-Champaign, nor the names of its contributors may be used to
     endorse or promote products derived from this Software without specific
     prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
Licensee’s use of the PCRE third party component is subject to the following terms and conditions:

------------
PCRE LICENCE
------------
PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.
Release 8 of PCRE is distributed under the terms of the "BSD" licence, as
specified below. The documentation for PCRE, supplied in the "doc"
directory, is distributed under the same terms as the software itself. The
basic library functions are written in C and are freestanding. Also
included in the distribution is a set of C++ wrapper functions, and a just-
in-time compiler that can be used to optimize pattern matching. These are
both optional features that can be omitted when the library is built.

THE BASIC LIBRARY FUNCTIONS
---------------------------
Written by:       Philip Hazel
Email local part: ph10
Email domain:     cam.ac.uk
University of Cambridge Computing Service,
Cambridge, England.
Copyright (c) 1997-2012 University of Cambridge
All rights reserved.

PCRE JUST-IN-TIME COMPILATION SUPPORT
-------------------------------------
Written by:       Zoltan Herczeg
Email local part: hzmester
Emain domain:     freemail.hu
Copyright(c) 2010-2012 Zoltan Herczeg
All rights reserved.

STACK-LESS JUST-IN-TIME COMPILER
--------------------------------
Written by:       Zoltan Herczeg
Email local part: hzmester
Emain domain:     freemail.hu
Copyright(c) 2009-2012 Zoltan Herczeg
All rights reserved.

THE C++ WRAPPER FUNCTIONS
-------------------------
Contributed by:   Google Inc.
Copyright (c) 2007-2012, Google Inc.
All rights reserved.
THE "BSD" LICENCE
-----------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

  * Neither the name of the University of Cambridge nor the name of Google
    Inc. nor the names of their contributors may be used to endorse or
    promote products derived from this software without specific prior
    written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Vasily Volkov and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2007-2009, Regents of the University of California

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the University of California, Berkeley nor
      the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Davide Barbieri and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2008-2009 Davide Barbieri @ University of Rome Tor Vergata.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * The name of the author may not be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were derived from code developed by the University of Tennessee and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2010 The University of Tennessee.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer listed in this license in the documentation and/or
      other materials provided with the distribution.
    * Neither the name of the copyright holders nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Jonathan Hogg and are subject to the Modified Berkeley Software Distribution License as follows:

Copyright (c) 2012, The Science and Technology Facilities Council (STFC).

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the STFC nor the names of its contributors
      may be used to endorse or promote products derived from this
      software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE STFC BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN
IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuBLAS library routines were written by or derived from code written by Ahmad M. Abdelfattah, David Keyes, and Hatem Ltaief, and are subject to the Apache License, Version 2.0, as follows:

-- (C) Copyright 2013 King Abdullah University of Science and Technology
 Authors:
 Ahmad Abdelfattah (ahmad.ahmad@kaust.edu.sa)
 David Keyes (david.keyes@kaust.edu.sa)
 Hatem Ltaief (hatem.ltaief@kaust.edu.sa)

 Redistribution  and  use  in  source and binary forms, with or without
 modification,  are  permitted  provided  that the following conditions
 are met:

 * Redistributions  of  source  code  must  retain  the above copyright
   notice,  this  list  of  conditions  and  the  following  disclaimer.
 * Redistributions  in  binary  form must reproduce the above copyright
   notice,  this list of conditions and the following disclaimer in the
   documentation  and/or other materials provided with the distribution.
 * Neither  the  name of the King Abdullah University of Science and
   Technology nor the names of its contributors may be used to endorse
   or promote products derived from this software without specific prior
   written permission.

 THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
Some of the cuSPARSE library routines were written by or derived from code written by Li-Wen Chang and are subject to the NCSA Open Source License as follows:

Copyright (c) 2012, University of Illinois.

All rights reserved.

Developed by: IMPACT Group, University of Illinois, http://impact.crhc.illinois.edu

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal with the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimers in the documentation and/or other materials provided
      with the distribution.
    * Neither the names of IMPACT Group, University of Illinois, nor
      the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior
      written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE
SOFTWARE.
Some of the cuRAND library routines were written by or derived from code written by Mutsuo Saito and Makoto Matsumoto and are subject to the following license:

Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
University. All rights reserved.

Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
University and University of Tokyo.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the Hiroshima University nor the names of
      its contributors may be used to endorse or promote products
      derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the cuRAND library routines were derived from code developed by D. E. Shaw Research and are subject to the following license:

Copyright 2010-2011, D. E. Shaw Research.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions, and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions, and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of D. E. Shaw Research nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Some of the Math library routines were written by or derived from code developed by Norbert Juffa and are subject to the following license:

Copyright (c) 2015-2017, Norbert Juffa
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Licensee’s use of the lz4 third party component is subject to the following terms and conditions:

Copyright (C) 2011-2013, Yann Collet.
BSD 2-Clause License (http://www.opensource.org/licenses/bsd-license.php)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
The NPP library uses code from the Boost Math Toolkit, and is subject to the following license:

Boost Software License - Version 1.0 - August 17th, 2003
. . . .

Permission is hereby granted, free of charge, to any person or
organization obtaining a copy of the software and accompanying
documentation covered by this license (the "Software") to use,
reproduce, display, distribute, execute, and transmit the Software,
and to prepare derivative works of the Software, and to permit
third-parties to whom the Software is furnished to do so, all
subject to the following:

The copyright notices in the Software and this entire statement,
including the above license grant, this restriction and the following
disclaimer, must be included in all copies of the Software, in whole
or in part, and all derivative works of the Software, unless such
copies or derivative works are solely in the form of machine-executable
object code generated by a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR
ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR
OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
Portions of the Nsight Eclipse Edition is subject to the following license:

The Eclipse Foundation makes available all content in this plug-in
("Content"). Unless otherwise indicated below, the Content is provided
to you under the terms and conditions of the Eclipse Public License
Version 1.0 ("EPL"). A copy of the EPL is available at http://
www.eclipse.org/legal/epl-v10.html. For purposes of the EPL, "Program"
will mean the Content.

If you did not receive this Content directly from the Eclipse
Foundation, the Content is being redistributed by another party
("Redistributor") and different terms and conditions may apply to your
use of any object code in the Content. Check the Redistributor's
license that was provided with the Content. If no such license exists,
contact the Redistributor. Unless otherwise indicated below, the terms
and conditions of the EPL still apply to any source code in the
Content and such source code may be obtained at http://www.eclipse.org.
Some of the cuBLAS library routines uses code from OpenAI, which is subject to the following license:

License URL
https://github.com/openai/openai-gemm/blob/master/LICENSE

License Text
The MIT License

Copyright (c) 2016 OpenAI (http://openai.com), 2016 Google Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
Licensee’s use of the Visual Studio Setup Configuration Samples is subject to the following license:

The MIT License (MIT)
Copyright (C) Microsoft Corporation. All rights reserved.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
Licensee’s use of linmath.h header for CPU functions for GL vector/matrix operations from lunarG is subject to the Apache License Version 2.0.

The DX12-CUDA sample uses the d3dx12.h header, which is subject to the MIT license .

Components of the driver and compiler used for binary management, including nvFatBin, nvcc, and cuobjdump, use the Zstandard library which is subject to the following license:

BSD License

For Zstandard software

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name Facebook, nor Meta, nor the names of its contributors may
      be used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
```

## 3FS - 0.1.5

- **Repository URL**: https://github.com/deepseek-ai/3FS
- **License URL**: https://github.com/deepseek-ai/3FS/blob/main/LICENSE
- **License name**: MIT

### License Text:
```
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Abseil - 20250512.0

- **Repository URL**: https://github.com/abseil/abseil-cpp
- **License URL**: https://github.com/abseil/abseil-cpp/blob/master/LICENSE
- **License name**: Apache 2.0
### License Text:
```

                                 Apache License
                           Version 2.0, January 2004
                        https://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       https://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## libaio - 0.4.0

- **Repository URL**: https://pagure.io/libaio
- **License URL**: https://pagure.io/libaio/blob/master/f/COPYING
- **License name**: LGPL 2.1
### License Text:
```

                  GNU LESSER GENERAL PUBLIC LICENSE
                       Version 2.1, February 1999

 Copyright (C) 1991, 1999 Free Software Foundation, Inc.
     59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

[This is the first released version of the Lesser GPL.  It also counts
 as the successor of the GNU Library Public License, version 2, hence
 the version number 2.1.]

                            Preamble

  The licenses for most software are designed to take away your
freedom to share and change it.  By contrast, the GNU General Public
Licenses are intended to guarantee your freedom to share and change
free software--to make sure the software is free for all its users.

  This license, the Lesser General Public License, applies to some
specially designated software packages--typically libraries--of the
Free Software Foundation and other authors who decide to use it.  You
can use it too, but we suggest you first think carefully about whether
this license or the ordinary General Public License is the better
strategy to use in any particular case, based on the explanations
below.

  When we speak of free software, we are referring to freedom of use,
not price.  Our General Public Licenses are designed to make sure that
you have the freedom to distribute copies of free software (and charge
for this service if you wish); that you receive source code or can get
it if you want it; that you can change the software and use pieces of
it in new free programs; and that you are informed that you can do
these things.

  To protect your rights, we need to make restrictions that forbid
distributors to deny you these rights or to ask you to surrender these
rights.  These restrictions translate to certain responsibilities for
you if you distribute copies of the library or if you modify it.

  For example, if you distribute copies of the library, whether gratis
or for a fee, you must give the recipients all the rights that we gave
you.  You must make sure that they, too, receive or can get the source
code.  If you link other code with the library, you must provide
complete object files to the recipients, so that they can relink them
with the library after making changes to the library and recompiling
it.  And you must show them these terms so they know their rights.

  We protect your rights with a two-step method: (1) we copyright the
library, and (2) we offer you this license, which gives you legal
permission to copy, distribute and/or modify the library.

  To protect each distributor, we want to make it very clear that
there is no warranty for the free library.  Also, if the library is
modified by someone else and passed on, the recipients should know
that what they have is not the original version, so that the original
author's reputation will not be affected by problems that might be
introduced by others.
^L
  Finally, software patents pose a constant threat to the existence of
any free program.  We wish to make sure that a company cannot
effectively restrict the users of a free program by obtaining a
restrictive license from a patent holder.  Therefore, we insist that
any patent license obtained for a version of the library must be
consistent with the full freedom of use specified in this license.

  Most GNU software, including some libraries, is covered by the
ordinary GNU General Public License.  This license, the GNU Lesser
General Public License, applies to certain designated libraries, and
is quite different from the ordinary General Public License.  We use
this license for certain libraries in order to permit linking those
libraries into non-free programs.

  When a program is linked with a library, whether statically or using
a shared library, the combination of the two is legally speaking a
combined work, a derivative of the original library.  The ordinary
General Public License therefore permits such linking only if the
entire combination fits its criteria of freedom.  The Lesser General
Public License permits more lax criteria for linking other code with
the library.

  We call this license the "Lesser" General Public License because it
does Less to protect the user's freedom than the ordinary General
Public License.  It also provides other free software developers Less
of an advantage over competing non-free programs.  These disadvantages
are the reason we use the ordinary General Public License for many
libraries.  However, the Lesser license provides advantages in certain
special circumstances.

  For example, on rare occasions, there may be a special need to
encourage the widest possible use of a certain library, so that it
becomes
a de-facto standard.  To achieve this, non-free programs must be
allowed to use the library.  A more frequent case is that a free
library does the same job as widely used non-free libraries.  In this
case, there is little to gain by limiting the free library to free
software only, so we use the Lesser General Public License.

  In other cases, permission to use a particular library in non-free
programs enables a greater number of people to use a large body of
free software.  For example, permission to use the GNU C Library in
non-free programs enables many more people to use the whole GNU
operating system, as well as its variant, the GNU/Linux operating
system.

  Although the Lesser General Public License is Less protective of the
users' freedom, it does ensure that the user of a program that is
linked with the Library has the freedom and the wherewithal to run
that program using a modified version of the Library.

  The precise terms and conditions for copying, distribution and
modification follow.  Pay close attention to the difference between a
"work based on the library" and a "work that uses the library".  The
former contains code derived from the library, whereas the latter must
be combined with the library in order to run.
^L
                  GNU LESSER GENERAL PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

  0. This License Agreement applies to any software library or other
program which contains a notice placed by the copyright holder or
other authorized party saying it may be distributed under the terms of
this Lesser General Public License (also called "this License").
Each licensee is addressed as "you".

  A "library" means a collection of software functions and/or data
prepared so as to be conveniently linked with application programs
(which use some of those functions and data) to form executables.

  The "Library", below, refers to any such software library or work
which has been distributed under these terms.  A "work based on the
Library" means either the Library or any derivative work under
copyright law: that is to say, a work containing the Library or a
portion of it, either verbatim or with modifications and/or translated
straightforwardly into another language.  (Hereinafter, translation is
included without limitation in the term "modification".)

  "Source code" for a work means the preferred form of the work for
making modifications to it.  For a library, complete source code means
all the source code for all modules it contains, plus any associated
interface definition files, plus the scripts used to control
compilation
and installation of the library.

  Activities other than copying, distribution and modification are not
covered by this License; they are outside its scope.  The act of
running a program using the Library is not restricted, and output from
such a program is covered only if its contents constitute a work based
on the Library (independent of the use of the Library in a tool for
writing it).  Whether that is true depends on what the Library does
and what the program that uses the Library does.

  1. You may copy and distribute verbatim copies of the Library's
complete source code as you receive it, in any medium, provided that
you conspicuously and appropriately publish on each copy an
appropriate copyright notice and disclaimer of warranty; keep intact
all the notices that refer to this License and to the absence of any
warranty; and distribute a copy of this License along with the
Library.

  You may charge a fee for the physical act of transferring a copy,
and you may at your option offer warranty protection in exchange for a
fee.

  2. You may modify your copy or copies of the Library or any portion
of it, thus forming a work based on the Library, and copy and
distribute such modifications or work under the terms of Section 1
above, provided that you also meet all of these conditions:

    a) The modified work must itself be a software library.

    b) You must cause the files modified to carry prominent notices
    stating that you changed the files and the date of any change.

    c) You must cause the whole of the work to be licensed at no
    charge to all third parties under the terms of this License.

    d) If a facility in the modified Library refers to a function or a
    table of data to be supplied by an application program that uses
    the facility, other than as an argument passed when the facility
    is invoked, then you must make a good faith effort to ensure that,
    in the event an application does not supply such function or
    table, the facility still operates, and performs whatever part of
    its purpose remains meaningful.

    (For example, a function in a library to compute square roots has
    a purpose that is entirely well-defined independent of the
    application.  Therefore, Subsection 2d requires that any
    application-supplied function or table used by this function must
    be optional: if the application does not supply it, the square
    root function must still compute square roots.)

These requirements apply to the modified work as a whole.  If
identifiable sections of that work are not derived from the Library,
and can be reasonably considered independent and separate works in
themselves, then this License, and its terms, do not apply to those
sections when you distribute them as separate works.  But when you
distribute the same sections as part of a whole which is a work based
on the Library, the distribution of the whole must be on the terms of
this License, whose permissions for other licensees extend to the
entire whole, and thus to each and every part regardless of who wrote
it.

Thus, it is not the intent of this section to claim rights or contest
your rights to work written entirely by you; rather, the intent is to
exercise the right to control the distribution of derivative or
collective works based on the Library.

In addition, mere aggregation of another work not based on the Library
with the Library (or with a work based on the Library) on a volume of
a storage or distribution medium does not bring the other work under
the scope of this License.

  3. You may opt to apply the terms of the ordinary GNU General Public
License instead of this License to a given copy of the Library.  To do
this, you must alter all the notices that refer to this License, so
that they refer to the ordinary GNU General Public License, version 2,
instead of to this License.  (If a newer version than version 2 of the
ordinary GNU General Public License has appeared, then you can specify
that version instead if you wish.)  Do not make any other change in
these notices.
^L
  Once this change is made in a given copy, it is irreversible for
that copy, so the ordinary GNU General Public License applies to all
subsequent copies and derivative works made from that copy.

  This option is useful when you wish to copy part of the code of
the Library into a program that is not a library.

  4. You may copy and distribute the Library (or a portion or
derivative of it, under Section 2) in object code or executable form
under the terms of Sections 1 and 2 above provided that you accompany
it with the complete corresponding machine-readable source code, which
must be distributed under the terms of Sections 1 and 2 above on a
medium customarily used for software interchange.

  If distribution of object code is made by offering access to copy
from a designated place, then offering equivalent access to copy the
source code from the same place satisfies the requirement to
distribute the source code, even though third parties are not
compelled to copy the source along with the object code.

  5. A program that contains no derivative of any portion of the
Library, but is designed to work with the Library by being compiled or
linked with it, is called a "work that uses the Library".  Such a
work, in isolation, is not a derivative work of the Library, and
therefore falls outside the scope of this License.

  However, linking a "work that uses the Library" with the Library
creates an executable that is a derivative of the Library (because it
contains portions of the Library), rather than a "work that uses the
library".  The executable is therefore covered by this License.
Section 6 states terms for distribution of such executables.

  When a "work that uses the Library" uses material from a header file
that is part of the Library, the object code for the work may be a
derivative work of the Library even though the source code is not.
Whether this is true is especially significant if the work can be
linked without the Library, or if the work is itself a library.  The
threshold for this to be true is not precisely defined by law.

  If such an object file uses only numerical parameters, data
structure layouts and accessors, and small macros and small inline
functions (ten lines or less in length), then the use of the object
file is unrestricted, regardless of whether it is legally a derivative
work.  (Executables containing this object code plus portions of the
Library will still fall under Section 6.)

  Otherwise, if the work is a derivative of the Library, you may
distribute the object code for the work under the terms of Section 6.
Any executables containing that work also fall under Section 6,
whether or not they are linked directly with the Library itself.
^L
  6. As an exception to the Sections above, you may also combine or
link a "work that uses the Library" with the Library to produce a
work containing portions of the Library, and distribute that work
under terms of your choice, provided that the terms permit
modification of the work for the customer's own use and reverse
engineering for debugging such modifications.

  You must give prominent notice with each copy of the work that the
Library is used in it and that the Library and its use are covered by
this License.  You must supply a copy of this License.  If the work
during execution displays copyright notices, you must include the
copyright notice for the Library among them, as well as a reference
directing the user to the copy of this License.  Also, you must do one
of these things:

    a) Accompany the work with the complete corresponding
    machine-readable source code for the Library including whatever
    changes were used in the work (which must be distributed under
    Sections 1 and 2 above); and, if the work is an executable linked
    with the Library, with the complete machine-readable "work that
    uses the Library", as object code and/or source code, so that the
    user can modify the Library and then relink to produce a modified
    executable containing the modified Library.  (It is understood
    that the user who changes the contents of definitions files in the
    Library will not necessarily be able to recompile the application
    to use the modified definitions.)

    b) Use a suitable shared library mechanism for linking with the
    Library.  A suitable mechanism is one that (1) uses at run time a
    copy of the library already present on the user's computer system,
    rather than copying library functions into the executable, and (2)
    will operate properly with a modified version of the library, if
    the user installs one, as long as the modified version is
    interface-compatible with the version that the work was made with.

    c) Accompany the work with a written offer, valid for at
    least three years, to give the same user the materials
    specified in Subsection 6a, above, for a charge no more
    than the cost of performing this distribution.

    d) If distribution of the work is made by offering access to copy
    from a designated place, offer equivalent access to copy the above
    specified materials from the same place.

    e) Verify that the user has already received a copy of these
    materials or that you have already sent this user a copy.

  For an executable, the required form of the "work that uses the
Library" must include any data and utility programs needed for
reproducing the executable from it.  However, as a special exception,
the materials to be distributed need not include anything that is
normally distributed (in either source or binary form) with the major
components (compiler, kernel, and so on) of the operating system on
which the executable runs, unless that component itself accompanies
the executable.

  It may happen that this requirement contradicts the license
restrictions of other proprietary libraries that do not normally
accompany the operating system.  Such a contradiction means you cannot
use both them and the Library together in an executable that you
distribute.
^L
  7. You may place library facilities that are a work based on the
Library side-by-side in a single library together with other library
facilities not covered by this License, and distribute such a combined
library, provided that the separate distribution of the work based on
the Library and of the other library facilities is otherwise
permitted, and provided that you do these two things:

    a) Accompany the combined library with a copy of the same work
    based on the Library, uncombined with any other library
    facilities.  This must be distributed under the terms of the
    Sections above.

    b) Give prominent notice with the combined library of the fact
    that part of it is a work based on the Library, and explaining
    where to find the accompanying uncombined form of the same work.

  8. You may not copy, modify, sublicense, link with, or distribute
the Library except as expressly provided under this License.  Any
attempt otherwise to copy, modify, sublicense, link with, or
distribute the Library is void, and will automatically terminate your
rights under this License.  However, parties who have received copies,
or rights, from you under this License will not have their licenses
terminated so long as such parties remain in full compliance.

  9. You are not required to accept this License, since you have not
signed it.  However, nothing else grants you permission to modify or
distribute the Library or its derivative works.  These actions are
prohibited by law if you do not accept this License.  Therefore, by
modifying or distributing the Library (or any work based on the
Library), you indicate your acceptance of this License to do so, and
all its terms and conditions for copying, distributing or modifying
the Library or works based on it.

  10. Each time you redistribute the Library (or any work based on the
Library), the recipient automatically receives a license from the
original licensor to copy, distribute, link with or modify the Library
subject to these terms and conditions.  You may not impose any further
restrictions on the recipients' exercise of the rights granted herein.
You are not responsible for enforcing compliance by third parties with
this License.
^L
  11. If, as a consequence of a court judgment or allegation of patent
infringement or for any other reason (not limited to patent issues),
conditions are imposed on you (whether by court order, agreement or
otherwise) that contradict the conditions of this License, they do not
excuse you from the conditions of this License.  If you cannot
distribute so as to satisfy simultaneously your obligations under this
License and any other pertinent obligations, then as a consequence you
may not distribute the Library at all.  For example, if a patent
license would not permit royalty-free redistribution of the Library by
all those who receive copies directly or indirectly through you, then
the only way you could satisfy both it and this License would be to
refrain entirely from distribution of the Library.

If any portion of this section is held invalid or unenforceable under
any particular circumstance, the balance of the section is intended to
apply, and the section as a whole is intended to apply in other
circumstances.

It is not the purpose of this section to induce you to infringe any
patents or other property right claims or to contest validity of any
such claims; this section has the sole purpose of protecting the
integrity of the free software distribution system which is
implemented by public license practices.  Many people have made
generous contributions to the wide range of software distributed
through that system in reliance on consistent application of that
system; it is up to the author/donor to decide if he or she is willing
to distribute software through any other system and a licensee cannot
impose that choice.

This section is intended to make thoroughly clear what is believed to
be a consequence of the rest of this License.

  12. If the distribution and/or use of the Library is restricted in
certain countries either by patents or by copyrighted interfaces, the
original copyright holder who places the Library under this License
may add an explicit geographical distribution limitation excluding those
countries, so that distribution is permitted only in or among
countries not thus excluded.  In such case, this License incorporates
the limitation as if written in the body of this License.

  13. The Free Software Foundation may publish revised and/or new
versions of the Lesser General Public License from time to time.
Such new versions will be similar in spirit to the present version,
but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number.  If the Library
specifies a version number of this License which applies to it and
"any later version", you have the option of following the terms and
conditions either of that version or of any later version published by
the Free Software Foundation.  If the Library does not specify a
license version number, you may choose any version ever published by
the Free Software Foundation.
^L
  14. If you wish to incorporate parts of the Library into other free
programs whose distribution conditions are incompatible with these,
write to the author to ask for permission.  For software which is
copyrighted by the Free Software Foundation, write to the Free
Software Foundation; we sometimes make exceptions for this.  Our
decision will be guided by the two goals of preserving the free status
of all derivatives of our free software and of promoting the sharing
and reuse of software generally.

                            NO WARRANTY

  15. BECAUSE THE LIBRARY IS LICENSED FREE OF CHARGE, THERE IS NO
WARRANTY FOR THE LIBRARY, TO THE EXTENT PERMITTED BY APPLICABLE LAW.
EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR
OTHER PARTIES PROVIDE THE LIBRARY "AS IS" WITHOUT WARRANTY OF ANY
KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE
LIBRARY IS WITH YOU.  SHOULD THE LIBRARY PROVE DEFECTIVE, YOU ASSUME
THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

  16. IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN
WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MAY MODIFY
AND/OR REDISTRIBUTE THE LIBRARY AS PERMITTED ABOVE, BE LIABLE TO YOU
FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE
LIBRARY (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING
RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A
FAILURE OF THE LIBRARY TO OPERATE WITH ANY OTHER SOFTWARE), EVEN IF
SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
DAMAGES.

                     END OF TERMS AND CONDITIONS
^L
           How to Apply These Terms to Your New Libraries

  If you develop a new library, and you want it to be of the greatest
possible use to the public, we recommend making it free software that
everyone can redistribute and change.  You can do so by permitting
redistribution under these terms (or, alternatively, under the terms
of the ordinary General Public License).

  To apply these terms, attach the following notices to the library.
It is safest to attach them to the start of each source file to most
effectively convey the exclusion of warranty; and each file should
have at least the "copyright" line and a pointer to where the full
notice is found.


    <one line to give the library's name and a brief idea of what it
does.>
    Copyright (C) <year>  <name of author>

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307  USA

Also add information on how to contact you by electronic and paper
mail.

You should also get your employer (if you work as a programmer) or
your
school, if any, to sign a "copyright disclaimer" for the library, if
necessary.  Here is a sample; alter the names:

  Yoyodyne, Inc., hereby disclaims all copyright interest in the
  library `Frob' (a library for tweaking knobs) written by James
Random Hacker.

  <signature of Ty Coon>, 1 April 1990
  Ty Coon, President of Vice

That's all there is to it!
```

## liburing - 2.9

- **Repository URL**: https://github.com/axboe/liburing
- **License URL**: https://github.com/axboe/liburing/blob/master/LICENSE
- **License name**: MIT
### License Text:
```
Copyright 2020 Jens Axboe

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Mooncake - v0.3.2.post1

- **Repository URL**: https://github.com/kvcache-ai/Mooncake
- **License URL**: https://github.com/kvcache-ai/Mooncake/blob/main/LICENSE
- **License name**: Apache-2.0
### License Text:
```

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Google flags - 2.2.2

- **Repository URL**: https://github.com/gflags/gflags
- **License URL**: https://github.com/gflags/gflags/blob/master/COPYING.txt
- **License name**: BSD3
### License Text:
```
Copyright (c) 2006, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Google test - 1.17.0

- **Repository URL**: https://github.com/google/googletest
- **License URL**: https://github.com/google/googletest/blob/main/LICENSE
- **License name**: BSD3
### License Text:
```
Copyright 2008, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.
    * Neither the name of Google Inc. nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## pybind11 - 2.13.6

- **Repository URL**: https://github.com/pybind/pybind11
- **License URL**: https://github.com/pybind/pybind11/blob/master/LICENSE
- **License name**: BSD3
### License Text:
```
Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Please also refer to the file .github/CONTRIBUTING.md, which clarifies licensing of
external contributions to this project including patches, pull requests, etc.
```

## NVSHMEM - 3.2.5

- **Repository URL**: https://developer.nvidia.com/nvshmem
- **License URL**: https://docs.nvidia.com/nvshmem/api/sla.html
- **License name**: NVIDIA Software License Agreement
### License Text:
```
This document is the Software License Agreement (SLA) for NVIDIA NVSHMEM. The following contains specific license terms and conditions for NVIDIA NVSHMEM. By using NVSHMEM, you agree to comply with all the terms and conditions applicable to the specific product(s) included herein.

LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS
This license agreement, including exhibits attached (“Agreement”) is a legal agreement between you and NVIDIA Corporation (“NVIDIA”) and governs your use of a NVIDIA software development kit (“SDK”).

Each SDK has its own set of software and materials, but here is a description of the types of items that may be included in a SDK: source code, header files, APIs, data sets and assets (examples include images, textures, models, scenes, videos, native API input/output files), binary software, sample code, libraries, utility programs, programming code and documentation.

This Agreement can be accepted only by an adult of legal age of majority in the country in which the SDK is used.

If you are entering into this Agreement on behalf of a company or other legal entity, you represent that you have the legal authority to bind the entity to this Agreement, in which case “you” will mean the entity you represent.

If you don’t have the required age or authority to accept this Agreement, or if you don’t accept all the terms and conditions of this Agreement, do not download, install or use the SDK.

You agree to use the SDK only for purposes that are permitted by (a) this Agreement, and (b) any applicable law, regulation or generally accepted practices or guidelines in the relevant jurisdictions.

1. License.
1.1 Grant

Subject to the terms of this Agreement, NVIDIA hereby grants you a non-exclusive, non-transferable license, without the right to sublicense (except as expressly provided in this Agreement) to:

Install and use the SDK,
Modify and create derivative works of source code delivered in the SDK, and
Distribute those portions of the SDK that are identified in this Agreement as distributable, as incorporated into a software application that meets the distribution requirements indicated in this Agreement.
1.2 Distribution Requirements

These are the distribution requirements for you to exercise the distribution grant:

Your application must have material additional functionality, beyond the included portions of the SDK.
The distributable portions of the SDK shall only be accessed by your application.
The following notice shall be included in modifications and derivative works of sample source code distributed: “This software contains source code provided by NVIDIA Corporation.”
Unless a developer tool is identified in this Agreement as distributable, it is delivered for your internal use only.
The terms under which you distribute your application must be consistent with the terms of this Agreement, including (without limitation) terms relating to the license grant and license restrictions and protection of NVIDIA’s intellectual property rights. Additionally, you agree that you will protect the privacy, security and legal rights of your application users.
You agree to notify NVIDIA in writing of any known or suspected distribution or use of the SDK not in compliance with the requirements of this Agreement, and to enforce the terms of your agreements with respect to distributed SDK.
1.3 Authorized Users

You may allow employees and contractors of your entity or of your subsidiary(ies) to access and use the SDK from your secure network to perform work on your behalf.

If you are an academic institution you may allow users enrolled or employed by the academic institution to access and use the SDK from your secure network.

You are responsible for the compliance with the terms of this Agreement by your authorized users. If you become aware that your authorized users didn’t follow the terms of this Agreement, you agree to take reasonable steps to resolve the non-compliance and prevent new occurrences.

1.4 Pre-Release SDK

The SDK versions identified as alpha, beta, preview or otherwise as pre-release, may not be fully functional, may contain errors or design flaws, and may have reduced or different security, privacy, accessibility, availability, and reliability standards relative to commercial versions of NVIDIA software and materials. Use of a pre-release SDK may result in unexpected results, loss of data, project delays or other unpredictable damage or loss. You may use a pre-release SDK at your own risk, understanding that pre-release SDKs are not intended for use in production or business-critical systems. NVIDIA may choose not to make available a commercial version of any pre-release SDK. NVIDIA may also choose to abandon development and terminate the availability of a pre-release SDK at any time without liability.

1.5 Updates

NVIDIA may, at its option, make available patches, workarounds or other updates to this SDK. Unless the updates are provided with their separate governing terms, they are deemed part of the SDK licensed to you as provided in this Agreement.

You agree that the form and content of the SDK that NVIDIA provides may change without prior notice to you. While NVIDIA generally maintains compatibility between versions, NVIDIA may in some cases make changes that introduce incompatibilities in future versions of the SDK.

1.6 Third Party Licenses

The SDK may come bundled with, or otherwise include or be distributed with, third party software licensed by a NVIDIA supplier and/or open source software provided under an open source license. Use of third party software is subject to the third-party license terms, or in the absence of third party terms, the terms of this Agreement. Copyright to third party software is held by the copyright holders indicated in the third-party software or license.

1.7 Reservation of Rights

NVIDIA reserves all rights, title and interest in and to the SDK not expressly granted to you under this Agreement.

2. Limitations.
The following license limitations apply to your use of the SDK:

2.1 You may not reverse engineer, decompile or disassemble, or remove copyright or other proprietary notices from any portion of the SDK or copies of the SDK.

2.2 Except as expressly provided in this Agreement, you may not copy, sell, rent, sublicense, transfer, distribute, modify, or create derivative works of any portion of the SDK.

2.3 Unless you have an agreement with NVIDIA for this purpose, you may not indicate that an application created with the SDK is sponsored or endorsed by NVIDIA.

2.4 You may not bypass, disable, or circumvent any encryption, security, digital rights management or authentication mechanism in the SDK.

2.5 You may not use the SDK in any manner that would cause it to become subject to an open source software license. As examples, licenses that require as a condition of use, modification, and/or distribution that the SDK be (i) disclosed or distributed in source code form; (ii) licensed for the purpose of making derivative works; or (iii) redistributable at no charge.

2.6 Unless you have an agreement with NVIDIA for this purpose, you may not use the SDK with any system or application where the use or failure of the system or application can reasonably be expected to threaten or result in personal injury, death, or catastrophic loss. Examples include use in nuclear, avionics, navigation, military, medical, life support or other life critical applications. NVIDIA does not design, test or manufacture the SDK for these critical uses and NVIDIA shall not be liable to you or any third party, in whole or in part, for any claims or damages arising from such uses.

2.7 You agree to defend, indemnify and hold harmless NVIDIA and its affiliates, and their respective employees, contractors, agents, officers and directors, from and against any and all claims, damages, obligations, losses, liabilities, costs or debt, fines, restitutions and expenses (including but not limited to attorney’s fees and costs incident to establishing the right of indemnification) arising out of or related to your use of the SDK outside of the scope of this Agreement, or not in compliance with its terms.

3. Ownership.
3.1 NVIDIA or its licensors hold all rights, title and interest in and to the SDK and its modifications and derivative works, including their respective intellectual property rights. This SDK may include software and materials from NVIDIA’s licensors, and these licensors are intended third party beneficiaries that may enforce this Agreement with respect to their intellectual property rights.

3.2 You may, but don’t have to, provide to NVIDIA suggestions, feature requests or other feedback regarding the SDK, including possible enhancements or modifications to the SDK. For any feedback that you voluntarily provide, you hereby grant NVIDIA and its affiliates a perpetual, non-exclusive, worldwide, irrevocable license to use, reproduce, modify, license, sublicense (through multiple tiers of sublicensees), and distribute (through multiple tiers of distributors) it without the payment of any royalties or fees to you. NVIDIA will use feedback at its choice.

4. No Warranties.
THE SDK IS PROVIDED BY NVIDIA “AS IS” AND “WITH ALL FAULTS.” TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES EXPRESSLY DISCLAIM ALL WARRANTIES OF ANY KIND OR NATURE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, OR THE ABSENCE OF ANY DEFECTS THEREIN, WHETHER LATENT OR PATENT. NO WARRANTY IS MADE ON THE BASIS OF TRADE USAGE, COURSE OF DEALING OR COURSE OF TRADE.

5. Limitations of Liability.
TO THE MAXIMUM EXTENT PERMITTED BY LAW, NVIDIA AND ITS AFFILIATES SHALL NOT BE LIABLE FOR ANY SPECIAL, INCIDENTAL, PUNITIVE OR CONSEQUENTIAL DAMAGES, OR ANY LOST PROFITS, LOSS OF USE, LOSS OF DATA OR LOSS OF GOODWILL, OR THE COSTS OF PROCURING SUBSTITUTE PRODUCTS, ARISING OUT OF OR IN CONNECTION WITH THIS AGREEMENT OR THE USE OR PERFORMANCE OF THE SDK, WHETHER SUCH LIABILITY ARISES FROM ANY CLAIM BASED UPON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY OR ANY OTHER CAUSE OF ACTION OR THEORY OF LIABILITY. IN NO EVENT WILL NVIDIA’S AND ITS AFFILIATES TOTAL CUMULATIVE LIABILITY UNDER OR ARISING OUT OF THIS AGREEMENT EXCEED US$10.00. THE NATURE OF THE LIABILITY OR THE NUMBER OF CLAIMS OR SUITS SHALL NOT ENLARGE OR EXTEND THIS LIMIT.

These exclusions and limitations of liability shall apply regardless if NVIDIA or its affiliates have been advised of the possibility of such damages, and regardless of whether a remedy fails its essential purpose. These exclusions and limitations of liability form an essential basis of the bargain between the parties, and, absent any of these exclusions or limitations of liability, the provisions of this Agreement, including, without limitation, the economic terms, would be substantially different.

6. Termination.
6.1 This Agreement will continue to apply until terminated by either you or NVIDIA as described below.

6.2 If you want to terminate this Agreement, you may do so by stopping to use the SDK.

6.3 NVIDIA may, at any time, terminate this Agreement if: (i) you fail to comply with any term of this Agreement and the non-compliance is not fixed within thirty (30) days following notice from NVIDIA (or immediately if you violate NVIDIA’s intellectual property rights); (ii) you commence or participate in any legal proceeding against NVIDIA with respect to the SDK; or (iii) NVIDIA decides to no longer provide the SDK in a country or, in NVIDIA’s sole discretion, the continued use of it is no longer commercially viable.

6.4 Upon any termination of this Agreement, you agree to promptly discontinue use of the SDK and destroy all copies in your possession or control. Your prior distributions in accordance with this Agreement are not affected by the termination of this Agreement. Upon written request, you will certify in writing that you have complied with your commitments under this section. Upon any termination of this Agreement all provisions survive except for the licenses granted to you.

7. General.
If you wish to assign this Agreement or your rights and obligations, including by merger, consolidation, dissolution or operation of law, contact NVIDIA to ask for permission. Any attempted assignment not approved by NVIDIA in writing shall be void and of no effect. NVIDIA may assign, delegate or transfer this Agreement and its rights and obligations, and if to a non-affiliate you will be notified.

You agree to cooperate with NVIDIA and provide reasonably requested information to verify your compliance with this Agreement.

This Agreement will be governed in all respects by the laws of the United States and of the State of Delaware as those laws are applied to contracts entered into and performed entirely within Delaware by Delaware residents, without regard to the conflicts of laws principles. The United Nations Convention on Contracts for the International Sale of Goods is specifically disclaimed. You agree to all terms of this Agreement in the English language.

The state or federal courts residing in Santa Clara County, California shall have exclusive jurisdiction over any dispute or claim arising out of this Agreement. Notwithstanding this, you agree that NVIDIA shall still be allowed to apply for injunctive remedies or an equivalent type of urgent legal relief in any jurisdiction.

If any court of competent jurisdiction determines that any provision of this Agreement is illegal, invalid or unenforceable, such provision will be construed as limited to the extent necessary to be consistent with and fully enforceable under the law and the remaining provisions will remain in full force and effect. Unless otherwise specified, remedies are cumulative.

Each party acknowledges and agrees that the other is an independent contractor in the performance of this Agreement.

The SDK is “commercial items” consisting of “commercial computer software” and “commercial computer software documentation” provided, to the extent consistent with a governing federal procurement contract, with RESTRICTED RIGHTS. Contractor/manufacturer is NVIDIA, 2788 San Tomas Expressway, Santa Clara, CA 95051.

The SDK is subject to United States export laws and regulations. You agree that you will not ship, transfer or export the SDK into any country, or use the SDK in any manner, prohibited by the United States Bureau of Industry and Security or economic sanctions regulations administered by the U.S. Department of Treasury’s Office of Foreign Assets Control (OFAC), or any applicable export laws, restrictions or regulations. These laws include restrictions on destinations, end users and end use. By accepting this Agreement, you confirm that you are not a resident or citizen of any country currently embargoed by the U.S. and that you are not otherwise prohibited from receiving the SDK.

Any notice delivered by NVIDIA to you under this Agreement will be delivered via mail, email or fax. You agree that any notices that NVIDIA sends you electronically will satisfy any legal communication requirements. Please direct your legal notices or other correspondence to NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, California 95051, United States of America, Attention: Legal Department.

This Agreement and any exhibits incorporated into this Agreement constitute the entire agreement of the parties with respect to the subject matter of this Agreement and supersede all prior negotiations or documentation exchanged between the parties relating to this SDK license. Any additional and/or conflicting terms on documents issued by you are null, void, and invalid. Any amendment or waiver under this Agreement shall be in writing and signed by representatives of both parties.

(v. September 3, 2019)

```

---
This file was automatically generated.