
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
## ucx - 1.18.0

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
---
This file was automatically generated.
