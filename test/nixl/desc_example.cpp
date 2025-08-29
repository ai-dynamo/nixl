/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <string>
#include <cassert>
#include "nixl.h"
#include "serdes/serdes.h"
#include "backend/backend_aux.h"

#include <sys/time.h>


void testPerf(){
    int desc_count = 24*64*1024;
    void* buf = malloc(256);
    nixl_xfer_dlist_t dlist (DRAM_SEG);

    struct timeval start_time, end_time, diff_time;

    gettimeofday(&start_time, NULL);

    for(int i = 0; i<desc_count; i++)
        dlist.addDesc(nixlBasicDesc((uintptr_t) buf, 256, 0));

    gettimeofday(&end_time, NULL);

    assert(dlist.descCount() == 24*64*1024);
    timersub(&end_time, &start_time, &diff_time);
    std::cout << "add desc mode, total time for " << 24*64*1024 << " descs: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    float time_per_desc = ((diff_time.tv_sec * 1000000) + diff_time.tv_usec);
    time_per_desc /=  (24*64*1024) ;
    std::cout << "time per desc " << time_per_desc << "us\n";


    nixl_xfer_dlist_t dlist2(DRAM_SEG, desc_count);

    gettimeofday(&start_time, NULL);

    for(int i = 0; i<desc_count; i++)
        dlist2[i] = nixlBasicDesc((uintptr_t) buf, 256, 0);

    gettimeofday(&end_time, NULL);

    assert(dlist.descCount() == 24*64*1024);
    timersub(&end_time, &start_time, &diff_time);
    std::cout << "Operator [] mode, total time for " << 24*64*1024 << " descs: "
              << diff_time.tv_sec << "s " << diff_time.tv_usec << "us \n";

    time_per_desc = ((diff_time.tv_sec * 1000000) + diff_time.tv_usec);
    time_per_desc /=  (24*64*1024) ;
    std::cout << "time per desc " << time_per_desc << "us\n";

    free(buf);
 }

int main()
{
    // nixlBasicDesc functionality
    nixlBasicDesc buff1;
    buff1.addr   = (uintptr_t) 1000;
    buff1.len    = 105;
    buff1.devId  = 0;

    nixlBasicDesc buff2 (2000,23,3);
    nixlBasicDesc buff3 (buff2);
    nixlBasicDesc buff4;
    buff4 = buff1;
    nixlBasicDesc buff5 (1980,21,3);
    nixlBasicDesc buff6 (1010,30,4);
    nixlBasicDesc buff7 (1010,30,0);
    nixlBasicDesc buff8 (1010,31,0);

    nixlBasicDesc importDesc(buff2.serialize());
    assert(buff2 == importDesc);

    assert (buff3==buff2);
    assert (buff4==buff1);
    assert (buff3!=buff1);
    assert (buff8!=buff7);

    assert (buff2.covers(buff3));
    assert (buff4.overlaps(buff1));
    assert (!buff1.covers(buff2));
    assert (!buff1.overlaps(buff2));
    assert (!buff2.covers(buff1));
    assert (!buff2.overlaps(buff1));
    assert (buff2.overlaps(buff5));
    assert (buff5.overlaps(buff2));
    assert (!buff2.covers(buff5));
    assert (!buff5.covers(buff2));
    assert (!buff1.covers(buff6));
    assert (!buff6.covers(buff1));
    assert (buff1.covers(buff7));
    assert (!buff7.covers(buff1));

    nixlBlobDesc stringd1;
    stringd1.addr   = 2392382;
    stringd1.len    = 23;
    stringd1.devId  = 4;
    stringd1.metaInfo = std::string("567");

    nixlBlobDesc importStringD(stringd1.serialize());
    assert(stringd1 == importStringD);

    std::cout << "\nSerDes Desc tests:\n";
    buff2.print("");
    std::cout << "this should be a copy:\n";
    importDesc.print("");
    std::cout << "\n";
    stringd1.print("");
    std::cout << "this should be a copy:\n";
    importStringD.print("");
    std::cout << "\n";

    nixlBlobDesc stringd2;
    stringd2.addr     = 1010;
    stringd2.len      = 31;
    stringd2.devId    = 0;
    stringd2.metaInfo = std::string("567f");

    nixlMetaDesc meta1;
    meta1.addr      = 56;
    meta1.len       = 1294;
    meta1.devId     = 0;
    meta1.metadataP = nullptr;

    nixlMetaDesc meta2;
    meta2.addr      = 70;
    meta2.len       = 43;
    meta2.devId     = 0;
    meta2.metadataP = nullptr;

    assert (stringd1!=buff1);
    assert (stringd2==buff8);
    nixlBasicDesc buff9 (stringd1);

    buff1.print("");
    buff2.print("");
    buff9.print("");
    stringd1.print("");
    stringd2.print("");


    // DescList functionality
    std::cout << "\n\n";
    nixlMetaDesc meta3 (10070, 43, 0);
    nixlMetaDesc meta4 (10070, 42, 0);
    meta3.metadataP = nullptr;
    meta4.metadataP = nullptr;

    nixl_meta_dlist_t dlist1 (DRAM_SEG);
    dlist1.addDesc(meta1);
    dlist1.addDesc(meta3);

    nixl_meta_dlist_t dlist2(VRAM_SEG, false);
    dlist2.addDesc(meta3);
    dlist2.addDesc(meta2);

    nixl_meta_dlist_t dlist3(VRAM_SEG, true);
    dlist3.addDesc(meta3);
    dlist3.addDesc(meta2);

    nixl_meta_dlist_t dlist4 (dlist1);
    nixl_meta_dlist_t dlist5 (VRAM_SEG);
    dlist5 = dlist3;

    dlist1.print();
    dlist2.print();
    dlist3.print();
    dlist4.print();
    dlist5.print();

    try {
        dlist1.remDesc(2);
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
    std::cout << dlist1.getIndex(meta3) << "\n";
    dlist1.remDesc(0);
    std::cout << dlist1.getIndex(meta3) << "\n";
    try {
        dlist2.remDesc(dlist2.getIndex(meta1));
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }
    dlist2.remDesc(dlist2.getIndex(meta3));
    assert(dlist2.getIndex(meta3)== NIXL_ERR_NOT_FOUND);
    assert(dlist3.getIndex(meta1)== NIXL_ERR_NOT_FOUND);
    try {
        dlist3.remDesc(dlist3.getIndex(meta4));
    } catch (const std::out_of_range& e) {
        std::cout << "Caught expected error: " << e.what() << std::endl;
    }

    dlist1.print();
    dlist2.print();
    dlist3.print();

    // Populate and unifiedAddr test
    std::cout << "\n\n";
    nixlBlobDesc s1 (10070, 43, 0);
    s1.metaInfo = "s1";
    nixlBlobDesc s2 (900, 43, 2);
    s2.metaInfo = "s2";
    nixlBlobDesc s3 (500, 43, 1);
    s3.metaInfo = "s3";
    nixlBlobDesc s4 (100, 43, 3);
    s4.metaInfo = "s4";

    nixlBasicDesc b1 (10075, 30, 0);
    nixlBasicDesc b2 (905, 30, 2);
    nixlBasicDesc b3 (505, 30, 1);
    nixlBasicDesc b4 (105, 30, 3);
    nixlBasicDesc b5 (305, 30, 4);
    nixlBasicDesc b6 (100, 30, 3);

    nixl_xfer_dlist_t dlist10(DRAM_SEG);
    nixl_xfer_dlist_t dlist11(DRAM_SEG);
    nixl_xfer_dlist_t dlist12(DRAM_SEG);
    nixl_xfer_dlist_t dlist13(DRAM_SEG);
    nixl_xfer_dlist_t dlist14(DRAM_SEG);

    nixl_reg_dlist_t dlist20(DRAM_SEG);

    dlist10.addDesc(b1);
    dlist10.addDesc(b2);
    dlist10.addDesc(b3);
    dlist10.addDesc(b4);

    dlist11.addDesc(b1);
    dlist11.addDesc(b2);
    dlist11.addDesc(b3);
    dlist11.addDesc(b4);

    dlist12.addDesc(b1);
    dlist12.addDesc(b2);
    dlist12.addDesc(b3);
    dlist12.addDesc(b4);

    dlist13.addDesc(b1);
    dlist13.addDesc(b2);
    dlist13.addDesc(b3);
    dlist13.addDesc(b5);

    dlist14.addDesc(b1);
    dlist14.addDesc(b2);
    dlist14.addDesc(b3);
    dlist14.addDesc(b6);

    dlist20.addDesc(s1);
    dlist20.addDesc(s2);
    dlist20.addDesc(s3);
    dlist20.addDesc(s4);

    dlist11.print();
    dlist12.print();
    dlist13.print();
    dlist14.print();

    std::cout << "\nSerDes DescList tests:\n";
    nixlSerDes* ser_des = new nixlSerDes();
    nixlSerDes* ser_des2 = new nixlSerDes();

    assert(dlist10.serialize(ser_des) == 0);
    nixl_xfer_dlist_t importList (ser_des);;
    assert(importList == dlist10);

    assert(dlist20.serialize(ser_des2) == 0);
    nixl_reg_dlist_t importSList (ser_des2);
    assert(importSList == dlist20);

    dlist10.print();
    std::cout << "this should be a copy:\n";
    importList.print();
    std::cout << "\n";
    dlist20.print();
    std::cout << "this should be a copy:\n";
    importSList.print();
    std::cout << "\n";

    nixl_reg_dlist_t dlist21(DRAM_SEG);
    nixl_reg_dlist_t dlist22(DRAM_SEG);
    nixl_reg_dlist_t dlist23(DRAM_SEG);
    nixl_reg_dlist_t dlist24(DRAM_SEG);
    nixl_reg_dlist_t dlist25(DRAM_SEG);

    testPerf();

    delete ser_des;
    delete ser_des2;

    return 0;
}
