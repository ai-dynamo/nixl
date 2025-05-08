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
#ifndef _NIXL_LIST_ELEM_H
#define _NIXL_LIST_ELEM_H


template <typename T>
class nixlLinkElem
{
private:
    T* _next;

public:
    nixlLinkElem() { _next = NULL; }

    ~nixlLinkElem()
    {
        _next = NULL;
        ;
    }

    /* Link this element into the chain after "elem" */
    void link(T* elem)
    {
        elem->_next = _next;
        _next = elem;
    }

    /* Exclude this element from the chain, return the new head */
    T* unlink()
    {
        T* ret = _next;
        /* Forget my place */
        _next = NULL;
        return ret;
    }

    T* next() { return _next; }
};

#endif
