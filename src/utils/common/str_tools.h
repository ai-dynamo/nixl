/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NIXL_SRC_UTILS_COMMON_STR_TOOLS_H
#define NIXL_SRC_UTILS_COMMON_STR_TOOLS_H

#include <string>

class strEqual
{
    public:
      bool operator() (const std::string &t1, const std::string &t2) const
      {
          size_t s1 = t1.size();
          size_t s2 = t2.size();

          if (s1 != s2) return false;
          if (((s1&7) != 0) || (s1>64)) return (t1 == t2);

          size_t i = 0;
          const char* d1 = t1.data();
          const char* d2 = t2.data();

          for (i=0; i<s1; i+=8) {
              if (*((uint64_t*) (d1 + i)) != *((uint64_t*) (d2 + i)))
                  return false;
          }
          return true;
      }
};

#endif // NIXL_SRC_UTILS_COMMON_STR_TOOLS_H
