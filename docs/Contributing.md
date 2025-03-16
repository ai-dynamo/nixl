<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contributing to NIXL
## Commits

Please follow the following format for your commits:

```
<component>: <title>

Detailed description of the fix.

Signed-off-by: <real name> <<email address>>
```

Please make commits with proper description and try to combine commits to smaller features
as possible. Each author should sign off at the bottom of the commit to certify that they contributed to the commit. If more than one author(s) sheperd the contribution, add their own attestation to the bottom of the commit also.

Example commit can be as shown below.

```
commit 067e922af48c0d9b45da507b5800c3951076c4e9
Author: Jane Doe <jane@daos.io>
Date:   Thu Jan 23 14:26:00 2024 +0800

    NIXL-001 include: Add new APIs

    Add awesome new APIs to the NIXL

    Signed-off-by: Jane Doe <jane@nixl.io>
    Signed-off-by: John Smith <jsmith@corp.com>
```