// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/imperative/flags.h"

#include "paddle/common/flags.h"

PHI_DEFINE_EXPORTED_uint64(dygraph_debug,
                           0,
                           "Debug level of dygraph. This flag is not "
                           "open to users");

namespace paddle::imperative {

bool IsDebugEnabled() { return FLAGS_dygraph_debug != 0; }

uint64_t GetDebugLevel() { return FLAGS_dygraph_debug; }

}  // namespace paddle::imperative
