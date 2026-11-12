# Copyright 2025 The JAX Authors.
#
# Modified by Hygon Information Technology Co., Ltd., 2026.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring

# XLA is pinned to the DAS fork (https://github.com/HYGON-AI/xla-das).
# To update XLA to a new revision,
# a) update XLA_COMMIT to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/HYGON-AI/xla-das/archive/{git_hash}.tar.gz | sha256sum
#    and update XLA_SHA256 with the result.

# buildifier: disable=module-docstring
XLA_COMMIT = "8e709b7753a3a135e80c7b1b2fdba601a06dd168"
XLA_SHA256 = "c1a3da7f2a602841f3a95f3bfe26502f458bc750dac99440e4c28c1253f0531e"
