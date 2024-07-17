/* Copyright 2024 Jonas Blenninger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "wrapper/handle.hpp"
#include "cuaoa/handle.hpp"

extern "C" {

CUAOAHandle *createHandle(size_t numNodes, int device, bool exact) {
  return new CUAOAHandle(numNodes, device, exact);
}

void destroyHandle(CUAOAHandle *handle) { delete handle; }
}
