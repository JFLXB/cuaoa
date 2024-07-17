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

#ifndef DEVICE_INFO_WRAPPER_HPP
#define DEVICE_INFO_WRAPPER_HPP

#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

struct CudaDeviceInfo {
  int id;
  char *name;
  size_t memory = 0; // in bytes
};

struct CudaDeviceInfoResult {
  size_t numDevices = 0;
  CudaDeviceInfo *devices = nullptr;
};

CudaDeviceInfoResult getCudaDevicesInfo();
CudaDeviceInfo *getCudaDevicesInfoData(CudaDeviceInfoResult *result);
void freeCudaDevicesInfoResult(CudaDeviceInfoResult *result);

#ifdef __cplusplus
}
#endif

#endif
