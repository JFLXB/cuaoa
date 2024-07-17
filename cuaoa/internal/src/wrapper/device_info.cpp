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

#include "wrapper/device_info.hpp"

extern "C" {

CudaDeviceInfoResult getCudaDevicesInfo() {
  int numDevices;
  cudaGetDeviceCount(&numDevices);

  CudaDeviceInfo *infos = new CudaDeviceInfo[numDevices];

  for (int i = 0; i < numDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties_v2(&prop, i);

    CudaDeviceInfo info;
    info.id = i;
    info.memory = prop.totalGlobalMem;
    info.name = prop.name;
    infos[i] = info;
  }

  CudaDeviceInfoResult result;
  result.numDevices = numDevices;
  result.devices = infos;
  return result;
}

CudaDeviceInfo *getCudaDevicesInfoData(CudaDeviceInfoResult *result) {
  return result ? result->devices : nullptr;
}

void freeCudaDevicesInfoResult(CudaDeviceInfoResult *result) {
  if (!result)
    return;

  delete[] result->devices;
  result->devices = nullptr;
  result->numDevices = 0;
}
}
