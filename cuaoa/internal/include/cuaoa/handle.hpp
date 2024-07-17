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

#ifndef HANDLE_HPP
#define HANDLE_HPP

#include <cstring>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <custatevec.h>
#include <iostream>

struct CUAOAHandle {
  int device;
  bool exact = false;
  cudaStream_t stream;

  cuDoubleComplex *deviceStatevector;
  double *deviceCostHamiltonian;
  size_t *devicePolyKeys;
  double *devicePolyVals;
  size_t *polykeys;
  double *polyvals;

  double *deviceIntermediate;

  cuDoubleComplex *devicePhi;
  double *deviceGradientHolder;
  double *pinnedHostGrad;

  size_t maxNumBits;
  size_t maxPolySize;

  custatevecHandle_t handle;

  size_t numNodes = 0;
  size_t numStates = 0;
  size_t polySize = 0;
  size_t blockSize = 0;
  size_t numBlocks = 0;

  CUAOAHandle(size_t maxBits, int cudaDevice, bool exact = false)
      : device(cudaDevice), exact(exact), stream(nullptr),
        deviceStatevector(nullptr), deviceCostHamiltonian(nullptr),
        devicePolyKeys(nullptr), devicePolyVals(nullptr), polykeys(nullptr),
        polyvals(nullptr), deviceIntermediate(nullptr), maxNumBits(maxBits) {
    cudaSetDevice(device);
    custatevecCreate(&handle);
    cudaStreamCreate(&stream);
    custatevecSetStream(handle, stream);

    size_t numStates = 1 << maxBits;

    if (!exact) {
      cudaMallocAsync(&deviceStatevector, numStates * sizeof(cuDoubleComplex),
                      stream);
    }
    cudaMallocAsync(&deviceCostHamiltonian, numStates * sizeof(double), stream);
  }

  ~CUAOAHandle() {
    cudaSetDevice(device);

    if (!exact) {
      cudaFreeAsync(deviceStatevector, stream);
    }

    cudaFreeAsync(deviceCostHamiltonian, stream);
    custatevecDestroy(handle);
    cudaStreamDestroy(stream);

    polykeys = nullptr;
    polyvals = nullptr;
    polySize = 0;

    numNodes = 0;
    numStates = 0;
    blockSize = 0;
    numBlocks = 0;
    exact = false;
  }

  void resetGradientHolder() {
    cudaMemsetAsync(deviceGradientHolder, 0, sizeof(double), stream);
  }

  void copyGradient(double *hostGradient) {
    cudaMemcpyAsync(pinnedHostGrad, deviceGradientHolder, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    std::memcpy(pinnedHostGrad, hostGradient, sizeof(double));
  }

  void mallocForGradient() {
    cudaMallocAsync(&devicePhi, numStates * sizeof(cuDoubleComplex), stream);
    cudaMallocAsync(&deviceGradientHolder, sizeof(double), stream);
  }

  void mallocPoly(size_t *polykeys, double *polyvals, size_t polysize) {
    cudaMallocAsync(&devicePolyKeys, polysize * sizeof(size_t), stream);
    cudaMallocAsync(&devicePolyVals, polysize * sizeof(double), stream);
    polykeys = polykeys;
    polyvals = polyvals;
    polySize = polysize;
  }

  void freePoly() {
    cudaFreeAsync(devicePolyKeys, stream);
    cudaFreeAsync(devicePolyVals, stream);
    polykeys = nullptr;
    polyvals = nullptr;
    polySize = 0;
    cudaStreamSynchronize(stream);
  }

  void freeForGradient() {
    cudaFreeAsync(devicePhi, stream);
    cudaFreeAsync(deviceGradientHolder, stream);
    cudaFreeHost(pinnedHostGrad);
    cudaStreamSynchronize(stream);
  }

  void mallocForExpval() {
    cudaMallocAsync(&deviceIntermediate, numStates * sizeof(double), stream);
  }

  void freeForExpval() {
    cudaFreeAsync(deviceIntermediate, stream);
    cudaStreamSynchronize(stream);
  }
};

#endif
