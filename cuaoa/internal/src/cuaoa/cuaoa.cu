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

#include "cuaoa/cuaoa.hpp"
#include "cuaoa/functions.hpp"
#include <cuda_runtime.h>
#include <iostream>

void core_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                size_t *polykeys, double *polyvals, size_t polysize,
                const double *betas, const double *gammas, size_t blockSize,
                const char *rxmethod) {
  cudaSetDevice(handle->device);
  initializeHandle(handle, numNodes, polykeys, polyvals, polysize, blockSize);

  initializeStatevector(handle);
  calcCostHamiltonian(handle);

  for (size_t p = 0; p < depth; p++) {
    applyCostHamiltonian(handle, gammas[p]);
    applyRxGate(handle, betas[p], 0, rxmethod);
  }

  handle->freePoly();
}

cuDoubleComplex *statevector_cuaoa(CUAOAHandle *handle, size_t numNodes,
                                   size_t depth, size_t *polykeys,
                                   double *polyvals, size_t polysize,
                                   double *betas, double *gammas,
                                   size_t blockSize, const char *rxmethod) {
  core_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize, betas,
             gammas, blockSize, rxmethod);
  return retrieveStatevector(handle);
}

double expectation_value_cuaoa(CUAOAHandle *handle, size_t numNodes,
                               size_t depth, size_t *polykeys, double *polyvals,
                               size_t polysize, double *betas, double *gammas,
                               size_t blockSize, const char *rxmethod) {
  core_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize, betas,
             gammas, blockSize, rxmethod);
  return calculateExpectationValue(handle);
}

SampleSet sample_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                       size_t *polykeys, double *polyvals, size_t polysize,
                       double *betas, double *gammas, size_t blockSize,
                       uint32_t maxShots, const uint32_t numShots,
                       const double *randnums, const char *rxmethod) {
  core_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize, betas,
             gammas, blockSize, rxmethod);
  return sampleStatevector(handle, maxShots, numShots, randnums);
}

void inner_gradients_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                           size_t *polykeys, double *polyvals, size_t polysize,
                           const double *betas, const double *gammas,
                           size_t blockSize, double *betaGradients,
                           double *gammaGradients, double *expectationValue,
                           const char *rxmethod) {
  core_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize, betas,
             gammas, blockSize, rxmethod);
  *expectationValue = calculateExpectationValue(handle);
  handle->mallocForGradient();
  calculateGradients(handle, depth, betas, gammas, betaGradients,
                     gammaGradients, rxmethod);
  handle->freeForGradient();
}

Gradient gradients_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                         size_t *polykeys, double *polyvals, size_t polysize,
                         const double *betas, const double *gammas,
                         size_t blockSize, const char *rxmethod) {
  Gradient grads;
  grads.size = depth;
  grads.beta = new double[depth];
  grads.gamma = new double[depth];
  inner_gradients_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize,
                        betas, gammas, blockSize, grads.beta, grads.gamma,
                        &grads.expectation_value, rxmethod);
  return grads;
}
