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

#include "cuaoa/functions.hpp"
#include "kernel/kernels.hpp"
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

cuDoubleComplex calcInitialValue(size_t numStates) {
  return make_cuDoubleComplex(1.0 / std::sqrt(numStates), 0.0);
}

void initializeHandle(CUAOAHandle *handle, size_t numNodes, size_t *polykeys,
                      double *polyvals, size_t polysize, size_t blockSize) {
  size_t numStates = 1 << numNodes;
  size_t numBlocks = (numStates + blockSize - 1) / blockSize;

  handle->mallocPoly(polykeys, polyvals, polysize);

  cudaMemcpyAsync(handle->devicePolyKeys, polykeys, polysize * sizeof(size_t),
                  cudaMemcpyHostToDevice, handle->stream);
  cudaMemcpyAsync(handle->devicePolyVals, polyvals, polysize * sizeof(double),
                  cudaMemcpyHostToDevice, handle->stream);
  handle->polykeys = polykeys;
  handle->polyvals = polyvals;
  handle->numNodes = numNodes;
  handle->numStates = 1 << numNodes;
  handle->polySize = polysize;
  handle->blockSize = blockSize;
  handle->numBlocks = numBlocks;
}

void initializeStatevector(CUAOAHandle *handle) {
  cuDoubleComplex initVal = calcInitialValue(handle->numStates);
  initializeSv(handle->stream, handle->numBlocks, handle->blockSize,
               handle->deviceStatevector, initVal, handle->numStates);
}

double calculateExpectationValue(CUAOAHandle *handle) {
  handle->mallocForExpval();
  double expval =
      calcExpval(handle->stream, handle->blockSize, handle->deviceStatevector,
                 handle->deviceCostHamiltonian, handle->deviceIntermediate,
                 handle->numStates);
  handle->freeForExpval();
  return expval;
}

cuDoubleComplex *retrieveStatevector(CUAOAHandle *handle) {
  cuDoubleComplex *hostStatevector;
  cudaMallocHost(&hostStatevector, handle->numStates * sizeof(cuDoubleComplex));
  cudaMemcpyAsync(hostStatevector, handle->deviceStatevector,
                  handle->numStates * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost, handle->stream);
  cudaStreamSynchronize(handle->stream);
  return hostStatevector;
}

double *retrieveCostHamiltonian(CUAOAHandle *handle) {
  double *hostCostHamiltonian;
  cudaMallocHost(&hostCostHamiltonian, handle->numStates * sizeof(double));
  cudaMemcpyAsync(hostCostHamiltonian, handle->deviceCostHamiltonian,
                  handle->numStates * sizeof(double), cudaMemcpyDeviceToHost,
                  handle->stream);
  return hostCostHamiltonian;
}

size_t *retrievePolykeys(CUAOAHandle *handle) {
  size_t *host;
  cudaMallocHost(&host, handle->polySize * sizeof(size_t));
  cudaMemcpyAsync(host, handle->devicePolyKeys,
                  handle->polySize * sizeof(size_t), cudaMemcpyDeviceToHost,
                  handle->stream);
  return host;
}

double *retrievePolyVals(CUAOAHandle *handle) {
  double *host;
  cudaMallocHost(&host, handle->polySize * sizeof(double));
  cudaMemcpyAsync(host, handle->devicePolyVals,
                  handle->polySize * sizeof(double), cudaMemcpyDeviceToHost,
                  handle->stream);
  return host;
}

void applyCostHamiltonian(CUAOAHandle *handle, double gamma) {
  applyHc(handle->stream, handle->numBlocks, handle->blockSize,
          handle->deviceStatevector, handle->deviceCostHamiltonian, gamma,
          handle->numStates);
}

void applyRxGate(CUAOAHandle *handle, double beta, int32_t adjoint,
                 const char *method) {
  if (strcmp(method, "custatevec") == 0) {
    applyRxGate(handle, beta, adjoint);
  } else if (strcmp(method, "qokit") == 0) {
    applyRxGateQO(handle, beta, adjoint);
  } else {
    applyRxGate(handle, beta, adjoint);
  }
}

void applyRxGate(CUAOAHandle *handle, double beta, int32_t adjoint) {
  applyRxGate(handle->handle, beta, handle->deviceStatevector, adjoint,
              handle->numNodes);
}

void applyRxGateQO(CUAOAHandle *handle, double beta, int32_t adjoint) {
  applyRxGateQO(handle->stream, handle->numBlocks, handle->blockSize, beta,
                handle->deviceStatevector, adjoint, handle->numNodes);
}

void calcCostHamiltonian(CUAOAHandle *handle) {
  calcHc(handle->stream, handle->numBlocks, handle->blockSize,
         handle->deviceCostHamiltonian, handle->devicePolyKeys,
         handle->devicePolyVals, handle->polySize, handle->numStates);
}

SampleSet sampleStatevector(CUAOAHandle *handle, uint32_t maxShots,
                            const uint32_t numShots, const double *randnums) {
  custatevecSamplerDescriptor_t sampler;
  size_t extraWorkspaceSizeInBytes = 0;
  custatevecStatus_t status_a = custatevecSamplerCreate(
      handle->handle, handle->deviceStatevector, CUDA_C_64F, handle->numNodes,
      &sampler, maxShots, &extraWorkspaceSizeInBytes);

  void *extraWorkspace = nullptr;
  if (extraWorkspaceSizeInBytes > 0) {
    cudaMallocAsync((void **)&extraWorkspace, extraWorkspaceSizeInBytes,
                    handle->stream);
  }

  custatevecStatus_t status_b = custatevecSamplerPreprocess(
      handle->handle, sampler, extraWorkspace, extraWorkspaceSizeInBytes);

  SampleSet sample_set;
  sample_set.bitStrings = new int64_t[numShots];

  int32_t *bitOrderingT = new int32_t[handle->numNodes];
  for (int32_t i = 0; i < handle->numNodes; i++) {
    bitOrderingT[i] = i;
  }
  const int32_t *bitOrdering = bitOrderingT;
  custatevecSamplerOutput_t output = CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER;

  custatevecStatus_t status_c = custatevecSamplerSample(
      handle->handle, sampler, (custatevecIndex_t *)sample_set.bitStrings,
      bitOrdering, handle->numNodes, randnums, numShots, output);

  custatevecStatus_t status_d = custatevecSamplerDestroy(sampler);
  cudaFreeAsync(extraWorkspace, handle->stream);

  double *d_energies;
  int64_t *d_samples;

  cudaMallocAsync((void **)&d_energies, numShots * sizeof(double),
                  handle->stream);
  cudaMallocAsync((void **)&d_samples, numShots * sizeof(int64_t),
                  handle->stream);

  cudaMemcpyAsync(d_samples, sample_set.bitStrings, numShots * sizeof(int64_t),
                  cudaMemcpyHostToDevice, handle->stream);

  findEnergies(handle->stream, handle->numBlocks, handle->blockSize,
               handle->deviceCostHamiltonian, d_samples, d_energies, numShots);

  sample_set.energies = new double[numShots];
  cudaMemcpyAsync(sample_set.energies, d_energies, numShots * sizeof(double),
                  cudaMemcpyDeviceToHost, handle->stream);
  cudaFreeAsync(d_energies, handle->stream);
  cudaFreeAsync(d_samples, handle->stream);

  return sample_set;
}

void calculateGradients(CUAOAHandle *handle, size_t depth, const double *betas,
                        const double *gammas, double *betaGradients,
                        double *gammaGradients, const char *rxmethod) {
  cudaStreamSynchronize(handle->stream);
  copyC(handle->stream, handle->numBlocks, handle->blockSize,
        handle->deviceStatevector, handle->devicePhi, handle->numStates);
  mulH(handle->stream, handle->numBlocks, handle->blockSize,
       handle->deviceStatevector, handle->deviceCostHamiltonian,
       handle->numStates);

  double *pinnedHostGrad;
  cudaMallocHost(&pinnedHostGrad, sizeof(double));

  for (int p = depth - 1; p >= 0; p--) {
    betaGradients[p] = 0.0;
    for (int q = 0; q < handle->numNodes; q++) {
      handle->resetGradientHolder();
      applyXGateSingle(handle->handle, handle->devicePhi, q, handle->numNodes);
      calcInnerProd(handle->stream, handle->numBlocks, handle->blockSize,
                    handle->deviceStatevector, handle->devicePhi,
                    handle->deviceGradientHolder, handle->numStates);
      applyXGateSingle(handle->handle, handle->devicePhi, q, handle->numNodes);
      cudaMemcpyAsync(pinnedHostGrad, handle->deviceGradientHolder,
                      sizeof(double), cudaMemcpyDeviceToHost, handle->stream);
      cudaStreamSynchronize(handle->stream);
      betaGradients[p] -= *pinnedHostGrad;
    }
    betaGradients[p] *= -2.0;

    if (rxmethod == "custatevec") {
      applyRxGate(handle->handle, betas[p], handle->deviceStatevector, 1,
                  handle->numNodes);
      applyRxGate(handle->handle, betas[p], handle->devicePhi, 1,
                  handle->numNodes);
    } else if (rxmethod == "qokit") {
      applyRxGateQO(handle->stream, handle->numBlocks, handle->blockSize,
                    betas[p], handle->deviceStatevector, 1, handle->numNodes);
      applyRxGateQO(handle->stream, handle->numBlocks, handle->blockSize,
                    betas[p], handle->devicePhi, 1, handle->numNodes);
    } else {
      applyRxGate(handle->handle, betas[p], handle->deviceStatevector, 1,
                  handle->numNodes);
      applyRxGate(handle->handle, betas[p], handle->devicePhi, 1,
                  handle->numNodes);
    }
    handle->resetGradientHolder();

    calcZZGrad(handle->stream, handle->numBlocks, handle->blockSize,
               handle->deviceStatevector, handle->devicePhi,
               handle->deviceCostHamiltonian, handle->deviceGradientHolder,
               handle->numStates);

    cudaMemcpyAsync(pinnedHostGrad, handle->deviceGradientHolder,
                    sizeof(double), cudaMemcpyDeviceToHost, handle->stream);
    cudaStreamSynchronize(handle->stream);
    gammaGradients[p] = -2.0 * *pinnedHostGrad;

    if (p > 0) {
      applyHc(handle->stream, handle->numBlocks, handle->blockSize,
              handle->deviceStatevector, handle->deviceCostHamiltonian,
              -gammas[p], handle->numStates);
      applyHc(handle->stream, handle->numBlocks, handle->blockSize,
              handle->devicePhi, handle->deviceCostHamiltonian, -gammas[p],
              handle->numStates);
    }
    cudaStreamSynchronize(handle->stream);
  }
}

int64_t *findMinima(CUAOAHandle *handle, double *minimum, size_t *numMatches) {
  *minimum = DBL_MAX;
  *numMatches = 0;

  int64_t *d_minIndices;
  int *d_numMatches;
  double *d_minCost;

  cudaMallocAsync(&d_minIndices, handle->numStates * sizeof(int64_t),
                  handle->stream);
  cudaMallocAsync(&d_numMatches, sizeof(int), handle->stream);
  cudaMallocAsync(&d_minCost, sizeof(double), handle->stream);

  cudaMemcpyAsync(d_minCost, minimum, sizeof(double), cudaMemcpyHostToDevice,
                  handle->stream);
  cudaMemcpyAsync(d_numMatches, numMatches, sizeof(int), cudaMemcpyHostToDevice,
                  handle->stream);

  findMinimum(handle->stream, handle->numBlocks, handle->blockSize,
              handle->deviceCostHamiltonian, d_minCost, handle->numStates);

  cudaMemcpyAsync(minimum, d_minCost, sizeof(double), cudaMemcpyDeviceToHost,
                  handle->stream);

  findIndicesOfMinimum(handle->stream, handle->numBlocks, handle->blockSize,
                       handle->deviceCostHamiltonian, *minimum, d_minIndices,
                       d_numMatches, handle->numStates);

  int numM;
  cudaMemcpyAsync(&numM, d_numMatches, sizeof(int), cudaMemcpyDeviceToHost,
                  handle->stream);

  *numMatches = (size_t)numM;
  int64_t *bitStrings;
  cudaMallocHost(&bitStrings, *numMatches * sizeof(int64_t));
  cudaMemcpyAsync(bitStrings, d_minIndices, *numMatches * sizeof(int64_t),
                  cudaMemcpyDeviceToHost, handle->stream);

  cudaFreeAsync(d_minIndices, handle->stream);
  cudaFreeAsync(d_numMatches, handle->stream);
  cudaFreeAsync(d_minCost, handle->stream);

  cudaStreamSynchronize(handle->stream);

  return bitStrings;
}
