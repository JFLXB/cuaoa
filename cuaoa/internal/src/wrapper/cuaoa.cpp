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
#include "wrapper/cuaoa.hpp"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

extern "C" {

AOAStatevectorResult statevecCuaoaWrapper(CUAOAHandle *handle, size_t numNodes,
                                          size_t depth, size_t *polykeys,
                                          double *polyvals, size_t polysize,
                                          double *betas, double *gammas,
                                          size_t blockSize,
                                          const char *rxmethod) {
  AOAStatevectorResult result;
  result.statevector =
      statevector_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize,
                        betas, gammas, blockSize, rxmethod);
  result.size = 1 << numNodes;
  return result;
}

double expvalCuaoaWrapper(CUAOAHandle *handle, size_t numNodes, size_t depth,
                          size_t *polynomialKeys, double *polynomialValues,
                          size_t polynomialSize, double *betas, double *gammas,
                          size_t blockSize, const char *rxmethod) {

  size_t *pinnedPolynomialKeys;
  cudaMallocHost(&pinnedPolynomialKeys, polynomialSize * sizeof(size_t));
  std::memcpy(pinnedPolynomialKeys, polynomialKeys,
              polynomialSize * sizeof(size_t));

  double *pinnedPolynomialValues;
  cudaMallocHost(&pinnedPolynomialValues, polynomialSize * sizeof(double));
  std::memcpy(pinnedPolynomialValues, polynomialValues,
              polynomialSize * sizeof(double));

  double *pinnedBetas;
  cudaMallocHost(&pinnedBetas, depth * sizeof(double));
  std::memcpy(pinnedBetas, betas, depth * sizeof(double));

  double *pinnedGammas;
  cudaMallocHost(&pinnedGammas, depth * sizeof(double));
  std::memcpy(pinnedGammas, gammas, depth * sizeof(double));

  double result = expectation_value_cuaoa(
      handle, numNodes, depth, pinnedPolynomialKeys, pinnedPolynomialValues,
      polynomialSize, pinnedBetas, pinnedGammas, blockSize, rxmethod);

  cudaFreeHost(pinnedPolynomialKeys);
  cudaFreeHost(pinnedPolynomialValues);
  cudaFreeHost(pinnedBetas);
  cudaFreeHost(pinnedGammas);

  return result;
}

AOAGradientsResult gradientsCuaoaWrapper(CUAOAHandle *handle, size_t numNodes,
                                         size_t depth, size_t *polykeys,
                                         double *polyvals, size_t polysize,
                                         const double *betas,
                                         const double *gammas, size_t blockSize,
                                         const char *rxmethod) {
  auto gradients =
      gradients_cuaoa(handle, numNodes, depth, polykeys, polyvals, polysize,
                      betas, gammas, blockSize, rxmethod);
  AOAGradientsResult result;
  result.betaGradients = gradients.beta;
  result.gammaGradients = gradients.gamma;
  result.size = gradients.size;
  result.expectationValue = gradients.expectation_value;
  return result;
}

AOASampleSet sampleCuaoaWrapper(CUAOAHandle *handle, size_t numNodes,
                                size_t depth, size_t *polykeys,
                                double *polyvals, size_t polysize,
                                double *betas, double *gammas,
                                uint32_t maxShots, const uint32_t numShots,
                                const double *randnums, size_t blockSize,
                                const char *rxmethod) {
  size_t *pinnedPolynomialKeys;
  cudaMallocHost(&pinnedPolynomialKeys, polysize * sizeof(size_t));
  std::memcpy(pinnedPolynomialKeys, polykeys, polysize * sizeof(size_t));

  double *pinnedPolynomialValues;
  cudaMallocHost(&pinnedPolynomialValues, polysize * sizeof(double));
  std::memcpy(pinnedPolynomialValues, polyvals, polysize * sizeof(double));

  double *pinnedBetas;
  cudaMallocHost(&pinnedBetas, depth * sizeof(double));
  std::memcpy(pinnedBetas, betas, depth * sizeof(double));

  double *pinnedGammas;
  cudaMallocHost(&pinnedGammas, depth * sizeof(double));
  std::memcpy(pinnedGammas, gammas, depth * sizeof(double));

  AOASampleSet sample_set;
  auto out =
      sample_cuaoa(handle, numNodes, depth, pinnedPolynomialKeys,
                   pinnedPolynomialValues, polysize, pinnedBetas, pinnedGammas,
                   blockSize, maxShots, numShots, randnums, rxmethod);

  cudaFreeHost(pinnedPolynomialKeys);
  cudaFreeHost(pinnedPolynomialValues);
  cudaFreeHost(pinnedBetas);
  cudaFreeHost(pinnedGammas);

  sample_set.bitStrings = out.bitStrings;
  sample_set.energies = out.energies;
  sample_set.size = numShots;
  return sample_set;
}
}
