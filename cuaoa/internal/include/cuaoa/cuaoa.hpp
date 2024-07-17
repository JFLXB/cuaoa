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

#ifndef CUAOA_HPP
#define CUAOA_HPP

#include "cuaoa/handle.hpp"
#include <cstddef>

struct SampleSet {
  int64_t *bitStrings = nullptr;
  double *energies = nullptr;
};

struct Gradient {
  double *beta;
  double *gamma;
  size_t size;
  double expectation_value;
};

cuDoubleComplex *statevector_cuaoa(CUAOAHandle *handle, size_t numNodes,
                                   size_t depth, size_t *polykeys,
                                   double *polyvals, size_t polysize,
                                   double *betas, double *gammas,
                                   size_t blockSize, const char *rxmethod);

double expectation_value_cuaoa(CUAOAHandle *handle, size_t numNodes,
                               size_t depth, size_t *polykeys, double *polyvals,
                               size_t polysize, double *betas, double *gammas,
                               size_t blockSize, const char *rxmethod);

SampleSet sample_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                       size_t *polykeys, double *polyvals, size_t polysize,
                       double *betas, double *gammas, size_t blockSize,
                       uint32_t maxShots, const uint32_t numShots,
                       const double *randnums, const char *rxmethod);

void inner_gradients_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                           size_t *polykeys, double *polyvals, size_t polysize,
                           const double *betas, const double *gammas,
                           size_t blockSize, double *betaGradients,
                           double *gammaGradients, double *expectationValue,
                           const char *rxmethod);

Gradient gradients_cuaoa(CUAOAHandle *handle, size_t numNodes, size_t depth,
                         size_t *polykeys, double *polyvals, size_t polysize,
                         const double *betas, const double *gammas,
                         size_t blockSize, const char *rxmethod);

#endif // CUAOA_HPP
