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

#include "wrapper/core.hpp"
#include <cuda_runtime.h>

extern "C" {

// BFSampleSetResult

void freeBFSampleSetResult(BFSampleSetResult *result) {
  if (!result)
    return;

  cudaFreeHost(result->bitStrings);
  result->bitStrings = nullptr;
  result->cost = 0.0;
  result->size = 0;
}

const int64_t *getBFSamples(BFSampleSetResult *result) {
  return result ? result->bitStrings : nullptr;
}

// AOAStatevectorResult

void freeAOAStatevectorResult(AOAStatevectorResult *result) {
  if (!result)
    return;
  cudaFreeHost(result->statevector);
  result->statevector = nullptr;
  if (result->error) {
    free(result->error);
    result->error = nullptr;
  }
}

const cuDoubleComplex *getStateVectorData(AOAStatevectorResult *result) {
  return result ? result->statevector : nullptr;
}

// AOAGradientsResult

void freeAOAGradientsResult(AOAGradientsResult *result) {
  if (!result)
    return;

  delete[] result->gammaGradients;
  result->gammaGradients = nullptr;
  delete[] result->betaGradients;
  result->betaGradients = nullptr;

  result->size = 0;
  result->expectationValue = 0;
}

const double *getGammaGradientsData(AOAGradientsResult *result) {
  return result ? result->gammaGradients : nullptr;
}
const double *getBetaGradientsData(AOAGradientsResult *result) {
  return result ? result->betaGradients : nullptr;
}

// OptimizationResult

void freeAOAOptimizationResult(AOAOptimizationResult *result) {
  if (!result)
    return;

  delete[] result->betas;
  result->betas = nullptr;
  delete[] result->gammas;
  result->gammas = nullptr;
  delete[] result->fx_log;
  result->fx_log = nullptr;
  delete[] result->x_log;
  result->x_log = nullptr;
  result->iteration = 0;
  result->n_evals = 0;
  result->status = 0;
  result->has_log = false;
}
const double *getGammas(AOAOptimizationResult *result) {
  return result ? result->gammas : nullptr;
}
const double *getBetas(AOAOptimizationResult *result) {
  return result ? result->betas : nullptr;
}
const double *getFxLog(AOAOptimizationResult *result) {
  return result ? result->fx_log : nullptr;
}
const double *getXLog(AOAOptimizationResult *result) {
  return result ? result->x_log : nullptr;
}

// AOASampleSet

void freeAOASampleSet(AOASampleSet *result) {
  if (!result)
    return;

  delete[] result->bitStrings;
  result->bitStrings = nullptr;
  delete[] result->energies;
  result->energies = nullptr;
  result->size = 0;
}

const int64_t *getAOASamples(AOASampleSet *result) {
  return result ? result->bitStrings : nullptr;
}

const double *getAOAEnergies(AOASampleSet *result) {
  return result ? result->energies : nullptr;
}
}
