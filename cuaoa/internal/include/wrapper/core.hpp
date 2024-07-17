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

#ifndef CORE_WRAPPER_HPP
#define CORE_WRAPPER_HPP

#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

// BFSampleSetResult

struct BFSampleSetResult {
  size_t size = 0;
  int64_t *bitStrings = nullptr;
  double cost = 0.0;
};

void freeBFSampleSetResult(BFSampleSetResult *result);
const int64_t *getBFSamples(BFSampleSetResult *result);

// AOAStatevectorResult

struct AOAStatevectorResult {
  char *error = nullptr;
  cuDoubleComplex *statevector = nullptr;
  size_t size = 0;
};

void freeAOAStatevectorResult(AOAStatevectorResult *result);
const cuDoubleComplex *getStateVectorData(AOAStatevectorResult *result);

// AOAGradientsResult

struct AOAGradientsResult {
  double *gammaGradients = nullptr;
  double *betaGradients = nullptr;
  size_t size = 0;
  double expectationValue = 0;
};

void freeAOAGradientsResult(AOAGradientsResult *result);
const double *getGammaGradientsData(AOAGradientsResult *result);
const double *getBetaGradientsData(AOAGradientsResult *result);

// AOAOptimizationResult

struct AOAOptimizationResult {
  double *gammas = nullptr;
  double *betas = nullptr;
  double *fx_log = nullptr;
  double *x_log = nullptr;
  bool has_log = false;
  int64_t iteration = 0;
  int64_t n_evals = 0;
  int status = 0;
};

void freeAOAOptimizationResult(AOAOptimizationResult *result);
const double *getGammas(AOAOptimizationResult *result);
const double *getBetas(AOAOptimizationResult *result);

// AOASampleSet

struct AOASampleSet {
  size_t size = 0;
  int64_t *bitStrings = nullptr;
  double *energies = nullptr;
};

void freeAOASampleSet(AOASampleSet *result);
const int64_t *getAOASamples(AOASampleSet *result);
const double *getAOAEnergies(AOASampleSet *result);

#ifdef __cplusplus
}
#endif
#endif // CORE_WRAPPER_HPP
