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

#include "wrapper/optimize.hpp"
#include "cuaoa/optimize.hpp"
#include <cstdio>

extern "C" {

AOAOptimizationResult optimizeCuaoaParametersWrapper(
    CUAOAHandle *handle, size_t numberOfNodes, size_t depth,
    size_t *polynomialKeys, double *polynomialValues, size_t polynomialSize,
    double *betas, double *gammas, int m, double epsilon, int past,
    double delta, int max_iterations, int linesearch, int max_linesearch,
    double min_step, double max_step, double ftol, double wolfe, double gtol,
    double xtol, double orthantwise_c, int orthantwise_start,
    int orthantwise_end, bool log, size_t blockSize, const char *rxmethod) {
  AOAOptimizationResult result;
  CuaoaOptimized optimized = optimize_cuaoa(
      handle, numberOfNodes, depth, polynomialKeys, polynomialValues,
      polynomialSize, betas, gammas, m, epsilon, past, delta, max_iterations,
      linesearch, max_linesearch, min_step, max_step, ftol, wolfe, gtol, xtol,
      orthantwise_c, orthantwise_start, orthantwise_end, log, blockSize,
      rxmethod);
  result.iteration = optimized.iteration;
  result.n_evals = optimized.n_evals;
  result.betas = optimized.betas;
  result.gammas = optimized.gammas;
  result.status = optimized.status;
  result.fx_log = optimized.fx_log;
  result.x_log = optimized.x_log;
  result.has_log = optimized.has_log;
  return result;
}
}
