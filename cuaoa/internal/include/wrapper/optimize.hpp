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

#ifndef OPTIMIZE_WRAPPER_HPP
#define OPTIMIZE_WRAPPER_HPP

#include "cuaoa/handle.hpp"
#include "wrapper/core.hpp"
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

AOAOptimizationResult optimizeCuaoaParametersWrapper(
    CUAOAHandle *handle, size_t numberOfNodes, size_t depth,
    size_t *polynomialKeys, double *polynomialValues, size_t polynomialSize,
    double *betas, double *gammas, int m, double epsilon, int past,
    double delta, int max_iterations, int linesearch, int max_linesearch,
    double min_step, double max_step, double ftol, double wolfe, double gtol,
    double xtol, double orthantwise_c, int orthantwise_start,
    int orthantwise_end, bool log, size_t blockSize, const char *rxmethod);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZE_WRAPPER_HPP
