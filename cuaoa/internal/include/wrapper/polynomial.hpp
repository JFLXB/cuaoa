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

#ifndef POLYNOMIAL_WRAPPER_HPP
#define POLYNOMIAL_WRAPPER_HPP

#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

struct PolynomialResult {
  size_t *keys = nullptr;
  size_t keysSize = 0;
  double *values = nullptr;
  size_t valuesSize = 0;
};

void freePolynomialResult(PolynomialResult result);

PolynomialResult makePolynomialsWrapper(const double *flat, size_t dimension);

#ifdef __cplusplus
}
#endif

#endif
