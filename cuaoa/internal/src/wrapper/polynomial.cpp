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

#include "polynomial.hpp"
#include "wrapper/polynomial.hpp"
#include <algorithm>
#include <cstddef>
#include <vector>

extern "C" {

PolynomialResult makePolynomialsWrapper(const double *flat, size_t dimension) {
  Polynomial polynomial = makePolynomialsfromAdjacencyMatrix(flat, dimension);
  PolynomialResult result;
  result.keysSize = polynomial.keys.size();
  result.keys = new size_t[polynomial.keys.size()];
  std::copy(polynomial.keys.begin(), polynomial.keys.end(), result.keys);
  result.valuesSize = polynomial.values.size();
  result.values = new double[polynomial.values.size()];
  std::copy(polynomial.values.begin(), polynomial.values.end(), result.values);
  return result;
}

void freePolynomialResult(PolynomialResult result) {
  if (result.keys != nullptr) {
    delete[] result.keys;
  }
  if (result.values != nullptr) {
    delete[] result.values;
  }
}
}
