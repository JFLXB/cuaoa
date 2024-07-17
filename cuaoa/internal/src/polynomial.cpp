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
#include <map>
#include <ranges>

Polynomial makePolynomialsfromAdjacencyMatrix(const double *flat,
                                              size_t dimension) {
  std::map<size_t, double> polynomial;

  for (size_t i = 0; i < dimension; i++) {
    double contrib = 0.0;
    for (size_t j = 0; j < dimension; j++) {
      size_t index = i * dimension + j;
      double value = flat[index];

      bool hasValue = value != 0;
      contrib += value;

      if (i == j) {
        continue;
      }

      if (hasValue) {
        size_t key = (1 << i) + (1 << j);
        if (polynomial.count(key) > 0) {
          polynomial[key] += value;
        } else {
          polynomial[key] = +value;
        }
      }
    }

    size_t key = (1 << i);
    if (contrib != 0) {
      if (polynomial.count(key) > 0) {
        polynomial[key] -= contrib;
      } else {
        polynomial[key] = -contrib;
      }
    }
  }

  auto keys = std::views::keys(polynomial);
  auto vals = std::views::values(polynomial);

  Polynomial pols;
  pols.keys = std::vector<size_t>{keys.begin(), keys.end()};
  pols.values = std::vector<double>{vals.begin(), vals.end()};
  return pols;
}
