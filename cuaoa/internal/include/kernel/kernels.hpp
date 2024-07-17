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

#ifndef KERNELS_HPP
#define KERNELS_HPP

#include "cuaoa/handle.hpp"
#include <cuComplex.h>
#include <custatevec.h>

void applyHc(cudaStream_t stream, size_t numBlocks, size_t blockSize,
             cuDoubleComplex *statevector, double *hamiltonian, double gamma,
             size_t N);

void calcHc(cudaStream_t stream, size_t numBlocks, size_t blockSize,
            double *diag, size_t *polykeys, double *polyvals, size_t polysize,
            size_t N);

void copyC(cudaStream_t stream, size_t numBlocks, size_t blockSize,
           cuDoubleComplex *statevector, cuDoubleComplex *other, size_t N);

double sumReduce(cudaStream_t stream, size_t blockSize, double *d_data,
                 size_t N);

double calcExpval(cudaStream_t stream, size_t blockSize, cuDoubleComplex *sv,
                  double *h, double *intermediate, size_t N);

void findValueForIndex(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                       double *d_h, int64_t *d_samples, double *d_energies,
                       size_t N);

void initializeSv(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                  cuDoubleComplex *statevector, cuDoubleComplex initialValue,
                  size_t N);

void calcInnerProd(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                   cuDoubleComplex *sv, cuDoubleComplex *phi, double *o,
                   size_t N);

void mulH(cudaStream_t stream, size_t numBlocks, size_t blockSize,
          cuDoubleComplex *statevector, double *hamiltonian, size_t N);

void applyRxGate(custatevecHandle_t handle, double beta, cuDoubleComplex *d_sv,
                 int32_t adjoint, size_t N);
void applyRxGateQO(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                   double beta, cuDoubleComplex *d_sv, int32_t adjoint,
                   size_t numQubits);

void applyXGateSingle(custatevecHandle_t handle, cuDoubleComplex *d_sv,
                      int qubit, size_t N);

void calcZZGrad(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                cuDoubleComplex *sv, cuDoubleComplex *phi, double *d_h,
                double *o, size_t N);

void findEnergies(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                  double *d_h, int64_t *d_samples, double *d_energies,
                  uint32_t numShots);

void findMinimum(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                 double *d_h, double *d_maxCost, size_t N);

void findIndicesOfMinimum(cudaStream_t stream, size_t numBlocks,
                          size_t blockSize, double *d_h, double maxVal,
                          int64_t *d_maxIndices, int *d_numMatches, size_t N);

#endif
