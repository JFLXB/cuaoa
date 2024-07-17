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

#include "kernel/kernels.hpp"
#include <cfloat>
#include <cstddef>
#include <cuComplex.h>
#include <cuda_runtime.h>

__global__ void applyHcKernel(cuDoubleComplex *statevector, double *hamiltonian,
                              double gamma, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double angle = gamma * hamiltonian[i];
    cuDoubleComplex multiplier = make_cuDoubleComplex(cos(angle), sin(angle));
    statevector[i] = cuCmul(statevector[i], multiplier);
  }
}

void applyHc(cudaStream_t stream, size_t numBlocks, size_t blockSize,
             cuDoubleComplex *statevector, double *hamiltonian, double gamma,
             size_t N) {
  applyHcKernel<<<numBlocks, blockSize, 0, stream>>>(statevector, hamiltonian,
                                                     gamma, N);
}

__global__ void copyCKernel(cuDoubleComplex *statevector,
                            cuDoubleComplex *other, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    other[i] = statevector[i];
  }
}

void copyC(cudaStream_t stream, size_t numBlocks, size_t blockSize,
           cuDoubleComplex *statevector, cuDoubleComplex *other, size_t N) {
  copyCKernel<<<numBlocks, blockSize, 0, stream>>>(statevector, other, N);
}

__global__ void calcHcKernel(double *diag, size_t *polykeys, double *polyvalues,
                             size_t polysize, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double buffer = 0.0;
    for (size_t j = 0; j < polysize; j++) {
      buffer += ((i & polykeys[j]) == polykeys[j]) * polyvalues[j];
    }
    diag[i] = buffer;
  }
}

void calcHc(cudaStream_t stream, size_t numBlocks, size_t blockSize,
            double *diag, size_t *polykeys, double *polyvals, size_t polysize,
            size_t N) {
  calcHcKernel<<<numBlocks, blockSize, 0, stream>>>(diag, polykeys, polyvals,
                                                    polysize, N);
}

__global__ void findValueForIndexKernel(double *d_h, int64_t *d_samples,
                                        double *d_energies, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    d_energies[i] = d_h[d_samples[i]];
  }
}

void findValueForIndex(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                       double *d_h, int64_t *d_samples, double *d_energies,
                       size_t N) {
  findValueForIndexKernel<<<numBlocks, blockSize, 0, stream>>>(d_h, d_samples,
                                                               d_energies, N);
}

__global__ void initializeSvKernel(cuDoubleComplex *statevector,
                                   cuDoubleComplex initialValue, size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    statevector[i] = initialValue;
  }
}

void initializeSv(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                  cuDoubleComplex *statevector, cuDoubleComplex initialValue,
                  size_t N) {
  initializeSvKernel<<<numBlocks, blockSize, 0, stream>>>(statevector,
                                                          initialValue, N);
}

__global__ void calcInnerProdKernel(cuDoubleComplex *sv, cuDoubleComplex *phi,
                                    double *gradientImag, size_t N) {
  extern __shared__ cuDoubleComplex temp[];

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N) {
    cuDoubleComplex a = sv[index];
    cuDoubleComplex aconj = make_cuDoubleComplex(cuCreal(a), -cuCimag(a));
    cuDoubleComplex b = phi[index];
    temp[tid] = cuCmul(aconj, b);
  } else {
    temp[tid] = make_cuDoubleComplex(0.0, 0.0);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      cuDoubleComplex elem = temp[tid];
      cuDoubleComplex o = temp[tid + s];
      cuDoubleComplex n = make_cuDoubleComplex(cuCreal(elem) + cuCreal(o),
                                               cuCimag(elem) + cuCimag(o));
      temp[tid] = n;
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(gradientImag, cuCimag(temp[0]));
  }
}

void calcInnerProd(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                   cuDoubleComplex *sv, cuDoubleComplex *phi, double *o,
                   size_t N) {
  size_t sharedMemSize = blockSize * sizeof(cuDoubleComplex);
  calcInnerProdKernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(sv, phi,
                                                                       o, N);
}

__global__ void mulHKernel(cuDoubleComplex *statevector, double *hamiltonian,
                           size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    cuDoubleComplex v = statevector[i];
    double h = hamiltonian[i];
    statevector[i] = make_cuDoubleComplex(cuCreal(v) * h, cuCimag(v) * h);
  }
}

void mulH(cudaStream_t stream, size_t numBlocks, size_t blockSize,
          cuDoubleComplex *statevector, double *hamiltonian, size_t N) {
  mulHKernel<<<numBlocks, blockSize, 0, stream>>>(statevector, hamiltonian, N);
}

__global__ void rxKernel(cuDoubleComplex *sv, size_t q, size_t N, double beta) {
  // Based on RX implementation from https://github.com/jpmorganchase/QOKit
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (N / 2)) {
    return;
  }
  cuDoubleComplex cosang = make_cuDoubleComplex(cos(beta), 0.0);
  cuDoubleComplex sinang = make_cuDoubleComplex(0.0, -sin(beta));

  size_t m1 = (1 << q) - 1;
  size_t m2 = m1 ^ ((N - 1) >> 1);
  size_t ia = (i & m1) | ((i & m2) << 1);
  size_t ib = ia | (1 << q);

  cuDoubleComplex ta = sv[ia];
  cuDoubleComplex tb = sv[ib];

  cuDoubleComplex naA = cuCmul(ta, cosang);
  cuDoubleComplex naB = cuCmul(tb, sinang);
  cuDoubleComplex na = make_cuDoubleComplex(cuCreal(naA) + cuCreal(naB),
                                            cuCimag(naA) + cuCimag(naB));

  cuDoubleComplex nbA = cuCmul(ta, sinang);
  cuDoubleComplex nbB = cuCmul(tb, cosang);
  cuDoubleComplex nb = make_cuDoubleComplex(cuCreal(nbA) + cuCreal(nbB),
                                            cuCimag(nbA) + cuCimag(nbB));

  sv[ia] = na;
  sv[ib] = nb;
}

void applyRxGateQO(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                   double beta, cuDoubleComplex *d_sv, int32_t adjoint,
                   size_t numQubits) {
  double b = adjoint == 0 ? beta : -beta;
  size_t N = 1 << numQubits;
  numBlocks = ((N / 2) + blockSize - 1) / blockSize;
  for (size_t q = 0; q < numQubits; ++q) {
    rxKernel<<<numBlocks, blockSize, 0, stream>>>(d_sv, q, N, b);
  }
}

void applyRxGate(custatevecHandle_t handle, double beta, cuDoubleComplex *d_sv,
                 int32_t adjoint, size_t N) {
  cuDoubleComplex RX_MATRIX[] = {
      {std::cos(beta), 0.0},
      {0.0, -std::sin(beta)},
      {0.0, -std::sin(beta)},
      {std::cos(beta), 0.0},
  };

  for (int i = 0; i < N; i++) {
    int targets[] = {i};
    custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, N, RX_MATRIX, CUDA_C_64F,
                          CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, targets, 1,
                          nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr,
                          0);
  }
}

void applyXGate(custatevecHandle_t handle, cuDoubleComplex *d_sv, size_t N) {
  cuDoubleComplex X_MATRIX[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
  for (int i = 0; i < N; i++) {
    int targets[] = {i};
    custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, N, X_MATRIX, CUDA_C_64F,
                          CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr,
                          nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0);
  }
}

void applyXGateSingle(custatevecHandle_t handle, cuDoubleComplex *d_sv,
                      int qubit, size_t N) {
  cuDoubleComplex X_MATRIX[] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};
  int targets[] = {qubit};
  custatevecApplyMatrix(handle, d_sv, CUDA_C_64F, N, X_MATRIX, CUDA_C_64F,
                        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets, 1, nullptr,
                        nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0);
}

__global__ void calcZZGradKernel(cuDoubleComplex *sv, cuDoubleComplex *phi,
                                 double *d_h, double *gradientImag, size_t N) {
  extern __shared__ cuDoubleComplex temp[];

  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < N) {
    cuDoubleComplex a = sv[index];
    cuDoubleComplex aconj =
        make_cuDoubleComplex(cuCreal(a) * d_h[index], -cuCimag(a) * d_h[index]);
    cuDoubleComplex b = phi[index];
    temp[tid] = cuCmul(aconj, b);
  } else {
    temp[tid] = make_cuDoubleComplex(0.0, 0.0);
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      cuDoubleComplex elem = temp[tid];
      cuDoubleComplex o = temp[tid + s];
      cuDoubleComplex n = make_cuDoubleComplex(cuCreal(elem) + cuCreal(o),
                                               cuCimag(elem) + cuCimag(o));
      temp[tid] = n;
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(gradientImag, cuCimag(temp[0]));
  }
}

void calcZZGrad(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                cuDoubleComplex *sv, cuDoubleComplex *phi, double *h, double *o,
                size_t N) {
  size_t sharedMemSize = blockSize * sizeof(cuDoubleComplex);
  calcZZGradKernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(sv, phi, h,
                                                                    o, N);
}

__global__ void findEnergiesKernel(double *d_h, int64_t *d_samples,
                                   double *d_energies, uint32_t numShots) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < numShots) {
    d_energies[i] = d_h[d_samples[i]];
  }
}

void findEnergies(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                  double *d_h, int64_t *d_samples, double *d_energies,
                  uint32_t numShots) {
  findEnergiesKernel<<<numBlocks, blockSize, 0, stream>>>(d_h, d_samples,
                                                          d_energies, numShots);
}

__device__ double atomicMin_double(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull, assumed,
        __double_as_longlong(min(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__global__ void findMinimumKernel(double *diag, double *minVal, size_t N) {
  extern __shared__ double sdata[];

  size_t tid = threadIdx.x;
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < N) ? diag[i] : DBL_MAX;
  __syncthreads();

  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = min(sdata[tid], sdata[tid + s]);
    }
  }

  if (tid == 0)
    atomicMin_double(minVal, sdata[0]);
}

void findMinimum(cudaStream_t stream, size_t numBlocks, size_t blockSize,
                 double *d_h, double *d_minCost, size_t N) {
  findMinimumKernel<<<numBlocks, blockSize, blockSize * sizeof(double),
                      stream>>>(d_h, d_minCost, N);
}

__global__ void findIndicesOfMinKernel(double *diag, double minVal,
                                       int64_t *minIndices, int *numMatches,
                                       size_t N) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (diag[i] == minVal) {
      int index = atomicAdd(numMatches, 1);
      minIndices[index] = i;
    }
  }
}

void findIndicesOfMinimum(cudaStream_t stream, size_t numBlocks,
                          size_t blockSize, double *d_h, double minVal,
                          int64_t *d_minIndices, int *d_numMatches, size_t N) {
  findIndicesOfMinKernel<<<numBlocks, blockSize, 0, stream>>>(
      d_h, minVal, d_minIndices, d_numMatches, N);
}
