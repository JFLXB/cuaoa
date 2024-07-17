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
#include <cstdio>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>

__inline__ __device__ double warpReduceSum(double sum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  return sum;
}

__global__ void preprocessKernel(cuDoubleComplex *c, double *h, double *output,
                                 size_t N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < N) {
    cuDoubleComplex amplitude = c[index];
    double realPart = cuCreal(amplitude);
    double imagPart = cuCimag(amplitude);
    double magnitude = realPart * realPart + imagPart * imagPart;
    double value = magnitude * h[index];
    output[index] = value;
  }
}

__global__ void reductionKernel(double *input, double *output, size_t N) {
  extern __shared__ double sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int index = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  double sum = 0;
  if (index < N)
    sum = input[index];
  if (index + blockDim.x < N)
    sum += input[index + blockDim.x];

  sdata[tid] = sum;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (blockDim.x >= 64 && tid < 32) {
    sdata[tid] = sum = sdata[tid] + sdata[tid + 32];
  }
  __syncthreads();

  if (tid < 32) {
    sum = warpReduceSum(sum);
    if (tid == 0) {
      output[blockIdx.x] = sum;
    }
  }
}

double sumReduce(cudaStream_t stream, size_t threads, cuDoubleComplex *d_a,
                 double *d_b, double *d_input, size_t N) {
  double *d_blockSums;
  double *d_finalSum;

  size_t preBlocks = (N + threads - 1) / threads;
  size_t blocks = (N + (threads * 2 - 1)) / (threads * 2);

  cudaMallocAsync(&d_blockSums, blocks * sizeof(double), stream);
  cudaMallocAsync(&d_finalSum, sizeof(double), stream);

  preprocessKernel<<<preBlocks, threads, 0, stream>>>(d_a, d_b, d_input, N);
  reductionKernel<<<blocks, threads, threads * sizeof(double), stream>>>(
      d_input, d_blockSums, N);

  size_t numElements = blocks;
  bool switched = true;
  while (numElements > 1) {
    size_t numBlocks = (numElements + (threads * 2 - 1)) / (threads * 2);
    if (switched) {
      reductionKernel<<<numBlocks, threads, threads * sizeof(double), stream>>>(
          d_blockSums, d_input, numElements);
    } else {
      reductionKernel<<<numBlocks, threads, threads * sizeof(double), stream>>>(
          d_input, d_blockSums, numElements);
    }
    numElements = numBlocks;
    switched = !switched;
  }
  double *pinnedFinalSum;
  cudaMallocHost(&pinnedFinalSum, sizeof(double));
  if (switched) {
    cudaMemcpyAsync(pinnedFinalSum, d_blockSums, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
  } else {
    cudaMemcpyAsync(pinnedFinalSum, d_input, sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
  }
  cudaStreamSynchronize(stream);
  double finalSum = *pinnedFinalSum;
  cudaFreeHost(pinnedFinalSum);
  cudaFreeAsync(d_blockSums, stream);
  cudaFreeAsync(d_finalSum, stream);
  cudaStreamSynchronize(stream);
  return finalSum;
}

double calcExpval(cudaStream_t stream, size_t blockSize, cuDoubleComplex *sv,
                  double *h, double *intermediate, size_t N) {
  return sumReduce(stream, blockSize, sv, h, intermediate, N);
}
