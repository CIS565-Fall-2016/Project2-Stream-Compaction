#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void printArr(int n, const int* data) {
  printf("    [ ");
  for (int i = 0; i < n; ++i) {
    printf("%3d ", data[i]);
  }
  printf("]\n");
}

/*
 * Performs one iteration of a naive scan of N elements. pow2d = 2^depth
 */
__global__ void naiveScanIteration(int N, int pow2d, int* odata, const int* idata) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= N) return;
  if (index >= pow2d) {
    odata[index] = idata[index] + idata[index - pow2d];
  }
  else {
    // we've already processed these elements. just copy them
    odata[index] = idata[index];
  }
}

__global__ void rshift(int n, int* odata, const int* idata) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= n) return;
  odata[index] = index == 0 ? 0 : idata[index - 1];
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int* dev_idata;
  int* dev_odata;
  cudaMalloc((void**)&dev_idata, n * sizeof(int));
  checkCUDAError("cudaMalloc dev_idata failed!");

  cudaMalloc((void**)&dev_odata, n * sizeof(int));
  checkCUDAError("cudaMalloc dev_odata failed!");

  cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy from idata to dev_data failed!");

  const int blockSize = 128;
  const int nBlocks = (n + blockSize - 1) / blockSize; //  n/blockSize, rounded up

  for (int pow2d = 1; pow2d < n; pow2d *= 2) {
    naiveScanIteration << < nBlocks, blockSize >> >(n, pow2d, dev_odata, dev_idata);
    std::swap(dev_idata, dev_odata);
  }

  // convert to exclusive scan
  rshift << <nBlocks, blockSize >> >(n, dev_odata, dev_idata);

  // we use dev_idata here because we swapped buffers
  cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_data to odata failed!");

  cudaFree(dev_idata);
  checkCUDAError("cudaFree dev_idata failed!");

  cudaFree(dev_odata);
  checkCUDAError("cudaFree dev_odata failed!");
}

}
}
