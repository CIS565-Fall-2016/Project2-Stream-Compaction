#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

/*
 * Performs one iteration of a naive scan of N elements. pow2d = 2^depth
 */
__global__ void naiveScanIteration(int N, int pow2d, int* odata, int* idata) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;
  if (index >= pow2d) {
    odata[index] = idata[index] + idata[index - pow2d];
  }
  else {
    // we've already processed these elements. just copy them
    odata[index] = idata[index];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int* dev_data;
  cudaMalloc((void**)&dev_data, n * sizeof(int));
  checkCUDAError("cudaMalloc dev_data failed!");
  cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy from idata to dev_data failed!");

  const int blockSize = 128;
  const int nBlocks = (n + blockSize - 1) / blockSize; //  n/blockSize, rounded up

  for (int d = 0; d < ilog2ceil(n); ++d) {
    naiveScanIteration << < nBlocks, blockSize >> >(n, pow(2, d), dev_data, dev_data);
  }

  cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_data to odata failed!");
  cudaFree(dev_data);
  checkCUDAError("cudaFree dev_data failed!");

  for (int i = n-1; i > 0; --i) {
    odata[i] = odata[i - 1];
  }
  odata[0] = 0;
}

}
}
