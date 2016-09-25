#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

__global__ void kernScan(int n, int dPow, int *odata, const int *idata) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;

  if (k >= dPow)
    odata[k] = idata[k - dPow] + idata[k];
  else
    odata[k] = idata[k];
}

__global__ void kernInclToExcl(int n, int *odata, const int *idata) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n-1)
    return;

  odata[k+1] = idata[k];
};

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int dPow = 1, dLogPow = 0;
  int *devData[2];
  cudaMalloc((void**)&devData[0], n*sizeof(int));
  cudaMalloc((void**)&devData[1], n*sizeof(int));
  cudaMemcpy(devData[0], idata, n*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(devData[1], 0, n*sizeof(int));

  dim3 blkDim(256);
  dim3 blkCnt((n + blkDim.x - 1)/blkDim.x);

  int dst, src;
  while (dPow/2 < n) {
    src = dLogPow % 2;
    dst = 1 - src;
    kernScan<<<blkCnt,blkDim>>>(n, dPow, devData[dst], devData[src]);
    dPow *= 2;
    dLogPow++;
    cudaDeviceSynchronize();
  }

  src = dLogPow % 2;
  dst = 1 - src;
  kernInclToExcl<<<blkCnt,blkDim>>>(n, devData[dst], devData[src]);
  cudaDeviceSynchronize();
  cudaMemcpy(odata, devData[dst], n*sizeof(int), cudaMemcpyDeviceToHost);
  odata[0] = 0;

  cudaFree(devData[0]);
  cudaFree(devData[1]);
}

}
}
