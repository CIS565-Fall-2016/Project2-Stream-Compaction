#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

double last_runtime;
int blkSize = 256;

// perform reduction
__global__ void kernScanUp(int n, int dPow, int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k % dPow != 0 || k + dPow - 1 >= n)
    return;

  data[k + dPow - 1] += data[k + dPow/2 - 1];
}

// perform reduction
__global__ void kernScanDown(int n, int dPow, int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k % dPow != 0 || k + dPow - 1 >= n)
    return;

  int t = data[k + dPow/2 - 1];
  data[k + dPow/2 - 1] = data[k + dPow - 1];
  data[k + dPow - 1] += t;
}

// mark nonzeroes
__global__ void kernMark(int n, int *keep, const int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;

  keep[k] = (data[k] != 0) ? 1 : 0;
}

__global__ void kernScatter(int n, int *out, const int *keep, const int *scan, const int *data) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n)
    return;

  if (keep[k]) {
    out[scan[k]] = data[k];
  }
}

static int getPot(int n) {
  unsigned int pot = n;
  pot--;
  pot |= pot >> 1;
  pot |= pot >> 2;
  pot |= pot >> 4;
  pot |= pot >> 8;
  pot |= pot >> 16;
  pot++;

  return pot;
}

static void devScanUtil(int n, int *devData) {
  int pot  = getPot(n);

  dim3 blkDim(blkSize);

  int dPow = 2;
  while (dPow/2 < n) {
    dim3 blkCnt((pot + blkDim.x - 1)/blkDim.x);
    kernScanUp<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow *= 2;
  }
  cudaMemset(&devData[pot-1], 0, sizeof(int));

  while (dPow > 1) {
    dim3 blkCnt((pot + blkDim.x - 1)/blkDim.x);
    kernScanDown<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow /= 2;
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int pot  = getPot(n);

  int *devData;
  cudaMalloc((void**)&devData, pot*sizeof(int));
  cudaMemset(devData, 0, pot*sizeof(int));
  cudaMemcpy(devData, idata, n*sizeof(int), cudaMemcpyHostToDevice);

  double t1 = clock();
  devScanUtil(n, devData);
  double t2 = clock();
  last_runtime = 1.0E6 * (t2-t1) / CLOCKS_PER_SEC;

  cudaMemcpy(odata, devData, n*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(devData);
}

/**
 * Performs stream compaction on idata, storing the result into odata.
 * All zeroes are discarded.
 *
 * @param n      The number of elements in idata.
 * @param odata  The array into which to store elements.
 * @param idata  The array of elements to compact.
 * @returns      The number of elements remaining after compaction.
 */
int compact(int n, int *odata, const int *idata) {
  int pot = getPot(n);

  // upload data
  int *devData;
  cudaMalloc((void**)&devData, n*sizeof(int));
  cudaMemcpy(devData, idata, n*sizeof(int), cudaMemcpyHostToDevice);

  dim3 blkDim(blkSize);
  dim3 blkCnt((n + blkDim.x - 1)/blkDim.x);

  // mark values to keep
  int *devKeep, *devScan;
  cudaMalloc((void**)&devKeep, pot*sizeof(int));
  cudaMalloc((void**)&devScan, pot*sizeof(int));
  cudaMemset(devKeep, 0, pot*sizeof(int));

  double t1 = clock();
  kernMark<<<blkCnt,blkDim>>>(n, devKeep, devData);
  cudaMemcpy(devScan, devKeep, pot*sizeof(int), cudaMemcpyDeviceToDevice);

  // scan boolean array
  devScanUtil(n, devScan);
  int nKeep;
  cudaMemcpy(&nKeep, &devScan[pot-1], sizeof(int), cudaMemcpyDeviceToHost);

  // scatter to output
  int *devOut;
  cudaMalloc((void**)&devOut, n*sizeof(int));
  kernScatter<<<blkCnt,blkDim>>>(n, devOut, devKeep, devScan, devData);
  double t2 = clock();
  last_runtime = 1.0E6 * (t2-t1) / CLOCKS_PER_SEC;
  cudaMemcpy(odata, devOut, nKeep*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(devOut);
  cudaFree(devData);
  cudaFree(devKeep);
  cudaFree(devScan);

  return nKeep;
}

}
}
