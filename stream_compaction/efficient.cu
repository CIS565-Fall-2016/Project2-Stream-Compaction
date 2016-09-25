#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

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

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  dim3 blkDim(256);
  dim3 blkCnt((n + blkDim.x - 1)/blkDim.x);

  unsigned int pot = n;
  pot--;
  pot |= pot >> 1;
  pot |= pot >> 2;
  pot |= pot >> 4;
  pot |= pot >> 8;
  pot |= pot >> 16;
  pot++;

  int *devData;
  cudaMalloc((void**)&devData, pot*sizeof(int));
  cudaMemset(devData, 0, pot*sizeof(int));
  cudaMemcpy(devData, idata, n*sizeof(int), cudaMemcpyHostToDevice);

  int dPow = 2;
  while (dPow/2 < n) {
    kernScanUp<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow *= 2;
  }
  cudaMemset(&devData[pot-1], 0, sizeof(int));

  while (dPow > 1) {
    kernScanDown<<<blkCnt,blkDim>>>(pot, dPow, devData);
    dPow /= 2;
  }
  cudaMemcpy(odata, devData, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 10; i++)
    printf("%d ", odata[i]);
  printf("\n");

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
    // TODO
    return -1;
}

}
}
