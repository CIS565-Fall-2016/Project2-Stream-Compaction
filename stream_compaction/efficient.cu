#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }

  int twoToLevel = powf(2, level);
  int twoToLevelPlusOne = powf(2, level + 1);
  if (tid % twoToLevelPlusOne == 0) {
    odata[tid + twoToLevelPlusOne - 1] += odata[tid + twoToLevel - 1];
  }
}

__global__ void downsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  int twoToLevel = powf(2, level);
  int twoToLevelPlusOne = powf(2, level + 1);

  if (tid % twoToLevelPlusOne == 0) {
    int t = odata[tid + twoToLevel - 1];
    odata[tid + twoToLevel - 1] = odata[tid + twoToLevelPlusOne - 1];
    odata[tid + twoToLevelPlusOne - 1] += t;
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
  int* dev_odata;

  int height = ilog2ceil(n);

  int ceilPower2 = pow(2, height);
  cudaMalloc((void**)&dev_odata, ceilPower2 * sizeof(int));
  cudaMemset(dev_odata, 0, ceilPower2 * sizeof(int));
  cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

  for (int level = 0; level < height; ++level) {
    upsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
  }

  cudaMemset(dev_odata + (ceilPower2 - 1), 0, sizeof(int));

  for (int level = height - 1; level >= 0; --level) {
    downsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
  }

  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(dev_odata);
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
