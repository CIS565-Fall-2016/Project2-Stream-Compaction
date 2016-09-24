#include "common.h"

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

/* Max reduction is really just the partial sum upsweep algorithm */
__global__ void maxReduction(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  
  int twoToLevel = powf(2, level);
  int twoToLevelPlusOne = powf(2, level + 1);
  if (tid % twoToLevelPlusOne == 0) {
    odata[tid + twoToLevelPlusOne - 1] = imax(odata[tid + twoToLevel - 1], odata[tid + twoToLevelPlusOne - 1]);
  }
}

int findMaxInDeviceArray(int n, int *dev_idata) {

  int height = ilog2ceil(n);


  for (int level = 0; level < height; ++level) {
	  maxReduction << <BLOCK_COUNT(n), BLOCK_SIZE >> >(n, level, dev_idata);
  }

  int maxValue = 0;
  cudaMemcpy(&maxValue, dev_idata + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

  return maxValue;
}

namespace StreamCompaction {
namespace Common {

  /**
   * Convert an inclusice scan result to an exclusive scan result
   *
   */
__global__ void inclusiveToExclusiveScanResult(int n, int* odata, const int* idata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }

  if (tid == 0) {
    odata[0] = 0;
    return;
  }

  odata[tid] = idata[tid - 1];
}



/**
 * Maps an array to an array of 0s and 1s for stream compaction. Elements
 * which map to 0 will be removed, and elements which map to 1 will be kept.
 */
__global__ void kernMapToBoolean(int n, int *bools, const int *idata) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n) {
    return;
  }

  bools[tid] = (bool)idata[tid];
}

/**
 * Performs scatter on an array. That is, for each element in idata,
 * if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
 */
__global__ void kernScatter(int n, int *odata,
        const int *idata, const int *bools, const int *indices) {
    // TODO
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= n) {
    return;
  }

  if (bools[tid] == 1) {
    odata[indices[tid]] = idata[tid];
  }
}

}
}
