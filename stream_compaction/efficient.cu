#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define PROFILE

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

__global__ void upsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }

  int twoToLevel = 1 << level;
  int twoToLevelPlusOne = 1 << (level + 1);
  if (tid % twoToLevelPlusOne == 0) {
    odata[tid + twoToLevelPlusOne - 1] += odata[tid + twoToLevel - 1];
  }
}

__global__ void downsweep(int n, int level, int* odata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  int twoToLevel = 1 << level;
  int twoToLevelPlusOne = 1 << (level + 1);

  if (tid % twoToLevelPlusOne == 0) {
    int t = odata[tid + twoToLevel - 1];
    odata[tid + twoToLevel - 1] = odata[tid + twoToLevelPlusOne - 1];
    odata[tid + twoToLevelPlusOne - 1] += t;
  }
}

// Should only be launched with 1 thread?
__global__ void remainingElementsCountForCompact(const int boolIndex, int* dev_indices, const int* dev_bools, int* remainingElementsCount) {
	*remainingElementsCount = dev_bools[boolIndex] == 1 ? boolIndex : boolIndex;
}

void deviceScan(int n, int* dev_odata) {

	int height = ilog2ceil(n); 
	int ceilPower2 = 1 << height;

	for (int level = 0; level < height; ++level) {
		upsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
		cudaThreadSynchronize();
	}

	// Set the root to zero
	cudaMemset(dev_odata + (ceilPower2 - 1), 0, sizeof(int));

	// Downsweep
	for (int level = height - 1; level >= 0; --level) {
		downsweep << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, level, dev_odata);
		cudaThreadSynchronize();
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
  int* dev_odata;
  int height = ilog2ceil(n);
  int ceilPower2 = 1 << height;
  cudaMalloc((void**)&dev_odata, ceilPower2 * sizeof(int));
  
	// Reset to zeros
  cudaMemset(dev_odata, 0, ceilPower2 * sizeof(int));

  // Copy idata to device memory
  cudaMemcpy(dev_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

#ifdef PROFILE
  // CUDA events for profiling
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif

#ifdef  PROFILE
  cudaEventRecord(start);
  // -- Start code to profile
#endif
  deviceScan(n, dev_odata);
#ifdef  PROFILE
  // -- End code to profile
  cudaEventRecord(stop);
#endif

 
#ifdef PROFILE
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Runtime: %d ns\n", (int)MS_TO_NS(milliseconds));
#endif
  // Transfer data back to host
  cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

  // Cleanup
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
    
  int height = ilog2ceil(n);
  int ceilPower2 = 1 << height;
  int *dev_bools, *dev_indices, *dev_odata, *dev_idata;
  cudaMalloc((void**)&dev_bools, sizeof(int) * ceilPower2);
  cudaMalloc((void**)&dev_idata, sizeof(int) * ceilPower2);
  cudaMalloc((void**)&dev_indices, sizeof(int) * ceilPower2);
  cudaMalloc((void**)&dev_odata, sizeof(int) * ceilPower2);

  // Transfer idata from host to device
  cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

#ifdef PROFILE
  // CUDA events for profiling
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
#endif
	
#ifdef PROFILE
  // -- Start code block to profile
  cudaEventRecord(start);
#endif

  // Set all non-zeros to 1s and zeros to 0s. This is our pass condition for an element to remain/discard after compaction
  Common::kernMapToBoolean << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_bools, dev_idata);
  
  // Compute indices of the out compacted stream
  // Reset to zeros
  cudaMemset(dev_indices, 0, ceilPower2 * sizeof(int));
  // Copy dev_bools to dev_indices to device memory
  cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
  StreamCompaction::Efficient::deviceScan(ceilPower2, dev_indices);

  // Move elements that are not discarded into appropriate slots based on scan result
  Common::kernScatter << <BLOCK_COUNT(ceilPower2), BLOCK_SIZE >> >(ceilPower2, dev_odata, dev_idata, dev_bools, dev_indices);

  // The max value of all the valid indices for the compacted stream is the number of remaining elements
  int remainingElementsCount = findMaxInDeviceArray(ceilPower2, dev_indices);
  
#ifdef PROFILE
  // -- End code block to profile
  cudaEventRecord(stop);
#endif

  // Transfer output back to host
  cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(dev_idata);
  cudaFree(dev_indices);
  cudaFree(dev_odata);
  
#ifdef PROFILE
  // Print runtime result
  cudaEventSynchronize(stop);
  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Runtime: %d ns\n", (int)MS_TO_NS(milliseconds));
#endif
  return remainingElementsCount;
}

}
}
