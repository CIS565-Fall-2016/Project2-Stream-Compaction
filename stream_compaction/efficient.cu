#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

__global__ void scan_up(int n, int pow2d, int *odata, const int *idata) {
  int index = 2 * pow2d * (blockIdx.x * blockDim.x + threadIdx.x + 1) - 1;
  if (index >= n) return;
  odata[index] = idata[index - pow2d] + idata[index];
}

__global__ void scan_down(int n, int pow2d, int *odata, const int *idata) {
  int index = n - 2 * pow2d * (blockIdx.x * blockDim.x + threadIdx.x) - 1;
  if (index < 0) return;

  int temp = idata[index - pow2d];
  odata[index - pow2d] = idata[index];
  odata[index] = temp + idata[index];
}

__global__ void zero(int n, int *odata) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 0) return;
  odata[index] = 0;
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
  int* dev_data;
  int sizePow2 = pow(2, ilog2ceil(n));

  const int blockSize = 128;
  int nBlocks;

  cudaMalloc((void**)&dev_data, sizePow2 * sizeof(int));
  checkCUDAError("cudaMalloc dev_data failed!");

  // fill with 0
  nBlocks = (sizePow2 + blockSize - 1) / blockSize;
  zero << <nBlocks, blockSize >> >(sizePow2, dev_data);

  cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy from idata to dev_data failed!");

  // scan up
  for (int pow2d = 1, int threads = sizePow2; pow2d < sizePow2 / 2; pow2d *= 2, threads /= 2) {
    nBlocks = (threads + blockSize - 1) / blockSize; // threads / blockSize, rounded up
    scan_up << < nBlocks, threads >> >(n, pow2d, dev_data, dev_data);
  }

  // set last item to 0
  int zero = 0;
  cudaMemcpy(&dev_data[sizePow2 - 1], &zero, sizeof(int), cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy copy zero failed!");

  // scan down
  for (int pow2d = pow(2, ilog2ceil(sizePow2) - 1), int threads = 1; pow2d >= 1; pow2d /= 2, threads *= 2) {
    nBlocks = (threads + blockSize - 1) / blockSize; // threads / blockSize, rounded up
    scan_down << < nBlocks, threads >> >(sizePow2, pow2d, dev_data, dev_data);
  }

  cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_data to odata failed!");

  cudaFree(dev_data);
  checkCUDAError("cudaFree dev_data failed!");
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
  int* dev_data;

  const int blockSize = 128;
  int nBlocks;

  cudaMalloc((void**)&dev_data, n * sizeof(int));
  checkCUDAError("cudaMalloc dev_data failed!");

  cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
  checkCUDAError("cudaMemcpy from idata to dev_data failed!");

  cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
  checkCUDAError("cudaMemcpy from dev_data to odata failed!");

  cudaFree(dev_data);
  checkCUDAError("cudaFree dev_data failed!");

  // TODO
  return -1;
}

}
}
