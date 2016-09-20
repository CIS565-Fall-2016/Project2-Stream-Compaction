#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

//#define DEBUG
#define BLOCK_SIZE 128
#define BLOCK_COUNT(n) ((n + BLOCK_SIZE - 1) / BLOCK_SIZE)

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

__global__ void inclusiveToExclusiveScan(int n, int* odata, const int* idata) {
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

__global__ void naiveScan(int n, int offset, int* odata, const int *idata) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  if (tid >= n) {
    return;
  }
  if (tid >= offset) {
    odata[tid] = idata[tid - offset] + idata[tid] ;
  } else {
    odata[tid] = idata[tid];
  }
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO

  int* dev_odata1;
  int* dev_odata2;

  cudaMalloc((void**)&dev_odata1, n * sizeof(int));
  cudaMalloc((void**)&dev_odata2, n * sizeof(int));

  cudaMemcpy(dev_odata1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_odata2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
#ifdef DEBUG
  int* temp = new int[5];
#endif

  int height = ilog2ceil(n);
  //height = 2;
  for (int level = 1; level <= height; ++level) {
    int offset = pow(2, level - 1);
    naiveScan << <BLOCK_COUNT(n), BLOCK_SIZE >> >(
      n, 
      offset, 
      (level % 2) == 0 ? dev_odata1 : dev_odata2, 
      (level % 2) == 0 ? dev_odata2 : dev_odata1
        );

#ifdef DEBUG
    printf("----\n");
    cudaMemcpy(temp, dev_odata1, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < 5; ++j) {
      printf("offset: %d, odata1[k]: %d\n", offset, temp[j]);
    }
    printf("\n");
    cudaMemcpy(temp, dev_odata2, 5 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int j = 0; j < 5; ++j) {
      printf("offset: %d, odata2[k]: %d\n", offset, temp[j]);
    }
#endif

  }

  if (height % 2 == 0) {
    inclusiveToExclusiveScan<<<BLOCK_COUNT(n), BLOCK_SIZE>>>(n, dev_odata2, dev_odata1);
    cudaMemcpy(odata, dev_odata2, n * sizeof(int), cudaMemcpyDeviceToHost);
  } else {
    inclusiveToExclusiveScan << <BLOCK_COUNT(n), BLOCK_SIZE >> >(n, dev_odata1, dev_odata2);
    cudaMemcpy(odata, dev_odata1, n * sizeof(int), cudaMemcpyDeviceToHost);
  }

  odata[0] = 0;

#ifdef DEBUG
  delete[] temp;
#endif

  cudaFree(dev_odata1);
  cudaFree(dev_odata2);
}

}
}
