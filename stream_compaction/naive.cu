#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__
__global__ void NaiveGPUScan(int n, int *odata, const int *idata,int step)
{
	int parallelCount = threadIdx.x+blockIdx.x*blockDim.x;
	
    if(parallelCount<n)
	{
	    if(parallelCount>=step)
		{
		    odata[parallelCount]=idata[parallelCount-step]+idata[parallelCount];
		}
	}
}

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
 void scan(int n, int *odata, const int *idata) {
    // TODO
  time_t start = clock();


  int* tempArray_1;
  int* tempArray_2;
  int tempCount=0;
  int step=0;

  cudaMalloc((void**)&tempArray_1, n * sizeof(int));
  cudaMalloc((void**)&tempArray_2, n * sizeof(int));

  //allocate the device space
  cudaMemcpy(tempArray_1, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(tempArray_2, idata, n * sizeof(int), cudaMemcpyHostToDevice);
  


  for (tempCount = 1; tempCount <= ilog2ceil(n); ++tempCount) {
    step=2^(tempCount-1);

    NaiveGPUScan << <n, BLOCK_SIZE >> >(n, (tempCount % 2) == 0 ? tempArray_1 : tempArray_2,  (tempCount % 2) == 0 ? tempArray_2 : tempArray_1,step);
  }

  if (ilog2ceil(n) % 2 == 0) {
    Common::inclusiveToExclusive << <n, BLOCK_SIZE >> >(n, tempArray_2, tempArray_1);

    cudaMemcpy(odata, tempArray_2, n * sizeof(int), cudaMemcpyDeviceToHost);
  } else {
    Common::inclusiveToExclusive << <n, BLOCK_SIZE >> >(n, tempArray_1, tempArray_2);

	cudaMemcpy(odata, tempArray_1, n * sizeof(int), cudaMemcpyDeviceToHost);
  }

  	 time_t end = clock();
	 printf("The running time is: %f ms. \n", double(end-start)*1000/CLOCKS_PER_SEC);
    cudaFree(tempArray_1);
    cudaFree(tempArray_2);
}

}
}
