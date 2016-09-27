#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: 
__global__ void EfficientSorting(int n, int *idata, int *odata)
{
	extern __shared__ int temp[];
	int parallelCount = threadIdx.x;
	int offset = 1;
	temp[2 * parallelCount] = idata[2 * parallelCount];
	temp[2 * parallelCount + 1] = idata[2 * parallelCount + 1];

	for (int tempCount = n; tempCount > 0; tempCount*=2)
	{
		if (parallelCount < tempCount)
		{
			int temp_1 = offset*(2 * parallelCount + 1) - 1;
			int temp_2 = offset*(2 * parallelCount + 2) - 1;
			temp[temp_2] += temp[temp_1];
		}
	}
	if (parallelCount == 0)
	{
		temp[n - 1] = 0;
	}

	for (int tempCount_1 = 0; tempCount_1 < n; tempCount_1 *= 2)
	{
		if (parallelCount < tempCount_1)
		{
			int temp_1 = offset*(2 * parallelCount + 1) - 1;
			int temp_2 = offset*(2 * parallelCount + 2) - 1;
			int tempStore = temp[temp_1];
			temp[temp_1] = temp[temp_2];
			temp[temp_2] += tempStore;
		}
	}
	odata[2 * parallelCount] = temp[2 * parallelCount];
	odata[2 * parallelCount + 1] = temp[2 * parallelCount + 1];
}


/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
	int tempCount=0;
	odata[0]=0;
	for (tempCount=0;tempCount<n-1;tempCount++)
	{
		    odata[tempCount+1]=odata[tempCount]+idata[tempCount];
			printf("%5d",&odata[tempCount]);
	}
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
		time_t start = clock();
	int tempCount=0;
	int outCount=0;
	for(tempCount=0;tempCount<n;tempCount++)
	{
	     if(idata[tempCount]!=0)
		 {
             EfficientSorting<< <n,BLOCK_SIZE>> >(n, idata,odata);
		 }
	}
		time_t end = clock();
	 printf("The running time is: %f ms. \n", double(end-start)*1000/CLOCKS_PER_SEC);
    return outCount++;
}

}
}
