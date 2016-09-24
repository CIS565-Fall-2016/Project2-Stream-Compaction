#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */

int *dev_Data;

__global__ void CudaUpSweep(int d, int *data)
{
	int thid = threadIdx.x;
	int m = 1 << (d + 1);
	if (!(thid % m))
		data[thid + m - 1] += data[thid + (m >> 1) - 1];
}

__global__ void CudaDownSweep(int d, int *data)
{
	int thid = threadIdx.x;
	int m = 1 << (d + 1);
	if (!(thid % m))
	{
		int temp = data[thid + (m >> 1) - 1];
		data[thid + (m >> 1) - 1] = data[thid + m - 1];
		data[thid + m - 1] += temp;
	}
}

void scan(int n, int *odata, const int *idata) {
 //   int n = 8;
	//int idata[8] ={0,1,2,3,4,5,6,7};
	//int odata[8];
	int nCeilLog = ilog2ceil(n);
	int nLength = 1 << nCeilLog;

	cudaMalloc((void**)&dev_Data, nLength * sizeof(int));
	checkCUDAError("cudaMalloc failed!");

	cudaMemcpy(dev_Data, idata, sizeof(int) * nLength, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy to device failed!");

	for (int i = 0; i < nCeilLog; i++)
		CudaUpSweep<<<1, nLength>>>(i, dev_Data);

	cudaMemset(dev_Data + nLength - 1, 0, sizeof(int));
	checkCUDAError("cudaMemcpy to device failed!");
	for (int i = nCeilLog - 1; i >= 0; i--)
	{
		CudaDownSweep<<<1, nLength>>>(i, dev_Data);
				//cudaMemcpy(odata, dev_Data, sizeof(int) * (nLength), cudaMemcpyDeviceToHost);

	}

	cudaMemcpy(odata, dev_Data, sizeof(int) * nLength, cudaMemcpyDeviceToHost);
		//	for (int j = 0; j < n; j++)
		//	printf("%d ", odata[j]);
		//printf("\n");
	checkCUDAError("cudaMemcpy to host failed!");	
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
