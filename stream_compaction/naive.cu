#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Naive {

// TODO: __global__

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
int threadPerBlock = 1024;
int BlockNum;

int *dev_Data[2];

__global__ void CudaScan(int d, int *in, int *out, int n)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= n)
		return;
	int m = 1 << (d - 1);
		
	if (thid >= m)
		out[thid] = in[thid] + in[thid - m];
	else
		out[thid] = in[thid];

}

void scan(int n, int *odata, const int *idata) {

	int nCeilLog = ilog2ceil(n);
	int nLength = 1 << nCeilLog;

	cudaMalloc((void**)&dev_Data[0], nLength * sizeof(int));
	cudaMalloc((void**)&dev_Data[1], nLength * sizeof(int));
	checkCUDAError("cudaMalloc failed!");

	cudaMemcpy(dev_Data[0], idata, sizeof(int) * nLength, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy to device failed!");

	int nOutputIndex = 0;
	float time_elapsed=0;
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start,0);
	for (int i = 1; i <= nCeilLog; i++)
	{
		nOutputIndex ^= 1;
		BlockNum = nLength / threadPerBlock + 1;
		CudaScan<<<BlockNum, threadPerBlock>>>(i, dev_Data[nOutputIndex ^ 1], dev_Data[nOutputIndex], nLength);
	}
		cudaEventRecord( stop,0);
cudaEventSynchronize(start);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_elapsed,start,stop);
//FILE* fp = fopen("efficient.txt", "a+");
//fprintf(fp, "%d %f\n", nCeilLog, time_elapsed);
//fclose(fp);
	odata[0] = 0;
	cudaMemcpy(odata + 1, dev_Data[nOutputIndex], sizeof(int) * (n - 1), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy to host failed!");	

	cudaFree(dev_Data[0]);
	cudaFree(dev_Data[1]);
}



}
}
