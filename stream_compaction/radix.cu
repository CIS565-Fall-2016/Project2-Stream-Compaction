#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
namespace Radix {

int *dev_Data[2];
int *dev_bArray;
int *dev_eArray;
int *dev_fArray;
int *dev_tArray;
int *dev_dArray;

__global__ void CudaUpSweep(int d, int *data, int addTimes)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= addTimes)
		return;
	data[(thid + 1) * (1 << (d + 1)) - 1] += data[(thid + 1) * (1 << (d + 1)) - 1 - (1 << d)];
}

__global__ void CudaDownSweep(int d, int *data, int addTimes)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= addTimes)
		return;
	int m = (thid + 1) * (1 << (d + 1));
	int temp = data[m - 1 - (1 << d)];
	data[m - 1 - (1 << d)] = data[m - 1];
	data[m - 1] += temp;
}

__global__ void CudaGetBEArray(int pass, int *in, int *out1, int *out2,  int n)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= n)
		return;
	out1[thid] = (in[thid] >> pass) & 1;
	out2[thid] = out1[thid] ^ 1;
}

__global__ void CudaGetTArray(int *in, int *e, int *out, int n)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	int totalFalses = in[n - 1] + e[n - 1];
	if (thid >= n)
		return;
	out[thid] = thid - in[thid] + totalFalses;
}

__global__ void CudaGetDArray(int *in1, int *in2, int *in3, int *out, int n)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= n)
		return;
	out[thid] = in1[thid] ? in2[thid] : in3[thid];
}

__global__ void CudaGetResult(int *in, int *out, int *data, int n)
{
	int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (thid >= n)
		return;
	out[in[thid]] = data[thid];
}

void RadixSort(int n, int *odata, int *idata)
{
	//int n = 8;
	//int idata[8] = {4,7,2,6,3,5,1,0};
	//int odata[8];

	int threadPerBlock = 512;
	int BlockNum;
	int nCeilLog = ilog2ceil(n);
	int nLength = 1 << nCeilLog;
	int pout = 1;
	cudaMalloc((void**)&dev_Data[0], n * sizeof(int));
	cudaMalloc((void**)&dev_Data[1], n * sizeof(int));
	cudaMalloc((void**)&dev_bArray, n * sizeof(int));
	cudaMalloc((void**)&dev_eArray, nLength * sizeof(int));
	cudaMalloc((void**)&dev_fArray, nLength * sizeof(int));
	cudaMalloc((void**)&dev_tArray, n * sizeof(int));
	cudaMalloc((void**)&dev_dArray, n * sizeof(int));
	cudaMemset(dev_eArray, 0, nLength * sizeof(int));
	cudaMemcpy(dev_Data[0], idata, sizeof(int) * n, cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy to device failed!");
	checkCUDAError("cudaMalloc failed!");

	int maxn = 0;
	for (int i = 0; i < n; i++)
		if (idata[i] > maxn)
			maxn = idata[i];
float time_elapsed = 0.0f;
cudaEvent_t start,stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord( start,0);
	int pass = 0;
	while (1)
	{
		int pin = pout ^ 1;
		if ((maxn >> pass) == 0)
			break;
		BlockNum = n / threadPerBlock + 1;
		CudaGetBEArray<<<BlockNum, threadPerBlock>>>(pass, dev_Data[pin], dev_bArray, dev_eArray, n);
		cudaMemcpy(dev_fArray, dev_eArray, nLength * sizeof(int), cudaMemcpyDeviceToDevice);
		checkCUDAError("cudaMemcpy device to device failed!");

		for (int i = 0; i < nCeilLog; i++)
		{
			int addTimes = 1 << (nCeilLog - 1 - i);
			BlockNum = addTimes / threadPerBlock + 1;
			CudaUpSweep<<<BlockNum, threadPerBlock>>>(i, dev_fArray, addTimes);
		}

		cudaMemset(dev_fArray + nLength - 1, 0, sizeof(int));
		checkCUDAError("cudaMemset failed!");
		for (int i = nCeilLog - 1; i >= 0; i--)
		{
			int addTimes = 1 << (nCeilLog - 1 - i);
			BlockNum = addTimes / threadPerBlock + 1;
			CudaDownSweep<<<BlockNum, threadPerBlock>>>(i, dev_fArray, addTimes);
		}

//printf("F:");
//cudaMemcpy(odata, dev_fArray, sizeof(int) * n, cudaMemcpyDeviceToHost);
//checkCUDAError("cudaMemcpy to host failed!");	
//for (int i = 0; i < n; i++)
//	printf("%d ", odata[i]);
//printf("\n");

		BlockNum = n / threadPerBlock + 1;
		CudaGetTArray<<<BlockNum, threadPerBlock>>>(dev_fArray, dev_eArray, dev_tArray, n);

//printf("T:");
//cudaMemcpy(odata, dev_tArray, sizeof(int) * n, cudaMemcpyDeviceToHost);
//checkCUDAError("cudaMemcpy to host failed!");	
//for (int i = 0; i < n; i++)
//	printf("%d ", odata[i]);
//printf("\n");

		CudaGetDArray<<<BlockNum, threadPerBlock>>>(dev_bArray, dev_tArray, dev_fArray, dev_dArray, n);
		CudaGetResult<<<BlockNum, threadPerBlock>>>(dev_dArray, dev_Data[pout], dev_Data[pin], n);
//printf("D:");
//cudaMemcpy(odata, dev_dArray, sizeof(int) * n, cudaMemcpyDeviceToHost);
//checkCUDAError("cudaMemcpy to host failed!");	
//for (int i = 0; i < n; i++)
//	printf("%d ", odata[i]);
//printf("\n");
//printf("output:");
//cudaMemcpy(odata, dev_Data[pout], sizeof(int) * n, cudaMemcpyDeviceToHost);
//checkCUDAError("cudaMemcpy to host failed!");	
//for (int i = 0; i < n; i++)
//	printf("%d ", odata[i]);
//printf("\n");
		pass++;
		pout ^= 1;
	}
cudaEventRecord( stop,0);
cudaEventSynchronize(start);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&time_elapsed,start,stop);
printf("Time:%f ms\n", time_elapsed);
	cudaMemcpy(odata, dev_Data[pout ^ 1], sizeof(int) * n, cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy to host failed!");	
	//for (int i = 0; i < n; i++)
	//	printf("%d ", odata[i]);
	//printf("\n");

	cudaFree(dev_Data[0]);
	cudaFree(dev_Data[1]);
	cudaFree(dev_bArray);
	cudaFree(dev_eArray);
	cudaFree(dev_fArray);
	cudaFree(dev_tArray);
	cudaFree(dev_dArray);
}


}
}
