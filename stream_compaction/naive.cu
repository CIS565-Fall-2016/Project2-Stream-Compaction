#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include <algorithm>

namespace StreamCompaction {
namespace Naive {

// TODO: __global__




#define blockSize 128

    int* dev_scandata;
    int* dev_scandata2;


    __global__ void kernscan(int n, int d, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;

        if (k >= 1 << (d - 1))
        {
            dev_scandata[k] = dev_scandata2[k] + dev_scandata2[k - (1 << (d - 1))];
            dev_scandata2[k] = dev_scandata[k];  //swap
        }
    }

    __global__ void kerninc2exc(int n, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;
        
        if (k == 0)
            dev_scandata[0] = 0;
        else
            dev_scandata[k] = dev_scandata2[k-1];
    }

    float timeKernScan(int n, int *odata, const int *idata)
    {
        int* dev_odata;
        int* dev_idata;
        cudaMalloc((void**)&dev_odata, n * sizeof(int));
        cudaMalloc((void**)&dev_idata, n * sizeof(int));

        cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);


        dim3 blocks((n + blockSize - 1) / blockSize);


        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
        int count = 0;
        if (n % 2 == 0)
        {
            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                kernscan << <blocks, blockSize >> >(n, d, dev_scandata, dev_scandata2);
                count++;
            }
        }
        cudaEventRecord(stop); cudaEventSynchronize(stop); float milliseconds = 0; cudaEventElapsedTime(&milliseconds, start, stop);
        //printf("\nELAPSED TIME = %f\n", milliseconds);

        cudaEventDestroy(start); cudaEventDestroy(stop);

        cudaFree(dev_odata);
        cudaFree(dev_idata);

        return milliseconds;
    }

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
    void scan(int n, int *odata, const int *idata) {
        // TODO
        //printf("TODO\n");


        cudaMalloc((void**)&dev_scandata, n * sizeof(int));
        cudaMalloc((void**)&dev_scandata2, n * sizeof(int));

        cudaMemcpy(dev_scandata2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_scandata, dev_scandata2, sizeof(int) * n, cudaMemcpyDeviceToDevice);

        dim3 blocks((n + blockSize - 1) / blockSize);

        int count = 0;
        if (n % ilog2(n) == 0)
        {
            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                kernscan << <blocks, blockSize >> >(n, d, dev_scandata, dev_scandata2);
                count++;
            }
        }
        else
        {
            for (int d = 1; d <= ilog2ceil(n)+1; d++)
            {
                kernscan << <blocks, blockSize >> >(n, d, dev_scandata, dev_scandata2);
                count++;
            }
        }

        
        if (n % ilog2(n) == 0)
        {
            kerninc2exc << <blocks, blockSize >> >(n, dev_scandata2, dev_scandata);
            cudaMemcpy(odata, dev_scandata2, sizeof(int) * n, cudaMemcpyDeviceToHost);
        }
        else
        {
            kerninc2exc << <blocks, blockSize >> >(n, dev_scandata, dev_scandata2);
            cudaMemcpy(odata, dev_scandata, sizeof(int) * n, cudaMemcpyDeviceToHost);
        }

    cudaFree(dev_scandata);
    cudaFree(dev_scandata2);
}

}
}
