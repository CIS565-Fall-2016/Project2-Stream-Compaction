#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
namespace Efficient {

// TODO: __global__
#define blockSize 128

    int* dev_idata;
    int* dev_odata;
    int* dev_bools;
    int* dev_indices;
    int* dev_scandata;


    __global__ void kernupsweep(int n, int d, int* dev_scandata)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;
        
        if (k % (1 << (d + 1)) == 0)
            dev_scandata[k + (1 << (d + 1)) - 1] += dev_scandata[k + (1 << d) - 1];// +dev_scandata[k + (1 << (d + 1)) - 1];
    }

    __global__ void kerndownsweep(int n, int d, int* dev_scandata)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;

        if (k % (1<<(d+1)) == 0)
        {
            int t = dev_scandata[k + (1<<d) - 1];
            dev_scandata[k + (1<<d) - 1] = dev_scandata[k + (1<<(d+1)) - 1];
            dev_scandata[k + (1<<(d+1)) - 1] = t + dev_scandata[k + (1<<(d+1)) - 1];
        }
    }

    __global__ void kernscan(int n, int d, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;

        if (k >= (1<<(d-1)))
            dev_scandata[k] = dev_scandata2[k] + dev_scandata2[k - (1<<(d-1))];
    }

    __global__ void kerninc2exc(int n, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;

        if (k == 0)
            dev_scandata[0] = 0;
        else
            dev_scandata[k] = dev_scandata2[k - 1];
    }


    void setup(int n, const int* idata)
    {
        cudaMalloc((void**)&dev_scandata, n * sizeof(int));
        cudaMemcpy(dev_scandata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    }

    void cleanup(void)
    {
        cudaFree(dev_scandata);
    }

    void setupcompact(int n, const int* idata)
    {
        cudaMalloc((void**)&dev_idata, n * sizeof(int));
        cudaMalloc((void**)&dev_odata, n * sizeof(int));
        cudaMalloc((void**)&dev_bools, n * sizeof(int));
        cudaMalloc((void**)&dev_indices, n * sizeof(int));
        cudaMalloc((void**)&dev_scandata, n * sizeof(int));

        cudaMemset(dev_bools, 0, sizeof(int) * n);
        cudaMemset(dev_indices, 0, sizeof(int) * n);
        cudaMemcpy(dev_scandata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
    }

    void cleanupcompact(void)
    {
        cudaFree(dev_idata);
        cudaFree(dev_odata);
        cudaFree(dev_bools);
        cudaFree(dev_indices);
        cudaFree(dev_scandata);
    }

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // TODO
    //printf("TODO\n");

    dim3 blocks((n + blockSize - 1) / blockSize);

    int nold = n;

    if (n % 2 == 0)
    {
        n = 1 << ilog2ceil(n);
        setup(n, idata);
        for (int d = 0; d <= ilog2ceil(n)-1; d++)
        {
            kernupsweep << <blocks, blockSize >> >(n, d, dev_scandata);
        }

        cudaMemset(dev_scandata+(n-1), 0, sizeof(int));
        for (int d = ilog2ceil(n); d>=0; d--)
        {
            kerndownsweep << <blocks, blockSize >> >(n, d, dev_scandata);
        }
        cudaMemcpy(odata, dev_scandata, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }
    else
    {
        n = 1 << ilog2ceil(n);
        setup(n, idata);
        
        for (int d = 0; d <= ilog2ceil(n) - 1; d++)
        {
            kernupsweep << <blocks, blockSize >> >(n, d, dev_scandata);
        }

        cudaMemset(dev_scandata+(n-1), 0, sizeof(int));
        for (int d = ilog2ceil(n); d >= 0; d--)
        {
            kerndownsweep << <blocks, blockSize >> >(n, d, dev_scandata);
        }
        cudaMemcpy(odata, dev_scandata, sizeof(int) * n, cudaMemcpyDeviceToHost);
    }

    cleanup();

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

    dim3 blocks((n + blockSize - 1) / blockSize);
    int count = 0;

    if (n % 2 == 0)
    {
        n = 1 << ilog2ceil(n);

        setupcompact(n, idata);

        int* bools = new int[n];
        int* indices = new int[n];
        memset(bools, 0, sizeof(int)*n);
        memset(indices, 0, sizeof(int)*n);
        cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odata, odata, sizeof(int)*n, cudaMemcpyHostToDevice);

        StreamCompaction::Common::kernMapToBoolean << <blocks, blockSize >> > (n, dev_bools, dev_scandata);

        scan(n, dev_indices, dev_bools);
        cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(indices, dev_indices, sizeof(int) * n, cudaMemcpyDeviceToHost);

        StreamCompaction::Common::kernScatter << <blocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

        cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);

        /*
        printf("\n%-10s", "bools: "); for (int i = 0; i < 20; i++)
        printf("%d ", bools[i]);
        printf("\n%-10s", "scans: "); for (int i = 0; i < 20; i++)
        printf("%d ", indices[i]);
        printf("\n%-10s", "idata: "); for (int i = 0; i < 20; i++)
        printf("%d ", idata[i]);
        printf("\n%-10s", "odata: "); for (int i = 0; i < 20; i++)
        printf("%d ", odata[i]);
        printf("\n");
        */

        count = indices[n - 1] +bools[n - 1];

        cleanupcompact();

        delete[] bools;
        delete[] indices;
    }
    else
    {
        n = 1 << ilog2ceil(n);

        setupcompact(n, idata);

        int* bools = new int[n];
        int* indices = new int[n];
        memset(bools, 0, sizeof(int)*n);
        memset(indices, 0, sizeof(int)*n);
        cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odata, odata, sizeof(int)*n, cudaMemcpyHostToDevice);

        StreamCompaction::Common::kernMapToBoolean << <blocks, blockSize >> > (n, dev_bools, dev_scandata);

        scan(n, dev_indices, dev_bools);
        cudaMemcpy(bools, dev_bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(indices, dev_indices, sizeof(int) * n, cudaMemcpyDeviceToHost);

        StreamCompaction::Common::kernScatter << <blocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

        cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);

        count = indices[n - 1] +bools[n - 1] - 1;

        cleanupcompact();

        delete[] bools;
        delete[] indices;
    }

    return count;
}

}
}
