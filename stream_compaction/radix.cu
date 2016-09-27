#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "radix.h"
#include "efficient.h"
#include "thrust.h"
#include "cpu.h"
#include <algorithm>

namespace StreamCompaction {
namespace Radix {


#define blockSize 128

    int getDigit(int number, int i)
    {
        int d = 0;
        if (i == 1)
        {
            d = number % (10 * i);
            return d;
        }
        else if (i > 1)
        {
            int currpower = (pow(10, i));
            int prevpower = (pow(10, i - 1));

            if (number < prevpower)
                return -1;

            int nextdigits = number % (prevpower);
            d = (number%currpower - nextdigits) / prevpower;

            return d;
        }
        return -1;
    }




    __global__ void kernDecimalsMap(int n, int sbit, int *decimals, const int *idata) {
        // TODO
        int index = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (index >= n)
            return;

        int number = idata[index];

        // get digit
        int d = 0;
        if (sbit == 1)
        {
            d = number % (10 * sbit);
        }
        else if (sbit > 1)
        {
            int currpower = (int)(pow(double(10), double(sbit)));
            int prevpower = (int)(pow(double(10), double(sbit - 1)));

            if (number < prevpower)
                d = -1;

            int nextdigits = number % (prevpower);
            d = (number%currpower - nextdigits) / prevpower;
        }
        else
            d = -1;
        // end get digit

        decimals[index] = d;
    }


    __global__ void kernexc2inc(int n, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n-1)
            return;

        dev_scandata[k] = dev_scandata2[k+1];
    }

    void exc2inc(int n, int *odata, const int *idata)
    {
        dim3 blocks((n + blockSize - 1) / blockSize);
        int* dev_idata;
        int* dev_odata;
        cudaMalloc((void**)&dev_idata, sizeof(int) * n);
        cudaMalloc((void**)&dev_odata, sizeof(int) * n);
        cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odata, odata, sizeof(int)*n, cudaMemcpyHostToDevice);

        kernexc2inc << <blocks, blockSize >> >(n, dev_odata, dev_idata);
        cudaMemcpy(odata, dev_odata, sizeof(int)*n, cudaMemcpyDeviceToHost);
        
        if (n>1)
            odata[n - 1] += odata[n-2];

        cudaFree(dev_idata);
        cudaFree(dev_odata);
    }

    int findMax(int n, const int* nums)
    {
        int max = 0;
        for (int i = 0; i < n; i++)
            max = max < nums[i] ? nums[i] : max;
        return max;
    }


    __global__ void kernFindMax(int n, int d, int* dev_scandata, int* dev_scandata2)
    {
        int k = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (k >= n)
            return;

        if (k >= 1 << (d - 1))
        {
            dev_scandata[k] = dev_scandata2[k] < dev_scandata2[k - (1 << (d - 1))] ? dev_scandata2[k - (1 << (d - 1))] : dev_scandata2[k];
            dev_scandata2[k] = dev_scandata[k];  //swap
        }
    }

    void findMaxGPU(int n, int *odata, const int *idata)
    {
        dim3 blocks((n + blockSize - 1) / blockSize);
        int* dev_scandata;
        int* dev_scandata2;
        cudaMalloc((void**)&dev_scandata, n * sizeof(int));
        cudaMalloc((void**)&dev_scandata2, n * sizeof(int));

        cudaMemcpy(dev_scandata2, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_scandata, dev_scandata2, sizeof(int) * n, cudaMemcpyDeviceToDevice);

        if (n % 2 == 0)
        {
            for (int d = 1; d <= ilog2ceil(n); d++)
                kernFindMax << <blocks, blockSize >> >(n, d, dev_scandata, dev_scandata2);
        }
        else
        {
            for (int d = 1; d <= ilog2ceil(n) + 1; d++)
                kernFindMax << <blocks, blockSize >> >(n, d, dev_scandata, dev_scandata2);
        }

        cudaMemcpy(odata, dev_scandata, sizeof(int) * n, cudaMemcpyDeviceToHost);
        cudaFree(dev_scandata);
        cudaFree(dev_scandata2);
    }
/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
    void radixCPU(int n, int *odata, const int *idata) {
        // TODO
        //printf("TODO\n");

        int* mapping = new int[10];
        int* idatacpy = new int[n];
        memcpy(idatacpy, idata, sizeof(int)*n);

        memset(odata, 0, sizeof(int)*n);
        
        int sbit = 1;
        bool notdone = true;

        while (notdone)
        {
            memset(mapping, 0, sizeof(int) * 10);
            notdone = false;
            for (int i = 0; i < n; i++)
            {
                int number = idatacpy[i];
                int val = getDigit(number, sbit);
                if (val != -1)
                {
                    mapping[val]++;
                    notdone = true;
                }
                else
                    mapping[0]++;
            }
            if (notdone == false)
                break;

            for (int i = 1; i < 10; i++)
                mapping[i] += mapping[i - 1];

            for (int i = n - 1; i >= 0; i--)
            {
                int number = idatacpy[i];
                int digit = getDigit(number, sbit);
                int mapidx = digit == -1 ? 0 : digit;
                mapping[mapidx] -= 1;
                int index = mapping[mapidx];
                odata[index] = number;
            }

            memcpy(idatacpy, odata, sizeof(int)*n);
            sbit += 1;
        }
}
    void radixGPU(int n, int *odata, const int *idata, gputesttype testtype) {
        // TODO
        //printf("TODO\n");

        dim3 blocks((n + blockSize - 1) / blockSize);

        int* dev_decmapping;
        int* dev_idata;
        cudaMalloc((void**)&dev_decmapping, sizeof(int) * n);
        cudaMalloc((void**)&dev_idata, sizeof(int) * n);
        cudaMemset(dev_decmapping, 0, sizeof(int) * n);
        cudaMemcpy(dev_idata, idata, sizeof(int)*n, cudaMemcpyHostToDevice);

        int* idatacpy = new int[n];
        int* decmapping = new int[n];
        int* deccount = new int[10];
        int* decscan = new int[10];
        memcpy(idatacpy, idata, sizeof(int) * n);
        memset(decmapping, 0, sizeof(int) * n);
        
        memset(odata, 0, sizeof(int)*n);


        findMaxGPU(n, odata, idata);
        int maxnum = odata[n - 1];

        memset(odata, 0, sizeof(int)*n);

        int maxbit = maxnum > 0 ? (int)log10((double)maxnum) + 1 : 1;

        for (int sbit = 1; sbit <= maxbit; sbit++)
        {

            memset(deccount, 0, sizeof(int) * 10);
            memset(decscan, 0, sizeof(int) * 10);

            kernDecimalsMap << <blocks, blockSize >> >(n, sbit, dev_decmapping, dev_idata);
            cudaMemcpy(decmapping, dev_decmapping, sizeof(int)*n, cudaMemcpyDeviceToHost);
            for (int i = 0; i < n; i++)
                deccount[decmapping[i]]++;

        
            if (testtype == 0)
                CPU::scan(10, decscan, deccount);
            else if (testtype == 1)
                Naive::scan(10, decscan, deccount);
            else if (testtype == 2)
                Thrust::scan(10, decscan, deccount);


            exc2inc(10, deccount, decscan);
        
        
            for (int i = n - 1; i >= 0; i--)
            {
                int number = idatacpy[i];
                int digit = getDigit(number, sbit);
                int mapidx = digit < 0 ? 0 : digit;
                deccount[mapidx] -= 1;
                int index = deccount[mapidx];
                if (index>=0)
                    odata[index] = number;
            }

            memcpy(idatacpy, odata, sizeof(int) * n);
        
        }

        memcpy(idatacpy, odata, sizeof(int) * n);

        cudaFree(dev_decmapping);
        cudaFree(dev_idata);

        delete[] decmapping;
        delete[] deccount;
        delete[] decscan;
        delete[] idatacpy;
    }
}
}
