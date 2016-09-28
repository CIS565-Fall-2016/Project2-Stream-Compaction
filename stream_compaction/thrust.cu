#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {


    float timeThrust(int n, int *odata, const int *idata)
    {
        int* dev_odata;
        int* dev_idata;
        cudaMalloc((void**)&dev_odata, n * sizeof(int));
        cudaMalloc((void**)&dev_idata, n * sizeof(int));

        cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_odata, dev_idata, sizeof(int) * n, cudaMemcpyDeviceToDevice);

        thrust::host_vector<int> dv_in(idata, idata + n);
        thrust::host_vector<int> dv_out(odata, odata + n);

        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start);
        thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
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
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

    //thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());


    thrust::host_vector<int> dv_in(idata, idata + n);
    thrust::host_vector<int> dv_out(odata, odata + n);

    thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

    //odata = &(dv_out.front());
    for (int i = 0; i < n; i++)
        odata[i] = dv_out[i];
}

}
}
