#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

    using StreamCompaction::Common::PerformanceTimer;
    PerformanceTimer& timer()
    {
        static PerformanceTimer timer;
        return timer;
    }

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
    // DONE use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());

    thrust::device_vector<int> thrust_idata(idata, idata + n);
    thrust::device_vector<int> thrust_odata(odata, odata + n);

    timer().startCpuTimer();
    timer().startGpuTimer();

    thrust::exclusive_scan(thrust_idata.begin(), thrust_idata.end(), thrust_odata.begin());

    timer().endGpuTimer();
    timer().endCpuTimer();

    //thrust::host_vector<int> thrust_host_odata = thrust_odata;
    thrust::copy(thrust_odata.begin(), thrust_odata.end(), odata);
}

void unstableSort(int* start, int* end)
{
    thrust::device_vector<int> thrust_data(start, end);

    timer().startCpuTimer();
    timer().startGpuTimer();

    thrust::sort(start, end);

    timer().endGpuTimer();
    timer().endCpuTimer();
    // GPU timer results in very small number?
}

void stableSort(int* start, int* end)
{
    thrust::device_vector<int> thrust_data(start, end);

    timer().startCpuTimer();
    timer().startGpuTimer();

    thrust::stable_sort(start, end);
   
    timer().endGpuTimer();
    timer().endCpuTimer();
}

}
}
