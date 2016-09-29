#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include "common.h"
#include "timer.h"
#include "thrust.h"

namespace StreamCompaction {
namespace Thrust {

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata)
{
		thrust::device_vector<int> dv_in(idata, idata + n);
		thrust::device_vector<int> dv_out(odata, odata + n);
	Timer::playTimer();
		thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	Timer::pauseTimer();
		thrust::copy(dv_out.begin(), dv_out.end(), odata);
}

void radixSort(int n, int *odata, const int *idata)
{
		thrust::device_vector<int> dv_in(idata, idata + n);
	Timer::playTimer();
		thrust::stable_sort(dv_in.begin(), dv_in.end());
	Timer::pauseTimer();
		thrust::copy(dv_in.begin(), dv_in.end(), odata);
}

}
}
