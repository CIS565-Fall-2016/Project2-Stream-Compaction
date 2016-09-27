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

/**
 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
 */
void scan(int n, int *odata, const int *idata) {
	thrust::device_vector<int> dv_in(idata, idata + n);
	thrust::device_vector<int> dv_out(odata, odata + n);

    thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	thrust::copy(dv_out.begin(), dv_out.end(), odata);
}

void TestScan(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	thrust::device_vector<int> dv_in(idata, idata + n);
	thrust::device_vector<int> dv_out(odata, odata + n);

	for (int i = 0; i < samp; i++) {


		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;


		time += elapsed_seconds.count() * 1000 / samp;
	}

	thrust::copy(dv_out.begin(), dv_out.end(), odata);
	printf("    %f\n", time);

}

void TestSortStable(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;

	for (int i = 0; i < samp; i++) {
		memcpy(odata, idata, n*sizeof(int));

		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		thrust::stable_sort(odata, odata + n);
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}
	printf("    %f\n", time);
}

void TestSortUnstable(int n, int *odata, const int *idata) {

	double time = 0;
	int samp = 1000;
	
	for (int i = 0; i < samp; i++) {
		memcpy(odata, idata, n*sizeof(int));

		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();

		thrust::sort(odata, odata + n);
		cudaThreadSynchronize(); // block until kernel is finished

		end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;

		time += elapsed_seconds.count() * 1000 / samp;
	}
	printf("    %f\n", time);

}

}
}
