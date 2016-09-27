#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}


namespace StreamCompaction {
namespace Common {

	const int BlockSize = 128;

    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);
	
	class Timer
	{
	public :
		Timer()
		{
			cudaEventCreate(&gpu_timer_start);
			cudaEventCreate(&gpu_timer_stop);
		}

		void startCpuTimer()
		{
			cpu_timer_start = std::chrono::high_resolution_clock::now();
		}

		void stopCpuTimer()
		{
			cpu_timer_stop = std::chrono::high_resolution_clock::now();
		}

		double getCpuElapsedTime()
		{
			std::chrono::duration<double, std::milli> duration = cpu_timer_stop - cpu_timer_start;

			return duration.count();
		}

		void startGpuTimer()
		{
			cudaEventRecord(gpu_timer_start);
		}

		void stopGpuTimer()
		{
			cudaEventRecord(gpu_timer_stop);
			cudaEventSynchronize(gpu_timer_stop);
		}

		double getGpuElapsedTime()
		{
			float elapsedTime;

			
			cudaEventElapsedTime(&elapsedTime, gpu_timer_start, gpu_timer_stop);
			return elapsedTime;
		}

		void printTimerInfo(const char* s, double elapsedTime)
		{
			printf("<==TIMER==> %s %.3lf ms <==TIMER==> \n", s, elapsedTime);
		}

	private:
		std::chrono::high_resolution_clock::time_point cpu_timer_start;
		std::chrono::high_resolution_clock::time_point cpu_timer_stop;

		cudaEvent_t gpu_timer_start = NULL;
		cudaEvent_t gpu_timer_stop = NULL;

	};

	static Timer timer;
}
}
