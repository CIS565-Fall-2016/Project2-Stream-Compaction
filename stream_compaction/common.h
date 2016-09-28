#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>
#include <exception>

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
    __global__ void kernMapToBoolean(int n, int *bools, const int *idata);

    __global__ void kernScatter(int n, int *odata,
            const int *idata, const int *bools, const int *indices);

	class MyCudaTimer
	{
	public:
		MyCudaTimer() : elapsedTimeMs(0.f), isActive(false)
		{
			cudaEventCreate(&begin);
			cudaEventCreate(&end);
		}

		virtual ~MyCudaTimer()
		{
			cudaEventDestroy(begin);
			cudaEventDestroy(end);
		}

		void start()
		{
			if (!isActive)
			{
				cudaEventRecord(begin);
				isActive = true;
			}
		}

		void stop()
		{
			if (isActive) cudaEventRecord(end);
			isActive = false;
		}

		void restart()
		{
			stop();
			clear();
			start();
		}

		void clear()
		{
			cudaEventSynchronize(end);
			elapsedTimeMs = 0.f;
		}

		float duration()
		{
			float et;

			cudaEventSynchronize(end);
			cudaEventElapsedTime(&et, begin, end);
			elapsedTimeMs += et;

			return elapsedTimeMs;
		}

	private:
		cudaEvent_t begin, end;
		float elapsedTimeMs;
		bool isActive;
	};
}
}
