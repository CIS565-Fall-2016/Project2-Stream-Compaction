#include <cstdio>
#include "cpu.h"
#include "common.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;

	#if PROFILE
		auto begin = std::chrono::high_resolution_clock::now();
	#endif

	for (int i = 1; i < n; i++)
	{
		odata[i] = odata[i - 1] + idata[i - 1];
	}

	#if PROFILE
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Time Elapsed for cpu scan(size " << n << "): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
	#endif
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	long j = 0;

	#if PROFILE
		auto begin = std::chrono::high_resolution_clock::now();
	#endif

	for (int i = 0; i < n; i++)
	{
		if (idata[i] != 0)
		{
			odata[j] = idata[i];
			j++;
		}
	}

	#if PROFILE
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Time Elapsed for compact without scan(size " << n << "): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
	#endif

	return j;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	int *temporal;
	int *pscan;
	long j = 0;
	temporal = (int*)malloc(sizeof(int)*n);
	pscan = (int*)malloc(sizeof(int)*n);

	#if PROFILE
		auto begin = std::chrono::high_resolution_clock::now();
	#endif
	for (int i = 0; i < n; i++)
	{
		temporal[i] = idata[i] ? 1 : 0;
	}
	scan(n, pscan, temporal);
	for (int i = 0; i < n; i++)
	{
		if (temporal[i] == 1)
		{
			odata[pscan[i]] = idata[i];
			j++;
		}
	}

	#if PROFILE
		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Time Elapsed for compact with scan(size " << n << "): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "ns" << std::endl;
	#endif
	
	return j;
}

}
}
