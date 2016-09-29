#include <cstdio>
#include "timer.h"
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
	void scan(int n, int *odata, const int *idata)
	{
		Timer::playTimer();
		{
			odata[0] = 0;
			for (size_t i = 1; i < n; ++i)
			{
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}
		Timer::pauseTimer();
	}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata)
{
	int j = 0;
	Timer::playTimer();
	{
		for (int i = 0; i < n; ++i)
		{
			if (idata[i] != 0)
				odata[j++] = idata[i];
		}
	}
	Timer::pauseTimer();
	return j;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) 
{
	Timer::playTimer();
		int* filterData = new int[n];
		//1) Compute temporary array containing 1 if corresponding element meets
		//criteria, 0 otherwise.
		for (int i = 0; i < n; ++i)
		{
			filterData[i] = (idata[i] != 0);
		}

		//2) Run exclusive scan on temporary array.
		scan(n, odata, filterData);
		const int numElementsAfterCompaction = odata[n - 1];

		//3) Result of scan is index into final array.
		//Only write an element if temporary array has a 1
		for (int i = 0; i < n; ++i)
		{
			if (filterData[i] == 1)
				odata[odata[i]] = idata[i];
		}
		delete[] filterData;
	Timer::pauseTimer();

	return numElementsAfterCompaction;
}

}
}
