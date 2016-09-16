#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	
	//exclusive scan: first element of output is 0
	odata[0] = 0;

	for (int i = 0; i < n - 1; ++i)
	{
		odata[i + 1] = odata[i] + idata[i];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
    
	int oIndex = 0;

	for (int i = 0; i < n; ++i)
	{
		if (idata[i])
		{
			odata[oIndex] = idata[i];
			++oIndex;
		}
	}

	return oIndex;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	
	int *temp = new int[n];
	int *scanResult = new int[n];
	// compute the temp array
	for (int i = 0; i < n; ++i)
	{
		if (idata[i])
			temp[i] = 1;
		else
			temp[i] = 0;
	}

	// Run exclusive scan on temp array
	scan(n, scanResult, temp);

	// result of scan is index into final array
	int oCnt = 0;
	for (int i = 0; i < n; ++i)
	{
		// only write if tmp array has 1
		if (temp[i])
		{
			odata[scanResult[i]] = idata[i];
			++oCnt;
		}
	}

	delete[] temp;
	delete[] scanResult;

	return oCnt;
}

}
}
