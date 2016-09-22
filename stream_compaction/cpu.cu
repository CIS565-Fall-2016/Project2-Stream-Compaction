#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	if (n <= 0 || !odata || !idata)
	{
		return;
	}

	int pre, cur;

	for (int i = 0; i < n; ++i)
	{
		cur = idata[i];

		if (i == 0)
		{
			odata[i] = 0;
		}
		else
		{
			odata[i] = pre + odata[i - 1];
		}

		pre = cur;
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	if (n <= 0 || !odata || !idata)
	{
		return -1;
	}

	const int *obegin = odata;

	for (int i = 0; i < n; ++i)
	{
		if (idata[i])
		{
			*odata++ = idata[i];
		}
	}

	return static_cast<int>(odata - obegin);
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	if (n <= 0 || !odata || !idata || odata == idata)
	{
		return -1;
	}

	if (n == 1 && idata[0])
	{
		odata[0] = idata[0];
		return 1;
	}

	for (int i = 0; i < n; ++i)
	{
		odata[i] = static_cast<int>(idata[i] != 0);
	}

	scan(n, odata, odata);

	int odataSize = odata[n - 1];

	for (int i = 0; i < n - 1; ++i)
	{
		if (odata[i] != odata[i + 1])
		{
			odata[odata[i]] = idata[i];
		}
	}

	if (idata[n - 1])
	{
		odata[odataSize] = idata[n - 1];
		++odataSize;
	}
	return odataSize;
}

}
}
