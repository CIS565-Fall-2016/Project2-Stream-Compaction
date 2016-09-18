#include <cstdio>
#include "cpu.h"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	odata[0] = 0;
	for (int i = 1; i < n; ++i) {
		odata[i] = idata[i - 1] + odata[i - 1];
	}
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int j = 0;
	for (int i = 0; i < n; ++i) {
		if (idata[i] != 0) {
			odata[j++] = idata[i];
		}

	}
    return (n - j);
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
	//Run the scan on temp array
	int *tmp = new int[n];
	int j = 0;
	scan(n, tmp, idata);

	for (int i = 0; i < n - 1; ++i) {
		if ((tmp[i] != tmp[i + 1])) {
			odata[j++] = idata[i];
		}
	}
	delete[] tmp;
	return n - j;
}

}
}
