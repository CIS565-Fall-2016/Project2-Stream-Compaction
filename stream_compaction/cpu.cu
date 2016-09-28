#include <cstdio>
#include "cpu.h"
#include "chrono"

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
double scan(int n, int *odata, const int *idata) {
	// record time
	auto start = std::chrono::system_clock::now();

    // TODO : finished
	if (n <= 0) return -1;
	odata[0] = 0;
	for (int i = 1; i < n; ++i)
	{
		odata[i] = odata[i - 1] + idata[i-1];
	}

	std::chrono::duration<double, std::milli> diff = (std::chrono::system_clock::now() - start);
	//printf("CPU scan took %fms\n", diff.count());

	return diff.count();
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata, double &time) {
    // TODO : finished
	// record time
	auto start = std::chrono::system_clock::now();

	int num_remain = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[num_remain++] = idata[i];
		}
	}

	std::chrono::duration<double, std::milli> diff = (std::chrono::system_clock::now() - start);
	//printf("CPU compact without scan took %fms\n", diff.count());
	time = diff.count();
	return num_remain;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata, double &time) {
    // TODO : finished
	// record time
	auto start = std::chrono::system_clock::now();

	// map data to 1 and 0 for non-zero and zero.
	int *tmp_data = new int[n];
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] == 0) tmp_data[i] = 0;
		else tmp_data[i] = 1;
	}

	// scan
	scan(n, odata, tmp_data);

	// scatter
	int num_remain = 0;
	for (int i = 0; i < n; ++i)
	{
		if (tmp_data[i] == 1)
		{
			odata[odata[i]] = idata[i];
			num_remain++;
		}
	}

	delete[] tmp_data;

	std::chrono::duration<double, std::milli> diff = (std::chrono::system_clock::now() - start);
	//printf("CPU compact with scan took %fms\n", diff.count());
	time = diff.count();
	return num_remain;
}

}
}
