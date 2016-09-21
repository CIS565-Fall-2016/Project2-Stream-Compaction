#include <cstdio>
#include "cpu.h"
#include <chrono>
#include <iostream>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
    
	odata[0] = 0;
	for (int i = 1; i < n; ++i)
		odata[i] = odata[i - 1] + idata[i - 1];

    //printf("CPU::scan done\n");
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	
	auto begin = std::chrono::high_resolution_clock::now();

	int index = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[index++] = idata[i];
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU::CompactWithoutScan --> Elapsed time = " << duration / 1000.f << "ms" << std::endl;

	return index;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {

	int* inputScan = new int[n];
	int* outputScan = new int[n];

	auto begin = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < n; ++i)
		inputScan[i] = (idata[i] == 0) ? 0 : 1;

	scan(n, outputScan, inputScan);

	int sum = 0;
	for (int i = 0; i < n; ++i)
	{
		if (idata[i] != 0)
		{
			odata[outputScan[i]] = idata[i];
			sum++;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU::CompactWithScan --> Elapsed time = " << duration / 1000.f<< "ms" <<std::endl;

	delete[] inputScan;
	delete[] outputScan;
	return sum;
}

}
}
