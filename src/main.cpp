/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */


#include <ctime>
#include <iostream>
#include <iomanip>
#include <cstdio>
#define MEASURE_EXEC_TIME
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>
#include "testing_helpers.hpp"


#define TEST_GROUP_SIZE 100


int main(int argc, char* argv[]) {
    const int SIZE = 1 << 20;
    const int NPOT = SIZE - 3;
	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];

    // Scan tests
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 5);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

	float accumExecTime = 0.f;
	zeroArray(SIZE, b);
	printDesc("parallel radix sort, power-of-two");
	try
	{
		for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		{
			accumExecTime += ParallelRadixSort::sort(SIZE, reinterpret_cast<uint32_t *>(b), reinterpret_cast<uint32_t *>(a), 0xffffffff);
		}
	}
	catch (std::exception &e)
	{
		std::cout << "    ParallelRadixSort::sort: Error: " << e.what() << std::endl;
	}
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
	printArray(SIZE, b, true);
	testSorted(SIZE, b);

	accumExecTime = 0.f;
	zeroArray(SIZE, b);
	printDesc("parallel radix sort, non power-of-two");
	try
	{
		for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		{
			accumExecTime += ParallelRadixSort::sort(NPOT, reinterpret_cast<uint32_t *>(b), reinterpret_cast<uint32_t *>(a), 0xffffffff);
		}
	}
	catch (std::exception &e)
	{
		std::cout << "    ParallelRadixSort::sort: Error: " << e.what() << std::endl;
	}
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
	printArray(NPOT, b, true);
	testSorted(NPOT, b);

	//system("pause");

	accumExecTime = 0.f;
	zeroArray(SIZE, b);
	printDesc("thrust sort, power-of-two");
	try
	{
		for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		{
			accumExecTime += ParallelRadixSort::thrustSort(SIZE, reinterpret_cast<uint32_t *>(b), reinterpret_cast<uint32_t *>(a));
		}
	}
	catch (std::exception &e)
	{
		std::cout << "    ParallelRadixSort::thrustSort: Error: " << e.what() << std::endl;
	}
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
	printArray(SIZE, b, true);
	testSorted(SIZE, b);

	accumExecTime = 0.f;
	zeroArray(SIZE, b);
	printDesc("thrust sort, non power-of-two");
	try
	{
		for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		{
			accumExecTime += ParallelRadixSort::thrustSort(NPOT, reinterpret_cast<uint32_t *>(b), reinterpret_cast<uint32_t *>(a));
		}
	}
	catch (std::exception &e)
	{
		std::cout << "    ParallelRadixSort::thrustSort: Error: " << e.what() << std::endl;
	}
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
	printArray(NPOT, b, true);
	testSorted(NPOT, b);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
	clock_t start = clock();
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		StreamCompaction::CPU::scan(SIZE, b, a);
	clock_t end = clock();
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << double(end - start) / CLOCKS_PER_SEC * 1000 / TEST_GROUP_SIZE << "ms\n";
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
	start = clock();
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		StreamCompaction::CPU::scan(NPOT, c, a);
	end = clock();
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << double(end - start) / CLOCKS_PER_SEC * 1000 / TEST_GROUP_SIZE << "ms\n";
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Naive::scan(SIZE, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Naive::scan(NPOT, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Efficient::scan(SIZE, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Efficient::scan(NPOT, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Thrust::scan(SIZE, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
	accumExecTime = 0.f;
	for (int i = 0; i < TEST_GROUP_SIZE; ++i)
		accumExecTime += StreamCompaction::Thrust::scan(NPOT, c, a);
	std::cout << "    Execution Time: " << std::fixed << std::setprecision(2) << accumExecTime / TEST_GROUP_SIZE << "ms\n";
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient compact, power-of-two");
	count = StreamCompaction::Efficient::compact(SIZE, c, a);
	printArray(count, c, true);
	printCmpLenResult(count, expectedCount, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient compact, non-power-of-two");
	count = StreamCompaction::Efficient::compact(NPOT, c, a);
	printArray(count, c, true);
	printCmpLenResult(count, expectedNPOT, b, c);

	delete[] a;
	delete[] b;
	delete[] c;

	system("pause");
	return 0;
}
