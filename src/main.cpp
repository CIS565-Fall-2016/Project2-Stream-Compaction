/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"
#include <string>
#include <algorithm>


void bench_mark()
{
	int run_times = 100;
	const int block_choice = 5;
	const int input_choice = 4;
	const int method_choice = 7;

	int blockSizes[] = { 32, 128, 256, 512, 1024 };
	int inputSizes[] = { 8, 16, 20, 24 };
	std::string methods[] = {
		"cpu scan",
		"naive scan",
		"eff scan",
		"thrust scan",
		"cpu compact no scan",
		"cpu compact",
		"gpu compact"
	};

	double result[method_choice * block_choice * input_choice];

	for (int i = 0; i < run_times; ++i)
	{
		printf("=========Running %d round ... \n", i+1);
		for (int j = 0; j < block_choice; ++j)
		{
			printf("==================Running on blockSize %d ... \n", blockSizes[j]);
			for (int k = 0; k < input_choice; ++k)
			{
				printf("=============================Running on inputSize 2^%d ... \n", inputSizes[k]);
				int idx;
				// generate input
				int SIZE = 1 << inputSizes[k];
				int *a = new int[SIZE];
				int *b = new int[SIZE];

				genArray(SIZE - 1, a, 1000);
				int cur_method = 0;

				// cpu scan
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				result[idx] += StreamCompaction::CPU::scan(SIZE, b, a);

				// naive scan
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				result[idx]
					+= StreamCompaction::Naive::scan(SIZE, b, a, blockSizes[j]);

				// work-efficient scan
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				result[idx]
					+= StreamCompaction::Efficient::scan(SIZE, b, a, blockSizes[j]);

				// thrust scan
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				result[idx]
					+= StreamCompaction::Thrust::scan(SIZE, b, a);

				// cpu compact no scan
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				double time;
				StreamCompaction::CPU::compactWithoutScan(SIZE, b, a, time);
				result[idx] += time;

				// cpu compact
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("==test on pos %d \n", idx);
				zeroArray(SIZE, b);
				StreamCompaction::CPU::compactWithScan(SIZE, b, a, time);
				result[idx] += time;

				// gpu compact
				idx = block_choice * input_choice * (cur_method++) + j * input_choice + k;
				printf("test on pos %d \n", idx);
				zeroArray(SIZE, b);
				StreamCompaction::Efficient::compact(SIZE, b, a, time);
				result[idx] += time;

				delete[] a;
				delete[] b;
			}
		}
	}

	// print result
	printf("===================== RESULTS ========================\n");
	for (int j = 0; j < block_choice; ++j)
	{
		printf("======= block size %d ===========\n", blockSizes[j]);

		for (int i = 0; i < method_choice; ++i)
		{
			printf("==== method %s ==== ", methods[i].c_str());
			for (int k = 0; k < input_choice; ++k)
			{
				printf(" %d input %f time ", inputSizes[k], result[block_choice * input_choice * i + j * input_choice + k] / run_times);
			}
			printf("\n");
		}

		printf("=====================================\n");
	}
}


int main(int argc, char* argv[]) {
    const int SIZE = 1 << 16;
    const int NPOT = SIZE - 3;
    //int a[SIZE], b[SIZE], c[SIZE];
	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];

    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
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

	double time;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a, time);
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a, time);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a, time);
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a, time);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a, time);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	zeroArray(SIZE, c);
	a[SIZE - 1] = 5;
	printDesc("work-efficient compact, power-of-two, last non-zero");
	count = StreamCompaction::Efficient::compact(SIZE, c, a, time);
	int *bb = new int[SIZE];
	int cpuCount = StreamCompaction::CPU::compactWithoutScan(SIZE, bb, a, time);
	//printArray(count, c, true);
	printCmpLenResult(count, cpuCount, bb, c);

	zeroArray(SIZE, c);
	a[SIZE - 1] = 0;
	printDesc("work-efficient compact, power-of-two, last zero");
	count = StreamCompaction::Efficient::compact(SIZE, c, a, time);
	//printArray(count, c, true);
	printCmpLenResult(count, expectedCount, b, c);

	printDesc("work-efficient compact, test on special case 1");
	int test[5] = { 1, 0, 1, 0, 1 };
	count = StreamCompaction::Efficient::compact(5, c, test, time);
	printCmpLenResult(count, 3, c, c);

	printDesc("work-efficient compact, test on special case 2");
	int test1[5] = { 1, 0, 1, 0, 0 };
	count = StreamCompaction::Efficient::compact(5, c, test1, time);
	printCmpLenResult(count, 2, c, c);

	printDesc("cpu compact without scan, test on special case 1");
	count = StreamCompaction::CPU::compactWithoutScan(5, c, test, time);
	printCmpLenResult(count, 3, c, c);



	//bench_mark();

	int testArr[] = { 0, 5, -2, 6, 3, 7, -5, 2, 7, 1 };
	int resultArr[10];
	int goalArr[] = { -5, -2, 0, 1, 2, 3, 5, 6, 7, 7 };

	StreamCompaction::Efficient::radix_sort(10, resultArr, testArr);
	printDesc("radix sort, test on special case");
	printArray(10, testArr, true);
	printf("  sorted:\n");
	printArray(10, resultArr, true);
	printCmpResult(10, goalArr, resultArr);

	genArray(SIZE, a, 10000);
	StreamCompaction::Efficient::radix_sort(SIZE, b, a);
	printDesc("radix sort, test");
	printArray(SIZE, a, true);
	printf("  sorted:\n");
	printArray(SIZE, b, true);
	std::sort(a, a + SIZE);
	printCmpResult(SIZE, a, b);


	delete[] a;
	delete[] b;
	delete[] c;
	delete[] bb;
}
