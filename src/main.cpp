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
#include <stream_compaction/sort.h>
#include <cstring>
#include <thrust/sort.h>
#include "testing_helpers.hpp"

int main(int argc, char* argv[]) {
	const int POW = 16;
	const int SIZE = 1 << POW;
    const int NPOT = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

	const int KEYSIZE = 16;
	int a_sm[8], b_sm[8], c_sm[8];
	genArray(8, a_sm, 1 << KEYSIZE);

	genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case

    // Scan tests
#if 0
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    //printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
	printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printArray(SIZE, c, true);
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
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	printf("\n");
	printf("**********************\n");
	printf("** RADIX SORT TESTS **\n");
	printf("**********************\n");


	zeroArray(8, b_sm);
	zeroArray(8, c_sm);
	printDesc("radix sort, power-of-two");
	printArray(8, a_sm, true);
	memcpy(b_sm, a_sm, 8 * sizeof(int));
	StreamCompaction::Sort::radix(8, KEYSIZE, c_sm, a_sm);
	thrust::sort(b_sm, b_sm + 8);
	printArray(8, c_sm, true);
	printCmpResult(8, b_sm, c_sm);

	int a_npot_sm[7];
	zeroArray(8, b_sm);
	zeroArray(8, c_sm);
	memcpy(a_npot_sm, a_sm, 7 * sizeof(int));
	printDesc("radix sort, non-power-of-two");
	printArray(7, a_npot_sm, true);
	memcpy(b_sm, a_npot_sm, 7 * sizeof(int));
	StreamCompaction::Sort::radix(7, KEYSIZE, c_sm, a_npot_sm);
	thrust::sort(b_sm, b_sm + 7);
	printArray(7, c_sm, true);
	printCmpResult(7, b_sm, c_sm);

#endif
#if 0
	int successes = 0;
	int tests = 1000;
	for (int n = 0; n < tests; n++){
		zeroArray(SIZE, b);
		zeroArray(SIZE, c);
		genArray(SIZE, a, 1 << POW);  // Leave a 0 at the end to test that edge case
		memcpy(b, a, SIZE*sizeof(int));
		thrust::sort(b, b + SIZE);
		StreamCompaction::Sort::radix(SIZE, KEYSIZE, c, a);

		if (!cmpArrays(SIZE, b, c)) successes++;
	}

	printf("\nSort passed %i/%i randomly generated verification tests.\n", successes, tests);

#endif

#if 1
	printf("\n");
	printf("**********************\n");
	printf("**   TIMING TESTS   **\n");
	printf("**********************\n");

#define ITER 1 << i

	zeroArray(SIZE, c);
	printDesc("naive scan, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Naive::TestScan(ITER, c, a);

	zeroArray(SIZE, c);
	printDesc("efficient scan, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Efficient::TestScan(ITER, c, a);


	zeroArray(SIZE, c);
	printDesc("Thrust scan, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Thrust::TestScan(ITER, c, a);
	



	zeroArray(SIZE, c);
	printDesc("CPU scan, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::CPU::TestScan(ITER, c, a);

	
	zeroArray(SIZE, c);
	printDesc("CPU compact, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::CPU::TestCompact(ITER, c, a);

	zeroArray(SIZE, c);
	printDesc("CPU compact without scan, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::CPU::TestCompactWithoutScan(ITER, c, a);


	zeroArray(SIZE, c);
	printDesc("efficient compact, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Efficient::TestCompact(ITER, c, a);



	zeroArray(SIZE, c);
	printDesc("Thrust stable sort, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Thrust::TestSortStable(ITER, c, a);


	zeroArray(SIZE, c);
	printDesc("Thrust unstable sort, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Thrust::TestSortUnstable(ITER, c, a);


	zeroArray(SIZE, c);
	printDesc("Radix sort, power-of-two");
	for (int i = 1; i <= POW; i++) StreamCompaction::Sort::TestSort(ITER, KEYSIZE, c, a);

#endif

}