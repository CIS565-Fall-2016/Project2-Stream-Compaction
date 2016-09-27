/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <chrono>
#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

#define TIMING 1

void comparePerformance(const int SIZE, const int RUNS) {
	printf("Timing performance with arrays of size %d, averaged over %d runs\n", SIZE, RUNS);
	int * a = new int[SIZE];
	int * b = new int[SIZE];
	genArray(SIZE, a, 2);

	for (int i = 0; i < RUNS; i++) {
		std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
		start = std::chrono::high_resolution_clock::now();

		StreamCompaction::CPU::scan(SIZE, b, a);

		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsedSeconds = end - start;
		printf("CPU scan: %lf milliseconds\n", elapsedSeconds.count() * 1000.0f);

		StreamCompaction::Naive::scan(SIZE, b, a);

		StreamCompaction::Efficient::scan(SIZE, b, a);

		StreamCompaction::Thrust::scan(SIZE, b, a);
	}

	delete a;
	delete b;
}

void runTests() {
	const int SIZE = 1 << 23;
	const int NPOT = SIZE - 17;
	int * a = new int[SIZE];
	int * b = new int[SIZE];
	int * c = new int[SIZE];

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

	delete a;
	delete b;
	delete c;
}


int main(int argc, char* argv[]) {
#if TIMING == 1
	comparePerformance(1 << 10, 3);
	comparePerformance(1 << 13, 3);
	comparePerformance(1 << 16, 3);
	comparePerformance(1 << 19, 3);
	comparePerformance(1 << 22, 3);
	comparePerformance(1 << 25, 3);
	comparePerformance(1 << 28, 3);
#else
	runTests();
#endif
}
