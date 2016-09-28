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
#include <chrono>
using namespace std::chrono;
#define RUNS 1

void runTimings() {
	
	high_resolution_clock::time_point start, end;
	for (int i = 15; i <= 15; i+=2) {
		const int SIZE = 1 << i;
		int *a = new int[SIZE];
		int *b = new int[SIZE];
		zeroArray(SIZE, a);
		zeroArray(SIZE, b);

		start = high_resolution_clock::now();
		StreamCompaction::CPU::scan(SIZE, b, a);
		end = high_resolution_clock::now();
		duration<double> duration = end - start;
		printf("CPU scan: %f ms\n", duration.count() * 1000.0f);

		StreamCompaction::Naive::scan(SIZE, b, a);
		StreamCompaction::Efficient::scan(SIZE, b, a);
		StreamCompaction::Thrust::scan(SIZE, b, a);

		delete a;
		delete b;


	}
}

void runTests() {
	const int SIZE = 1 << 8;
	const int NPOT = SIZE - 3;
	int *a, *b, *c;
	a = new int[SIZE];
	b = new int[SIZE];
	c = new int[SIZE];

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
	//printArray(SIZE, b, true);
	//printf("%lf\n", Timer::getCPUTiming("cpu_scan"));

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
	printArray(SIZE, c, true);
	printCmpResult(NPOT, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient scan, power-of-two");
	StreamCompaction::Efficient::scan(SIZE, c, a);
	printArray(SIZE, c, true);
	printCmpResult(SIZE, b, c);

	zeroArray(SIZE, c);
	printDesc("work-efficient scan, non-power-of-two");
	StreamCompaction::Efficient::scan(NPOT, c, a);
	printArray(NPOT, c, true);
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
}

int main(int argc, char* argv[]) {
	runTests();
	//runTimings();
}
