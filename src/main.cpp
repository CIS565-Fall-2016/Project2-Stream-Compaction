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
#include "testing_helpers.hpp"
#include <iostream>
#include <chrono>

#define ITER 1

int main(int argc, char* argv[]) {
    const size_t SIZE = 1 << 20;
    const int NPOT = SIZE - 3;
    //int a[SIZE], b[SIZE], c[SIZE];
	
	int *a = new int[SIZE];
	int *b = new int[SIZE];
	int *c = new int[SIZE];

	float time = 0.f, totalTime = 0.f;
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
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; ++i){
		StreamCompaction::CPU::scan(SIZE, b, a);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (float)duration << " ms total, average : " << (float)duration / ITER << " ms." << std::endl;
    //printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    //printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

	totalTime = 0.f;
    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    //printArray(SIZE, c, false);
	for (int i = 0; i < ITER; ++i) {
		StreamCompaction::Naive::scan(SIZE, c, a, time);
		totalTime += time;
	}
	std::cout << "total time to run naive scan, power-of-two: " << totalTime << " in ms, aver: "  << totalTime / ITER << std::endl;
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a, time);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

	totalTime = 0.f;
    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
	for (int i = 0; i < ITER; ++i) {
		StreamCompaction::Efficient::scan(SIZE, c, a, time);
		totalTime += time;
	}
	std::cout << "total time to run efficient scan, power-of-two: " << totalTime << " in ms, aver: " << totalTime / ITER << std::endl;
    //printArray(SIZE, c, false);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
	StreamCompaction::Efficient::scan(NPOT, c, a, time);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; ++i){
		StreamCompaction::Thrust::scan(SIZE, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (float)duration << " ms total thrust, average : " << (float)duration / ITER << " ms." << std::endl;
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
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; ++i){
		count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (float)duration << " ms total cpu w/o scan, average : " << (float)duration / ITER << " ms." << std::endl;
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
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; ++i){
		count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (float)duration << " ms total cpu w scan, average : " << (float)duration / ITER << " ms." << std::endl;
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

	totalTime = 0.f;
    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
	for (int i = 0; i < ITER; ++i) {
		count = StreamCompaction::Efficient::compact(SIZE, c, a, time);
		totalTime += time;
	}
	std::cout << "total time to run efficient compact, power-of-two: " << totalTime << " in ms, aver: " << totalTime / ITER << std::endl;
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
	count = StreamCompaction::Efficient::compact(NPOT, c, a, time);
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

	printf("\n");
	printf("*****************************\n");
	printf("** RADIX SORT TESTS **\n");
	printf("*****************************\n");

	// SORT tests

	genArray(SIZE, a, 7);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);

	totalTime = 0.f;
	zeroArray(SIZE, b);
	printDesc("radix sort, power-of-two");
	for (int i = 0; i < ITER; ++i) {
		StreamCompaction::Sort::sort(SIZE, b, a, time);
		totalTime += time;
	}
	std::cout << "total time to run radix, power-of-two: " << totalTime << " in ms, aver: " << totalTime / ITER << std::endl;
	printArray(SIZE, b, true);


	zeroArray(SIZE, b);
	printDesc("thrust sort, power-of-two");
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < ITER; ++i){
		StreamCompaction::Thrust::sort(SIZE, a);
	}
	end = std::chrono::high_resolution_clock::now();
	float fduration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
	std::cout << (float)fduration << " ms total thrust sort, average : " << (float)fduration / ITER << " ms." << std::endl;

	delete[] a;
	delete[] b;
	delete[] c;
}
