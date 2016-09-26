/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include <fstream>
#include "testing_helpers.hpp"

const int SIZE = 1 << 24;
const int NPOT = SIZE - 3;
int a[SIZE], b[SIZE], c[SIZE];

int main(int argc, char* argv[]) {
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
	auto startTime = std::chrono::high_resolution_clock::now();
    StreamCompaction::CPU::scan(SIZE, b, a);
	auto endTime = std::chrono::high_resolution_clock::now();
    printArray(SIZE, b, true);
	std::chrono::duration<double, std::milli> eclipsed = endTime - startTime;
	double delta = eclipsed.count();
	printf("CPU scan power-of-two number time is %f ms\n", delta);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
	startTime = std::chrono::high_resolution_clock::now();
    StreamCompaction::CPU::scan(NPOT, c, a);
	endTime = std::chrono::high_resolution_clock::now();
	eclipsed = endTime - startTime;
	delta = eclipsed.count();
    printArray(NPOT, b, true);	
    printCmpResult(NPOT, b, c);
	printf("CPU scan non-power-of-two number time is %f ms\n", delta);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
 // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
 // printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
 // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
 // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
 // printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
 // printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

	printf("\n");
	printf("*********************************************\n");
	printf("*************** EXTRA CREDIT ****************\n");
	printf("************* RADIX SORT TESTS **************\n");
	printf("*************** POWER-OF-TWO ****************\n");
	printf("*********************************************\n");
	
	genArray(SIZE, a, SIZE);
	printArray(SIZE, a, true);
	memcpy(b, a, SIZE*sizeof(int));

	printDesc("std sort for comparasion");
	std::sort(a, a + SIZE);
	printArray(SIZE, a, true);
	printf("\n");

	printDesc("Extra : RadixSort");
	StreamCompaction::Radix::RadixSort(SIZE, b, SIZE);
	printCmpResult(SIZE, b, a);
	printArray(SIZE, b, true);
	printf("\n");

	printf("\n");
	printf("*********************************************\n");
	printf("*************** EXTRA CREDIT ****************\n");
	printf("************* RADIX SORT TESTS **************\n");
	printf("************* NON-POWER-OF-TWO **************\n");
	printf("*********************************************\n");

	//zeroArray(SIZE, c);
	//printDesc("work-efficient scan, power-of-two");
	//StreamCompaction::Efficient::scan(SIZE, c, a);
	//// printArray(SIZE, c, true);
	//printCmpResult(SIZE, b, c);

	//zeroArray(SIZE, c);
	//printDesc("work-efficient scan, non-power-of-two");
	//StreamCompaction::Efficient::scan(NPOT, c, a);
	//// printArray(NPOT, c, true);
	//printCmpResult(NPOT, b, c);

	genArray(SIZE, a, SIZE);
	printArray(SIZE, a, true);
	memcpy(b, a, NPOT * sizeof(int));

	printDesc("std sort for comparasion");
	std::sort(a, a + NPOT);
	printArray(NPOT, a, true);
	printf("\n");

	printDesc("Extra : RadixSort");
	StreamCompaction::Radix::RadixSort(NPOT, b, SIZE);
	printCmpResult(NPOT, b, a);
	printArray(NPOT, b, true);
	printf("\n");

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
	startTime = std::chrono::high_resolution_clock::now();
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
	endTime = std::chrono::high_resolution_clock::now();
	eclipsed = endTime - startTime;
	delta = eclipsed.count();
	expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);
	printf("CPU compact without scan power-of-two number time is %f ms\n", delta);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
	startTime = std::chrono::high_resolution_clock::now();
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
	endTime = std::chrono::high_resolution_clock::now();
	eclipsed = endTime - startTime;
	delta = eclipsed.count();
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
	printf("CPU compact without scan non-power-of-two number time is %f ms\n", delta);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
	startTime = std::chrono::high_resolution_clock::now();
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
	endTime = std::chrono::high_resolution_clock::now();
	eclipsed = endTime - startTime;
	delta = eclipsed.count();
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
	printf("CPU compact with scan time is %f ms\n", delta);

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
