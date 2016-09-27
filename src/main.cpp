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
#include <iostream>
#include "testing_helpers.hpp"
#include <stream_compaction/profilingcommon.h>

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 16;
    const int NPOT = SIZE - 3;
	int a[SIZE];
	int b[SIZE];
	int c[SIZE];

#ifdef PROFILE
	float timeElapsedMs = 0;
	float totalTimeElapsedMs = 0;

	printDesc("PROFILING ON");
	printf("\n\n");
#endif

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

#ifdef PROFILE
	auto begin = std::chrono::high_resolution_clock::now();
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif

		StreamCompaction::CPU::scan(SIZE, b, a);

#ifdef PROFILE
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin)/PROFILE_ITERATIONS).count() / 1000000.0f << " ms" << std::endl;
#endif

	printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");

#ifdef PROFILE
	begin = std::chrono::high_resolution_clock::now();
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif
		StreamCompaction::CPU::scan(NPOT, c, a);
#ifdef PROFILE
	}
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin) / PROFILE_ITERATIONS).count() / 1000000.0f << " ms" << std::endl;
#endif
	
	printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");

#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Naive::scan(SIZE, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Naive::scan(NPOT, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");

#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Efficient::scan(SIZE, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif

	printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Efficient::scan(NPOT, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Thrust::scan(SIZE, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		StreamCompaction::Thrust::scan(NPOT, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
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

    int count = 0, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
#ifdef PROFILE
	begin = std::chrono::high_resolution_clock::now();
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif
		count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
#ifdef PROFILE
	}
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin) / PROFILE_ITERATIONS).count() / 1000000.0f << " ms" << std::endl;
#endif
	expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
#ifdef PROFILE
	begin = std::chrono::high_resolution_clock::now();
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif
		count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
#ifdef PROFILE
	}
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin) / PROFILE_ITERATIONS).count() / 1000000.0f << " ms" << std::endl;
#endif
	expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
#ifdef PROFILE
	begin = std::chrono::high_resolution_clock::now();
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
#endif
		count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
#ifdef PROFILE
	}
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Runtime: " << std::chrono::duration_cast<std::chrono::nanoseconds>((end - begin) / PROFILE_ITERATIONS).count() / 1000000.0f << " ms" << std::endl;
#endif
	printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		count = StreamCompaction::Efficient::compact(SIZE, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
#ifdef PROFILE
	totalTimeElapsedMs = 0;
	for (auto it = 0; it < PROFILE_ITERATIONS; ++it) {
		timeElapsedMs = 0;
#endif
		count = StreamCompaction::Efficient::compact(NPOT, c, a, &timeElapsedMs);
#ifdef PROFILE
		totalTimeElapsedMs += timeElapsedMs;
	}
	std::cout << "Runtime: " << totalTimeElapsedMs / PROFILE_ITERATIONS << " ms" << std::endl;
#endif
	printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
}
