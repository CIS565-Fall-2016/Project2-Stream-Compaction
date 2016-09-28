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
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 15;
    const int NPOT = SIZE - 3;
    int a[SIZE], b[SIZE], c[SIZE];

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
	printf("****************************\n");
	printf("** SCAN PERFORMANCE TESTS **\n");
	printf("****************************\n");
	uint32_t iterations = 100;
	zeroArray(SIZE, c);
	auto begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::CPU::scan(SIZE, c, a);
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU POW SCAN TIME ELAPSED : " << (float)(duration / iterations) * 0.000001 << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::CPU::scan(NPOT, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU NPOT SCAN TIME ELAPSED : " << (float)(duration / iterations) * 0.000001 << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	float timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Naive::scan(SIZE, c, a);
	}
	std::cout << "NAIVE POW SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Naive::scan(NPOT, c, a);
	}
	std::cout << "NAIVE NPOT SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Efficient::scan(SIZE, c, a);
	}
	std::cout << "EFFICIENT POW SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Efficient::scan(NPOT, c, a);
	}
	std::cout << "EFFICIENT NPOT SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Thrust::scan(SIZE, c, a);
	}
	std::cout << "THRUST POW SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		timer += StreamCompaction::Thrust::scan(NPOT, c, a);
	}
	std::cout << "THRUST NPOT SCAN TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

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
	printf("*****************************************\n");
	printf("** STREAM COMPACTION PERFORMANCE TESTS **\n");
	printf("*****************************************\n");

	zeroArray(SIZE, c);
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::CPU::compactWithoutScan(SIZE, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU COMPACT NOSCAN POW TIME ELAPSED : " << (float)(duration / iterations) * 0.000001 << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU COMPACT NOSCAN NPOT TIME ELAPSED : " << (float)(duration / iterations) * 0.000001 << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	begin = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::CPU::compactWithScan(SIZE, c, a);
	}
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
	std::cout << "CPU COMPACT SCAN TIME ELAPSED : " << (float)(duration / iterations) * 0.000001 << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::Efficient::compact(SIZE, c, a, &timer);
	}
	std::cout << "EFFICIENT POW COMPACT TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;

	zeroArray(SIZE, c);
	timer = 0.0f;
	for (int i = 0; i < iterations; ++i)
	{
		StreamCompaction::Efficient::compact(NPOT, c, a, &timer);
	}
	std::cout << "EFFICIENT NPOT COMPACT TIME ELAPSED : " << timer / iterations << " milliseconds." << std::endl;
}
