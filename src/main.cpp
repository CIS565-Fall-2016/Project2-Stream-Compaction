/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>

#include <stream_compaction/timer.h>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>

#include "testing_helpers.hpp"

int main(int argc, char* argv[])
{
	const int  SIZE = 1 << 16;
	const int  NPOT = SIZE - 3;
	int* a = new int[SIZE];
	int* b = new int[SIZE];
	int* c = new int[SIZE];

	Timer::initializeTimer();
	const int numTestRepeat = 1000;

	// Scan tests
	printf("****************\n");
	printf("** TESTS INFO **\n");
	printf("****************\n");
	printf("\n");
	printf("ArraySize = %d; ArrayOddSize: %d\n", SIZE, NPOT);

	printf("****************\n");
	printf("** SCAN TESTS **\n");
	printf("****************\n");

	genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);

	printDesc("cpu scan, power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, b);
			StreamCompaction::CPU::scan(SIZE, b, a);
		}
		printArray(SIZE, b, true);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}
	
	printDesc("cpu scan, non-power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::CPU::scan(NPOT, c, a);
		}
		printArray(NPOT, b, true);
		printCmpResult(NPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("naive scan, power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::Naive::scan(SIZE, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("naive scan, non-power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::Naive::scan(NPOT, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(NPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("work-efficient scan, power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::Efficient::scan(SIZE, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("work-efficient scan, non-power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::Efficient::scan(NPOT, c, a);
		}
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("thrust scan, power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::Thrust::scan(SIZE, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("thrust scan, non-power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);

			StreamCompaction::Thrust::scan(NPOT, c, a);
		}
		//printArray(NPOT, c, true);
		printCmpResult(NPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

	printDesc("cpu compact without scan, power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, b);

			count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
			expectedCount = count;
		}
		printArray(count, b, true);
		printCmpLenResult(count, expectedCount, b, b);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("cpu compact without scan, non-power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);

			count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
			expectedNPOT = count;
		}
		printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("cpu compact with scan");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
		}
		printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("work-efficient compact, power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			count = StreamCompaction::Efficient::compact(SIZE, c, a);
		}
		//printArray(count, c, true);
		printCmpLenResult(count, expectedCount, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("work-efficient compact, non-power-of-two");
	{
		Timer::resetTimer();
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			count = StreamCompaction::Efficient::compact(NPOT, c, a);
		}
		//printArray(count, c, true);
		printCmpLenResult(count, expectedNPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printf("\n");
	printf("*****************************\n");
	printf("** RADIX SORT TESTS **\n");
	printf("*****************************\n");

	// RdixSort tests
	genArray(SIZE - 1, a, SIZE);  // Leave a 0 at the end to test that edge case
	a[SIZE - 1] = 0;
	printArray(SIZE, a, true);

	printDesc("thrust radix-sort, power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, b);
			StreamCompaction::Thrust::radixSort(SIZE, b, a);
		}
		printArray(SIZE, b, true);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("my radix-sort, power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::RadixSort::sort(SIZE, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(SIZE, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("thrust radix-sort, non-power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, b);
			StreamCompaction::Thrust::radixSort(NPOT, b, a);
		}
		printArray(NPOT, b, true);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	printDesc("my radix-sort, non-power-of-two");
	{
		Timer::resetTimer(false);
		for (size_t i = 0; i < numTestRepeat; ++i)
		{
			zeroArray(SIZE, c);
			StreamCompaction::RadixSort::sort(NPOT, c, a);
		}
		//printArray(SIZE, c, true);
		printCmpResult(NPOT, b, c);
		Timer::printTimer(NULL, 1.0f / numTestRepeat);
	}

	Timer::shutdownTimer();

	delete[] a;
	delete[] b;
	delete[] c;
}
