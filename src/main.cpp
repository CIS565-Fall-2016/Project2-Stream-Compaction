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
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"
#include <algorithm>
#include <time.h>
#include <iostream>
#include <windows.h>


int comparator(const void* a, const void* b)
{
    return (*(int*)a - *(int*)b);
}

int main(int argc, char* argv[]) {
    const int SIZE = 1 << 8;
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

    printDesc("cpu radix sort");
    memcpy(b, a, sizeof(int)*SIZE);
    memcpy(c, a, sizeof(int)*SIZE);
    std::qsort(b, SIZE, sizeof(int), comparator);
    //printArray(SIZE, b, true);
    StreamCompaction::Radix::radixCPU(SIZE, c, a);
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);


    printDesc("gpu radix sort cpu scan");
    memcpy(c, a, sizeof(int)*SIZE);
    std::qsort(b, SIZE, sizeof(int), comparator);
    //printArray(SIZE, b, true);
    StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::CPU);
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    printDesc("gpu radix sort naive scan");
    memcpy(c, a, sizeof(int)*SIZE);
    std::qsort(b, SIZE, sizeof(int), comparator);
    //printArray(SIZE, b, true);
    StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::NAIVE);
    printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    printDesc("gpu radix sort thrust scan");
    memcpy(c, a, sizeof(int)*SIZE);
    std::qsort(b, SIZE, sizeof(int), comparator);
    //printArray(SIZE, b, true);
    StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::THRUST);
    printArray(SIZE, c, true);
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












    {
        // bechmarks

        // Scan tests

        printf("\n");
        printf("****************\n");
        printf("** BENCHMARKS **\n");
        printf("****************\n");

        
        clock_t t;
        int f;
        LARGE_INTEGER start, end, freq;



        
        int iterations = 5;
        int maxiterations = 25;
        for (int i = 8; i < maxiterations;  i += 1)
        {
            int SIZE = (1 << i);
            int NPOT = SIZE - 3;

            printf("\n** SIZE = %d **\n", SIZE);
            printf("****************\n");

            int* a = new int[SIZE];
            int* b = new int[SIZE];
            int* c = new int[SIZE];
            zeroArray(SIZE, a);
            zeroArray(SIZE, b);
            zeroArray(SIZE, c);

            genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
            a[SIZE - 1] = 0;
            printArray(SIZE, a, true);

            //t = clock();

            //t = clock() - t;
            //printf("Elapsed time %f second(s).\n", t, ((float)t) / CLOCKS_PER_SEC);

            Sleep(1);
            zeroArray(SIZE, b);
            printDesc("cpu scan, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::CPU::scan(SIZE, b, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("cpu scan, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::CPU::scan(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";
            */


            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("naive scan, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Naive::scan(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("naive scan, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Naive::scan(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";
            */



            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("work-efficient scan, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Efficient::scan(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("work-efficient scan, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Efficient::scan(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";
            */


            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("thrust scan, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Thrust::scan(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("thrust scan, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Thrust::scan(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";
            */


            Sleep(1);
            printDesc("cpu radix sort");
            memcpy(c, a, sizeof(int)*SIZE);
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Radix::radixCPU(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            Sleep(1);
            printDesc("gpu radix sort cpu scan");
            memcpy(c, a, sizeof(int)*SIZE);
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::CPU);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";
            
            Sleep(1);
            printDesc("gpu radix sort naive scan");
            memcpy(c, a, sizeof(int)*SIZE);
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::NAIVE);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";

            Sleep(1);
            printDesc("gpu radix sort thrust scan");
            memcpy(c, a, sizeof(int)*SIZE);
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
                StreamCompaction::Radix::radixGPU(SIZE, c, a, StreamCompaction::Radix::THRUST);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart)/iterations << " microSeconds\n";




            // compaction


            genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
            a[SIZE - 1] = 0;

            Sleep(1);
            zeroArray(SIZE, b);
            printDesc("cpu compact without scan, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
            count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart) / iterations << " microSeconds\n";


            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("cpu compact without scan, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
            count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart) / iterations << " microSeconds\n";
            */


            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("cpu compact with scan");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
            count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart) / iterations << " microSeconds\n";

            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("work-efficient compact, power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
            count = StreamCompaction::Efficient::compact(SIZE, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart) / iterations << " microSeconds\n";

            /*
            Sleep(1);
            zeroArray(SIZE, c);
            printDesc("work-efficient compact, non-power-of-two");
            QueryPerformanceFrequency(&freq);
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++)
            count = StreamCompaction::Efficient::compact(NPOT, c, a);
            QueryPerformanceCounter(&end);
            std::cout << "Time:  " << ((end.QuadPart - start.QuadPart) * 1000000 / freq.QuadPart) / iterations << " microSeconds\n";
            */












            delete[] a;
            delete[] b;
            delete[] c;

            Sleep(1);

        }

    }






    return 0;
}
