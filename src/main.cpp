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
#include <stream_compaction/real_efficient.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

#include <cstdlib>
#include <ctime>


int main(int argc, char* argv[]) {
    double t1,t2;

    int sizeExp = 19;
    int blkSize = 256;
    if (argc >= 3) {
      sizeExp = atoi(argv[1]);
      blkSize = atoi(argv[2]);
    }
    int SIZE = 1 << sizeExp;
    int NPOT = SIZE - 3;
    int *a = new int[SIZE], *b = new int[SIZE], *c = new int[SIZE];

    StreamCompaction::Naive::blkSize = blkSize;
    StreamCompaction::Efficient::blkSize = blkSize;
    StreamCompaction::RealEfficient::blkSize = blkSize;

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
    double tCpuScanPot = StreamCompaction::CPU::last_runtime;

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);
    double tCpuScanNpot = StreamCompaction::CPU::last_runtime;

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    double tNaiveScanPot = StreamCompaction::Naive::last_runtime;

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);
    double tNaiveScanNpot = StreamCompaction::Naive::last_runtime;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    double tEffScanPot = StreamCompaction::Efficient::last_runtime;

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    double tEffScanNpot = StreamCompaction::Efficient::last_runtime;


    zeroArray(SIZE, c);
    printDesc("real work-efficient scan, power-of-two");
    StreamCompaction::RealEfficient::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    double tRealEffScanPot = StreamCompaction::RealEfficient::last_runtime;

    zeroArray(SIZE, c);
    printDesc("real work-efficient scan, non-power-of-two");
    StreamCompaction::RealEfficient::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    double tRealEffScanNpot = StreamCompaction::RealEfficient::last_runtime;


    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);
    double tThrustScanPot = StreamCompaction::Thrust::last_runtime;

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);
    double tThrustScanNpot = StreamCompaction::Thrust::last_runtime;

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
    double tCpuCompNoscanPot = StreamCompaction::CPU::last_runtime;

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    double tCpuCompNoscanNpot = StreamCompaction::CPU::last_runtime;

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printCmpLenResult(count, expectedCount, b, c);
    double tCpuCompScanPot = StreamCompaction::CPU::last_runtime;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    double tEffCompScanPot = StreamCompaction::Efficient::last_runtime;

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    double tEffCompScanNpot = StreamCompaction::Efficient::last_runtime;

    zeroArray(SIZE, c);
    printDesc("real work-efficient compact, power-of-two");
    count = StreamCompaction::RealEfficient::compact(SIZE, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);
    double tRealEffCompScanPot = StreamCompaction::RealEfficient::last_runtime;

    zeroArray(SIZE, c);
    printDesc("real work-efficient compact, non-power-of-two");
    count = StreamCompaction::RealEfficient::compact(NPOT, c, a);
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);
    double tRealEffCompScanNpot = StreamCompaction::RealEfficient::last_runtime;

    fprintf(stderr, "[%d, %d, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
      SIZE, blkSize,
      tCpuScanPot, tNaiveScanPot, tEffScanPot, tRealEffScanPot, tThrustScanPot,
      tCpuCompNoscanPot, tCpuCompScanPot, tEffCompScanPot, tRealEffCompScanPot
      );

    delete a;
    delete b;
    delete c;

    return 0;
}
