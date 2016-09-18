/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya, Ruoyu Fan
 * @date      2015, 2016
 * @copyright University of Pennsylvania
 */

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix_sort.h>

#include "testing_helpers.hpp"
#include <iterator>
#include <algorithm>

int main(int argc, char* argv[]) {
    const static int SIZE = 1 << 8;
    const static int NPOT = SIZE - 3;
    const static int SORT_SIZE = 1 << 10;
    const static int SORT_NPOT = SORT_SIZE - 3;

    int a[SIZE], b[SIZE], c[SIZE], d[SORT_SIZE], e[SORT_SIZE], f[SORT_SIZE];

    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // result for edge case of 0 looks fine
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
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // result for edge case of 0 looks fine
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
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");

    genArray(SORT_SIZE - 1, d, 100);  
    d[SORT_SIZE - 1] = 0;
    printArray(SORT_SIZE, d, true);

    printDesc("std::sort");
    std::copy(std::begin(d), std::end(d), std::begin(e));
    StreamCompaction::CPU::stdSort(std::begin(e), std::end(e));
    printArray(SORT_SIZE, e, true);

    printDesc("radix sort");
    std::copy(std::begin(d), std::end(d), std::begin(f));
    StreamCompaction::RadixSort::radixSort(std::begin(f), std::end(f));
    printArray(SORT_SIZE, f, true);
    printCmpLenResult(count, expectedNPOT, e, f);

}
