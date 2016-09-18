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
#include <iostream>

const int SIZE = 1 << 26;
const int NPOT = SIZE - 3;
const int SCAN_MAX = 50;
const int COMPACTION_MAX = 4;

const int SORT_SIZE = 1 << 26;
const int SORT_NPOT = SORT_SIZE - 3;
const int SORT_MAX = 100;

int a[SIZE], b[SIZE], c[SIZE], d[SORT_SIZE], e[SORT_SIZE], f[SORT_SIZE];

int main(int argc, char* argv[]) {

    // Scan tests
    
    std::cout << "CIS-565 HW2 CUDA Stream Compaction Test (Ruoyu Fan)";
    std::cout << std::endl;
    std::cout << "    Block size for naive scan: " << StreamCompaction::Naive::getNaiveScanBlockSize() << std::endl;
    std::cout << "    Block size for up-sweep: " << StreamCompaction::Efficient::getUpSweepBlockSize() << std::endl;
    std::cout << "    Block size for down-sweep: " << StreamCompaction::Efficient::getDownSweepBlockSize() << std::endl;
    std::cout << "    Block size for boolean mapping: " << StreamCompaction::Common::getMapToBooleanBlockSize() << std::endl;
    std::cout << "    Block size for scattering: " << StreamCompaction::Common::getScatterBlocksize() << std::endl;
    std::cout << "    Block sizes for radix sort: " 
        << StreamCompaction::RadixSort::getComputeBArrayBlockSize() << " "
        << StreamCompaction::RadixSort::getComputeDArrayBlockSize() << " "
        << StreamCompaction::RadixSort::getComputeEArrayBlockSize() << " "
        << StreamCompaction::RadixSort::getReshuffleBlockSize() << std::endl;

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");
    std::cout << "Array size (power of two): " << SIZE << std::endl;
    std::cout << "Array size (non-power of two): " << NPOT << std::endl;

    genArray(SIZE - 1, a, SCAN_MAX);  // result for edge case of 0 looks fine
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printArray(NPOT, b, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printArray(NPOT, c, true);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printArray(NPOT, c, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printArray(SIZE, c, true);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printElapsedTime(StreamCompaction::Thrust::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printArray(NPOT, c, true);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printElapsedTime(StreamCompaction::Thrust::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");
    std::cout << "Array size (power of two): " << SIZE << std::endl;
    std::cout << "Array size (non-power of two): " << NPOT << std::endl;
    // Compaction tests

    genArray(SIZE - 1, a, COMPACTION_MAX);  // result for edge case of 0 looks fine
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    expectedCount = count;
    printArray(count, b, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    expectedNPOT = count;
    printArray(count, c, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printArray(count, c, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printArray(count, c, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printArray(count, c, true);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("*****************************\n");
    std::cout << "Array size (power of two): " << SORT_SIZE << std::endl;
    std::cout << "Array size (non-power of two): " << SORT_NPOT << std::endl;
    std::cout << "Max value: " << SORT_MAX << std::endl;

    genArray(SORT_SIZE - 1, d, SORT_MAX);
    d[SORT_SIZE - 1] = 0;
    printArray(SORT_SIZE, d, true);

    printDesc("std::sort, power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(e));
    StreamCompaction::CPU::stdSort(std::begin(e), std::end(e));
    printArray(SORT_SIZE, e, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    printDesc("std::sort, power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(e));
    StreamCompaction::CPU::stdSort(std::begin(e), std::end(e));
    printArray(SORT_SIZE, e, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    printDesc("thrust unstable sort, power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(f));
    StreamCompaction::Thrust::unstableSort(std::begin(f), std::end(f));
    printArray(SORT_SIZE, f, true);
    printElapsedTime(StreamCompaction::Thrust::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SORT_SIZE, e, f);

    printDesc("thrust stable sort, power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(f));
    StreamCompaction::Thrust::stableSort(std::begin(f), std::end(f));
    printArray(SORT_SIZE, f, true);
    printElapsedTime(StreamCompaction::Thrust::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SORT_SIZE, e, f);

    printDesc("radix sort, power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(f));
    StreamCompaction::RadixSort::radixSort(std::begin(f), std::end(f), SORT_MAX);
    printArray(SORT_SIZE, f, true);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SORT_SIZE, e, f);

    // must be after all power-of-two sorts since it is standard value
    printDesc("std::sort, non power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(e));
    StreamCompaction::CPU::stdSort(std::begin(e), std::begin(e) + SORT_NPOT);
    printArray(SORT_NPOT, e, true);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    printDesc("radix sort, non power-of-two");
    std::copy(std::begin(d), std::end(d), std::begin(f));
    StreamCompaction::RadixSort::radixSort(std::begin(f), std::begin(f) + SORT_NPOT, SORT_MAX);
    printArray(SORT_NPOT, f, true);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SORT_NPOT, e, f);
}
