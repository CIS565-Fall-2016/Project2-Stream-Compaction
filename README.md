**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 2 - Strean Compaction**

* Trung Le
* Windows 10 Home, i7-4790 CPU @ 3.60GHz 12GB, GTX 980 Ti (Person desktop)

### Stream compaction

**---- General information for CUDA device ----**
- Device name: GeForce GTX 980 Ti
- Compute capability: 5.2
- Compute mode: Default
- Clock rate: 1076000
- Integrated: 0
- Device copy overlap: Enabled
- Kernel execution timeout: Enabled
 
**---- Memory information for CUDA device ----**

- Total global memory: 6442450944
- Total constant memory: 65536
- Multiprocessor count: 22
- Shared memory per multiprocessor: 98304
- Registers per multiprocessor: 65536
- Max threads per multiprocessor: 2048
- Max grid dimensions: [2147483647, 65535, 65535]
- Max threads per block: 1024
- Max registers per block: 65536
- Max thread dimensions: [1024, 1024, 64]
- Threads per block: 512

# Analysis

Roughly optimize the block sizes of each of your implementations for minimal run time on your GPU.

(You shouldn't compare unoptimized implementations to each other!)
Compare all of these GPU Scan implementations (Naive, Work-Efficient, and Thrust) to the serial CPU version of Scan. Plot a graph of the comparison (with array size on the independent axis).

For timing GPU, I wrapped cuda events between kernel launches and for timing CPU, I used the C++11 std::chrono API. Each configuration is run 1000 times, then taken the average as displayed below:

![Scan performance](https://github.com/trungtle/Project2-Stream-Compaction/blob/master/images/ScanPerformaceAnalysis.png "Scan performance")


![Compaction performance](https://github.com/trungtle/Project2-Stream-Compaction/blob/master/images/CompactPerformaceAnalysis.png "Compaction performance")

As we can see, the CPU version is outperformed by the rest. Thrust is clearly a winner here (probably due to the fact that it was implemented properly). It occurs to me that the 'efficient' version is in fact a bit slower than the naive but is still faster than the CPU version. There are a couple reasons for this:
- We're not taking advantage of shared memory inside each block to store the partial sum results.
- Each level of upsweep/downsweep currently launches a new kernel. It would be ideal to use the same kernel and compute the next level there without having to transfer the control back to the CPU.
- At deeper level in the upsweep/downsweep calls, there are a lot of idle threads not doing work. This is wasting a lot of GPU cycles.
- In the stream compaction phase, in order to find the number of remaining elements after compaction, I launched a new kernel to search for the maximum value in the prefix-sum array that is used to index into the output array. This could be a potential bottle neck but I haven't tested a different version to compare.
- There are quite a bit of memory transfering between GPU & CPU, which initially slowed the application down alot. So I rewrote my scan and compaction functions to minimize this memory transfer.

When testing with different block sizes, I found it pretty interesting that at size 128, it seems to be the most optimal. So I decided to use this block size for the rest of profiling 

![Block sizes performance](https://github.com/trungtle/Project2-Stream-Compaction/blob/master/images/BlockSizePerformanceAnalysis.png "Block sizes performance")

For more details on the data collected, see [link](https://docs.google.com/spreadsheets/d/1mtohoQ4BtD_RamWI2KeV-HhkSYDMmendWos7sQgdVR8/edit?usp=sharing).

I also used NSight to profile thrust performance. It seems that thrust does take advantage of shared memory (24,528 bytes per block). It's occupancy is also lower (50.0%) and it uses more registers per threads compare to my efficient implementation.

![Thrust performance](https://github.com/trungtle/Project2-Stream-Compaction/blob/master/images/ThrustCapture.PNG "Thrust performance")


# Test output

```
==== PROFILING ON ====
****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
Runtime: 0.1365 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
==== cpu scan, non-power-of-two ====
Runtime: 0.1402 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== naive scan, power-of-two ====
Runtime: 0.0925244 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== naive scan, non-power-of-two ====
Runtime: 0.0927348 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
Runtime: 1.72386 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== work-efficient scan, non-power-of-two ====
Runtime: 1.79924 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== thrust scan, power-of-two ====
Runtime: 0.0006529 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== thrust scan, non-power-of-two ====
Runtime: 0.0006317 ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
Runtime: 0.1463 ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
Runtime: 0.1484 ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
Runtime: 0.47 ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
Runtime: 2.01726 ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
Runtime: 2.01408 ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
```

## Note
### Modified test
I added a #define PROFILE and #define PROFILE_ITERATIONS flags in a new header file "profilingcommon.h". When this is on, running main() will also iterate through each function call PROFILE_ITERATIONS number of times, then measure the execution time and average it for profiling analysis.

### Modified CMakeList.txt
- Added "ProfilingCommon.h"
- Changed to -arch=sm_52
