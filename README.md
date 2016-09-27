CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Nischal K N
* Tested on: Windows 10, i7-2670QM @ 2.20GHz 8GB, GTX 540 2GB (Personal)

### SUMMARY
This project implements scan, stream compaction and sorting algorithms implemented on GPU and its performance compared against a CPU implementation. Two kinds of exclusive scans are implemented, viz., naive and efficient. Stream compaction is implemented using the efficient GPU scan implementation. Additionally Radix sort is also implemented using efficient stream compaction. The performance of these algorithms are measured against CPU and thrust implementations and are documented in Analysis section.

### BUILD
* To setup cuda development environment use [this guide](https://github.com/nischalkn/Project0-CUDA-Getting-Started/blob/master/INSTRUCTION.md#part-1-setting-up-your-development-environment)
* Once the environment is set up, to Build and run the project follow [this](https://github.com/nischalkn/Project0-CUDA-Getting-Started/blob/master/INSTRUCTION.md#part-3-build--run)

### PARAMETERS
* `SIZE` - Size of array to be scanned, compacted and sorted
* `PROFILE` - `1` or `0`, Print execution times

### PERFORMANCE ANALYSIS
The time taken to perform exclusive scans on arrays of different sizes ranging from `2^8` or `2^16` were recorded with a fixed block size of `128` (program crashed for array sizes beyond `2^16`). It was seen that the time taken by the CPU was very small to be recorded accurately for small array sizes but would increase exponentially as the array size increases. It is also seen that the naive scan outperforms work-efficient scan at larger array sizes because of the overhead of upsweep and downsweep steps which increase logarithmically with increase in array size. Most of the threads do not do any work in the extreme stages of the upsweep and downsweep. Many threads are launched but do not perform any work. This overhead causes a dramatic increase in execution time. It was also observed that the thrust implementation consistently was the slowest. The following are the time taken for different kind of scans in milliseconds.

|Scan                             |	2^8 |	2^10  |	2^12  |	2^14  |	2^16  |
|:--------------------------------|-----|-------|-------|-------|-------|
|cpu scan, power-of-two           |	0   |	0     |	0     |	0	    | 0     |
|cpu scan, non-power-of-two       |	  0	|0      |	0	    |0      |	0.5004|
|naive scan, power-of-two         |	0.5357	|0.7699|	0.8006|	1.2946|	2.4013|
|naive scan, non-power-of-two     |	0.6124|	0.6252|	0.6991|	1.0961|	2.4055|
|work-efficient scan, power-of-two|	0.0750|	0.1054|	0.2469|	0.8709|	3.5367|
|work-efficient scan, non-power-of-two|	0.0745|	0.1050|	0.2465|	0.8685|	3.5334|
|thrust scan, power-of-two        |	6.7073|	6.8454|	7.4349|	10.7565|	28.5635|
|thrust scan, non-power-of-two    |	0.7849|	0.8866|	1.6581|	4.8936|	16.7807|

![](images/scan_array.png)

An other experiment was conducted with various block sizes for a constant array length of `2^16`. It is seen that the best performance is obtained with a block size of `128`.
![](images/scan_block.png)
CPU compactions for array sizes less than `2^14` take negligible amount of time. The work-efficient compactions is the slowest because of the same reason as mentioned above. A number of threads are launched which do not do any work.

|Compact                                  |	2^8 |	2^10  |	2^12  |	2^14  |	2^16  |
|:----------------------------------------|-----|-------|-------|-------|-----|
|cpu compact without scan, power of two	|0	|0	|0	|0.5029	|0.500066667|
|cpu compact without scan, non-power of two	|0	|0	|0	|0.5019	|0.50035|
|cpu compact with scan	|0	|0	|0	|0.4998	|0.999733333|
|work-efficient compact, power-of-two	|0.089568	|0.123530667	|0.280234667|	0.946997333	|3.793076667|
|work-efficient compact, non-power-of-two	|0.086741333	|0.118101333|	0.276192	|0.947157333	|3.79866|


![](images/compact_array.png)
Again running the compactions on different block sizes result in a best block size of `128`.
![](images/compact_block.png)
A radix sort was implemented using the work-efficient scan and its performance was compared with the `thrust::sort`. It was seen that for small array sizes, the radix sort outperforms thrust sort. But however as the array size increases, the work-efficient scan becomes slow as explained above and reduces the effciency of sorting algorithm.
![](images/sort.png)

### FUTURE IMPROVEMENTS
* Work-efficient scan can be optimized to reduce the number of threads launched at each stage of upsweep and downsweep process.
* Early termination of threads would also improve the scan performance.

### OUTPUT
```

****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  35   0 ]
==== cpu scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
==== cpu scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== naive scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== naive scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== work-efficient scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed
==== thrust scan, power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604374 1604409 ]
    passed
==== thrust scan, non-power-of-two ====
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 1604305 1604316 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed
==== cpu compact with scan ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   1   1 ]
    passed
==== work-efficient compact, non-power-of-two ====
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   3   1 ]
    passed

*****************************
***** RADIX SORT TESTS ******
*****************************
==== thrust sort small array ====
    [   0   1   2   3   4   5   6   7   8   9 ]
==== Radix Sort small array ====
    [   0   1   2   3   4   5   6   7   8   9 ]
    passed
==== thrust sort large array ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 199 199 ]
==== Radix Sort large array ====
    [   0   0   0   0   0   0   0   0   0   0   0   0   0 ... 199 199 ]
    passed
```

### MODIFICATIONS
* radixSort was added to StreamCompaction module containing the sort function.
* sort function was also added to thrust.cu module to perform `thrust::sort`.
* 4 tests are added to the main function for sorting. 2 test to sort an array using `thrust::sort` and 2 tests to sort the array using `StreamCompaction::radixSort`.
* CMakeLists.txt of StreamCompaction was edited to include `radixSort.h` and	`radixSort.cu`.
