CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Bowen Bao
* Tested on: Windows 10, i7-6700K @ 4.00GHz 32GB, GTX 1080 8192MB (Personal Computer)

## Overview

Here's the list of features of this project:

1. CPU Scan and Stream Compaction
2. Naive GPU Scan
3. Efficient GPU Scan and Stream Compaction
4. Thrust Scan
5. Optimize efficient GPU Scan
6. Radix Sort based on GPU Scan
7. Benchmark suite

## Instruction to Run

I made a few changes to the function headers to add more flexible benchmarking capabilities, such as able to return process times, change block size without re-compile, etc. The only change that is visible to the user is that they need to pass in a double parameter as reference to be able to receive the logged process time. 

I added a benchmark suite for testing the run time of each implementation under different parameter settings. Also, I inserted a few tests for radix sort into the original main function.

## Performance Analysis
### Performance of different implementation

![](/image/process_time.png)

Here's the test result for each of the methods. The tests are run with the block size of 256(which is decided as near optimal after testing on numerous values). For each methods, I ran 100 independent tests, and calculated their average process time.

We can observe indeed that the GPU version of scan has a better performance than CPU scan. 

### Performance of GPU methods under different block size

![](/image/process_time_blocksize.png)

The tests are run with the stream length of 2^24, each method is tested 100 times and recorded the average. Observe that the performance starts to decrease after blocksize getting over 256. 

## Extra Credits
### Improving GPU Scan
See part 3 in Questions.

### Radix Sort
I followed the algorithm in the slides, and implemented a radix sort method based on the GPU Scan function. One interesting note is that when checking bits of the numbers, numbers with 1 on the first bit are actually smaller than those with 0, as on these occasions they turned out to be negative, which is the reverse case against situations on other bits. I tested my radix sort function with a special hand crafted case containing negative numbers, and with a random large test case.

## Questions
* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)

See Performance Analysis.

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * You should use CUDA events for timing GPU code. Be sure **not** to include
    any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your
    performance measurements, for comparability. Note that CUDA events cannot
    time CPU code.
  * You can use the C++11 `std::chrono` API for timing CPU code. See this
    [Stack Overflow answer](http://stackoverflow.com/a/23000049) for an example.
    Note that `std::chrono` may not provide high-precision timing. If it does
    not, you can either use it to time many iterations, or use another method.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

See Performance Analysis.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

One problem with "naive" efficient GPU scan is that there are too many threads wasted(after being checked that their index mod interval is not zero). One way of improving this is to assign the index as the divided result of the original index by the interval, and compute back the actual index later in that thread. With this improvement, we can save a lot of useless works done by threads, and note that waste grows exponentially with the number of elements in stream in the original implementation.

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

See Output.

## Output

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
	    passed
	==== naive scan, non-power-of-two ====
	    passed
	==== work-efficient scan, power-of-two ====
	    passed
	==== work-efficient scan, non-power-of-two ====
	    passed
	==== thrust scan, power-of-two ====
	    passed
	==== thrust scan, non-power-of-two ====
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
	    passed
	==== work-efficient compact, non-power-of-two ====
	    passed
	==== work-efficient compact, power-of-two, last non-zero ====
	    passed
	==== work-efficient compact, power-of-two, last zero ====
	    passed
	==== work-efficient compact, test on special case 1 ====
	    passed
	==== work-efficient compact, test on special case 2 ====
	    passed
	==== cpu compact without scan, test on special case 1 ====
	    passed
	==== radix sort, test on special case ====
	    [   0   5  -2   6   3   7  -5   2   7   1 ]
	  sorted:
	    [  -5  -2   0   1   2   3   5   6   7   7 ]
	    passed
	==== radix sort, test ====
	    [  38 7719 1238 2437 8855 1797 8365 2285 450 612 5853 8100 1142 ... 5085 6505 ]
	  sorted:
	    [   0   0   0   0   0   0   0   1   1   1   1   1   1 ... 9999 9999 ]
	    passed