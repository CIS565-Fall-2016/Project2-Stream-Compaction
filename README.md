CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Austin Eng
* Tested on: Windows 10, i7-4770K @ 3.50GHz 16GB, GTX 780 3072MB (Personal Computer)

## Analysis

**Note: Reported graphs are the result of 100 trials, averaged. Also note that input sizes are at increasing powers of two. Furthermore, since the algorithm is exponential in growth, both axes are displayed at a log scale**

![Scan Analysis](https://docs.google.com/spreadsheets/d/1x1MppbyAceIIrwhDLsmV7unUYS2RYU_I20_wK0ReORY/pubchart?oid=175703576&format=image)

![Compaction Analysis](https://docs.google.com/spreadsheets/d/1x1MppbyAceIIrwhDLsmV7unUYS2RYU_I20_wK0ReORY/pubchart?oid=477396612&format=image)

## Analysis

For smaller input sizes, the CPU implementation for both Scan and Stream Compaction is much, much faster than the GPU implementation. When dealing with contiguous buffers of memory, the CPU reaps large benefits from cache which makes it very fast. However, at around 2^19 in input size, the more efficient GPU implementations begin to outperform the CPU. With only a single core, CPU performance becomes worse as the number of computations required increases exponentially.

Meanwhile, on the GPU, the exponent of this algorithmic growth is divided by the number of cores so there is much slower growth. However, there is a larger cost from memory access so the GPU implementations are much slower for lower input sizes because of this memory overhead. Memory usage, however, increases linearly not exponentially, so for larger sets of data, the GPU wins with performance.

In comparing the Naive and Efficient GPU implementations, we see that for smaller datasets, the Naive implementation is faster. This is probably because there are fewer kernel invocations are made. Even though there are an exponential number of additions, this still takes less time. However, as the input size increases, the much more computationally-efficient method performs better.

I did not see much difference between power-of-two input sizes and non-power-of-two data sizes. This is likely because my implementation just increases the size of non-power-of-two inputs to be power-of-two inputs.

### Why is Thrust So Fast?

It seems like the Thrust implementation receives a big performance boost from using shared memory. From the names of the function calls: `accumulate_tiles, exclusive_scan, exclusive_downsweep` it seems like Thrust is doing the same thing as the Efficient implementation except the `accumulate_tiles` calls have 32 and 4064 static and dynamic bytes of shared memory, respectively. `exclusive_scan`: 48 and 12240. `exclusive_downsweep`: 32 and 6880. This probably allows for much more efficient memory access in the kernel. Analysis also shows that each of the kernels is called twice, notably wrapped in `cuda_task` and `parallel_group`. This is probably done because the computation needs to be split into multiple pieces since shared memory can only be so large.

## Test Output
```
****************
** SCAN TESTS **
****************
    [  38  19  38  37   5  47  15  35   0  12   3   0  42 ...  10   0 ]
==== cpu scan, power-of-two ====
Elapsed: 10.0046ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
==== cpu scan, non-power-of-two ====
Elapsed: 9.0066ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed
==== naive scan, power-of-two ====
Elapsed: 9.708448ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== naive scan, non-power-of-two ====
Elapsed: 9.713088ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ...   0   0 ]
    passed
==== work-efficient scan, power-of-two ====
Elapsed: 4.019968ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== work-efficient scan, non-power-of-two ====
Elapsed: 3.999136ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed
==== thrust scan, power-of-two ====
Elapsed: 0.906560ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473618 205473628 ]
    passed
==== thrust scan, non-power-of-two ====
Elapsed: 1.042912ms
    [   0  38  57  95 132 137 184 199 234 234 246 249 249 ... 205473524 205473568 ]
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   3   2   1   3   1   1   1   2   0   1   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
Elapsed: 17.0074ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact without scan, non-power-of-two ====
Elapsed: 17.0071ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== cpu compact with scan ====
Elapsed: 6.0037ms
Elapsed: 31.0118ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, power-of-two ====
Elapsed: 5.496416ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
==== work-efficient compact, non-power-of-two ====
Elapsed: 5.449856ms
    [   2   3   2   1   3   1   1   1   2   1   2   1   1 ...   2   2 ]
    passed
```