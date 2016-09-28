CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

For this project we compare several implementations of two fundamental GPU algorithms:

- `scan`: similar to `fold` in functional programming. `scan` takes an input array and a binary operator and returns an array where each element is the reduction of the preceding elements in the input array using the binary operator. For example, if the binary operator is addition, as it was in our implementation, each element in the output array is the sum of all preceding elements in the input array.

- `stream-compact`: equivalent to `filter` in functional programming. `stream-compact` takes an input array and a test function and returns a shortened version of the input array containing only elements that pass the test function. In our implementations, we use an implicit test function that passes only if a number is not equal to zero. In other words, our functions filters out all zeros from the input array.

## Scan
We implemented three versions of `scan`:

- CPU scan: this implementation does not use the GPU. Instead it simply iterates through the input array on the CPU accumulating sums and writing them to the output array as it goes.

- Naive scan: this implementation iteratively adds elements at different strides: on the first iteration, the algorithm adds all elements that are directly adjacent; On the second iteration, the algorithm adds elements that are separated by a stride of two; on the third, elements are added at a stride of four. Strides continue to double until only one addition is performed. The following picture depicts this process:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/parallel%20scan.png)
Since the process terminates when the doubling strides exceed the length of the input array, the total number of iterations is O(log n), where n is the number of elements in the input array. Since each thread performs a single addition, the time complexity of the algorithm as a whole is O(log n), _assuming that there are O(n) threads_.

- Efficient scan: Naive scan is perfectly effective when there are as many threads as elements in the input array. However, this is not the case for larger arrays and instead threads must round-robin through waiting kernels. In this case, it is advantageous to minimize the number of threads launched at once. Naive scan actually performs _O(n log n)_ addition operations and consequently launches a total of _O(n log n)_ threads. Since the CPU version performs only _n_ addition operations, this suggests that we can do better. Efficient scan uses a clever upsweep/downsweep approach that achieves the logarithmic time complexity of the naive version but also performs only one addition per element in the array.

- Thrust scan: this is an implementation from Thrust, a C++ library of GPU algorithms.

The following diagram compares performance between these implementations:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_2.png)

Note that the x-axis in this diagram is logarithmic (that is, at each tick along the x-axis, the number of elements doubles).

There are a few curiosities about these results. First, we observe that efficient scan, naive scan, and Thrust scan, all appear to be running with linear time complexity. This is the case because the GeForce GTX 1070 only has 1920 CUDA cores. In any GPU, the number of threads is constant and does not increase with the size of the inputs. However in more powerful GPUs, this can still lead to substantial performance increases by parallelizing a large fraction of the operations and dividing the total execution time by a large constant number. However as the number elements in the array increases relative to the number of threads executing in parallel, this improvement becomes less evident. In fact, the constant time improvement may be offset by the shortcomings of the GPU, namely its lack of optimizations for sequential operations (e.g. pipelining).

This explains why the performance of the GPU implementations is comparable to (and in many cases worse than) the performance of the CPU implementation. It still does not explain the fact that naive scan outperforms all the other GPU algorithms, and efficient scan performs worst of all. One possible reason for this is that efficient scan reduces the total number of addition operations by O(log n). However, it requires the operation to be split into an upsweep and a downsweep, the latter having to wait for the former to complete. At the end of both these operations very few cores are active -- in fact, at the very end, only one core is active since approximately only one addition operation is performed on the last iteration. This is a problem for the naive implementation as well but the naive implementation only encounters this situation once, whereas the efficient implementation encounters this situation twice. Consequently the naive version actually has higher hardware saturation though it is also doing more work overall.

Another likely explanation is that the efficient implementation performs almost twice as many memory accesses per kernel invocation -- it performs three in the upsweep kernel and four in the downsweep whereas the naive implementation performs only two memory accesses per kernel invocation. Since the GPU can only perform a limited number of simultaneous memory accesses, this might be a bottleneck that hinders performance on the efficient implementation.

## Stream Compaction
We also implemented three versions of `stream-compact`.

- efficient stream-compact: this version takes three steps:

  1. We call a kernel to populate a new array of booleans that determines whether an element passes our test function (in our case, whether an element is nonzero).
  2. We perform `scan` on the array of booleans. We use the efficient version of `scan` described in the previous section.
  3. The output of `scan` actually corresponds to indices where the nonzero elements of the input array should be assigned. We use this information to assign elements of the input array to the output array in parallel.

- compact with scan: in order to emulate the GPU version, we implement a version on the CPU that uses the CPU scan algorithm. As the diagram below demonstrates, this has no performance benefit on the CPU.

- compact without scan: this final implementation is a straightforward CPU implementation that simply iterates through the input array and assigns all nonzero elements to the output array.

The following diagram compares performance across these implementations. Again, the x-axis is logarithmic.

![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_3.png).
In this case, we observe that the GPU implementation does outperform the CPU implementations. However, as discussed in the previous section, this cannot be credited to the `scan` operation, and must have more to do with steps 1. and 3. -- the parallel testing and assignment of each element in the input array to the output array.

## Large numbers bug
A strange bug that remains unresolved in this project is that our GPU algorithms simply would not run for arrays larger than 2^16. The following picture displays the result of running an array with 2^17 elements:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/BugPNG.PNG)
This appears to be a compiler error, since the program encounters no errors for any of the smaller arrays that we tested. One unfortunate consequence of this issue is that it was impossible to compare our GPU implementations with the CPU algorithms on very large arrays, where the GPU may have indeed had the advantage. We were able to compare the Thrust implementation of scan with the CPU version at sizes as large as 2^29. The following graph depicts the results:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_1.png)
It is interesting to note that the CPU _still_ outperforms even the optimized Thrust implementation for the GPU. Also, the Thrust implementation causes the same bug that we described earlier for arrays larger than 2^29. Again we reason that the poor performance of the GPU can be credited to the poor throughput of the GeForce GTX 1070.

## blockSize optimizations
On both the naive and efficient implementations, we experimented with different block sizes and record the results in the charts below:

![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_4.png)
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_5.png)

Because of these experiments, we ran all of the earlier experiments using block sizes of 256.
