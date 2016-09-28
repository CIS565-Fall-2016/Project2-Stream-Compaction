CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Ethan Brooks
* Tested on: Windows 7, Intel(R) Xeon(R), GeForce GTX 1070 8GB (SIG Lab)

For this project we compare several implementations of two fundamental GPU algorithms:

- `scan`: similar to `fold` in functional programming. `scan` takes an input array and a binary operator and returns an array where each element is the reduction of the preceding elements in the input array using the binary operator. For example, if the binary operator is addition, as it was in our implementation, each element in the output array is the sum of all preceding elements in the input array.

- `stream-compact`: equivalent to `filter` in functional programming. `stream-compact` takes an input array and a test function and returns a shortened version of the input array containing only elements that pass the test function. In our implementations, we use an implicit test function that passes only if a number is not equal to zero. Therefore our functions filter out all zeros from the input array.

## Scan
We implemented three versions of `scan`:

- CPU scan: this implementation does not use the GPU. Instead it simply iterated through the input array on the CPU accumulating sums and writing them to the output array on the way.

- Naive scan: this implementation adds the elements of the input array in by iterating over the array and adding elements next to each other at different strides. In other words, on the first iteration, the algorithm adds all elements that are directly adjacent. On the second iteration, the algorithm adds elements that are separated by a stride of two. On the third, elements are added at a stride of four. Strides continue to double until only one addition is performed. The following picture depicts this process:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/parallel%20scan.png)
Since the process terminates when the doubling strides exceed the length of the input array, the total number of iterations is O(log n), where n is the number of elements in the input array. Since each thread performs a single addition, the time complexity of the algorithm as a whole is O(log n), _assuming that there are O(n) thread_.

- Efficient scan: Naive scan is perfectly effective when there are as many threads as elements in the input array. However, this is not the case for larger arrays and instead threads must round-robin through waiting kernels. In this case, it is advantageous to try to launch as few threads at once as possible. Naive scan actually performs O(n log n) addition operations and consequently launches a total of O(n log n) threads. Since the CPU version actually performs only n addition operations, this suggests that we can do better. Efficient scan uses a clever upsweep/downsweep approach that achieves the logarithmic time complexity of the naive version but also performs only one addition per element in the array.

- Thrust scan: this is an implementation from Thrust, a C++ library of GPU algorithms.

The following diagram compares performance between these techniques:
![alt text] (https://github.com/lobachevzky/Project2-Stream-Compaction/blob/master/Profiling_Page_2.png)

Note that the depiction of array sizes in this diagram is logarithmic (that is, at each tick along the x-axis, the number of elements doubles).

There are a few curiosities about these results. First, we observe that efficient scan, naive scan, and Thrust scan, all appear to be running with linear time complexity. This is the case because the GeForce GTX 1070 only has 1920 CUDA cores. In any GPU, the number of threads is constant and does not increase with the size of the inputs. However in more powerful GPUs, this can still lead to substantial performance increases by parallelizing a large fraction of the operations and dividing the total execution time by a large constant number. However as the number elements in the array increases relative to the number of threads executing in parallel, this improvement becomes less evident. In fact, the constant time improvement may be offset by the shortcomings of the GPU, namely its lack of optimizations for sequential operations.

This explains why the performance of the GPU scans is still comparable to the performance of the CPU scan. It still does not explain the fact that naive scan outperforms all the other GPU algorithms, and efficient scan performs worst of all. One possible reason for this is that efficient scan reduces the total number of addition operations by O(log n). However, it requires the operation to be split into an upsweep and a downsweep, the latter having to wait for the former to complete. At the end of both these operations very few cores are active -- in fact, at the very end, only one core is active since approximately only one addition operation is performed on the last iteration. This is a problem for the naive implementation as well but it only happens once. In contrast the efficient implementation actually encounters this situation twice.
