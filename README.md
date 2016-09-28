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


