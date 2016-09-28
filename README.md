CUDA Stream Compaction
======================

***PAGE UNDER CONSTRUCTION***

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Gabriel Naghi
* Tested on: Windows 7, i7-6700 @ 3.70GHz 16GB, Quadro K620 222MB (Moore 100C Lab)

## Overview

In this project, I implemented three Stream Compation algorithms.

Stream compaction is the process of compressing a list of elements, removing any elements 
which don't match some criteria and replacing the "good" elements in their original ordering. 
This algorithm will be useful in the future when dong ray-tracing, where once rays have escaped 
the scene, we no longer need to process them and thus wish to elmiate them from our list.

This is equivalent to the python dictionary comprehension one-liner:

output_list = [ elt for elt in input_list if elt != 0 ]

While this process is easily done in an iterative fashion, we can also employ some parallel algorithms 
to compute the compacted array more quickly. These parallel algorithms require that first a temporary boolean 
mapping of the list must be created, which then undergoes an "exclusive scan". 

A "scan" is an operation that creates and ouput list such that for each index an input list, the output list 
contains the sums of all elements preceeding it in the input list. The term "exclusive" means that the first 
element of the output array is always 0, and thus the last element of the input array is excluded. This contrasts 
with an "inclusive" scan, which begins with the first element of the input array.

It is here that we can divide our algorithms into naive and efficientimplementations. For comparison's sake, 
the scan method was also implemented as a CPU function.

For the purposes of our implementation, the criteria for inclusion in the output list in non-zero value.

### CPU Implementation

The CPU implementation functions in the most straightforward way possible. At each index, it simply adds the value at that index plus the preceeding calculated value, much like a fibbonacci sequence. 

The only optimization I was able to make here was that, instead of re-summing all input elements 0 through j-1 to compute 
element j, I simply add input element j-1 to output element j-2. We will see later in performace analysis, however, that
optimizations are inherit to the CPU implementation due to hardware features such as memory cacheing. 





### Naive Parallel Implementation

### Efficient Parallel Implementation 


## Performance Analysis

***PICTORAL DEPICTIONS OF ALGORITHMS DRAWN FROM CIS565 LECTURE NOTES***
