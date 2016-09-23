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
This algorithm will be useful in the future when dong ray-racing, where once rays have escaped 
the scene, we no longer need to process them and thus wish to elmiate them from our list.

While this process is easily done in an iteraive fashion, we can also employ some parallel algorithms 
to compute the compacted array more quickly. These parallel algorithms require that first a temporary boolean 
mapping of the list must be created, which then undergoes an "exclusive scan". It is here that we can divide 
our algorithms into naive and efficientimplementations. For comparison's sake, the scan method was also 
employed in a CPU implementation.

For the purposes of our implementation, the criteria for inclusion in the output list in non-zero value.

### CPU Implementation

The CPU implementation simply iterates through each index in the array and places it at the end of a
list if it meets the criteria. This is equivalent to the python dictionary comprehension one-liner:

output_list = [ elt for elt in input_list if elt != 0 ]



### Naive Parallel Implementation

### Efficient Parallel Implementation 


## Performance Analysis