# University of Pennsylvania, CIS 565: GPU Programming and Architecture
## Project 2 - Stream Compaction
* Liang Peng
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz, 8GB, GTX 960M (Personal Computer)

## Screenshots
* Result
<br>![](http://i.imgur.com/fDr2pRK.jpg)

## Analysis
* Time measurement with std::chrono
<blockquote>
high_resolution_clock::time_point t1;<br>
kernel<<<..., ...>>>(...);<br>
cudaDeviceSynchronize();<br>
high_resolution_clock::time_point t2;<br>
duration t = t1 = t2;<br>
print t.count();<br>
</blockquote>

* Array size
<br>![](http://i.imgur.com/fDr2pRK.jpg)

* Block size 
<br>![](http://i.imgur.com/fDr2pRK.jpg)
