# University of Pennsylvania, CIS 565: GPU Programming and Architecture
## Project 2 - Stream Compaction
* Liang Peng
* Tested on: Windows 10, i7-6700HQ @ 2.6GHz, 8GB, GTX 960M (Personal Computer)

## Screenshots
* Result
<br><img src="img/Capture1.PNG" width="500"></img>

## Analysis
* Time measurement with std::chrono
<blockquote>
high_resolution_clock::time_point t1;<br>
kernel<<<..., ...>>>(...);<br>
cudaDeviceSynchronize();<br>
high_resolution_clock::time_point t2;<br>
duration t = t2 - t1;<br>
print t.count();<br>
</blockquote>

* Array size
<br><img src="img/Capture2.PNG" width="500"></img>

* Block size 
<br><img src="img/Capture3.PNG" width="500"></img>
