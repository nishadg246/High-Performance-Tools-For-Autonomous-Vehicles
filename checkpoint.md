---
layout: default
---

# 15-418 Final Project Checkpoint
## High Performance Video Processing Suite For Autonomous Vehicles
Nishad Gothoskar and Cyrus Tabrizi

### Updated Schedule
- Tue April 25th (11:59pm) -- Project Checkpoint Report Due
- Fri Apr 28th - Finish preliminary GPU implementation for each algorithm
- Mon May 1st - Complete first full GPU vs CPU benchmark
- Wed May 3rd - Push CPU and GPU versions closer to real-time
- Sat May 6th - Combine algorithms to build pipeline
- Tue, May 9th -- Final performance tuning for pipeline, prepare video of demo
- Wed May 10th (9:00am) -- Project pages are made available to judges for finalist selection.
- Thurs May 11th (3pm) -- Finalists announced, presentation time sign ups
- Fri May 12th -- Parallelism Competition Day + Project Party / Final Report Due at 11:59pm

### Checkpoint Update

Progress has been made on four different fronts.
The first component is a CUDA implementation of k-means clustering for 3D data points. This is important so that we can recognize the clusters as objects. To implement k-means, we’re using the standard iterative approach, but computing clusters and membership in parallel across points.  Given that k-means requires that all computation for a stage is completed before the reassignment of clusters is done, a large amount of synchronization is necessary. 
The second CUDA implementation is of the Canny edge detection algorithm, Canny edge detection is also a key component of feature extraction on RGB image data. Canny edge detection involves applying Gaussian filters using convolution and then detecting gradient directions. These are just the first two steps in the process, but are highly parallelizable across pixels and regions. 
	The third component has been feature comparison and feature matching across images. This feature enables us to identify the location of a given object across multiple frames. Combining this with our depth data will eventually enable us to build a map as we move through the scene. Currently, we’re able to evaluate a similarity score for a region in two different images as well as identify the region in one image that is most similar to a specified region in another. This has been implemented sequentially on a CPU without any data reuse and it is already clear what the first steps should be for optimizing the CPU version.
	The last component has been to set up OpenCV and a basic timing framework. Currently, the feature comparison and feature matching functions have been implemented inside of this framework, giving me an executable that I can run on various images and get timing values as a result.
  
The main thing that changed between our proposal and this checkpoint has been our shift from working on a set of independent vision functions meant for generic use on autonomous vehicle sensor data (functions like “Edge Detection”) to working on a vision pipeline that will specifically support real-time Simultaneous Localization and Mapping (SLAM) on that same autonomous vehicle sensor data. We knew from the beginning that we were interested in supporting some kind of vision work in autonomous vehicles but only now have we figured out the specific way we wanted to do that.
Making this shift towards a real-time SLAM pipeline has been good for us as it helped us direct our efforts. Previously, when we knew only that we wanted to support a generic library of “real-time” vision functions, the “real-time” requirement was hard to define. Now that we know that those functions have to work together to support SLAM, we can determine exactly what timing requirements our pipeline must meet in order for the vehicle to stop given a map-based visual cue.

Our updated demo will show the sensor data at the original data capture rate and generate a viewable map at the same rate and within a latency that would allow a car to brake safely at the sight of an obstacle. Since we can’t collect our own sensor data and demo this on an actual vehicle, we will allow the data capture rate to be adjusted during the demo to simulate travelling at different speeds. Being able to control the data capture rate as we tune our pipeline further will allow us to estimate the resulting braking distance, which is the measure by which we’ll evaluate the performance of our GPU and CPU SLAM implementations. We’ll also make it possible to explore the resolution/performance trade-offs with our SLAM pipeline since it’s possible a lower-resolution map output might be an acceptable compromise to improve timing requirements.

The main issue we’ve faced so far was our unfamiliarity with the specific details of SLAM. Implementing SLAM on a short deadline required understanding enough about existing approaches to start working on its individual components without knowing exactly how those components would need to perform or behave once combined. It’s a risky pipeline to implement because we’re implementing and learning about both those individual components for the first time as well as the pipeline overall. Another issue is that our work on those individual components can only be evaluated meaningfully once combined into a working SLAM demo. The individual algorithms behind SLAM may be challenging and we’ll still be able to compare optimized CPU and optimized GPU versions, but without them working together, there won’t be anything particularly exciting to show for.

Even though we’re behind schedule according to our original deadlines, our decision to move towards a specific kind of functionality has helped us catch up. It’s still our intention to have a complete working demo by the competition day even though it’s unknown exactly what additional algorithms we’ll need to make SLAM work in real-time. This unknown is concerning but also exciting. At this point, it’s mostly a matter of getting the work done.


