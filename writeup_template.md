## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./test_images/test4.jpg "Test Image"
[image2]: ./output_images/test4_result.png "Vehicles Detected 1"
[image3]: ./output_images/test4_plot.png "Vehicle Detection Plot 1"
[image4]: ./output_images/test_images_plot.png "Vehicle Classes"
[image5]: ./output_images/hog_image_plot.png "Hog Image"
[image6]: ./output_images/test5_result.png "Vehicles Detected 2"
[image7]: ./output_images/test5_plot.png "Vehicle Detection Plot 2"
[image8]: ./output_images/slide_windows.png "Slide Windows"
[image9]: ./output_images/video_clip_false_positive.png "False positive"
[video1]: ./project_video_output.mp4 "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell under the section "Initialize list of functions to be used inside the project" and sub-section titled "HOG features and visualization" of the IPython notebook "Vehicle Detection.ipynb". 

This code cell contains the function `get_hog_features()` to compute HOG features.

The code cell under the title "Compute features of a single image" takes in a single image and computes not only the HOG Features but also the Binned Spatial features and Color Histogram features if the flags are set. This function `single_img_features()` is same as the one mentioned inside lecture notes.

Next function `extract_features()` under "Extract features from a list of images" computes the features for a list of images. 

Next set of logic to extract HOG features for `vehicle` and `non-vehicle` images is under the section "Classifier Training". I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image4]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image5]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including multiple color spaces like `RGB` and `YCrCb` but going over multiple online resoures, I found out that many settled on the color space `YCrCb`. I have also gone through the [HOG research paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) where they have used `RGB` to identify persons but when I used that color space in this project, I got a lot of false positives. Even slight imperfections on the road were identified as cars. It was too cumbersome to apply the `heat` parameter to filter these out.

When I switched to `YCrCb` the false positives dropped significantly, probably because now there is a good linear seperation between `vehicles` and `non-vechicles`. 

Regarding the other parameters like `orient`, `pix_per_cell` and `cell_per_block`, I followed the discussion in the research paper mentioned above. One variation for `orient` is I used `9` instead of `8` orientations because I felt it gave a better result - less false positives.

Finally for `hog_channel`, I prefered to go with `ALL` channels as this gave more feature vectors to train the classifier.

The final choice of parameters is in the code cell under "Global Parameters".

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using just HOG Features as it was enough to properly differentiate between `vehicles` and `non-vehicles`. The training logic can be found under the section "Training the Classifier". Here are the steps followed:

1. Extract features for cars and not cars
2. Apply StandardScaler to features
3. Define labels vectors with '1's for cars and '0's for non-cars
4. Split data into training and testing - 20% for testing and randomize the data
5. Use a Linear Support Vector Machines Classifier to train data
6. Print Accuracy

The final feature vector length with `9 orientations`, `8 pixels per cell` and `2 cells per block` had `5292` feature vectors. It took just less than `17sec` to train the classifier and the accuracy came up to `98.17%` which is pretty good. Here is the final output:

```
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5292
16.65 Seconds to train SVC...
Test Accuracy of SVC =  0.9817
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented inside the function `slide_window()` same as the one mentioned inside lecture notes. This function returns the list of windows given the start and stop positions in `x` and `y`, the size of the window and the overlap. I have settled on a window size of `64x64` after trying other combinations like `96x96` like in the lecture notes. I found out that `64x64` gave a pretty good prediction whereas `96x96` was only good when the cars were a bit closer to the vehicle. For this project `64x64` seemed like a good option to settle on.

For the amount of overlap, I used the default `50%` overlap as used inside lecture notes. It was a good overlap to start with.

Regarding the scales, I first started with just a scale of `1.0` and found out that the scale was enough to identify cars over `90%` of the time inside the images. This was good enough to proceed forward so I decided not to implement any additional scales.

![alt text][image8]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on just one scale using `YCrCb` `ALL`-channel HOG features, which provided a nice result.  Here are some example images:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have started with defining the functions `add_heat()`, `apply_threshold()` and `draw_labeled_bboxes()` under the title "Heat threshold to an image to filter false positives" in the ipython notebook. These are the same functions as in the lecture notes that will help get the heatmap of an image.

One thing I have noticed is that the number of `hot_windows` I got as the output from `search_windows()` function is that the number of false positives were negligible in the complete video. The false positives are actually cars identified on the other side of the highway so they are not the ones pointing to the road or trees. You can see an example below:

![alt text][image9]

Also, looking at one of the test images (below), we see that the heatmap for the identified cars will filter out even true positives. Also keeping in mind that the false positives are really not "true" false positives and the occurence is not frequent, I decided to not apply a filter or heat threshold to the images.

![alt text][image7]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

So as mentioned earlier, I spent some time going through the [HOG research paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) and tried to use the same parameters as discussed. The only variation from these parameters to the ones I chose was the color space - `YCrCb` instead of `RGB`. The reason being that I got a lot of false positives with using an `RGB` color space - keeping all other parameters same. Because of the number of false positives, I tried to implement a heat threshold filter including the multiple scales solution, I wasn't able to completely filter out the false positives. In many cases, it was filtering out the cars before the false positives. `YCrCb` gave a much better solution.

I haven't implemented a smoothing algorithm that will take an average of all windows in a series of video frames and add a single window (along with threshold) because I think the current solution was good enough to showcase that the chosen parameters were enough to detect vehicles most of the times during the video. But adding a smoothing algorithm will definetely make it a cleaner vehicle detection algorithm, also eliminating any existing false positives.



