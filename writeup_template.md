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
[image6]: ./output_images/slide_windows.png "Slide Windows"
[image7]: ./output_images/video_clip_false_positive.png "False positive"
[image8]: ./output_images/find_cars.png "Area searched for slide windows"
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
5. Use a Linear SVM Classifier to train data and apply GridSearchCV() to obtain the best 'C' value
6. Print Accuracy

The final feature vector length with `9 orientations`, `8 pixels per cell` and `2 cells per block` had `5292` feature vectors. It took about a minute to train the classifier and the accuracy came up to `97.78%%` which is pretty good. Here is the final output:

```
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5292
61.27 Seconds to train SVC...
Test Accuracy of SVC =  0.9778
Best Parameters chosen =  {'C': 1}
```

I have used `sklearn.model_selection.GridSearchCV` with the `C : [1, 10]` parameter to find the best `C` value to tune the Linear SVM classifier. In the end, `C:1` was the bext parameter.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented inside the function `slide_window()` same as the one mentioned inside lecture notes. This function returns the list of windows given the start and stop positions in `x` and `y`, the size of the window and the overlap. I have settled on a window size of `64x64` after trying other combinations like `96x96` like in the lecture notes. I found out that `64x64` gave a pretty good prediction whereas `96x96` was only good when the cars were a bit closer to the vehicle. For this project `64x64` seemed like a good option to settle on.

For the amount of overlap, I used the default `50%` overlap as used inside lecture notes. It was a good overlap to start with.

Regarding the scales, I first started with just a scale of `1.0` and found out that the scale was enough to identify cars over `90%` of the time inside the images. This scale was giving very few windows of identified cars and as mentioned, 10% of the time, the cars were not getting identified. I therefore used a set of scales `[1, 1.5, 2, 2.5]. This set gave a good amount of search windows for the identified cars - along with some false positives. Finally, using a heat threshold of `4` I was able to cleanly remove the false positives.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the classifier (already mentioned above), I used the `sklearn.model_selection.GridSearchCV` to fine tune the `LinearSVM` classifier. Because we are using a linear classifier, we can only play around with the `C` parameter (as mentioned inside lecture notes). 

Also as mentioned earlier, I used just the HOG features to train my classifier as it was good enough to give a great accuracy on the test data. I used `ALL` HOG channels and a `YCrCb` color space. You can see the final heatmap of one of the test images here after applying multiple scales and calculating the list of windows on just HOG features:

![alt text][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

As you can see, multiple scales `[1, 1.5, 2, 2.5]` were applied for performing a window search for vehicles. This significantly increased the number of true positives but also added few false positives which were handled by the heat threshold.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I have started with defining the functions `add_heat()`, `apply_threshold()` and `draw_labeled_bboxes()` under the title "Heat threshold to an image to filter false positives" in the ipython notebook. These are the same functions as in the lecture notes that will help get the heatmap of an image.

Apart from applying multiple scales and heat threshold, I have also implemented a smoothing mechanism that will get a list of all search windows for a set of 6 consecutive video frames and makes an average heatmap that can later be used to further filter out any false negatives. This logic is present inside the cell below `Define pipeline for the video` title. I have used 2 `collections.deque` variables `hot_windows_queue` and `hot_windows_set_count`. The `hot_windows_queue` variable keeps a list of all windows that came positive in car search. The list includes the list of windows only from the last 6 frames. To keep a count of how many windows were added at each video frame, I used the `hot_windows_set_count` variable to track these numbers. Starting 7th frame, I remove the list of windows that were added inside `hot_windows_queue` first. This follows the FIFO model of a queue data structure.

Also, to properly tune the heatmap threshold parameter, I took a clip of the project_video.mp4 named [part_clip.mov](./part_clip.mov) and fine tuned my algorithm along with the parameters. The final result of this clip is in the [part_clip_output.mp4](./part_clip_output.mp4) file.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

So as mentioned earlier, I spent some time going through the [HOG research paper](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf) and tried to use the same parameters as discussed. The only variation from these parameters to the ones I chose was the color space - `YCrCb` instead of `RGB`. The reason being that I got a lot of false positives with using an `RGB` color space - keeping all other parameters same. Because of the number of false positives, I tried to implement a heat threshold filter including the multiple scales solution, I wasn't able to completely filter out the false positives. In many cases, it was filtering out the cars before the false positives. `YCrCb` gave a much better solution.

Further improvements could involve a much bigger training dataset that gives the classifier a better prediction on what is a car and non-car. 


