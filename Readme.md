# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./example_data/11_rigtoffway_atnextintersection_32x32x3.jpg "Traffic Sign 1"
[image5]: ./example_data/12_priority_road_32x32x3.jpg "Traffic Sign 2"
[image6]: ./example_data/13_yield.jpg "Traffic Sign 3"
[image7]: ./example_data/17_noentry_32x32x3.jpg "Traffic Sign 4"
[image8]: ./example_data/31_wildanimalscrossing_32x32x3.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? Ans: 34799
* The size of the validation set is ? Ans: 4410
* The size of test set is ? Ans: 12630
* The shape of a traffic sign image is ? Ans: (32, 32, 3)
* The number of unique classes/labels in the data set is ? Ans: 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are distributed.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I did a very simple preprocess in this case.
For one thing, I decided to nomarlized the RGB values of images from (0~256) to (-1~+1) for float accuracy.

For another thing, I shuffle the images distribution to avoid the possible ordered interference.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Initially, I set the needed arguments such as weights, bias, BATCH_SIZE, keep_prob and so on.
And, I defined some useful convolution function as the classes example.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| flattenlize	      	| outputs 5x5x16 								|
| Fully connected		| output  400      								|
| RELU & dropout			|											|
| Fully connected		| output  120      								|
| RELU & dropout			|											|
| Fully connected		| output  84      								|
| RELU & dropout			|											|
| Fully connected		| output  43      								|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, the parameters are setuped as:

EPOCHS = 10 
BATCH SIZE = 128
learning rate = 0.001

The training accuracy are varied as:

EPOCH 1 ...
Validation Accuracy = 0.843

EPOCH 2 ...
Validation Accuracy = 0.897
        .
        .
        .
        
EPOCH 18 ...
Validation Accuracy = 0.949

EPOCH 19 ...
Validation Accuracy = 0.939

EPOCH 20 ...
Validation Accuracy = 0.946

When the EPOCH comes to 20, the values aren't changed largely and become stable.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.946
* test set accuracy of 0.938

Initially, I set a too low dropout rate of 0.7 (I didn't notice it), so the accuracy was very poor. And, my intuition solution is that I increase the training epoch to 30 or higher, but the result was still not ideal. In the meanwhile, I changed the first convolution from 5*5 to 3*3, max pooling function, adding an additional layer and so on, but all didn't improve the situation too much. Finally, I found I forgot to change the core paremeter of dropout function from constant to a argument(keep_prob), and I setup it as 0.9 during traning process. Magically, the final result only need 15 epochs or more can reach the accuracy of higher than 0.93. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Image	         			|     Prediction	  									| 
|:-----------------------------:|:-----------------------------------------------------:| 
| Right of way at next intersection         				| Right of way at next intersection				|                             
| Priority road  					| Priority  road	 									|
| Yield				| Yield														|
| No entry		      			| No entry									|
| Wild animals crossing				    | Wild animals crossing				|


The model was able to correctly guess all traffic signs, which gives an Test Accuracy = 1.000. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is very sure that this is a Right of way at next intersection (probability of 1.0). The top five soft max probabilities were

| Probability         			|     Prediction	  									| 
|:-----------------------------:|:-----------------------------------------------------:| 
| 1.0000         				| Right of way at next intersection  					|                             
| 4.13169e-13  					| Beware of ice/snow 									|
| 8.51969e-14					| Double curve											|
| 2.26864e-14	      			| Pedestrians											|
| 1.27746e-14				    | End of no passing by vehicles over 3.5 metric tons 	|




