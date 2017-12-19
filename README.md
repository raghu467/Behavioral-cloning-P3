# Self-Driving Car Engineer Nanodegree
## Deep Learning
In this project  we train the Deep Neural Network  to clone human driving behavior.  We have used the car simulator provided by UDACITY base using Unity. The car in this simulator can be driven using the keyboard arrow keys (up, down, left, and right) while the user drived the car in the simulator it record the images of what is visible from the car wind-sheild and corresponging steering angle. we use this recorded data during the training phase to train the neural network. During the taining phase we save the model and weights which are later used to validate the model. To validate the model we run the simulator in the Autonomus mode now the drive.py script in the repository is used to run the controller(simple PI) and validate the model. The controller running in the drive.py script communicates the steering angle and speed to the simulator using web-sockets.

## Dataset Statistics
The training dataset contains 8600 images also the  training track contains a lot of straight roads and shallow turns. since we have lot of stright roads in the simulator the majority steering angles are zeros. 
To normalize this training data we use the following pipeline to pre-process the images and steering angles before training the Neural network.
The following is the distribution of the steering angle in the training data set collected.<br>
![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/steering_distribution.png)<br>

It is evident from the distribution plot above that  most of the training data captured has steerng angle 0. to improve this data distribution we apply random shearing to the training images and the corresponding steering angles. Also random shearing helps to reduce the impact of the image data out-side the road on the training.

## Data Processing Pipeline

Before the images are fed into the network for training they are pre-processed. The following are the steps followed for the pre-processing of the data. These are the output images corresponding to each stage.
1.put Iage:<br>

![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/wr_image.jpg)

2.Seared Image:<br>
3.Cropped Image:<br>
During this stage the top_crop_percent=0.35, bottom_crop_percent=0.1 to eliminate<br>
The following is the output of the crop and shear stage.<br>
![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/wr_shear_image.jpg)
4.Resize Image:<br>
During this stage we resize the image  to 64x64 in order to reduce the training time and also the data size<br>
The follwing is the output of the rezise stage<br>
![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/wr_cropped_image.jpg)<br>
5.Flipped Image:<br>
During this stage the images are randomly flipped (prob=0.5). The training tack has more left turns compared to the right turns . Hence<br> to normalize the data and we randomly flip the image. The following image is the output of the flip stage<br>
![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/wr_flip_image.jpg)<br>
6.Random Brightened Image:<br>
![alt tag](https://github.com/raghu467/Behavioral-cloning-P3/blob/master/Readme_images/wr_bright_image.jpg)<br>


## Network Architecture
The CNN architecture was inspired from NVIDIA's End to End Learning for Self-Driving Cars paper. One main difference is that in our model we use max polling 2x2 after the each convolution layer.

[Reference:]( https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)<br>
![alt tag](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)



## Training
We used the the following two generators which provide pre-processed images to the CNN netowrk on the fly.<br>

#generators for training and validation<br>
train_generators = data_prep.generate_next_batch()<br>
validation_generators = data_prep.generate_next_batch()<br>
Due to the memory constraints we have used fit_generator API in the keras Library to train the mode.
We used Adam optimizer with 0.0001 learning rate. After trying out other EPOCH numbers we used 8 which works well with the following loss at the end of the trianing EPOCH's(8).<br>

20032/20032 [==============================] - 132s - loss: 0.0092 - val_loss: 0.0090<br>

# Result
After training the network with the vast amount of pre-processed data the network was able to steer the car with decent accuracy and maintaining the car on the road. 
Note: During the Intial stages when only one lap worth of data was ued to train the network the car was going all over the track later when a bigger data-set was used to train the performance Improved. 
Mainly data agumentation and new data-set worked very well for the training purpose. This new data helped improve the performance of the model really well.<br>

Talking about extentions we can improve the performance even better by experimenting with new data adumentation techniques also try out advanced netowork architectures.




