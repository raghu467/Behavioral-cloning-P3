# Self-Driving Car Engineer Nanodegree
## Deep Learning
In this project  we train the Deep Neural Network  to clone human driving behavior.  We have used the car simulator provided by UDACITY base using Unity. The car in this simulator can be driven using the keyboard arrow keys (up, down, left, and right) while the user drived the car in the simulator it record the images of what is visible from the car wind-sheild and corresponging steering angle. we use this recorded data during the training phase to train the neural network. During the taining phase we save the model and weights which are later used to validate the model. To validate the model we run the simulator in the Autonomus mode now the drive.py script in the repository is used to run the controller(simple PI) and validate the model. The controller running in the drive.py script communicates the steering angle and speed to the simulator using web-sockets.

## Dataset Statistics
The training dataset contains 8600 images also the  training track contains a lot of straight roads and shallow turns. since we have lot of stright roads in the simulator the majority steering angles are zeros. 
To normalize this training data we use the following pipeline to pre-process the images and steering angles before training the Neural network.
The following is the distribution of the steering angle in the training data set collected.
![alt tag]()

## Data Processing Pipeline

Before the images are fed into the network for training they are pre-processed. The following are the steps followed for the pre-processing of the data.
