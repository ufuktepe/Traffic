# S80 Project 5

I have performed over 60 trials, each time modifying a single parameter. Below is a summary of my observations.

## Number of 2D convolutional + MaxPooling2D layers
As a starting point I used a single convolutional layer followed by a max pooling layer. However, while testing different numbers of convolutional + max pooling layers I have observed that using three convolutional + max pooling layers performs the best. Therefore, I used three convolutional + max pooling layers where the number of filters in the first, second, and third convolutional layers are 64, 128, and 256, respectively.

## Kernel Size
I used odd-sized kernels because when the kernel size is odd the previous layer pixels will be symmetric around the output pixel. Otherwise, we would have to account for distortions between layers.

Also, I used a kernel size of 3x3 in all convolutional layers so that finer details are captured in all layers. As an alternative, I have also tried using a kernel size of 7x7, 5x5, and 3x3 in the first, second, and third convolutional layers, respectively, in order to capture larger features in the early layers and finer details in the following layers. However, for this dataset, using a kernel size of 3x3 in all layers performed the best.

## Normalization
I normalized the pixel values by dividing them by 255. This way, the model does not require as much computational resources and it is able to converge faster.

## Activation Function
For the convolutional layers and the hidden dense layer I have tried using the rectified linear unit (ReLU), exponential linear unit (ELU), scaled exponential linear unit (SELU), and hyperbolic tangent (tanh) activation functions. In the majority of the cases, RELU performed better than the other activation functions.

## Hidden Dense Layer
For the hidden dense layer I have tried using 300, 400, 500, and 600 units. All these configurations resulted in an accuracy over 98%. However, using 400 units yielded the greatest accuracy.

## Dropout
Dropout layers with rates 0.3, 0.4, 0.5, 0.6, and 0.7 are tested. It is observed that using a dropout rate less than or equal to 0.5 results in overfitting and a dropout rate of 0.6 yields the best accuracy.


## Results
Epoch 1/10
500/500 [==============================] - 13s 26ms/step - loss: 2.1074 - accuracy: 0.4068
Epoch 2/10
500/500 [==============================] - 13s 25ms/step - loss: 0.4292 - accuracy: 0.8640
Epoch 3/10
500/500 [==============================] - 14s 28ms/step - loss: 0.1826 - accuracy: 0.9453
Epoch 4/10
500/500 [==============================] - 13s 27ms/step - loss: 0.1154 - accuracy: 0.9653
Epoch 5/10
500/500 [==============================] - 13s 27ms/step - loss: 0.0771 - accuracy: 0.9767
Epoch 6/10
500/500 [==============================] - 14s 28ms/step - loss: 0.0633 - accuracy: 0.9805
Epoch 7/10
500/500 [==============================] - 15s 30ms/step - loss: 0.0485 - accuracy: 0.9855
Epoch 8/10
500/500 [==============================] - 14s 27ms/step - loss: 0.0382 - accuracy: 0.9889
Epoch 9/10
500/500 [==============================] - 15s 29ms/step - loss: 0.0439 - accuracy: 0.9870
Epoch 10/10
500/500 [==============================] - 15s 29ms/step - loss: 0.0356 - accuracy: 0.9887
333/333 - 3s - loss: 0.0277 - accuracy: 0.9929 - 3s/epoch - 8ms/step