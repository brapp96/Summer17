from __future__ import print_function
import numpy as np
import tflearn

# Input dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels.astype(int)
x_test = mnist.test.images
y_test = mnist.test.labels.astype(int)

# Build neural network
input_layer = tflearn.input_data(shape=[None, x_train.shape[1]],name='input')
Hidden1 = tflearn.fully_connected(input_layer, 1000, activation='softmax',name='Hidden1')
Hidden2 = tflearn.fully_connected(Hidden1, 10, activation='softmax',name='Hidden2')
net = tflearn.regression(Hidden2,name='regression')

# Define model
model = tflearn.DNN(net,tensorboard_verbose=0, tensorboard_dir='/tmp/tflearn_logs/')
# Start training (apply gradient descent algorithm)
model.fit(x_train, y_train, n_epoch=5,validation_set=(x_test, y_test),show_metric=True, snapshot_epoch=True)

# Get layer weights and biases
print("Hidden1 layer weights:")
print(model.get_weights(Hidden1.W).shape)
print("Hidden1 layer biases:")
print(model.get_weights(Hidden1.b).shape)

print("Hidden2 layer weights:")
print(model.get_weights(Hidden2.W).shape)
print("Hidden2 layer biases:")
print(model.get_weights(Hidden2.b).shape)

# Predict classes
#pred = model.predict(x_test)
#print(pred.shape)
#print(np.reshape(pred[0],(10,1)))
#print(np.reshape(y_test[0],(10,1)))
