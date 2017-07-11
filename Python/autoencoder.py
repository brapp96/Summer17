import numpy as np
import matplotlib.pyplot as plt
import tflearn
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

# Data loading and preprocessing
import tflearn.datasets.mnist as MNIST_data
X, Y, testX, testY = MNIST_data.load_data(one_hot=False)

# Number of datapoints to use
n = 10000

# Reduce dataset size
if n != X.shape[0]:
	X = X[0:n,:]
	Y = Y[0:n]
	testX = testX[0:n,:]
	testY = testY[0:n]

# Building the encoder
encoder = tflearn.input_data(shape=[None, 784])
encoder1 = tflearn.fully_connected(encoder, 100)
encoder2 = tflearn.fully_connected(encoder1, 2,name='features')

# Building the decoder
decoder = tflearn.fully_connected(encoder2, 100)
decoder = tflearn.fully_connected(decoder, 784, activation='sigmoid')

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001, loss='mean_square', metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=10, show_metric=True, batch_size=256)

# Redefine model to output feature layer
model = tflearn.DNN(encoder2)

# Get low dimensional feature representation for test data
output = model.predict(testX)

# Use TSNE to get 2D embedding
#lowDim = TSNE(n_components=2, random_state=0)
#np.set_printoptions(suppress=True)
#output = lowDim.fit_transform(X) 

# Cluster features using k-means 
#kmeans = KMeans(n_clusters=10).fit(output)
kmeans = KMeans(n_clusters=10).fit(output)
predY = kmeans.labels_

# Plot 2D representation
#for i in range(0,10):
#	selected = 0
#	selected = output[predY == i]
#	plt.scatter(selected[:,0], selected[:,1], c=str(1/(i+1)))
#	#plt.show()
#	#plt.hold(True)
#	#plt.pause(1)
#plt.show()

plt.figure(1)
plt.scatter(output[:,0], output[:,1], c=predY)
plt.show()


## Get features for X[0]
#f1 = model.predict([X[0]])
#
## Compare original images with their reconstructions
#fig, (sub1,sub2) = plt.subplots(2, 1)
#original = X[0]
#sub1.matshow(np.reshape(original,(28,28)))
##sub2.matshow(np.reshape(y1,(28,28)))
#sub2.matshow(np.reshape(f1,(16,16)))
#
#fig.show()
#plt.draw()
#plt.waitforbuttonpress()


