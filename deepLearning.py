#step 1:- Download the database of image from MNIST website.
import urllib.request # this for downalod things from the web
import gzip  # this unzips the file and loads the data.
import numpy as np
import os

# Using python's mnist library to load the data
# from mnist import MNIST
#
# emnist_data = MNIST(path = '/Users/shahzebkhalid/Documents/CIS490ML/Project4', return_type='numpy')
# emnist_data.select_emnist('letters')
# X, y = emnist_data.load_training()

# This Code section is for Recoginzing Digits.
def load_database ():
    from urllib.request import urlopen

    def download(filename,source = "/Users/shahzebkhalid/Documents/CIS490ML/Project4" ):
        try:
            urlopen(source)
        except:
            pass

    def load_mnist_images(filename):
        # this open the file and reads the binary file.
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 16)
            print()
        ''' This data will have 2 issues, data can not be one long string
        1) it has to be an array which representing the image so each element
        in teh arry should represent one image.
        2) numbers are in the form of bytes we need to convert it in the form of floats.'''

        ''' each image is 28x28 in Mnist data base. so need to reshape the one long
        array in to the array of images by using numpy. The array would have
        4 dimensions.
        1st dimension  represents the number of images are there by making it -1 this
        just means that it doesnt know how many images are there so just ask the
        other dimentions for it lenght and the numbers.

        2nd dimension represents how many channle there are which is 1
        since we want monochrome and for colors it would be 3/4 channles.
        3rd & 4th dimension represents the pixels'''

        data = data.reshape(-1,1,28,28)
        # print(data)

        # this is converting the byte value to a floart32 in the range [0,1]
        return data/np.float32(256)
    def load_mnist_labels(filename):
        # read the labes which are in binary file.
        with gzip.open(filename,'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset = 8)
        return data

    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test  = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    return X_train, y_train, X_test, y_test
# calling the funtions form the funtion load_database and giving it to vairables
X_train, y_train, X_test, y_test = load_database()

import matplotlib.pyplot as plt
# plt.show(plt.imshow(X_train[1][0]))


#Step2 Setting up a neural network with the required numbers of layers and nodes
# also telling the network how to train it shelf.

''' Use 2 packages called Theano and lasagne
Theano: it allows you to define and perform mathimatical computaion
    on higher dimention arrays which are also called Tensors

Lasagne is a library that uses Theano heavily and it allows you to
    build neural network. It sets up layers by the use of functions. Alos
    defines error functions train neural network.'''
import lasagne
import theano
import theano.tensor as Tes

def build_NN(input_var = None):
    # Creating a neural network with 2 hidden layers of 800 nodes each
    # The output layer would have 10 nodes - the nodes are numbered 0-9 and teh output
    # at each node will be value b/w 0-1. The node witht the higher value will be the
    # predicted output.

    #  First we have input layer the expected input shape is 1x28x28 (for 1 image)
    #  we will link this input layer to the input_var(which will be the array of image that we'll
    #  pass in later on).

    l_in = lasagne.layers.InputLayer(shape = (None,1,28,28), input_var = input_var)

    # we'll ass a 20% dropout - this means that randomly 20% of edges b/w the inputs & the next
    # layer will be dropped - this is done to avoid overfitting
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p = 0.2)

    # Add a layer with 800 nodes. Initially this will be dense/fully - connected
    # i.e edges possiable will be drawn
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units = 800,
                                        nonlinearity = lasagne.nonlinearities.rectify,
                                        W = lasagne.init.GlorotUniform())
    # This layer has been initialized with weights. There are some schemes to initialize
    # the weight so that training will be done faster, Glorot's schemes is one of them

    # we will ass a dropout of 50% to hidden layer1
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p = 0.5)

    # Adding another layer, it will works the same way
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units = 800,
                                        nonlinearity = lasagne.nonlinearities.rectify,
                                        W = lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p = 0.5)

    # now add teh final output layer
    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units = 10,
                                        nonlinearity = lasagne.nonlinearities.softmax )

    # the output layer has 10 units. softmax specifies that each of those output is between 0-1 and
    # the max of those will be the final prediction.
    return l_out
# now th enetwork is setup, so now have to tell the network how to train it shelf
# i.e. how should it find values of all the weights it need to find

# By initializing some empty arrays wich will act as placeholders for thetrain/test data
# that will be given to the network
input_var = Tes.tensor4('inputs') # An empty 4 dimention array
target_var = Tes.ivector('targets') # an empty 1 dimentional array to represent
# Both are empty and tey are representing the shapes of what the input and the labels are going to be.

network = build_NN(input_var)#calls the functions and inializes the netural network.

# a. compute an error funtion
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction,target_var)

# categorical cross entropy is one of the standard error fucntions with calssification problems
loss = loss.mean()

# b. tell the network how to update all its weights based on the value of error function.
params = lasagne.layers.get_all_params(network,trainable = True) #current valyes of all weights
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate = 0.01, momentum = 0.9)

# Nesterov momentum is one of the options that lasagne offers for updating the weights in the training
#  step this based on stochastic Gradients Decent

#  Theano to compile a funtion tha tis going to represent a single traing step i.e compute error,
# find the current weight update the weight
train_fn = theano.function([input_var, target_var], loss, updates = updates)

# Step 3: Feed the traing data to neural network.

# num_training_steps = 4 #  identifing to train for few 100-200 steps would be optimal
#
# for step in range(num_training_steps):
#     train_err = train_fn(X_train, y_train)
#     print("Current step is " + str(step))


# Step 4: Check how the output is for 1 image
test_prediction = lasagne.layers.get_output(network)
val_fn = theano.function([input_var], test_prediction)

# print("val_fn","\n", val_fn([X_test[0]]))
print("\n")
print("Final output: ",y_test[102])

plt.show(plt.imshow(X_test[102][0]))


# Step 5: feed a data set of 10000 images to the trained neural network to check it accuracy
# set up a function that will take in image and their labels, feed images to
# our network and compute its accuracy

test_prediction = lasagne.layers.get_output(network)
test_acc = Tes.mean(Tes.eq(Tes.argmax(test_prediction, axis = 1), target_var),dtype = theano.config.floatX)

acc_fn = theano.function([input_var,target_var], test_acc)
print(acc_fn(X_test,y_test))


'''
    Refrances:-
        •   Nielsen, and Michael A. “Neural Networks and Deep Learning.”
            Neural Networks and Deep Learning, Determination Press, 1 Jan.
            1970, neuralnetworksanddeeplearning.com/chap1.html.

        •   lecun, yann. “THE MNIST DATABASE.” MNIST Handwritten Digit Database,
            Yann LeCun, Corinna Cortes and Chris Burges,
            yann.lecun.com/exdb/mnist/index.html.
            '''
