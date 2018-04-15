# MLP using theano
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import load_iris
import pandas as pd
from URule import sgd, rmsprop, momentums
from Layers import Stacked

iris = load_iris()
train_x = iris.data
print("train x shape:", train_x.shape)
train_y = iris.target
print("train y shape: ", train_y.shape)
nn_input_dim = train_x.shape[1]
print("nn_input_dim:", nn_input_dim)
"""
location = "c:/data/mnist_train.csv"
locationtest = "c:/data/mnist_test.csv"
data = pd.read_csv(location)
datatest = pd.read_csv(locationtest)
train_x = data.iloc[:, 1:].values
train_y = data.iloc[:, 0].values
test_x = datatest.iloc[:, 1:].values
test_y = datatest.iloc[:, 0].values
print("train x: ",train_x.shape)
print("train y: ",train_y.shape)
print("test x: ",test_x.shape)
print("test y: ",test_y.shape)
train_x = np.multiply(train_x, 1.0/255.0)
test_x = np.multiply(test_x, 1.0/255.0)
#train_y = T.cast(train_y, 'int32')
#test_y = T.cast(test_y, 'int32')
nn_input_dim = train_x.shape[1]
print(nn_input_dim)

#nn_output_dim = len(iris.target_names)
nn_output_dim = 10
print("nn_output_dim: ",nn_output_dim)
"""
nn_hdim = 128
epsilon = 0.1
batch_size = 128

x = T.matrix('x')
y = T.lvector('y')


act_ff = T.nnet.softmax

# Autoencoder
enc_layer_1 = Stacked(x, nn_input_dim, 128, activation=act_ff)
dec_layer_1 = Stacked(enc_layer_1.output, 128, 64, activation=act_ff)
enc_layer_2 = Stacked(dec_layer_1.output, 64, 4, activation=act_ff)
loss = T.nnet.categorical_crossentropy(enc_layer_2.output, y).mean()
prediction = T.argmax(enc_layer_2.output, axis=1)
forward_prop = theano.function([x], enc_layer_2.output, allow_input_downcast=True)
#param = [enc_layer_1.paramsW, enc_layer_1.paramsb,dec_layer_1.paramsW, dec_layer_1.paramsb, enc_layer_2.paramsW, enc_layer_2.paramsb]
param = enc_layer_1.params + dec_layer_1.params + enc_layer_2.params
calculate_loss = theano.function([x, y], loss, allow_input_downcast=True)
predict = theano.function([x], prediction, allow_input_downcast=True)
accuracy = theano.function([x], T.sum(T.eq(prediction, train_y)), allow_input_downcast=True)
#accuracytest = theano.function([x], T.sum(T.eq(prediction, test_y)), allow_input_downcast=True)


grad = T.grad(loss, wrt = param)

gradient_step = theano.function(
    [x, y],
    updates = rmsprop(l_rate =0.1,
                    parameters =param,
                    grads = grad), allow_input_downcast=True)


def build_model(num_passes=500000):
    np.random.seed(0)
    
    for i in range(0, num_passes):

        batch_indices = np.random.randint(train_x.shape[0],size=128)
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        gradient_step(batch_x, batch_y)
        
        if i % 1000 == 0:
            print("Loss after iteration {0}: {1}".format(i, calculate_loss(train_x, train_y)))
            print("Train Accuracy: ", accuracy(train_x)/train_x.shape[0])
            #print("Test Accuracy: ", accuracytest(test_x)/test_x.shape[0])
            print("---------------------------------")

build_model()