# MLP using theano
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import load_iris
import pandas as pd
from URule import sgd, rmsprop, momentums
from Layers import Stacked
"""
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

nn_hdim = 128
nn_hdim2 = 128

epsilon = 0.1
batch_size = 128

x = T.matrix('x')
y = T.lvector('y')



W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
b1 = theano.shared(np.zeros(nn_hdim), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_hdim2), name='W2')
b2 = theano.shared(np.zeros(nn_hdim2), name='b2')
W3 = theano.shared(np.random.randn(nn_hdim2, nn_output_dim), name='W2')
b3 = theano.shared(np.zeros(nn_output_dim), name='b2')

z1 = x.dot(W1) + b1
a1 = T.nnet.softmax(z1)
z2 = a1.dot(W2) + b2
a2 = T.nnet.softmax(z2)
z3 = a2.dot(W3) + b3
a3 = T.nnet.softmax(z3)


loss = T.nnet.categorical_crossentropy(a3, y).mean()
prediction = T.argmax(a3, axis=1)
forward_prop = theano.function([x], a3, allow_input_downcast=True)
calculate_loss = theano.function([x, y], loss, allow_input_downcast=True)
predict = theano.function([x], prediction, allow_input_downcast=True)
accuracy = theano.function([x], T.sum(T.eq(prediction, train_y)), allow_input_downcast=True)
accuracytest = theano.function([x], T.sum(T.eq(prediction, test_y)), allow_input_downcast=True)

param = [W1, b1, W2, b2, W3, b3 ]

dW3 = T.grad(loss, W3)
db3 = T.grad(loss, b3)
dW2 = T.grad(loss, W2)
db2 = T.grad(loss, b2)
dW1 = T.grad(loss, W1)
db1 = T.grad(loss, b1)

grad = T.grad(loss, wrt = param)

gradient_step = theano.function(
    [x, y],
    updates=((W3, W3 - epsilon * dW3),
             (W2, W2 - epsilon * dW2),
             (W1, W1 - epsilon * dW1),
             (b3, b3 - epsilon * db3),
             (b2, b2 - epsilon * db2),
             (b1, b1 - epsilon * db1)), allow_input_downcast=True)


def build_model(num_passes=500000):
    np.random.seed(0)
    
    W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
    b1.set_value(np.zeros(nn_hdim))
    W2.set_value(np.random.randn(nn_hdim, nn_hdim2) / np.sqrt(nn_hdim))
    b2.set_value(np.zeros(nn_hdim2))
    W3.set_value(np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2))
    b3.set_value(np.zeros(nn_output_dim))
    

    for i in range(0, num_passes):

        batch_indices = np.random.randint(train_x.shape[0],size=128)
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        gradient_step(batch_x, batch_y)
        """
        batch_indicest = np.random.randint(test_x.shape[0],size=128)
        batch_xt, batch_yt = test_x[batch_indicest], test_y[batch_indices]
        forward_prop(batch_xt)
        """
        if i % 1000 == 0:
            print("Loss after iteration {0}: {1}".format(i, calculate_loss(train_x, train_y)))
            print("Train Accuracy: ", accuracy(train_x)/train_x.shape[0])
            print("Test Accuracy: ", accuracytest(test_x)/test_x.shape[0])
            print("---------------------------------")

build_model()