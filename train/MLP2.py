# MLP using theano
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import load_iris
import pandas as pd
from URule import rmsprop
from URule import momentums
from URule import sgd
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
x = T.matrix('x', dtype=theano.config.floatX)
y = T.lvector('y')


lr = T.scalar(name = "Learning Rate", dtype = theano.config.floatX)
act_f = T.nnet.sigmoid
act_ff =T.nnet.softmax
#Layers
enc_layer1 = Stacked(x, train_x.shape[1], 64, activation = act_ff)
enc_layer2 = Stacked(enc_layer1.output, 64, 32, activation = act_ff)
dec_layer1 = Stacked(enc_layer2.output, 32, 64, activation = act_ff)
out_layer = Stacked(dec_layer1.output, 64, 3, activation = act_ff)

loss = T.nnet.categorical_crossentropy(out_layer.output, y).mean()
prediction = T.argmax(out_layer.output, axis=1)

forward_prop = theano.function([x], out_layer.output, allow_input_downcast=True)
calculate_loss = theano.function([x, y], loss,allow_input_downcast=True)
predict = theano.function([x], prediction, allow_input_downcast=True)

params = enc_layer1.params + enc_layer2.params + dec_layer1.params + out_layer.params
grad = T.grad(loss, wrt = params)


gradient_step = theano.function(
    [x, y],
    updates=rmsprop(l_rate =0.1,
                    parameters =params,
                    grads = grad), allow_input_downcast=True)


def build_model(num_passes=50000):
    

    for i in range(0, num_passes):

        batch_indices = np.random.randint(train_x.shape[1],size=30)
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        gradient_step(batch_x, batch_y)

        if i % 1000 == 0:
            print("Loss after iteration {0}: {1}".format(i, calculate_loss(train_x, train_y)))
            print(accuracy(train_x))


build_model()