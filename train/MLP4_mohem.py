# MLP using theano
import numpy as np
import theano
import theano.tensor as T
from sklearn.datasets import load_iris
import pandas as pd
from URule import *
from Layers import *
import timeit
import matplotlib.pyplot as plt
import pylab

start = timeit.default_timer()
location = "c:/data/mnist_train.csv"
locationtest = "c:/data/mnist_test.csv"
data = pd.read_csv(location)
datatest = pd.read_csv(locationtest)
data = data.sample(59999, random_state=35)
datatest = datatest.sample(9999, random_state=35)
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
#print(nn_input_dim)
out_class = len(np.unique(test_y))
#nn_output_dim = len(iris.target_names)
#nn_output_dim = 10
#print("nn_output_dim: ",nn_output_dim)

x = T.matrix('x')
y = T.lvector('y')
act_ff = T.nnet.softmax
act_f = T.nnet.sigmoid
act_fff = theano.tensor.tanh
# MLP
enc_layer_1 = Stacked(x, nn_input_dim, 128, activation_fn=act_f)
#enc_layer_1 = Dropout(enc_layer_1.output, 64, 64, activation_fn=act_f)
dec_layer_1 = Stacked(enc_layer_1.output, 128, 64, activation_fn=act_f)
#dec_layer_1 = Dropout(dec_layer_1.output, 64, 64, activation_fn=act_f)
enc_layer_2 = Stacked(dec_layer_1.output, 64, out_class, activation_fn=act_ff)

#param = [enc_layer_1.paramsW, dec_layer_1.paramsW, dec_layer_1.paramsb, enc_layer_1.paramsb, enc_layer_2.paramsW, enc_layer_2.paramsb]
param = enc_layer_1.params + enc_layer_2.params + dec_layer_1.params

loss = T.nnet.categorical_crossentropy(enc_layer_2.output, y).mean()
prediction = T.argmax(enc_layer_2.output, axis=1)
forward_prop = theano.function([x], enc_layer_2.output, allow_input_downcast=True)


calculate_loss = theano.function([x, y], loss, allow_input_downcast=True)
predict = theano.function([x], prediction, allow_input_downcast=True)
accuracy = theano.function([x], T.sum(T.eq(prediction, train_y)), allow_input_downcast=True)
accuracytest = theano.function([x], T.sum(T.eq(prediction, test_y)), allow_input_downcast=True)

grad = T.grad(loss, wrt = param)

gradient_step = theano.function(
    [x, y],
    updates = amsgrad(learning_rate =0.01,
                    params =param,
                    loss_or_grads = grad), allow_input_downcast=True)


def build_model(num_passes=15000):
    np.random.seed(0)
    print("**Training Started**")
    loss_errors = np.ndarray(num_passes)
    train_acc = np.ndarray(num_passes)
    test_acc = np.ndarray(num_passes)
    
    for i in range(0, num_passes):

        batch_indices = np.random.randint(train_x.shape[0],size=128)
        batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
        gradient_step(batch_x, batch_y)
        
        if i % 1000 == 0:
            losses = calculate_loss(train_x, train_y)
            acctrain = accuracy(train_x)/train_x.shape[0]
            acctest = accuracytest(test_x)/test_x.shape[0]
            print("Train Loss after iteration {0}: {1}".format(i, losses))
            print("Train Accuracy: {:.6f} ".format(acctrain))
            print("Test Accuracy: {:.6f} ".format(acctest))
            
            stop = timeit.default_timer()
            print("Time elapsed:", stop - start)
            print("---------------------------------")
            
        loss_errors[i] = losses
        train_acc[i] = acctrain
        test_acc[i] = acctest
    #print(loss_errors) 
    
    #final_errs.append(losses)
    #print(final_errs)
    stop = timeit.default_timer()
    print("Time elapsed:", stop - start)
    fig = plt.figure()
    plt.ylim(0., 1.)
    plt.plot(range(num_passes), loss_errors, 'b-', label='Train loss')
    plt.plot(range(num_passes), train_acc, 'r-', label='Train Accuracy')
    plt.plot(range(num_passes), test_acc,'g-', label='Test Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Plot')
    pylab.legend(loc='center right')
    plt.show()
    
build_model()