import matplotlib.cm as cm	
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from sklearn.model_selection import train_test_split
from URule import rmsprop
from URule import momentum
from Layers import Stacked
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from logistic_sgd import LogisticRegression

location = "c:/data/mnist_train.csv"
locationtest = "c:/data/mnist_test.csv"
data = pd.read_csv(location)
datatest = pd.read_csv(locationtest)
train_x = data.iloc[:, 1:].values
train_y = data.iloc[:, 0].values
test_x = datatest.iloc[:, 1:].values
test_y = datatest.iloc[:, 0].values
train_x = train_x.astype(theano.config.floatX)
test_x = test_x.astype(theano.config.floatX)
train_y = train_y.astype(theano.config.floatX)
test_y = test_y.astype(theano.config.floatX)
train_y = T.cast(train_y, 'int32')
test_y = T.cast(test_y, 'int32')

#Concepts
X = T.matrix(name = "X", dtype = theano.config.floatX)
y = T.lvector('y')
print(y.dtype.startswith('int'))
lr = T.scalar(name = "Learning Rate", dtype = theano.config.floatX)
act_f = T.nnet.sigmoid
act_ff =T.nnet.softmax
#Layers
enc_layer1 = Stacked(X, 784, 512, activation = act_ff)
enc_layer2 = Stacked(enc_layer1.output, 512, 256, activation = act_ff)
dec_layer1 = Stacked(enc_layer2.output, 256, 512, activation = act_ff)
out_layer = Stacked(dec_layer1.output, 512, 784, activation = act_ff)


#train and test
pred_y = T.argmax(out_layer.output, axis=1)
params = enc_layer1.params + enc_layer2.params + dec_layer1.params + out_layer.params
cost = T.nnet.categorical_crossentropy(out_layer.output, y).mean()
grad = T.grad(cost, wrt = params)

#functions

forward_prop = theano.function([X], out_layer.output, allow_input_downcast=True)
calculate_loss = theano.function([X, y], cost, allow_input_downcast=True)
predict = theano.function([X], pred_y, allow_input_downcast=True)
accuracy = theano.function([X], T.sum(T.eq(pred_y, train_y)), allow_input_downcast=True)
gradient_step = theano.function(inputs = [X, y, lr], outputs = cost, updates = rmsprop(l_rate =lr, parameters =params, grads = grad), allow_input_downcast=True)

"""
num_passes=50000
for i in range(0, num_passes):

    batch_indices = np.random.randint(train_x.shape[1],size=30)
    batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
    batch_y = T.cast(batch_y, 'int32')
    gradient_step(batch_x, batch_y)

    if i % 1000 == 0:
        print("Loss after iteration {0}: {1}".format(i, calculate_loss(train_x, train_y)))
        print(accuracy(train_x))
"""
def train(trainset_x, trainset_y, testx_input, testy_input, learning_rate = 0.1, batch_size=512, epochs = 200):
    
    train_val = len(trainset_x)
    test_val = len(testx_input)
    print('---Training Model---')
    
    for epoch in range(epochs):
        print('Currently on epoch {}'.format(epoch+1))
    
   
        costij = gradient_step(trainset_x, trainset_y, learning_rate)
        train_cost += costij
        print('The train loss err is= {:.6f}'.format(train_cost/batch_size))
        #test_accuracy  = np.mean(np.argmax(testy_input) == predict(testx_input))
        PRE =predict(testx_input)
        print("test accuracy: ", PRE)
        print('--------------------------')
   
def main():
       
    train(train_x, train_y, test_x, test_y)
    
if __name__ == "__main__":
    main()
