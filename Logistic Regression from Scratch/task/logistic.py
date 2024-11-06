##   The development of the below multi-layer neural network is work in progress
##   The below building blocks are tested:
## - Xavier initialization of weights matrix
## - Training loop that incluedes the below steps
## - Forward step algorithm to copmute Prediction in each layer of the neural network
## - Backpropagation step algorithm to update the weights matrix given the predictions and truth values
## - Gradient descent algorithm to minimize the prediction error
## - Scaling of input variables and target variables 
## - Activation function (sigmoid) and compution of gradient 
## - MSE error computation and gradient of error func
## - Numpy (and pandas) to perform matrix multiplications
## - Thus, no use of library routines, rather all matrix computations are performed.


from scipy.special import softmax
import sys
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import time
import requests
from matplotlib import pyplot as plt



def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train


def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    #print('X_train shape: ', X_train[:, 0])
    # write your code here

def scale(X_train, X_test):
    # if X_train, X_test were DataFrames:
    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        return X_train.divide((X_train.max()).max()), X_test.divide((X_test.max()).max())
    elif isinstance(X_train, np.ndarray) and isinstance(X_test, np.ndarray):
        return np.divide(X_train, X_train.max()), np.divide(X_test, X_test.max())
	# Rescaled X_train, X_test

def xavier(n_in, n_out):
    # resulting matrix of weights
    return np.random.uniform(((-1) * np.sqrt(6)) / np.sqrt(n_in + n_out), np.sqrt(6) / np.sqrt(n_in + n_out), (n_in, n_out))
    #np.random.seed(0)
    #fan_in = n_in
    #fan_out = n_out
    #limit = np.sqrt(6 / (fan_in + fan_out))
    #return np.random.uniform(-limit, limit, (n_out, n_in))

def sigmoid(t: float) -> float:
    return 1 / (1 + np.exp(-t))

###σ′(x)=σ(x)⋅(1−σ(x))
def sigmoidGrad(t: float) -> float:
    return sigmoid(t) * (1 - sigmoid(t))


# MSE and MSE Gradient

def mseGradient(y, y_train):
    ##Assume np.array
    mseGrad = np.zeros(len(y))
    sqrErr = 0
    len_y
    for i in range(len_y):
        mseGrad[i] = 2 * (y[i] - y_train[i]) / len_y
        err_per_class[i] = y[i] - y_train[i] 
        mse  += np.square(y[i] - y_train[i])/len_y
    return sqrErr / len(y), mseGrad
    #return mseGrad


scaled_X_train, scaled_X_test = scale(X_train, X_test)
#print([scaled_X_train[2,778], scaled_X_test[0,774]], xavier(2, 3).flatten().tolist(), [sigmoid(x) for x in [-1, 0, 1, 2]])

class TwoLayerNeural:


    def __init__(self, n_features, n_neurons, n_classes):
        # Initiate weights and biases using Xavier
        # n_in = n_features, n_out = n_classes
        self.n_features = n_features
        self.n_classes = n_classes
        self.h1_layer = n1_neurons
        ###  Had to transpose the matrix for the solution to work -- Not switch order of args for matrix dims
        self.w = xavier(n_neurons, n_classes).T ## M X N array - ith row corresponds to weights of ith output neuron
        self.w1 = xavier(n_features, n_neurons).T ## M X N array - ith row corresponds to weights of ith h1 neuron
        #slef.w = np.load("path/to/small_array", mmap_mode="r")
        self.bias = xavier(n_classes, 1)  ## column vector (if to lay out the matrix operations by hand
        self.h1_bias = xavier(n_neurons, 1)  ## column vector (if to lay out the matrix operations by hand
        #slef.bias = np.load("path/to/small_array", mmap_mode="r")
        self.output_asArr  = np.zeros((n_classes, 1)) ##
        self.output_asList  = []
        self.output = self.output_asArr

        #print('bias.shape = {0}'.format(self.bias.shape))

    def forward(self, X):
        # Perform a forward step
        # https://hyperskill.org/learn/step/47283 Multilayer perceptron
        n_samples = len(X)
        output_asList = [] ## empty list
        output_asArr = np.zeros((self.n_classes, n_samples))
        for n in range(n_samples):
            a1 = X[n]  # a1 is the inputs array -- a col vector
            h1 = np.zeros((self.h1_layer))  ## for naming consistency h1 is output of h1 (hidden layer)
            # compute: a: a col vector -
            a = np.zeros((self.n_classes))
            #print('X: {0}'.format(X))
            #sys.exit()
            #print('lenW = {0}, lenA1 = {1}'.format(len(self.w[0]), len(a1)))
            #print('shapeW[0] = {0}, shapeA1 = {1}'.format((self.w[0].shape), a1.shape))
            for i in range(len(h1)):   ## Compute h1 (output of hidden layer)
                h1[i] = sigmoid(np.dot(self.w1[i], a1) + self.h1_bias[i, 0])
            for i in range(len(a)):   ## Compute a (final output)
                a[i] = sigmoid(np.dot(self.w[i], h1) + self.bias[i, 0])
            s = softmax(a)
            a = np.zeros(self.n_classes)
            a[np.argmax(s)] = 1
        #print('a: {0}'.format(a))
        #print('argmax s: {0}'.format(np.argmax(s)))
        #sys.exit()
            output_asList = output_asList + a.tolist()
            output_asArr[:, n] = a
        #print('self.out: {0}', output_asArr.T)
        #sys.exit()

        self.output_asList = output_asList
        self.output_asArr = output_asArr
        self.output = self.output_asArr
        #return(a[:, np.newaxis]) # return a col vector to concat to self.output
        return output_asList, output_asArr
    
    def mseGradient2(y, y_train, err_per_sample_cls):
        ##Assume np.array
        len_y = len(y)
        mseGrad = np.zeros(len_y)
        err_per_sample_cls = np.zeros(len_y)
        for i in range(len_y):
            err_per_sample_cls[i] += y[i] - y_train[i] 
            #mse  += np.square(y[i] - y_train[i])/len_y

        #mean_mseGrad = 2 * err_per_sample_class / len_y
        #mean_mse = np.mean(err_per_sample_class)

    def backprop(self, X, y, alpha):

    # Calculating gradients for each of
    # weights and biases.
        #mse = np.zeros(self.output.shape[1]) ## for number samples
        #mseGrad = np.zeros((self.output.shape[0],self.output.shape[1]))
        len_y = len(y)
        err_per_sample_cls = np.zeros(self.output.shape[0])
        sigGrad = np.zeros((self.output.shape[0],self.output.shape[1]))
        for k in range(self.output.shape[1]):
            #mse[k], mseGrad[: ,k] = mseGradient(self.output[:, k], y[k])
            err_per_sample_cls += self.output[:, k] - y[k]
            #sigGrad = np.zeros(self.n_classes)
            for i in range(self.n_classes):
                sigGrad[i, k] =  sigmoidGrad(self.output[i, k])

    #Calc means 
        err_per_sample_cls = err_per_sample_cls /  len_y
        mse = np.mean(np.square(err_per_sample_cls))
        mseGrad =   2 * err_per_sample_cls / len_y
        #mean_mseGrad = np.mean(mseGrad,axis=1)
        mean_sigGrad =  np.mean(sigGrad,axis=1)
        mean_X = np.mean(X, axis=0)
        #mean_X = X[len(X)-1]
        #print('err_per:{0}, mseGrad.shape:{1}'.format(err_per_sample_cls, mseGrad.shape))
        #print('Mean_sigGrad.shape:{0},Mean_X.shape:{1}'.format(mean_sigGrad.shape, mean_X.shape))
        #sys.exit()
        
 
    #Updating weights and biases.
        #print("mseGrad: ",mseGrad.shape, "sigGrad: ",mean_sigGrad.shape)
        # ∂L/∂w
        dL_dw = np.zeros((n_classes, n_features))
        dL_dbias = np.zeros(n_classes)
        for i in range(self.n_classes):  # ∂L/∂w
            dL_dbias[i] = mseGrad[i] * mean_sigGrad[i]
            dL_dw[i, :] = (mseGrad[i] * mean_sigGrad[i]) * mean_X
        #print(dL_dw.shape)

        for i in range(self.n_classes):
            for j in range(self.n_features):
                self.w[i, j] = self.w[i, j] - alpha * dL_dw[i, j]
        for i in range(self.n_classes):
            self.bias[i] = self.bias[i] - alpha * dL_dbias[i]
        return(mse, mseGrad, mean_sigGrad)


    def accuracy(self, y, output):
        len_y = len(y)
        corrects = 0
        for i in range(len_y):
            corrects += (y[i] == output[i]).all()
            #print('output: {0}, y: {1}'.format(output[i], y[i]))
        #sys.exit()
        return corrects/len_y
            
       #return np.mean(np.argmax(y, axis=-1) == np.argmax(output.T, axis=-1))
       #return np.mean(np.argmax(y) == np.argmax(output))
       #self.forward(X)
       #return np.round(np.mean(y == output), 2) 
       #return np.mean(y == output)
       #return accuracy_score(y, output.round(),normalize=False)
       #print('len y: ', y)
       #print('len output: ', output)

    def train(self, X_train, y_train, X_test, y_test, forward_run=False, epochs=20, 
              batch_size=100, optimizer='momentum', l_rate=0.5, beta=.9):
        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size

        start_time = time.time()
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        template = "Epoch {}: {:.2f}s, train acc={}, train loss={:.2f}"

        # Train
        X =  scaled_X_train
        y =  y_train
        
        if X_test is None:
            # Shuffle training data
            num_batches = (scaled_X_train.shape[0] // self.batch_size)
            #permutation = np.random.permutation(X_train.shape[0])
            #print('Permut1: {0}'.format(permutation))
            #sys.exit()
            #X = X_train_shuffled = scaled_X_train[permutation]
            #y = y_train_shuffled = y_train[permutation]
        else:
            # Shuffle test data
            X =  scaled_X_test
            y =  y_test
            
            num_batches = (scaled_X_test.shape[0] // self.batch_size)
            #permutation = np.random.permutation(X_test.shape[0])
            #X = X_test_shuffled = scaled_X_test[permutation]
            #print('X_test_shuffle: {0}'.format(X))
            #print('X== X_test_shuffled: {0}'.format((X==X_test_shuffled).all()))
            #sys.exit()
            #y = y_test_shuffled = y_test[permutation]
            #print('y== y_test_shuffled: {0}'.format((y==y_test_shuffled).all()))
            #sys.exit()
            
        train_acc = []

        #for i in range(epochs):
        i = 0
        while i < epochs:
            if forward_run:
                self.forward(X)
                #train_loss = mseGradient(self.output.flatten(order='C'), y.flatten())[0]
                #fwd_acc = self.accuracy(y.flatten(), self.output.flatten(order='C'))
                fwd_acc = self.accuracy(y, self.output.T)
                #print(template.format(i+1, time.time()-start_time, train_acc, train_loss))
                #sys.exit()
                return fwd_acc


            for j in range(num_batches):
                # Batch
                begin = j * batch_size
                if y_test is None:
                    end = min(begin + self.batch_size, scaled_X_train.shape[0]-1)
                    X = scaled_X_train[begin:end]
                    y = y_train[begin:end]
                else:
                    end = min(begin + self.batch_size, scaled_X_test.shape[0]-1)
                    X = scaled_X_test[begin:end]
                    y = y_test[begin:end]

                # Forward -- moved to self.accuracy
                self.forward(X)

                # Backprop
                mse, mseGrad, sigGrad = self.backprop(X, y, 0.5)
            
            #print('mse: ', mse)
            #print('mseGrad: ', mseGrad)
            #print('sigGrad: ', sigGrad)

            # Evaluate performance
            # Training data
            #output = self.feed_forward(x_train)
            #print('output shape:{0}'.format(self.output.shape))
            #print('flat output:{0}'.format(self.output.flatten(order='C')))
            #train_acc = train_acc + [self.accuracy(self.output.flatten(order='C'), y.flatten())] ## 'C' column major
            if i == 0:
                prev_acc = self.accuracy(self.output.T, y)
                new_acc = prev_acc
                train_acc = train_acc + [new_acc]
                i += 1
                #print('1st epoch acc: {0}'.format(train_acc))
                #sys.exit()
            else: 
                new_acc = self.accuracy(self.output.T, y) 
                if new_acc > prev_acc:
                   train_acc = train_acc + [new_acc]
                   #print('epoch: {0}'.format(i))
                   #print('Acc list: {0}'.format(train_acc))
                   prev_acc = new_acc
                   i += 1
                else:
                   #i -= 1
                   pass
                     
            #train_acc= train_acc + [self.accuracy(self.output.T, y)]
            #print(self.accuracy(y.flatten(), self.output.flatten(order='C')))
            #train_loss = mseGradient(self.output.flatten(order='C'), y.flatten())[0]
            # Test data
            #output = self.feed_forward(x_test)
            #test_acc = self.accuracy(y_test, output)
            #test_loss = self.cross_entropy_loss(y_test, output)
            #print(template.format(i+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))
            #print(template.format(i+1, time.time()-start_time, train_acc, train_loss))
            #if  i == epochs - 1:
                #print(train_acc)
        return train_acc


n_features = len(raw_train.columns) - 1 ## .columns is an index obj
n_classes = 10
n1_neurons = 64
tln = TwoLayerNeural(n_features, n1_neurons, n_classes)
#print('scaledX[0] = {0}'.format(scaled_X_train[0]))

'''
#mse,mseGrad = mseGradient(np.array([-1,0,1,2]), np.array([4,3,2,1]))
#print([mse], mseGrad.tolist(), [sigmoidGrad(var) for var in [-1,0,1,2]], [mseGradient(oln.output[:,0], y_train[0])[0]])
#print([mse], mseGrad.tolist(), [sigmoidGrad(var) for var in [-1,0,1,2]], [mseGradient(oln.output[:,0], y_train[1])[0]])

'''
#oln.train(scaled_X_train, y_train, scaled_X_test, y_test, epochs=1, batch_size=64, optimizer='momentum', l_rate=0.1, beta=.9)
fwd_acc = tln.train(None, None, scaled_X_test, y_test,True, epochs=20, batch_size=100, optimizer='momentum', l_rate=0.5, beta=.9)
#train_acc = oln.train(scaled_X_train,y_train, None, None,False, epochs=20, batch_size=100, optimizer='momentum', l_rate=0.5, beta=.9)
###train_acc = oln.train(None, None, scaled_X_test, y_test,False, epochs=20, batch_size=100, optimizer='momentum', l_rate=0.5, beta=.9)
#print('FWD accu: {0}'.format(fwd_acc))
#sys.exit()
print([fwd_acc])
#print([fwd_acc],train_acc)
#np.savetxt('a.txt', a,'%.7e', ',', ',')
#np.savetxt('fname', oln.w)
#large_array[some_slice] = np.load("path/to/small_array", mmap_mode="r")
