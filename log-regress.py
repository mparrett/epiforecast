import math
import random
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
# https://sourceforge.net/projects/scipy/files/scipy/0.16.1/

from pprint import pprint as pp

# http://www.scholarpedia.org/article/Seizure_prediction

#filename = './data/training_data_only_seizures.txt';
#filename = './data/training_data_hybrid.txt';
#filename = './data/training_data_seizures_plus_some_non.txt';
#filename = './data/training_data.txt' # non-seizure days and seizure days

learn_rate = 0.003
init_lambda = 1.0 # Overfitting penalizer


filename = './data/training_data.txt';
theta = [-3.0, 0.3] # Starting point
#theta = [0.1, 0.1]
test_set =  [[1, 2], [1, 3], [1, 31], [1, 50], [1, 100]]
alldata = pd.read_csv(filename, header=None, names=["DaysSince", "WasSeizure"], delim_whitespace=True)

#filename = './data/ex2data1.txt';
#theta = [0.1, 0.1, 0.1]
#test_set =  [[1, 20, 30], [1, 40, 50], [1, 99, 99]]
#alldata = pd.read_csv(filename, header=None, names=["Ex1", "Ex2", "WasAdmit"], delim_whitespace=False)

# Set the training data
train_data = alldata

# Shuffle if we are going to split our training data
seed = int(time.time())
#seed = 123 # Fixed seed
alldata = alldata.sample(frac=1, random_state=seed).reset_index(drop=True)
m = len(alldata)
train_data = alldata.iloc[m/4:, :] # First 75%
test_data = alldata.iloc[:m/4, :] # Last 25%

# Add bias (intercept) term
test_data.insert(0, 'Ones', 1)
train_data.insert(0, 'Ones', 1)

print("Using %s training examples and %s test examples" % (len(train_data), len(test_data)))
print(alldata.head())

m = len(train_data)
cols = train_data.shape[1]
theta = np.matrix(theta)

X = train_data.iloc[:,0:cols-1]
y = train_data.iloc[:,cols-1:cols]

Xtest = test_data.iloc[:,0:cols-1]
ytest = test_data.iloc[:,cols-1:cols]


X = np.matrix(X.values)
y = np.matrix(y.values)

Xtest = np.matrix(Xtest.values)
ytest = np.matrix(ytest.values)

def precision(true_pos, false_pos):
    if true_pos + false_pos == 0:
        return 0.0
    return true_pos / (true_pos + false_pos)


def recall(true_pos, false_neg):
    if true_pos + false_neg == 0:
        return 0.0
    return true_pos / (true_pos + false_neg)


def f1score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_cost(theta, X, y, reg_lambda):
    """
    Logistic cost function
    """
    # y = 1: should pred 1, so log(pred) = 0 (no penalty)
    # y = 0: should pred 0, so log(1 - pred) = 0 (no penalty)
    # as prediction deviates, the log cost increases towards infinity
    
    theta = np.matrix(theta)
    
    hx = X * theta.T # hx = pred
    first = np.multiply(-y, np.log(sigmoid(hx)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(hx)))
    
    # Regularize
    temp = theta
    temp[0] = 0  # theta 1 ... n (do not regularize 0)

    reg_cost = 0.5 * reg_lambda * np.asscalar(temp * temp.T) # Why asscalar needed?

    return (np.sum(first - second) + reg_cost) / len(X)


def predict(theta, X):
    X = np.matrix(X)
    theta = np.matrix(theta)
    prob = sigmoid(X * theta.T)
    return prob


def gradient(theta, X, y, reg_lambda):
    theta = np.matrix(theta)
    
    params = int(theta.ravel().shape[1])
    grad = np.zeros(params)
    
    error = sigmoid(X * theta.T) - y
    
    # Regularization term, does not include j=0
    temp = theta
    temp[0] = 0

    reg = reg_lambda / len(X) * temp.T
    
    return X.T * error / len(X) + reg


# Minimize cost function
def learn(theta, lbda):
    result = opt.fmin_tnc(func=log_cost, x0=theta, fprime=gradient, args=(X, y, lbda), disp=False)
    return result


for lbda in (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2):
    result = learn(theta, lbda)
    #print(lbda, result[0])
    #preds = predict(result[0], test_set)
    #print(preds)

print result[0]
preds = [1 if x >= 0.5 else 0 for x in predict(result[0], Xtest)]

print(preds)
#print(Xtest)
#print(ytest)

true_pos = [1 if (a == 1 and b == 1) else 0 for (a, b) in zip(preds, ytest)]
true_neg = [1 if (a == 0 and b == 0) else 0 for (a, b) in zip(preds, ytest)]
false_pos = [1 if (a == 1 and b == 0) else 0 for (a, b) in zip(preds, ytest)]
false_neg = [1 if (a == 0 and b == 1) else 0 for (a, b) in zip(preds, ytest)]

print 'TP {}'.format(sum(true_pos))
print 'FP {}'.format(sum(false_pos))
print 'TN {}'.format(sum(true_neg))
print 'FN {}'.format(sum(false_neg))

prec = precision(sum(true_pos), sum(false_pos))
rec = recall(sum(true_pos), sum(false_neg))
f1 = f1score(prec, rec)

print 'precision: {:.2f}'.format(prec)
print 'recall: {:.2f}'.format(rec)
print 'f1score: {:.2f}'.format(f1)

correct = sum(true_pos) + sum(true_neg)
accuracy = 100 * correct / float(len(Xtest))

print 'accuracy = {:.2f}%'.format(accuracy)  

#test_set =  [[1, 50, 50]]

# Todo: normalization of data
# Todo: auto selection of lambda
# Todo: auto selection of polynomial terms
# Todo: add features (day of week)
# Todo: precision/recall evaluation (F1)

# Compute the x and y coordinates for points on a sigmoid curve
#x = np.arange(-10, 10, 0.1)
#y = sigmoid(x)

# Plot the points using matplotlib
#plt.plot(x, y)
#plt.show()  # You must call plt.show() to make graphics appear.