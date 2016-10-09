import math
import random
import time
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint as pp

#filename = './data/training_data.txt';
#filename = './data/training_data_only_seizures.txt';
filename = './data/training_data_hybrid.txt';
#filename = './data/training_data_seizures_plus_some_non.txt';
test_filename = './data/training_data.txt' # non-seizure days and seizure days
split_input = False
run_tests = True
# http://www.scholarpedia.org/article/Seizure_prediction

learn_rate = 0.0003
reg_lambda = 1.0 # Overfitting penalizer
max_iters = 15000

# Starting point
theta = [-3.0, 0.3]
#theta = [-3.0, 0.3, 0.1]


def read_file_to_array(filename):
    arr = []
    with open(filename) as f:
        lines = f.read().splitlines()
        for line in lines:
            splt = line.split(' ')
            arr.append([float(splt[0]), int(splt[1])])
    return arr

alldata = read_file_to_array(filename)
train_data = alldata
m = len(train_data)

# Shuffle
if split_input:
    seed = time.time() # seed = 123
    print "Using seed %s" % str(seed)
    random.seed(seed)
    random.shuffle(alldata)
    m = len(alldata)
    train_data = alldata[m/4:]
    test_data = alldata[:m/4]
else:
    test_data = read_file_to_array(test_filename)


def sigmoid(z):
    return 1 / (1 + math.exp(-z))

# Hypothesis(theta)
def hyp_theta_x(x):
    return (
        theta[0] 
        + theta[1] * x
        #+ theta[2] * math.sqrt(x)
    )

# Logistic cost
def log_cost(x, y, hfn):
    hx = sigmoid(hfn(x))
    if hx == 1.0:
        hx -= 0.0000000001 # math.log(0) is undefined
    # hx = pred
    # y = 1: should pred 1, so log(pred) = 0 (no penalty)
    # y = 0: should pred 0, so log(1 - pred) = 0 (no penalty)
    # as prediction deviates, the log cost increases towards infinity
    return y * math.log(hx) + (1 - y) * math.log(1 - hx)

def total_cost():
    total_cost = 0
    for (x, y) in train_data:
        total_cost += log_cost(x, y, hyp_theta_x)
    # regularization
    reg_cost = reg_lambda / (2 * m) * (theta[1] * theta[1]) # theta 1 ... n (do not regularize 0)
    return -total_cost / m + reg_cost

def sum_error(hfn):
    err = 0
    for (x, y) in train_data:
        err += (sigmoid(hfn(x)) - y) * x # Partial deriv
    return err

# Do gradient descent and hope it converges
def descend(iters):
    for iter in xrange(0, iters):
        last_err = total_cost()    
        #print iter, theta[0], theta[1], theta[2], last_err
        print iter, theta[0], theta[1], last_err
        theta0_new = theta[0] - (learn_rate * sum_error(hyp_theta_x))
        theta1_reg = (1 - learn_rate * reg_lambda / m)
        theta1_new = (theta[1] * theta1_reg) - (learn_rate * (sum_error(hyp_theta_x)))
        #theta2_new = theta[2] - (learn_rate * sum_error(hyp_theta_x))
        theta[0] = theta0_new
        theta[1] = theta1_new
        #theta[2] = theta2_new
        if abs(total_cost() - last_err) < 0.0000000001:
            print "Converged after {} iterations delta={}".format(iter+1, abs(total_cost() - last_err))
            break

descend(max_iters)

if not run_tests:
    sys.exit(0)

print "RUNNING TEST"
wrongs = 0
rights = 0
seizures = 0
swrongs = 0
nswrongs = 0
nseizures = 0
for (x,y) in test_data:
    if y == 1:
        seizures = seizures + 1
    else:
        nseizures = nseizures + 1
    pS = sigmoid(hyp_theta_x(x))
    pred = ''
    #pS = 0.5;
    # 0.5 / 0.5 : equally likely
    # 0.333 / 0.666 : 2x more likely
    # 0.25 / 0.75 : 3x more likely
    # 0.20 / 0.80 : 4x more likely
    # 0.165 / 0.835 : 5x more likely
    if pS > 0.5 and y == 1:
       print "Okay"
       rights = rights + 1
    elif pS < 0.5 and y == 1:
       print "Not okay - missed seizure"
       wrongs = wrongs + 1
    elif pS == 0.5 and y == 1:
       print "Unsure"
    elif pS > 0.5 and y == 0:
       print "Flat wrong but passing because this is a warning system"
       #wrongs = wrongs + 1
    elif pS < 0.5 and y == 0:
       print "Okay"
       rights = rights + 1
    else:
        pass   
    #if y == 0 and pS < 0.25:
    #   rights = rights + 1
    #   pred = "RIGHT (predicted no seizure)"
    #elif y == 0 and pS >= 0.75:
    #   nswrongs = nswrongs + 1
    #   wrongs = wrongs + 1
    #   pred = "WRONG (predicted seizure but no seizure)"
    #elif y == 1 and pS < 0.75:
    #   wrongs = wrongs + 1
    #   swrongs = swrongs + 1
    #   pred = "WRONG (missed seizure)"
    #elif y == 1 and pS >= 0.75:
    #   rights = rights + 1
    #   pred = "RIGHT (predicted seizure)"
    #else:
    #   pred = "UNKNOWN"
    print "days=%d %.4f %d %s" % (x, pS, y, pred)


print "%d of %d, %.4f%% total wrong during test" % (wrongs, (wrongs+rights), float(100 * wrongs / (wrongs + rights)))
#print "%d of %d missed seizures (%.4f%%)" % (swrongs, seizures, 100*swrongs/seizures)
#print "%d of %d missed non-seizures (%.4f%%)" % (nswrongs, nseizures, 100*nswrongs/nseizures)
#print 
print theta[0], theta[1], total_cost()
