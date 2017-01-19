import csv
import urllib2
import theano
import theano.tensor as T
import lasagne as L
import gc
import random
import numpy as np
import scipy.special as special
import scipy.stats as stats
import matplotlib.pyplot as plt
from six.moves import cPickle


def normalize(obs):
    z_scores = stats.zscore(obs)
    return special.ndtr(z_scores)

symmap = {}
datemap = {}
close = []
window = 25

with open('WIKI_20161229.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    for row in reader:
        if row[0] not in symmap:
            symmap[row[0]] = None
        if row[1] not in datemap:
            datemap[row[1]] = None
        close.append([row[0], row[1], row[5]])
n_dates = 0
for key, value in sorted(datemap.items()):
    datemap[key] = n_dates
    n_dates += 1
n_syms = 0
for key, value in sorted(symmap.items()):
    symmap[key] = n_syms
    n_syms += 1

np_close = np.full((n_syms,n_dates),np.nan)
for i in range(0, len(close)):
    try:
        np_close[symmap[close[i][0]],datemap[close[i][1]]] = float(close[i][2])
    except ValueError:
        continue

#compute changes and daily averages
np_change = np.full(np_close.shape, np.nan)
for i in range(0,np_change.shape[0]):
    for j in range(0,np_change.shape[1]-window):
        np_change[i,j+window] = float(np_close[i,j+window]/np_close[i,j])

#np_change = (np_close[:,window:]/np_close[:,0:-window])   #make same size as np_close
np_average = np.full(np_change.shape[1], np.nan)
print 'np_change.shape',np_change.shape
print 'np_average.shape',np_average.shape
print 'np_change',np_change[0,-50:]
for k in range(window, np_average.shape[0]):
    np_average[k] = np.nanmean(np_change[:,k])

#np_adjchange = np.full(np_change.shape,np.nan)
for k in range(window, np_change.shape[1]):
    np_change[:,k] = (np_change[:,k]-np_average[k])

print 'np_change',np_change[0,-50:]
#normalize dataset by probability
np_data = np.full(np_change.shape,np.nan,dtype='float32')
for i in range(0, np_change.shape[0]):
    temp = []
    for j in range(0,np_change.shape[1]):
        if np.isnan(np_change[i,j]) or np.isinf(np_change[i,j]):
            continue
        else:
            temp.append(np_change[i,j])
    if len(temp) == 0:
        break
    temp2 = normalize(temp)
    k = 0
    for j in range(0,np_change.shape[1]):
        if np.isnan(np_change[i,j]) or np.isinf(np_change[i,j]):
            continue
        else:
            np_data[i,j] = temp2[k]
            k+=1

#bin normalized dataset
bins = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1])
for i in range(0,np_data.shape[0]):
    for j in range(0,np_data.shape[1]):
        if np.isnan(np_data[i,j]):
            continue
        else:
            np_data[i,j] = np.digitize(np_data[i,j],bins,right=True)

print np_data[0,-50:]
train_split = 0.8
test = np_data[:,int(train_split*np_data.shape[1]):np_data.shape[1]]
test_windows = []

#define network
input_var = T.matrix(dtype=theano.config.floatX)
target_var = T.vector(dtype='int32')
network = L.layers.InputLayer((None, window), input_var)
network = L.layers.DenseLayer(network, 1000)
network = L.layers.DenseLayer(network, 1000)
network = L.layers.DenseLayer(network, 1000)
network = L.layers.DenseLayer(network, 1000)
network = L.layers.DenseLayer(network, 1000)
network = L.layers.DenseLayer(network, 20, nonlinearity=L.nonlinearities.softmax)
prediction = L.layers.get_output(network)
loss = L.objectives.aggregate(L.objectives.categorical_crossentropy(prediction, target_var), mode='mean')
params = L.layers.get_all_params(network, trainable=True)
updates = L.updates.adam(loss, params, learning_rate=0.0001)
scaled_grads,norm = L.updates.total_norm_constraint(T.grad(loss,params), np.inf, return_norm=True)
train_fn = theano.function([input_var, target_var], [loss,norm], updates=updates)
test_fn = theano.function([input_var], L.layers.get_output(network, deterministic=True))

#load network
f = open('model.pickle', 'rb')
m = cPickle.load(f)
f.close()
L.layers.set_all_param_values(network, m)

myfile = open('output1.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
for ticker, i in sorted(symmap.items()):
    for date, j in sorted(datemap.items()):
        if j-window>=int(train_split*np_data.shape[1]):
            #current_window = test[i,j:j+window]
            #print j-window
            #print int(train_split*np_data.shape[1])
            current_window = np.reshape(np_data[i,j-window:j],(1,window))
            #print 'i',i,'j',j,'current_window.shape',current_window.shape,'ticker',ticker,'date',date,'np_data.shape',np_data.shape
            val_output = test_fn(current_window)
            val_prediction = np.argmax(val_output[0])
            if (np.isnan(current_window).sum()>0):
                continue
                #print 'NAN', 'ticker', ticker, 'date', date, 'current window', current_window
                #wr.writerow([ticker,date,-1])
            else:
                print 'ticker', ticker, 'date', date, 'val_prediction', val_prediction, 'current window', current_window
                wr.writerow([ticker,date,val_prediction,np_change[i,j-window]])
                test_windows.append(current_window)

#np_test_windows = np.array(test_windows, dtype='float32')



