# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python example010m.py
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
import csv
import time
from six.moves import cPickle
# generate data set
def normalize(obs):
    z_scores = stats.zscore(obs)
    return special.ndtr(z_scores)

symmap = {}
datemap = {}
close = []
window = 25

#load dataset
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
np_change = (np_close[:,window:]/np_close[:,0:-window])
np_average = np.full(n_dates-window, np.nan)
print np_change.shape
print np_average.shape
for k in range(0, n_dates-window):
    np_average[k] = np.nanmean(np_change[:,k])
np_adjchange = np.full((n_syms,n_dates-window),np.nan)
for k in range(0, n_dates-window):
    np_adjchange[:,k] = (np_change[:,k]-np_average[k])

#normalize dataset by probability
np_data = np.full(np_adjchange.shape,np.nan)
for i in range(0, np_adjchange.shape[0]):
    temp = []
    for j in range(0,np_adjchange.shape[1]):
        if np.isnan(np_adjchange[i,j]) or np.isinf(np_adjchange[i,j]):
            continue
        else:
            temp.append(np_adjchange[i,j])
    if len(temp) == 0:
        break
    temp2 = normalize(temp)
    k = 0
    for j in range(0,np_adjchange.shape[1]):
        if np.isnan(np_adjchange[i,j]) or np.isinf(np_adjchange[i,j]):
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

print np_data
print np_data.shape

#split into train and test
train_split = 0.8
train = np_data[:,0:int(train_split*np_data.shape[1])]
test = np_data[:,int(train_split*np_data.shape[1]):np_data.shape[1]]
#print 'train', train, 'test', test
#exit(0)
#save memory
del np_data
del np_adjchange
del np_change
del symmap
del datemap
del close
del np_average
gc.collect()
#print gc.garbage
#exit(0)
#create windows
train_windows = []
train_label = []
test_windows = []
test_label = []
for i in range(0,train.shape[0]):
    for j in range(0,train.shape[1]-window-1):
        current_window = train[i,j:j+window]
        if (np.isnan(current_window).sum()>0) or (np.isnan(train[i,j+window+1])):
            continue
        else:
            train_windows.append(current_window)
            train_label.append(train[i,j+window+1])

np_train_windows = np.array(train_windows, dtype='float32')
del train_windows #save
gc.collect()

np_train_labels = np.array(train_label, dtype='int32')
del train_label
gc.collect()

for i in range(0,test.shape[0]):
    for j in range(0,test.shape[1]-window-1):
        current_window = test[i,j:j+window]
        if (np.isnan(current_window).sum()>0) or (np.isnan(test[i,j+window+1])):
            continue
        else:
            test_windows.append(current_window)
            test_label.append(test[i,j+window+1])

np_test_windows = np.array(test_windows, dtype='float32')
del test_windows
gc.collect()

np_test_labels = np.array(test_label, dtype='int32')
del test_label
gc.collect()

print 'train windows ', np_train_windows.shape, ', train labels ', np_train_labels.shape, ', test windows ', np_test_windows.shape, ', test labels ', np_test_labels.shape
#exit(0)

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

t0 = time.time()
t = time.time()
print np.isnan(np_train_windows).sum(), np.isinf(np_train_windows).sum()
print np.isnan(np_train_labels).sum(), np.isinf(np_train_labels).sum()
print np.isnan(np_test_windows).sum(), np.isinf(np_test_windows).sum()
print np.isnan(np_test_labels).sum(), np.isinf(np_test_labels).sum()
for epoch in range(0,100):
    rng_state = np.random.get_state()
    np.random.shuffle(np_train_windows)
    np.random.set_state(rng_state)
    np.random.shuffle(np_train_labels)

    bin_acc = np.zeros(20)
    bin_total = np.zeros(20)
    n_loss = 0
    t_loss = 0
    t_norm = 0
    for i in range(0,np_train_windows.shape[0]-1000,1000):
        b_loss,b_norm = train_fn(np_train_windows[i:i+1000],np_train_labels[i:i+1000])
        t_loss += b_loss
        t_norm += b_norm
        n_loss += 1.0
    hist = np.zeros(20)
    tacc = 0.0
    for i in range(0,np_test_windows.shape[0]):
        val_output = test_fn([np_test_windows[i]])
#        print 'val_output.shape', val_output.shape
        val_predictions = np.argmax(val_output[0])
#        print 'val_predictions', val_predictions
        hist[val_predictions] += 1
        tacc += abs(val_predictions-np_test_labels[i])
        if val_predictions == np_test_labels[i]:
            bin_acc[np_test_labels[i]] += 1
            bin_total[np_test_labels[i]] += 1
        else:
            bin_total[np_test_labels[i]] += 1
    tacc = tacc/float(np_test_windows.shape[0])
    
    print 'epoch', epoch, 't_loss', t_loss/n_loss, 't_norm', t_norm/n_loss, 'tacc', tacc, 'hist', hist, 'epoch time', int(time.time()-t), 'time since t0', int(time.time()-t0)
    for i in range(0,20):
        print i, float(bin_acc[i]/bin_total[i])
    t = time.time()
    f = open('model.pickle', 'wb')
    cPickle.dump(L.layers.get_all_param_values(network), f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

