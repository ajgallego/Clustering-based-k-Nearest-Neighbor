#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import pandas as pd
import numpy as np
import os, re
import random
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ----------------------------------------------------------------------------
def print_error(str):
    print('\033[91m' + str + '\033[0m')
    
# ----------------------------------------------------------------------------
def print_tabulated(list):
    print '\t'.join('%.4f' % x if type(x) is np.float64 or type(x) is float else str(x) for x in list)

# ----------------------------------------------------------------------------
def mkdirp(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# ----------------------------------------------------------------------------
# Return the list of files in folder
def list_dirs(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, f))]

# ----------------------------------------------------------------------------
# Return the list of files in folder
# ext param is optional. For example: 'jpg' or 'jpg|jpeg|bmp|png'
def list_files(directory, ext=None):
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and ( ext==None or re.match('([\w_-]+\.(?:' + ext + '))', f) )]

# ----------------------------------------------------------------------------
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
def load_csv(path, sep=',', header=None):
	df = pd.read_csv(path, sep=sep, header=header)
	return df.values

#------------------------------------------------------------------------------
def load_ABCDE_datasets(path, fold, label_noise, attribute_noise):
    CVNAMES = ['A', 'B', 'C', 'D', 'E']
    train = np.array([])
    test = train = np.array([])
    for p in range(5):
        data = load_csv(os.path.join(path, CVNAMES[p] + '.txt'))
        print('Size db %d: %s' % (p, str(data.shape)))
        if p == fold:
            test = data
        else:
            train = np.concatenate((train, data), axis=0) if train.size else data

    X_train = train[:,1:].astype('float32')
    X_test  = test[:,1:].astype('float32')
    Y_train = train[:,0].astype('int')
    Y_test  = test[:,0].astype('int')
    
    # Generate noise
    if label_noise > 0:
        print(' - Generating {}% of label noise...'.format(label_noise))
        generateLabelNoise(Y_train, label_noise)
        
    elif attribute_noise > 0:
        print(' - Generating {}% of attribute noise...'.format(attribute_noise))
        generateAttributeNoise(X_train, attribute_noise)
    
    return X_train, X_test, Y_train, Y_test

#------------------------------------------------------------------------------
def generateLabelNoise(Y, percent):
    assert percent >= 0 and percent <= 100
    assert Y.ndim == 1 and len(Y) > 0

    arrayIndexes = range(len(Y))
    nb_changes = (percent * len(Y) / 100) / 2
    
    for i in range(0, nb_changes):
        posit1 = random.randrange(len(arrayIndexes))
        index1 = arrayIndexes[posit1]
        del arrayIndexes[posit1] 

        while True:  # search a distinct label
            posit2 = random.randrange(len(arrayIndexes))
            index2 = arrayIndexes[posit2]
            if Y[index1] != Y[index2]:            
                del arrayIndexes[posit2] 
                break
        aux = Y[index1]
        Y[index1] = Y[index2]
        Y[index2] = aux
        
#------------------------------------------------------------------------------
def generateAttributeNoise(X, percent):
    assert percent >= 0 and percent <= 100
    assert X.ndim == 2 and len(X) > 0 and len(X[0]) > 0
    
    minVal = np.zeros((len(X[0])))
    maxVal = np.zeros((len(X[0])))
    for c in range(len(X[0])):
        minVal[c] = min(X[:,c])
        maxVal[c] = max(X[:,c])
        #print(c, 'min', minVal[c], 'max', maxVal[c])
    
    numChanges = 0
    total = 0
    for r in range(len(X)):
        for c in range(len(X[r])):
            total += 1
            if random.randint(0, 100) < percent:
                X[r,c] = random.uniform(minVal[c], maxVal[c])
                numChanges += 1

#------------------------------------------------------------------------------
def write_data_to_csv_file(filename, x, labels):
    n_dim = len(x[0])
    with open(filename, 'w') as f:
        for i in range(len(x)):
            str_features = ','.join("{}".format(x[i,j]) for j in range(n_dim))
            f.write('{},{}\n'.format(labels[i], str_features))

#------------------------------------------------------------------------------
def labels_encoder(labels, encoder=None):
    if encoder == None:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    return y, encoder

#------------------------------------------------------------------------------
def l2norm(X):
    norm = 0
    for i in range(len(X)):
        if X[i] < 0:
            X[i] = 0
        else:
            norm += X[i] * X[i]
    if norm != 0:
        norm = math.sqrt(norm)
        X /= norm

#------------------------------------------------------------------------------
def scaler_standard(data, scaler=None):
    if scaler == None:
        scaler = StandardScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler
    
#------------------------------------------------------------------------------
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn.preprocessing.MinMaxScaler
def scaler_min_max(data, scaler=None):
    if scaler == None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    return data, scaler

#------------------------------------------------------------------------------
def mean_normalization(data, mean=None):
    if mean == None:
        mean = np.mean(data)
    data -= mean
    return data, mean

#------------------------------------------------------------------------------
def std_normalization(data, mean=None):
    if mean == None:
        mean = np.std(data)
    data -= mean
    return data, mean

#------------------------------------------------------------------------------
def preprocess_data(mode, data, scaler=None):
    if mode == None or mode == 'None':  return data, scaler
    elif mode == '255':         data /= 255
    elif mode == 'standard':    data, scaler = scaler_standard(data, scaler)
    elif mode == 'minmax':      data, scaler = scaler_min_max(data, scaler)
    elif mode == 'mean':        data, scaler = mean_normalization(data, scaler)
    elif mode == 'std':         data, scaler = std_normalization(data, scaler)
    else:
        print_error('\nError: Wrong preprocessing mode')
        quit()

    return data, scaler
