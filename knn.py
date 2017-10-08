#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import time
import numpy as np
import argparse
import util
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

np.random.seed(0)  # for reproducibility    

# -----------------------------------------------------------------------------
# Load dataset fron train / test files
def load_dataset(path):
    train = util.load_csv(os.path.join(path, 'train.txt'))
    test = util.load_csv(os.path.join(path, 'test.txt'))

    X_train = train[:,1:].astype('float32')
    X_test  = test[:,1:].astype('float32')
    Y_train = train[:,0].astype('int')
    Y_test  = test[:,0].astype('int')
    
    return X_train, X_test, Y_train, Y_test


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='knn experimenter')
parser.add_argument('-path',        required=True,                      help='Path to the dataset')
parser.add_argument('-cv', default=-1, choices=[-1,0,1,2,3,4], type=int, help='Fold to use as test [0-4]. If -1 train/test files will be used and noise will not be generated.')
parser.add_argument('-values',      default='1,3,5,7,9',    type=str,   help='List of values to test')  # rf: 5,10,20,50,100
parser.add_argument('--l2',         action='store_true',                help='Apply L2 norm')
parser.add_argument('-lnoise',      default=0,              type=int,   help='Percentaje of label noise')
parser.add_argument('-anoise',      default=0,              type=int,   help='Percentaje of attribute noise')
parser.add_argument('-jobs',        default=4,              type=int)
args = parser.parse_args()

assert (args.lnoise==0 and args.anoise==0) or (args.lnoise>0 and args.anoise==0) or (args.lnoise==0 and args.anoise>0)

arrayValues = args.values.split(',')    # Array of values to iterate
dbname = os.path.basename(os.path.dirname(args.path))  # Penultimate folder:  datasets/mnist/original --> "mnist"
l2norm = "yes" if args.l2 else "no"


# Load datasets
if args.cv != -1:
    label_noise = attr_noise = 0
    X_train, X_test, Y_train, Y_test = util.load_ABCDE_datasets(args.path, args.cv, label_noise, attr_noise)
else:
    X_train, X_test, Y_train, Y_test = load_dataset(args.path)


# L2 norm
if args.l2:
    for i in range(len(X_train)):
        util.l2norm(X_train[i,:])
    for i in range(len(X_test)):
        util.l2norm(X_test[i,:])


# Generate noise
if args.lnoise > 0 and args.cv != -1:
    print('Generating {}% of label noise...'.format(args.lnoise ))
    util.generateLabelNoise(Y_train, args.lnoise)

elif args.anoise > 0 and args.cv != -1:
    print(' - Generating {}% of attribute noise...'.format(args.anoise))
    util.generateAttributeNoise(X_train, args.anoise)


print(80*'-')
print('\t'.join(('dbname', 'lnoise', 'anoise', 'l2', 'cv','tr_size','te_size','k','score','preci.','recall','f1','tmp.sec','total_dist','m_dist')))
    
for k in arrayValues:
    k = int(k)    
        
    # Run experiment
    start_time = time.time()

    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=args.jobs)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)

    total_time = time.time() - start_time
    
    
    # Report results
    score = accuracy_score(Y_test, Y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(Y_test, Y_pred, average=None)

    
    util.print_tabulated( ( dbname, args.lnoise, args.anoise, l2norm, (args.cv+1), 
                            X_train.shape[0], X_test.shape[0], k, 
                            score,
                            np.average(precision), 
                            np.average(recall), 
                            np.average(f1), 
                            total_time, 
                            X_train.shape[0] * X_test.shape[0],
                            X_train.shape[0]))


