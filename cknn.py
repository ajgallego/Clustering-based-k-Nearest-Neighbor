#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import time
import numpy as np
import argparse
import warnings
import util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

np.random.seed(0)  # for reproducibility

TRAIN_SIZE = 0 
TEST_SIZE = 1 
SCORE = 2 
PRECISION = 3 
RECALL = 4 
F1 = 5 
TIME = 6 
TOTAL_DISTANCES = 7 
AVG_DISTANCES = 8


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
def calculate_clusters(b, X_train, X_test, n_jobs=1):
    kmeans = KMeans(n_clusters=b, random_state=0, n_jobs=n_jobs)
    kmeans.fit(X_train)

    start_time = time.time()
    test_predictions = kmeans.predict(X_test)
    initial_time = float(time.time() - start_time)

    train_clusters = np.zeros((len(X_train),b), dtype=int)
    test_clusters = np.zeros((len(X_test),b), dtype=int)

    for c in range(b):
        train_clusters[:,c] = kmeans.labels_ == c
        test_clusters[:,c] = test_predictions == c
    
    return train_clusters, test_clusters, initial_time


# -----------------------------------------------------------------------------
def increase_clusters(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, n_jobs=1):
    knn = KNeighborsClassifier(n_neighbors=(k+1), n_jobs=n_jobs)
    knn.fit(X_train, Y_train)
    neighbors_list = knn.kneighbors(X_train, None, False)
    
    for item_neighbors in neighbors_list:
        item_cluster = np.flatnonzero( train_clusters[ item_neighbors[0] ] )[0]
        
        for i in range(1, (k+1)):
            neighbour_cluster = np.flatnonzero( train_clusters[ item_neighbors[i] ] )[0]
            
            if neighbour_cluster != item_cluster:
                train_clusters[item_neighbors[i], item_cluster] = 1

    #print( np.max( np.sum(train_clusters, axis=1) ) )


# -----------------------------------------------------------------------------
def run_cknn(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, n_jobs=1):
    n_clusters = train_clusters.shape[1]
    total = np.zeros((n_clusters,9))

    for c in range(n_clusters):
        X_train_cluster = X_train[train_clusters[:,c] == 1]
        Y_train_cluster = Y_train[train_clusters[:,c] == 1]
        X_test_cluster = X_test[test_clusters[:,c] == 1]
        Y_test_cluster = Y_test[test_clusters[:,c] == 1]
        
        if X_train_cluster.shape[0] > 0 and X_test_cluster.shape[0] > 0:  # Non-empty cluster
            start_time = time.time()
            
            k_value = k
            if X_train_cluster.shape[0] < k:  # if the size of the cluster is smaller than k value
                k_value = X_train_cluster.shape[0]
        
            knnbc = KNeighborsClassifier(n_neighbors=k_value, algorithm='brute', n_jobs=n_jobs)
            knnbc.fit(X_train_cluster, Y_train_cluster)
            Y_pred = knnbc.predict(X_test_cluster)

            knn_time = time.time() - start_time
            
            # Save results    
            score = accuracy_score(Y_test_cluster, Y_pred)
            with warnings.catch_warnings(record=True): # to ignore warnings
                precision, recall, f1, support = precision_recall_fscore_support(Y_test_cluster, Y_pred, average=None)

            total[c][TRAIN_SIZE]      = X_train_cluster.shape[0]
            total[c][TEST_SIZE]       = X_test_cluster.shape[0]
            total[c][SCORE]           = score
            total[c][PRECISION]       = np.average(precision, None, support)
            total[c][RECALL]          = np.average(recall, None, support)
            total[c][F1]              = np.average(f1, None, support)
            total[c][TIME]            = knn_time
            total[c][TOTAL_DISTANCES] = (n_clusters * X_test_cluster.shape[0]) + X_train_cluster.shape[0] * X_test_cluster.shape[0]
            total[c][AVG_DISTANCES]   = n_clusters + X_train_cluster.shape[0]

    return total


# -----------------------------------------------------------------------------
def print_result(dbname, method, cv, b, k, args, X_train, X_test, initial_time, result):
    util.print_tabulated( ( dbname, method, args.lnoise, args.anoise, (cv+1), 
                            X_train.shape[0], X_test.shape[0], b, k, 
                            np.average(result[:,SCORE],     None, result[:,TEST_SIZE]),
                            np.average(result[:,PRECISION], None, result[:,TEST_SIZE]), 
                            np.average(result[:,RECALL],    None, result[:,TEST_SIZE]), 
                            np.average(result[:,F1],        None, result[:,TEST_SIZE]), 
                            initial_time + np.sum( result[:,TIME] ), 
                            np.sum( result[:,TOTAL_DISTANCES] ),
                            np.average(result[:,AVG_DISTANCES],None, result[:,TEST_SIZE]),
                            '-','-','-',
                            np.std(result[:,SCORE]),
                            np.std(result[:,PRECISION]),
                            np.std(result[:,RECALL]),
                            np.std(result[:,F1]),
                            np.std(result[:,TIME]),
                            '-',
                            np.std(result[:,TRAIN_SIZE])))



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='KNNbc experimenter')
parser.add_argument('-path',        required=True,                      help='path to the dataset')
parser.add_argument('-cv',          default=-1,     type=int, choices=[-1,0,1,2,3,4],   help='Fold to use as test [0-4]. If -1 train/test files will be used and noise will not be generated.')
parser.add_argument('-lnoise',      default=0,      type=int,           help='Percentaje of label noise')
parser.add_argument('-anoise',      default=0,      type=int,           help='Percentaje of attribute noise')
parser.add_argument('-jobs', default=4, type=int)
args = parser.parse_args()

assert (args.lnoise==0 and args.anoise==0) or (args.lnoise>0 and args.anoise==0) or (args.lnoise==0 and args.anoise>0)

dbname = os.path.basename(os.path.dirname(args.path))  # Penultimate folder:  datasets/mnist/original --> "mnist"


# Load datasets
if args.cv != -1:
    label_noise = attr_noise = 0
    X_train, X_test, Y_train, Y_test = util.load_ABCDE_datasets(args.path, args.cv, label_noise, attr_noise)
else:
    X_train, X_test, Y_train, Y_test = load_dataset(args.path)


# Generate noise
if args.lnoise > 0 and args.cv != -1:
    print(' - Generating {}% of label noise...'.format(args.lnoise ))
    util.generateLabelNoise(Y_train, args.lnoise)

elif args.anoise > 0 and args.cv != -1:
    print(' - Generating {}% of attribute noise...'.format(args.anoise))
    util.generateAttributeNoise(X_train, args.anoise)


for b in (10,15,20,25,30,100,500,1000): 

    print(80*'-')
    print('b=%d' % (b))
    print('\t'.join(('dbname', 'alg.', 'lnoise', 'anoise', 'cv','tr_size','te_size','b','k','score','preci.','recall','f1','tmp.sec','total_dist','m_dist','Std:','tr_size','te_size','score','preci.','recall','f1','tmp.sec','t_dist','m_dist')))
    
    # KMeans
    train_clusters, test_clusters, initial_time = calculate_clusters(b, X_train, X_test, args.jobs)

    for k in (1,3,5,7,9):

        # Run experiment ckNN
        result_1 = run_cknn(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, args.jobs)

        print_result(dbname, 'ckNN', args.cv, b, k, args, X_train, X_test, initial_time, result_1)
        
        
        # Run experiment ckNN Plus
        increase_clusters(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, args.jobs)
        
        result_2 = run_cknn(X_train, Y_train, X_test, Y_test, train_clusters, test_clusters, k, args.jobs)
       
        print_result(dbname, 'ckNN+', args.cv, b, k, args, X_train, X_test, initial_time, result_2)




