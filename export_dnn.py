#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import math
import os
import argparse
import warnings
import util, utilKerasModels
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras import backend as K

np.random.seed(125)  # for reproducibility
warnings.filterwarnings('ignore')

if K.backend() == 'tensorflow':
    import tensorflow as tf    # Memory control with Tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    K.set_session(sess)
    

#------------------------------------------------------------------------------
def run_test(model, X_test, Y_test, batch_size):
    print('Testing...')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    # Predict on test
    Y_pred = model.predict(X_test, batch_size=batch_size, verbose=1)
    
    # Report
    Y_pred1 = np.argmax(Y_pred, axis=1)
    Y_test1 = list()
    for i in Y_test:
        Y_test1.append(np.argmax(i))
    
    accuracy = accuracy_score(Y_test1, Y_pred1)
    
    print("Accuracy: %0.3f" % (accuracy))
    print(classification_report(Y_test1, Y_pred1))
        
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    

#------------------------------------------------------------------------------
def calculate_output_features(get_layer_output, X):
    learning_phase = 0 #test_mode    
    output = np.array([])
    psize = 500
    n_pages = int( math.ceil( len(X)/float(psize) ) )
    
    for p in range(n_pages):  # Calculamos las output features por trozos
        pfrom = p * psize
        pto = pfrom + psize
        if pfrom > len(X) - psize:  # resto
            pto = len(X)
        aux = get_layer_output([X[pfrom:pto, :], learning_phase])[0]
        output = np.concatenate((output, aux), axis=0) if output.size else aux
        
    return output


#------------------------------------------------------------------------------
def prv_export_vc(output_folder, model, X_train, X_test, Y_train, Y_test): 
    print('Exporting vector codes to files...')
    print('# Output data to folder:', output_folder)
    print('# model num layers:', len(model.layers))
    util.mkdirp(output_folder)
    
    n_layer = len(model.layers) - 2
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[n_layer].output])
    
    X_train_nc = calculate_output_features(get_layer_output, X_train)
    X_test_nc = calculate_output_features(get_layer_output, X_test)
    
    print('NC Train size:', X_train_nc.shape)
    print('NC Test size:', X_test_nc.shape)

    util.write_data_to_csv_file(output_folder + '/train.txt', X_train_nc, Y_train)
    util.write_data_to_csv_file(output_folder + '/test.txt', X_test_nc, Y_test)


# -----------------------------------------------------------------------------
def reshape(X_train, X_test , flat, channels, img_rows, img_cols):
    if flat == True:
        input_shape = X_train.shape[1]
    elif K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    return X_train, X_test, input_shape


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Export DNN features')
parser.add_argument('-path',   required=True,                              help='basepath to the dataset')
parser.add_argument('-cv',     default=0, choices=[0,1,2,3,4], type=int,   help='Fold to use as test [0-4]')
parser.add_argument('-m',      default=1,    dest='model',                 help='model number [1-6]')
parser.add_argument('-c',      default=1,    dest='channels',  type=int,   help='number of channels')
parser.add_argument('-e',      default=200,  dest='nb_epoch',  type=int,   help='number of epochs')
parser.add_argument('-b',      default=128,  dest='batch',     type=int,   help='mini batch size')
parser.add_argument('-pre',    default=None,                   type=str,   help='Preprocess data [255, standard, minmax, mean, std]')
parser.add_argument('-lnoise', default=0,                      type=int,   help='Percentaje of label noise')
parser.add_argument('-anoise', default=0,                      type=int,   help='Percentaje of attribute noise')
parser.add_argument('--flat',                action='store_true',          help='Use a flat vector')
parser.add_argument('--load',                action='store_true',          help='Load weights from file')
parser.add_argument('--save',                action='store_true',          help='Save the output vector codes')
args = parser.parse_args()

assert (args.lnoise==0 and args.anoise==0) or (args.lnoise>0 and args.anoise==0) or (args.lnoise==0 and args.anoise>0)

dbname = os.path.basename(os.path.dirname(os.path.normpath(args.path)))  # Penultimate folder:  datasets/mnist/original --> "mnist"
fulldbname = args.path.strip("/").replace("/", "_")

sufix_fold = '_cv' + str(args.cv)
sufix_noise = ''
if args.lnoise > 0 or args.anoise > 0: 
    noise_type = '_label_noise' if args.lnoise > 0 else '_attr_noise'
    noise_level = args.lnoise if args.lnoise > 0 else args.anoise
    sufix_noise = noise_type + str(noise_level)

output_folder = 'datasets/' + dbname + '/m' + str(args.model) + sufix_fold + sufix_noise
weights_filename = 'MODELS/model_'+ fulldbname +'_m'+ str(args.model) + sufix_fold + sufix_noise +'.h5'
util.mkdirp('MODELS')



# Load dataset and prepare data

print('Loading dataset...')
X_train, X_test, Y_train, Y_test = util.load_ABCDE_datasets(args.path, args.cv, args.lnoise, args.anoise)

nb_classes = len(np.unique(Y_train))
img_rows = img_cols = int( math.sqrt( X_train.shape[1] / args.channels ) )

Yc_train = np_utils.to_categorical(Y_train, nb_classes)
Yc_test = np_utils.to_categorical(Y_test, nb_classes)

if args.pre != None and args.pre != 'None':
    X_train, scaler = util.preprocess_data(args.pre, X_train)
    X_test, scaler  = util.preprocess_data(args.pre, X_test, scaler)


X_train, X_test, input_shape = reshape(X_train, X_test, args.flat, 
                                       args.channels, img_rows, img_cols)


print('dbname:', dbname)
print('full dbname:', fulldbname)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('nb_classes:', nb_classes)
print('channels:', args.channels)
print('input:', input_shape)
print('preprocess:', args.pre)
print('Label noise:', args.lnoise)
print('Attribute noise:', args.anoise)
print('model:', args.model)
print('flat:', args.flat)
print('batch_size:', args.batch)
print('nb_epoch:', args.nb_epoch)


# Load model and fit
model = getattr(utilKerasModels, 'get_model_v'+str(args.model))(input_shape, nb_classes)
print(model.summary())


early_stopping = EarlyStopping(monitor='val_loss', patience=10)

if args.load == True:
    print('Loading weights...')    
    model.load_weights(weights_filename)
else:
    print('Fiting...')
    model.fit(X_train, Yc_train, 
              batch_size=args.batch, 
              nb_epoch=args.nb_epoch,
              verbose=2, 
              validation_split=0.1,
              shuffle=True,
              callbacks=[early_stopping] )

    model.save_weights(weights_filename, overwrite=True)


# Run tests
run_test(model, X_test, Yc_test, args.batch)


# Get last layer features and save to files
if args.save == True:
    prv_export_vc(output_folder, model, X_train, X_test, Y_train, Y_test)


