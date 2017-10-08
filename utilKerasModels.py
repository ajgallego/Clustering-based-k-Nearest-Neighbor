#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D

#------------------------------------------------------------------------------
def get_model_v1(input_shape, nb_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=input_shape, activation='relu', name='input'))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv_2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, activation='softmax', name='output'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return model

#------------------------------------------------------------------------------
def get_model_v2(input_shape, nb_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', name='input', input_shape=(input_shape,)))
    model.add(Dropout(0.2))  # .2
    model.add(Dense(1024, activation='relu', name='dense_1'))
    model.add(Dropout(0.2)) #
    model.add(Dense(512, activation='relu', name='dense_2'))
    model.add(Dropout(0.2)) #
    model.add(Dense(256, activation='relu', name='dense_3'))
    model.add(Dropout(0.2)) #
    model.add(Dense(128, activation='relu', name='dense_4'))
    model.add(Dropout(0.2)) #
    model.add(Dense(nb_classes, activation='softmax', name='output'))
        
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return model

#------------------------------------------------------------------------------
def get_model_v3(input_shape, nb_classes):
    model = Sequential()
    model.add(Dense(4096, activation='relu', name='input', input_shape=(input_shape,)))
    model.add(Dropout(0.5))  # .2
    model.add(Dense(2048, activation='relu', name='dense_1'))
    model.add(Dropout(0.5)) #
    model.add(Dense(1024, activation='relu', name='dense_2'))
    model.add(Dropout(0.5)) #
    model.add(Dense(512, activation='relu', name='dense_3'))
    model.add(Dropout(0.5)) #
    model.add(Dense(256, activation='relu', name='dense_4'))
    model.add(Dropout(0.5)) #
    model.add(Dense(128, activation='relu', name='dense_5'))
    model.add(Dropout(0.5)) #
    model.add(Dense(nb_classes, activation='softmax', name='output'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model

#------------------------------------------------------------------------------
def get_model_v4(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(64, 1, 1, border_mode='valid', input_shape=input_shape, activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))   
    
    model.add(Convolution2D(64, 2, 2, border_mode='valid', activation='relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Convolution2D(64, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))    
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax', name='output')) 
  
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return model

#------------------------------------------------------------------------------
def get_model_v5(input_shape, nb_classes):
    input_tensor = Input(shape=input_shape)
    base_model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    return model

#------------------------------------------------------------------------------
def get_model_v6(input_shape, nb_classes):
    model = Sequential()
    model.add(Convolution2D(256, 3, 3, border_mode='valid', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))    
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model



