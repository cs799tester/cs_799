# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 22:22:29 2016

@author: Moeman
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import glob, os
import re
from rdkit.Chem.Fingerprints import FingerprintMols
from keras.utils import np_utils
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from subprocess import call


#create giant pdf for all pcba assays
dataset_directory = "datasets/"

pcba_file = "pcba_mmtn_canon.csv.gz"
dude_file = "dude_mmtn_canon.csv.gz"
muv_file = "muv_mmtn_canon.csv.gz"
tox_file = "tox21_mmtn_canon.csv.gz"

#read data sets
data_pd = pd.read_csv(dataset_directory+ muv_file, dtype=str)

#now using the pandas df, construct the training set and test set
X_train = np.zeros(shape=(data_pd.shape[0], 1024))

for index, row in data_pd.iterrows():
    fps = list(row['1024_fingerprint'])
    
    X_train[index] = np.array(fps)


y_train = np.array(data_pd.as_matrix(columns = data_pd.columns[3:]))

X_train = X_train.astype(float)
y_train = y_train.astype(float)

#split into testing and training sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                X_train, y_train, test_size=0.4, random_state=0)

nb_classes = len(data_pd.columns[3:])

#the following is keras code for the structure of the network.
#see their website for more info
model = Sequential()

model.add(Dense(1000, input_dim=1024, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes, init='normal', activation='softmax'))

sgd = SGD(lr=.0003, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              class_mode='binary')


#save model architecture for later use
json_string = model.to_json()
open('muv_model_architecture.json', 'w').write(json_string)

#now train and save weights every 50,000 iterations
total_epochs = 200000
save_iters = 50000
iters = int(total_epochs/save_iters)

for i in range(iters): 
    model.fit(X_train, y_train,
              nb_epoch=save_iters,
              batch_size = 256,
              show_accuracy=True)
              
    
    model.save_weights("my_model_weights_iter_" + str(iters) + ".h5")

    call(["git",  "add",  "--all"])
    call(["git" , "commit",  "-m" , "\"updating weights\""])
    call(["git",  "push", "https://cs799tester:tester55555@github.com/cs799tester/cs_799.git", "--all"])

#generate score for training and validation set
score1 = model.evaluate(X_train, y_train, show_accuracy=True)
score2 = model.evaluate(X_test, y_test, show_accuracy=True)

roc_auc_score(y_train, np.array(model.predict(X_train) > 0.5, dtype=float))