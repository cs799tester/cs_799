# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 22:06:37 2016

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

dataset_directory = "datasets"

os.chdir(dataset_directory)

model = Sequential()

model.add(Dense(1000, input_dim=1024, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(50, init='normal', activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, init='normal', activation='softmax'))

sgd = SGD(lr=.0003, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              class_mode='binary')


#save model architecture for later use
json_string = model.to_json()
open('muv_model_architecture.json', 'w').write(json_string)
model.save_weights("my_model_weights.h5")

call(["git add --all"])
call(["git commit -m \"updating weights\""])
call(["git push https://cs799tester:tester55555@github.com/cs799tester/cs_799.git --all"])