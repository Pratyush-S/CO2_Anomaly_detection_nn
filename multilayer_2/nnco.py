import tensorflow as tf

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam

import os.path
import json
from keras.models import model_from_json


# For reproducibility - splitting train and test sets
seed = 123
np.random.seed(seed)
print("A")

# Load data from Excel sheets
#dataset2 = pd.read_excel('Uberset_02.xlsx')
dataset1 = pd.read_excel('Uberset.xlsx')
print("load")
#Combine datasets into one single data file
print("sda")
print(dataset1.describe())

