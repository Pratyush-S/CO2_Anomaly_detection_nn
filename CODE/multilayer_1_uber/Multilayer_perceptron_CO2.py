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


#for tensorboard
from time import time
from tensorflow.python.keras.callbacks import TensorBoard

# For reproducibility - splitting train and test sets
seed = 123
np.random.seed(seed)



# Load data from Excel sheets
dataset2 = pd.read_excel(r'D:\PS!\Dataset\UBER\Uberset_02.xlsx')
dataset1 = pd.read_excel(r'D:\PS!\Dataset\UBER\Uberset_01.xlsx')

#Combine datasets into one single data file
frames=[dataset1,dataset2]
dataset = pd.concat(frames)

#  Shuffle Data
dataset = dataset.sample(frac=1).reset_index(drop=True)
X_complete=dataset.drop('class',axis=1)
y_complete=dataset['class']

print("Unnormalized Data", "\n", X_complete[:5], "\n")


# Feature scaling according to training set data 
col=list(X_complete.columns.values)

for i in col:
    avg=X_complete[str(i)].mean()
    sd=X_complete[str(i)].std()
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))

print("Normalized Data\n", X_complete[:5], "\n")


# covert to array for processing
X_complete=X_complete.values

#One hot encoding
y_complete = pd.get_dummies(y_complete).values

# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(3, activation='softmax'))


#log directory to save tensorboard outputs
tensorboard=TensorBoard(log_dir="log/{}".format(time()))

# Compile model
model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('#multilayer_perceptron_weights_CO2.h5'):

    # Model reconstruction from JSON file
    json_file = open('model_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('multilayer_perceptron_weights_CO2.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    model.fit(X_train,y_train,epochs=10,batch_size=256,verbose=1,validation_data=(X_test,y_test),callbacks=[tensorboard])
    
    # Save parameters to JSON file
    model_json = model.to_json()
    with open("model_architecture.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('multilayer_perceptron_weights_CO2.h5')


model.summary()



# Model predictions for test set
y_pred = model.predict(X_test)
y_test_class = np.argmax(y_test,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)


print(y_test_class,y_pred_class)

# Evaluate model on test data
score = model.evaluate(X_test, y_test, batch_size=128,verbose=1)
print(score)

# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))
