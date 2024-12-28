import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
import os.path
import json
from keras.models import model_from_json



# For reproducibility - splitting train and test sets
seed = 127
np.random.seed(seed)

# Load data from Excel sheets
#dataset2 = pd.read_excel('Uberset_02.xlsx')
dataseta = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\1.xlsx')

dataset2a = dataseta.sample(frac=1).reset_index(drop=True)

X_completea=dataseta.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)

#y_complete=dataset['Class','zone_1']
y_completea=dataseta[['class_0','class_1','class_2','class_3','class_4']]


#y_complete=pd.concat([y0,y1,y2,y3,y4], axis=1, sort=False)
#y_complete=y1+y2+y0+y3+y4;

print("Unnormalized Data", "\n", X_completea[:5], "\n")
print("Unnormalized Data", "\n", y_completea[:5], "\n")

# Feature scaling according to training set data 
cola=list(X_completea.columns.values)

for i in range(0,12,1):
    print(i)
#    avg=X_complete[str(i)].mean()
 #   sd=X_complete[str(i)].std()
    X_complete[col[i]]=X_complete[col[i]]-
   
    
    
print("Normalized Data\n", X_complete[:5], "\n")


# covert to array for processing
X_complete=X_complete.values

#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_complete=y_complete.values

# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.2, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(8, input_dim=16, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('mlp_weights_CO2_b.h5'):

    # Model reconstruction from JSON file
    json_file = open('mlp_arch_2019_b.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('mlp_weights_CO2_b.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    model.fit(X_train,y_train,epochs=30,batch_size=200,verbose=1,validation_data=(X_test,y_test))
    
    # Save parameters to JSON file
    model_json = model.to_json()
    with open("mlp_arch_2019_b.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('mlp_weights_CO2_b.h5')


model.summary()




# Model predictions for test set
y_pred = model.predict(X_complete)
y_test_class = np.argmax(y_complete,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)

#print(y_test_class,y_pred_class)
print(y_pred_class)


# Evaluate model on test data
score = model.evaluate(X_complete, y_complete, batch_size=128,verbose=1)
 
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))


    
#