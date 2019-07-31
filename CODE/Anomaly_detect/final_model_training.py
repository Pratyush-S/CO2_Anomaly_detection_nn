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
dataset1 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_0.xlsx')
dataset2 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_47.xlsx')
dataset3 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_94.xlsx')
dataset4 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_141.xlsx')
dataset5 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_186.xlsx')

dataset6 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class1_b\training_set_4hr_pascnt_47.xlsx')
dataset7 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class1_b\training_set_4hr_pascnt_93.xlsx')

dataset8 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class2_b\training_set_4hr_pascnt_0.xlsx')
dataset9 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class2_b\training_set_4hr_pascnt_93.xlsx')
dataset10 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class2_b\training_set_4hr_pascnt_140.xlsx')
dataset11 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class2_b\training_set_4hr_pascnt_186.xlsx')

dataset12 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_0.xlsx')
dataset13 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_47.xlsx')
dataset14 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_93.xlsx')
dataset15 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_140.xlsx')
dataset16 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_20.xlsx')
dataset17 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_70.xlsx')
dataset18 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_163.xlsx')
dataset19 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class3_b\training_set_4hr_pascnt_170.xlsx')

#Combine datasets into one single data file
frames=[dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16, dataset17,dataset18,dataset19]
dataset = pd.concat(frames)

#  Shuffle Data
#The frac keyword argument specifies the fraction of rows to return in the random sample
#so frac=1 means return all rows (in random order)
#https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
dataset = dataset.sample(frac=1).reset_index(drop=True)


X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)

#y_complete=dataset['Class','zone_1']
y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]

#assigning different values to each output class
y0=dataset['class_0']*1;
y1=dataset['class_1']*2;
y2=dataset['class_2']*3;
y3=dataset['class_3']*4;
y4=dataset['class_4']*5;

#y_complete=pd.concat([y0,y1,y2,y3,y4], axis=1, sort=False)
#y_complete=y1+y2+y0+y3+y4;

print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")

# Feature scaling according to training set data 
col=list(X_complete.columns.values)

for i in col:
    avg=X_complete[str(i)].mean()
    sd=X_complete[str(i)].std()
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))
    print(avg)
    print(sd)
    print(i)
    
    
    
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

print(y_test_class,y_pred_class)

# Evaluate model on test data
score = model.evaluate(X_complete, y_complete, batch_size=128,verbose=1)
 
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))


    
#