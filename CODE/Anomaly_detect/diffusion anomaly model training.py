#importing libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD,Adam
import os.path
import json
import matplotlib.pyplot as plt

seed = 124
np.random.seed(seed)

#Importing datasets
#class1
dataset1 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P10_labeled.xlsx')
dataset2 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P30_labeled.xlsx')
dataset3 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P50_labeled.xlsx')
dataset4 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P70_labeled.xlsx')
dataset5 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P90_labeled.xlsx')
dataset6 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P110_labeled.xlsx')
dataset7 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P130_labeled.xlsx')
dataset8 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P150_labeled.xlsx')
dataset9 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P170_labeled.xlsx')
dataset10 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C1_P186_labeled.xlsx')

        
#class2
dataset11 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P10_labeled.xlsx')
dataset12 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P30_labeled.xlsx')
dataset13 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P50_labeled.xlsx')
dataset14 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P70_labeled.xlsx')
dataset15 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P90_labeled.xlsx')
dataset16 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P110_labeled.xlsx')
dataset17 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P130_labeled.xlsx')
dataset18 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P150_labeled.xlsx')
dataset19 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P170_labeled.xlsx')
dataset20 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C2_P186_labeled.xlsx')


#class3
dataset21 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P10_labeled.xlsx')
dataset22 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P30_labeled.xlsx')
dataset23 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P50_labeled.xlsx')
dataset24 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P70_labeled.xlsx')
dataset25 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P90_labeled.xlsx')
dataset26 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P110_labeled.xlsx')
dataset27 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P130_labeled.xlsx')
dataset28 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P150_labeled.xlsx')
dataset29 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P170_labeled.xlsx')
dataset30 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P186_labeled.xlsx')




#Combine datasets into one single data file
frames=[dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16, dataset17, dataset18, dataset19, dataset20, dataset21, dataset22, dataset23, dataset24, dataset25, dataset26, dataset27, dataset28, dataset29, dataset30]
dataset = pd.concat(frames)


avg_val=(dataset['CO2_Zone_1']+dataset['CO2_Zone_2']+dataset['CO2_Zone_3']+dataset['CO2_Zone_4']+dataset['CO2_Zone_5']+dataset['CO2_Zone_6'])/6
#avg_slope=(dataset['dz1']+dataset['dz2']+dataset['dz3']+dataset['dz4']+dataset['dz5']+dataset['dz6'])/6
#assigning different values to each output class

diffz1=pd.DataFrame(avg_val-dataset['CO2_Zone_1'])
diffz2=pd.DataFrame(avg_val-dataset['CO2_Zone_2'])
diffz3=pd.DataFrame(avg_val-dataset['CO2_Zone_3'])
diffz4=pd.DataFrame(avg_val-dataset['CO2_Zone_4'])
diffz5=pd.DataFrame(avg_val-dataset['CO2_Zone_5'])
diffz6=pd.DataFrame(avg_val-dataset['CO2_Zone_6'])

#creating dataframe for deviation from mean
frames2=[diffz1,diffz2,diffz3,diffz4,diffz5,diffz6]
dataset_2=pd.concat(frames2, axis=1)
dataset_2.columns=['dif1','dif2','dif3','dif4','dif5','dif6']

dataset=pd.concat([dataset,dataset_2],axis=1)



#  Shuffle Data
#The frac keyword argument specifies the fraction of rows to return in the random sample
#so frac=1 means return all rows (in random order)
#https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
dataset = dataset.sample(frac=1).reset_index(drop=True)
col=dataset.columns.values
dataset=dataset.drop(col[0],axis=1)



#creating datasets for input output for training and validation
X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]
print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")

# Feature scaling according to training set data 
col=list(X_complete.columns.values)

#Normalisation
for i in col:
    print(i)
    avg=X_complete[str(i)].mean()
    sd=X_complete[str(i)].std()
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))
    print(avg)
    print(sd)
    print(i)



    
# covert to array for processing
X_complete=X_complete.values
#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_complete=y_complete.values

# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(10, input_dim=22, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('@anomaly_detect.h5'):
    # Model reconstruction from JSON file
    json_file = open('anomaly_detect.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('anomaly_detect.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    history=model.fit(X_train,y_train,epochs=100,batch_size=200,verbose=1,validation_data=(X_test,y_test))
    
 
         # Save parameters to JSON file
    model_json = model.to_json()
    with open("diffusion_anomaly_detect.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('diffusion_anomaly_detect.h5')


model.summary()




# Model predictions for test set
y_pred = model.predict(X_complete)
y_test_class = np.argmax(y_complete,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
print(y_test_class,y_pred_class)
#print(y_pred_class)


# Evaluate model on test data
score = model.evaluate(X_complete,y_complete, batch_size=128,verbose=1)
 
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))





plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'test'], loc='upper left')














