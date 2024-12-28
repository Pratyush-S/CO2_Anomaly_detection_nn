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

###############################################################################################################################

#Importing datasets
avg_array=[1713.5323763512292,
 0.08760651186291118,
 1694.4256119631375,
 0.08627750036701946,
 1691.2298889705223,
 0.08776716632013695,
 1691.6902062117085,
 0.08812124568601862,
 1691.880886499942,
 0.08725316141932586,
 1688.407582924399,
 0.08677699855205569,
 21.9976924259261,
 80.00000888888754,
 0.0909368849628335,
 99.6]

sd_array=[1643.5227474594326,
 4.644740193277256,
 1637.9669659562214,
 4.683437399603012,
 1622.3815110560924,
 4.593012563995032,
 1622.0594629738184,
 4.639638381040414,
 1645.7269775537643,
 4.652622700633985,
 1621.673275566828,
 4.6061488048463515,
 0.049069506919712146,
 0.000219179502789205,
 0.07979031620288202,
 56.82824973050747]

###############################################################################################################################

##Importing datasets
##class1
dataset1 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P10_labeled.xlsx')
dataset2 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P30_labeled.xlsx')
dataset3 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P50_labeled.xlsx')
dataset4 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P70_labeled.xlsx')
dataset5 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P90_labeled.xlsx')
dataset6 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P110_labeled.xlsx')
dataset7 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P130_labeled.xlsx')
dataset8 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P150_labeled.xlsx')
dataset9 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P170_labeled.xlsx')
dataset10 = pd.read_excel(r'D:\PS!\Dataset\final training data\C1_P186_labeled.xlsx')        
#class2
dataset11 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P10_labeled.xlsx')
dataset12 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P30_labeled.xlsx')
dataset13 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P50_labeled.xlsx')
dataset14 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P70_labeled.xlsx')
dataset15 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P90_labeled.xlsx')
dataset16 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P110_labeled.xlsx')
dataset17 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P130_labeled.xlsx')
dataset18 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P150_labeled.xlsx')
dataset19 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P170_labeled.xlsx')
dataset20 = pd.read_excel(r'D:\PS!\Dataset\final training data\C2_P186_labeled.xlsx')
#class3
dataset21 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P10_labeled.xlsx')
dataset22 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P30_labeled.xlsx')
dataset23 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P50_labeled.xlsx')
dataset24 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P70_labeled.xlsx')
dataset25 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P90_labeled.xlsx')
dataset26 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P110_labeled.xlsx')
dataset27 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P130_labeled.xlsx')
dataset28 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P150_labeled.xlsx')
dataset29 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P170_labeled.xlsx')
dataset30 = pd.read_excel(r'D:\PS!\Dataset\final training data\C3_P186_labeled.xlsx')
###############################################################################################################################




#Combine datasets into one single data file
frames=[dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16, dataset17, dataset18, dataset19, dataset20, dataset21, dataset22, dataset23, dataset24, dataset25, dataset26, dataset27, dataset28, dataset29, dataset30]
dataset = pd.concat(frames)


col=dataset.columns.values

print(col)


dataset=dataset.drop(col[0],axis=1)

dataset = dataset.reset_index(drop=True)


#############################################################################################################################################################################




#creating datasets for input output for training and validation
X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]
print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")


a=X_complete
#############################################################################################

col=list(X_complete.columns.values)    
#Normalisation
j=0

for i in col:
    print(i)
    print(sd_array[j])   
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg_array[j])/(sd_array[j]))
    j=j+1
        
        
    
#############################################################################################
'''
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
'''
#############################################################################################


    
# covert to array for processing
X_complete=X_complete.values
#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_complete=y_complete.values



# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(10, input_dim=16, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('diffusion_anomaly_detect.h5'):
    # Model reconstruction from JSON file
    json_file = open('diffusion_anomaly_detect.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('diffusion_anomaly_detect.h5')
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
y_pred = model_anomaly.predict(X_complete)
y_test_class = np.argmax(y_complete,axis=1)
y_pred_class = y_pred
print(y_test_class,y_pred_class)
#print(y_pred_class)
# Evaluate model on test data
score = model_anomaly.evaluate(X_complete,y_complete, batch_size=128,verbose=1)
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))




# Model predictions for test set
y_pred = model_anomaly.predict(X_complete)
y_test_class = np.argmax(y_complete,axis=1)
y_pred_class = np.argmax(y_pred,axis=1)
print(y_test_class,y_pred_class)
#print(y_pred_class)


# Evaluate model on test data
score = model_anomaly.evaluate(X_complete,y_complete, batch_size=128,verbose=1)
 
# Compute stats on the test set and Output all results
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test_class,y_pred_class))
print(confusion_matrix(y_test_class,y_pred_class))





plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['train', 'test'], loc='upper left')





loadmodel_1()

loadmodel_2()








