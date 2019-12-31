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
###############################################################################################################################
################################################################################################################
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

###############################################################################################################################



#Combine datasets into one single data file
frames=[dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16, dataset17, dataset18, dataset19, dataset20, dataset21, dataset22, dataset23, dataset24, dataset25, dataset26, dataset27, dataset28, dataset29, dataset30]
dataset = pd.concat(frames)



col=dataset.columns.values
print(col)


dataset=dataset.drop(col[0],axis=1)
dataset = dataset.reset_index(drop=True)


#############################################################################################################################################################################


def class_to_lim(clas1):
    return {
        1:20,
        2:0.5,
        3:2,
        4:6,
        5:8
            }.get(clas1,20)


def cal_severity(clas,single_dev):
    severity=0
    lower_th=0.02
    upper_th=class_to_lim(clas)
   
    max_dev=max([abs(x) for x in single_dev])

    if(clas>1):
        if(max_dev>upper_th):    
            
            severity=2
        elif(max_dev<=lower_th):
            
            severity=0
        else:                   
            
            severity=1
    else:
        severity=0
    #print("Severity:         "+str(severity))
    #print("Error threshold: "+str(error_th)+"%")
    #print("------------------------------------------")
    return severity

#############################################################################################################################################################################


class_total=dataset['class_0']*0+dataset['class_1']*1+dataset['class_2']*2+dataset['class_3']*3+dataset['class_4']*4   

all_dev=dataset[['dz1','dz2','dz3','dz4','dz5','dz6']].values.tolist()

#To generate empty datasets
#level_0=dataset['class_1']*0
#level_1=dataset['class_1']*0
#level_2=dataset['class_1']*0

severity_ary=dataset['class_1']*0

#col_dz=all_dev.columns

for i in range(0,dataset.shape[0]):
       #single_dev=all_dev[i:i+1].tolist()
       
       #single_dev=[all_dev[i:i+1].values[0][0],all_dev[i:i+1].values[0][1],all_dev[i:i+1].values[0][2],all_dev[i:i+1].values[0][3],all_dev[i:i+1].values[0][4],all_dev[i:i+1].values[0][5]]
       single_dev=all_dev[i]
       clas=class_total[i]
       severity_ary[i]= cal_severity(clas,single_dev)
       
print(severity_ary[0:11])
print(class_total[0:11]) 

#############################################################################################################################################################################
frame=[class_total,severity_ary]
dataset_new_data=pd.concat(frame, axis=1)
dataset_new_data.columns=['class_total','severity']
dataset=pd.concat([dataset,dataset_new_data],axis=1)
col=dataset.columns.values


#############################################################################################################################################################################

#X_complete=dataset.drop([ 'class_0', 'class_1', 'class_2','class_3', 'class_4','dif1', 'dif2', 'dif3', 'dif4', 'dif5', 'dif6'],axis=1)
X_complete=dataset.drop([ 'class_0', 'class_1', 'class_2','class_3', 'class_4','severity'],axis=1)

#y_complete=dataset['Class','zone_1']
y_complete=dataset[[ 'severity']]
print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")



#############################################################################################
# Feature scaling according to training set data 
col=list(X_complete.columns.values)
avg=0
sd=0
for i in col:#[:-1]:
    avg=X_complete[str(i)].mean()
    sd=X_complete[str(i)].std()
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))
    print(avg)
    print(sd)
    print(i)


#############################################################################################



col=list(X_complete.columns.values)    
#Normalisation of all except class total
j=0
for i in col[:-1]:
    print(i)    
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg_array[j])/(sd_array[j]))
    j=j+1
        
        
    
#############################################################################################

  

    
print("Normalized Data\n", X_complete[:5], "\n")

# covert to array for processing
X_complete=X_complete.values

#One hot encoding
#_complete= pd.get_dummies(y_complete).values

y_complete=pd.get_dummies(y_complete['severity']).values



# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.2, random_state=seed)



# Define Neural Network model layers
model = Sequential()
model.add(Dense(10, input_dim=17, activation='relu'))
#model.add(Dense(10, input_dim=11, activation='softmax'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(Adam(lr=0.0001),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('@diffusion_severity_detect2.h5'):

    # Model reconstruction from JSON file
    json_file = open('diffusion_severity_detect2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('diffusion_severity_detect2.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.0001),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    history=model.fit(X_train,y_train,epochs=200,batch_size=200,verbose=1,validation_data=(X_test,y_test))
    
 
         # Save parameters to JSON file
    model_json = model.to_json()
    with open("diffusion_severity_detect2.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('diffusion_severity_detect2.h5')


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















