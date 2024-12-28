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
global model
global k
global err
global model
global X_test
global y_pred
global y_pred_class
 
    
k=0
err = []

global Pas,press,temp,humid  #new variables

def loadmodel():
    global model
    
    # Model reconstruction from JSON file
    json_file = open('arc.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('wei.h5')
    model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])

    print('model loaded')

loadmodel()



def Control(Class, Setpoint, avgco2):
    global k
    global ErP
    global C1, C2, C3, C4, C5, C6
    global dC1, dC2, dC3, dC4, dC5, dC6 #added addditional global variables
    global vef
    global initial,final
    global err
    global cf
    global cs
    global SetPoint
  
    
    if (Class == 1):
        kp = -0.007
       # kd = -0.00016
        kd=0
        ki = -0.000009

        ErCR = SetPoint - avgco2
        err.append(ErCR)

        P = kp * ErCR
        I = ki * sum(err)
        CS = P + I

        if(CS > 1):
            CS = 1
        if (CS < 0.2):
            CS = 0.2
        ratio = 0.5
    if (Class == 2):
        ratio = 0.3
        CS = 1
    if (Class == 3):
        ratio= 0.7
        CS= 1
    if (Class == 4):
        ratio = 0.4
        CS = 1
    if (Class == 0):
        ratio = 0.5
        CS = 1
        
    return CS, ratio
   


 
def predict(co1,dz1,co2,co3,co4,co5,co6,temp,press,humid,pas):

#def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,temp,press,humid,pas):
#def predict(a,da,b,db,c,dc,d,dd,e,de,f,df,g,h,i,j):

    control_flag = 0
    avgco2 = (co1 + co2 + co3 + co4 + co5 + co6)/6
    setpoint = 310 + (9*pas)
                 
  
    
    a=((co1-1457.7026375956484)/1141.2473423740632)               #changed group avd and sd
  #  da=((dz1-0.0638638733267406)/2.1427683118956535)
    
    b=((co2-1456.6391278541234)/1136.5534767844747)
   # db=((dz2-0.0638654702130458)/2.108189678851196)
    
    
    c=((co3-1457.0185787395235)/1139.6414395961863)
   # dc=((dz3-0.06386691438591093)/2.1175639313283745)
    
    
    d=((co4-1457.0723934194505)/1137.3779508854207)
  #  dd=((dz4-1456.0606374994725)/2.1196403497694223)
    
    
    e=((co5-1456.0606374994725)/1135.1186412360885)
   # de=((dz5-0.06387351647381606)/2.096588174371372)
    
    
    f=((co6-1457.220173709606)/1137.8027890096632)
    #df=((dz6-0.06388384196958394)/2.11233577808761)
   
    g=((temp-21.99985054933706)/0.0011882719873640968)
    h=((press-80.00001721478221)/0.00020207148656728872)
    i=((humid-0.08515214196256149)/0.06272724186119283)
    j=((pas-91.05336583796903)/61.95548201661293)                                                
    
    V_X=pd.DataFrame({'CO2_Zone_1':a,'dz1':dz1,'CO2_Zone_2':b,'CO2_Zone_3':c,'CO2_Zone_4':d,'CO2_Zone_5':e,'CO2_Zone_6':f,'temp_f':g,'press_f':h,'humid_f':i,'pass_f':j},index=[0])
    #V_X=pd.DataFrame({'CO2_Zone_1':a,'dz1':da,'CO2_Zone_2':b,'dz2':db,'CO2_Zone_3':c,'dz3':dc,'CO2_Zone_4':d,'dz4':dd,'CO2_Zone_5':e,'dz5':de,'CO2_Zone_6':f,'dz6':df,'temp_f':g,'press_f':h,'humid_f':i,'pass_f':j},index=[0])
   # V_X=pd.DataFrame({a,da,b,db,c,dc,d,dd,e,de,f,df,g,h,i,j})
 
    X_test = V_X.values
    y_pred = model.predict(X_test)
  
    y_pred_class = np.argmax(y_pred,axis=1)

    print(y_pred_class)

#    ans = y_pred_class[0]
    #ans = ans-1
  #  print("class",ans)
    

predict(1337.300772,1,1337.385909,1339.559835,1340.25132,1341.445219,1343.639,22,80,0.058074,70)

dataset1 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\training dataset\class0_b\training_set_4hr_pascnt_0.xlsx')
dataset = dataset1.sample(frac=1).reset_index(drop=True)
#dataseta = dataset1.sample(frac=1).reset_index(drop=True)


X_complete=dataset.drop(['dz2','dz3','dz4','dz5','dz6','class_0','class_1','class_2','class_3','class_4'],axis=1)

#y_complete=dataset['Class','zone_1']
y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]

#assigning different values to each output class
y0=dataset['class_0']
y1=dataset['class_1']
y2=dataset['class_2']
y3=dataset['class_3']
y4=dataset['class_4']
X_complete['dz1']=(y4*(-1))+y1+y2+y3
X_complete['temp_f']=dataset['dz1']

#y_complete=pd.concat([y0,y1,y2,y3,y4], axis=1, sort=False)
#y_complete=y1+y2+y0+y3+y4;

print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")

# Feature scaling according to training set data 
col=list(X_complete.columns.values)

meanz=1457.702637595667
sdz=1141.2473423740623
meandz=0.0638638733267413
sddz=2.142768311895675

X_complete['CO2_Zone_1']=(X_complete['CO2_Zone_1']-meanz)/sdz
X_complete['CO2_Zone_2']=(X_complete['CO2_Zone_2']-meanz)/sdz
X_complete['CO2_Zone_3']=(X_complete['CO2_Zone_3']-meanz)/sdz
X_complete['CO2_Zone_4']=(X_complete['CO2_Zone_4']-meanz)/sdz
X_complete['CO2_Zone_5']=(X_complete['CO2_Zone_5']-meanz)/sdz
X_complete['CO2_Zone_6']=(X_complete['CO2_Zone_6']-meanz)/sdz

X_complete['temp_f']=((X_complete['temp_f']-21.99985054933706)/0.0011882719873640968)
X_complete['press_f']=((X_complete['press_f']-80.00001721478221)/0.00020207148656728872)
X_complete['humid_f']=((X_complete['humid_f']-0.08515214196256149)/0.06272724186119283)
X_complete['pass_f']=((X_complete['pass_f']-91.05336583796903)/61.95548201661293)



# For reproducibility - splitting train and test sets
seed = 127
np.random.seed(seed)
  
print("Normalized Data\n", X_complete[:5], "\n")


# covert to array for processing
X_complete=X_complete.values

#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_complete=y_complete.values

# Creating a Train and a Test Dataset
X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.1, random_state=seed)


# Define Neural Network model layers
model = Sequential()
model.add(Dense(10, input_dim=11, activation='softmax'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(10, activation='softmax'))
model.add(Dense(8, activation='softmax'))
model.add(Dense(5, activation='softmax'))

# Compile model
model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])




if os.path.isfile('wei.h5'):

    # Model reconstruction from JSON file
    json_file = open('arc.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into the new model
    model.load_weights('wei.h5')
    print("Model weights loaded from saved model data.")

    model.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])
else:
    print("Model weights data not found. Model will be fit on training set now.")

    # Fit model on training data - try to replicate the normal input
    model.fit(X_train,y_train,epochs=30,batch_size=200,verbose=1,validation_data=(X_test,y_test))
    
    # Save parameters to JSON file
    model_json = model.to_json()
    with open("mlp_arch_2019_c.json", "w") as json_file:
        json_file.write(model_json)

    # Save model weights to file
    model.save_weights('mlp_weights_CO2_c.h5')


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


    