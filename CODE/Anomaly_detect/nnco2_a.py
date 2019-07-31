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
    json_file = open('mlp_arch_2019_c.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('mlp_weights_CO2_c.h5')
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
                 
    meanz=1457.702637595667
    sdz=1141.2473423740623
    meandz=0.0638638733267413
    sddz=2.142768311895675
    
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
    

predict(855.6928945,0.00E+00,854.610339,855.5876456,855.1140273,855.1140271,856.1477167,22,80,0.058074,70)
predict(852.7624429,0,851.4769346,852.4768195,852.1535818,852.1536071,853.0971088,22,80,0.058074,70)
predict(929.7872431,0,928.2047187,929.4113141,929.009091,929.0692154,930.0270955,22,80,0.058074,70)
predict(352.001738,0,352.9066922,354.1902945,355.247421,354.1311484,357.5279227,22,80,0.04,93)
predict(1138.797016,1,1149.043175,1140.043684,1140.339404,1138.898064,1141.471035,22,80,0.077504,93)
predict(2438.904134,1,2438.76041,2439.488991,2472.918331,2437.611359,2439.998065,22,80,0.077504,93)
predict(1008.733312,1,1009.185481,1011.725907,1012.783569,1014.343207,1016.90226,22,80,0.058074,70)
predict(1109.678648,1,1110.21176,1113.065902,1114.579772,1116.414867,1119.20647,22,80,0.058074,70)
predict(1163.535359,-1,1161.983229,1162.536606,1161.827356,1161.449327,1161.958862,22,80,0.058074,70)
predict(941.3463566,-1,939.7509211,940.9446255,940.5295224,940.576782,941.5218136,22,80,0.058074,70)
predict(1283.28098,-1,1282.934635,1283.317869,1283.00972,1280.965144,1282.935649,22,80,0.077504,93)
