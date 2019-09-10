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
 
global da,db,dc,de,df,dd 
k=0
err = []

global Pas,press,temp,humid  #new variables

def loadmodel():
    global model_anomaly
    global model_severity    
    
    # Model reconstruction from JSON file
    json_file = open('new_model_severity.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model_anomaly = model_from_json(loaded_model_json)
    model_anomaly.load_weights('new_model_severity.h5')
    model_anomaly.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])

    print('Anomaly model loaded')
    
        # Model reconstruction from JSON file
    json_file = open('new_model_severity_error_th.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model_severity = model_from_json(loaded_model_json)
    model_severity.load_weights('new_model_severity_error_th.h5')
    model_severity.compile(Adam(lr=0.001),'categorical_crossentropy',metrics=['accuracy'])

    print('Severity model loaded')





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
    
    
    
    
def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,T,Pcabin,H,P):

#def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,temp,press,humid,pas):
#def predict(a,da,b,db,c,dc,d,dd,e,de,f,df,g,h,i,j):

    control_flag = 0
    setpoint = 310 + (9*P)
                 
    meanz=1457.702637595667
    sdz=1141.2473423740623
    meandz=0.0638638733267413
    sddz=2.142768311895675
    
    avg_val=(co1+co2+co3+co4+co5+co6)/6;
   
    
    dif1=avg_val-co1;
    dif2=avg_val-co2;
    dif3=avg_val-co3;
    dif4=avg_val-co4;
    dif5=avg_val-co5;
    dif6=avg_val-co6;
    
    
    a=((co1-1429.0864)/1200.3404);
    da=((dz1-0.05919)/2.1899);
    b=((co2-1427.9556)/1195.3391);
    db=((dz2-0.05919)/2.1519);
    c=((co3-1428.4016)/1198.6366);
    dc=((dz3-0.05919)/2.1621);
    d=((co4-1428.4000)/1196.2201);
    dd=((dz4-0.05921)/2.16417);
    e=((co5-1427.2792)/1193.7958);
    de=((dz5-0.0592)/2.1387);
    f=((co6-1428.5712)/1196.6787);
    df=((dz6-0.0592)/2.1555);
    g=((T-21.9998)/0.00125);
    h=((Pcabin-80.000021)/0.000227);
    ii=((H-0.07916)/0.0636);
    j=((P-79.2352)/61.2388);
              
    a= ((co1-1450.3748089341545)/1105.3766530927035);
    da=((dz1-0.06730150028018703)/2.216723342719636);
    b= ((co2-1450.3748089341545)/1105.3766530927035);
    db=((dz2-0.06730150028018703)/2.216723342719636);
    c= ((co3-1450.3748089341545)/1105.3766530927035);
    dc=((dz3-0.06730150028018703)/2.216723342719636);
    d= ((co4-1450.3748089341545)/1105.3766530927035);
    dd=((dz4-0.06730150028018703)/2.216723342719636);
    e= ((co5-1450.3748089341545)/1105.3766530927035);
    de=((dz5-0.06730150028018703)/2.216723342719636);
    f= ((co6-1450.3748089341545)/1105.3766530927035);
    df=((dz6-0.06730150028018703)/2.216723342719636);

 

    g=((T-21.9998)/0.269869707571444);
    h=((Pcabin-80.000021)/0.000227);
    ii=((H-0.09891256672715397)/0.1116383178067835);
    j=((P-91.40854447325829)/60.20954481801682);
    
    d1=(dif1+0.8916)/8.4227;
    d2=(dif2+0.8916)/8.4227;
    d3=(dif3+0.8916)/8.4227;
    d4=(dif4+0.8916)/8.4227;
    d5=(dif5+0.8916)/8.4227;
    d6=(dif6+0.8916)/8.4227;
        
    V_X=pd.DataFrame({'CO2_Zone_1':a, 'dz1':da, 'CO2_Zone_2':b, 'dz2':db, 'CO2_Zone_3':c, 'dz3':dc,'CO2_Zone_4':d, 'dz4':dd, 'CO2_Zone_5':e, 'dz5':de, 'CO2_Zone_6':f, 'dz6':df, 'temp_f':g,'press_f':h, 'humid_f':ii, 'pass_f':j,'dif1':d1, 'dif2':d2, 'dif3':d3, 'dif4':d4, 'dif5':d5, 'dif6':d6},index=[0])
    #V_X=pd.DataFrame({'CO2_Zone_1':a,'dz1':da,'CO2_Zone_2':b,'dz2':db,'CO2_Zone_3':c,'dz3':dc,'CO2_Zone_4':d,'dz4':dd,'CO2_Zone_5':e,'dz5':de,'CO2_Zone_6':f,'dz6':df,'temp_f':g,'press_f':h,'humid_f':i,'pass_f':j},index=[0])
    # V_X=pd.DataFrame({a,da,b,db,c,dc,d,dd,e,de,f,df,g,h,i,j})
    X_test = V_X.values
    y_pred = model_anomaly.predict(X_test)
    y_pred_class = int(np.argmax(y_pred,axis=1))
    
    
    V_X2=pd.DataFrame({'CO2_Zone_1':a, 'dz1':da, 'CO2_Zone_2':b, 'dz2':db, 'CO2_Zone_3':c, 'dz3':dc,'CO2_Zone_4':d, 'dz4':dd, 'CO2_Zone_5':e, 'dz5':de, 'CO2_Zone_6':f, 'dz6':df, 'temp_f':g,'press_f':h, 'humid_f':ii, 'pass_f':j,'dif1':d1, 'dif2':d2, 'dif3':d3, 'dif4':d4, 'dif5':d5, 'dif6':d5,'class_total':y_pred_class},index=[0])
    X_test2 = V_X2.values
    y_pred2 = model_severity.predict(X_test2)
    y_pred_severity = int(np.argmax(y_pred2,axis=1))
    

    print("Anomaly class :    "+str(y_pred_class))
    print("Severity level:    "+str(y_pred_severity))
    
    print("-----------------------------------------------------")
    
    
    



#############################################################################################################################################################################
predict(343.4863274,9.684301762,344.6962192,9.678534821,347.1003993,9.668560406,347.706396,9.65671256,350.2386199,9.644776152,351.3131187,9.633560496,22,80,0.04,160)
predict(1697.24077,0.071674095,1696.455284,0.07159404,1697.699878,0.0715141,1696.918126,0.071434224,1697.504794,0.071354455,1696.803844,0.071274758,22,80,0.13081,160)
predict(3653.03038,7.556504699,3653.022682,7.555660511,3655.3465,7.554819522,3655.519044,7.553975763,3657.868916,7.553134997,3658.342047,7.552291826,22.0086,80.0002,1,160)
predict(1675.504186,1.241767933,1674.906828,1.240658447,1676.196519,1.239550366,1675.673992,1.238442878,1676.519715,1.23733668,1676.089565,1.236231192,22,80,0.1633,160)
predict(1038.166451,0.004388306,1039.400108,0.004385857,1039.581245,0.004383409,1038.604612,0.004380962,1039.402224,0.004378517,1039.307701,0.004376073,22.0003,79.9998,0.079275,47)
predict(808.4665807,1.202308663,809.9794633,1.201637822,810.4773507,1.200967141,809.4753521,1.200296517,810.6909625,1.199626736,810.832689,1.198957102,22.0003,79.9998,0.079275,47)
predict(2295.792298,1.014074428,2295.808519,1.012941074,2296.696844,1.011809144,2330.290543,1.010644522,2295.138225,1.009548655,2297.684099,1.00842081,22,80,0.077504,93)
predict(1079.759749,2.489745361,1090.072384,2.486943523,1081.138116,2.484194181,1081.499581,2.481422979,1080.123932,2.478654018,1082.762477,2.475890036,22,80,0.077504,93)
predict(2102.03798,30.9760538,2106.307054,30.94144022,2231.60422,30.90321708,2116.407914,30.87235656,2120.904708,30.83785983,2126.471253,30.80340782,22,80,0.13887,170)
predict(3317.872996,2.950641046,3318.279189,2.947342271,3371.184362,2.943892318,3318.594213,2.940756321,3318.630285,2.937468405,3319.772682,2.934184739,22,80,0.13887,170)
predict(1962.062073,3.077445561,1962.067681,3.074008475,1962.178007,3.070575286,1963.393106,3.06714653,1963.366984,3.063720931,1964.21254,3.06029963,22,80,0.13887,170)
predict(2097.402449,4.439709819,2098.047965,4.434751491,2098.286637,4.429798383,2099.396763,4.42485149,2100.002295,4.419909728,2100.508028,4.414973407,22,80,0.13887,170)
predict(1510.128099,9.395606443,1511.344173,9.385112434,1513.182607,9.374631179,1516.064445,9.364163354,1516.298048,9.353702838,1518.651475,9.343257515,22,80,0.077504,93)
predict(1211.980538,3.670537517,1212.389732,3.666437933,1213.309507,3.662343259,1215.160883,3.658253758,1214.590735,3.654167259,1215.947101,3.650086569,22,80,0.077504,93)


