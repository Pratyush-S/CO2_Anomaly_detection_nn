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
    model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])

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
   


 

def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,temp,press,humid,pas):
    global model
    global X_test
    global y_pred
    global y_pred_class
    control_flag = 0
    avgco2 = (co1 + co2 + co3 + co4 + co5 + co6)/6
    setpoint = 310 + (9*pas)
                 
    meanz=1457.702637595667
    sdz=1141.2473423740623
    meandz=0.0638638733267413
    sddz=2.142768311895675
    
    a=((co1-meanz)/sdz)               #changed group avd and sd
    da=((dz1-meandz)/sddz)
    
    b=((co2-meanz)/sdz)
    db=((dz2-meandz)/sddz)
    
    
    c=((co3-meanz)/sdz)
    dc=((dz3-meandz)/sddz)
    
    
    d=((co4-meanz)/sdz)
    dd=((dz4-meandz)/sddz)
    
    
    e=((co5-meanz)/sdz)
    de=((dz5-meandz)/sddz)
    
    
    f=((co6-meanz)/sdz)
    df=((dz6-meandz)/sddz)
   
    g=((temp-21.9998)/0.00125)
    h=((press-80.000021)/0.000227)
    i=((humid-0.07916)/0.0636)
    j=((pas-91.0533)/61.2388)                                                
    
    
    V_X=pd.DataFrame({'CO2_Zone_1':a,'dz1':da,'CO2_Zone_2':b,'dz2':db,'CO2_Zone_3':c,'dz3':dc,'CO2_Zone_4':d,'dz4':dd,'CO2_Zone_5':e,'dz5':de,'CO2_Zone_6':f,'dz6':df,'temp_f':g,'press_f':h,'humid_f':i,'pass_f':j},index=[0])
    X_test = V_X.values
    y_pred = model.predict(X_test)


   
    y_pred_class = np.argmax(y_pred,axis=1)
    ans = y_pred_class[0]
    #ans = ans-1
    print("class",ans)
    if(ans >0):
        control_flag = 1
   # flowrate, ratio = Control(ans,setpoint,avgco2)
    flowrate=1
    ratio=y_pred
    #return control_flag ,ans, flowrate, ratio
#0




predict(306.9607572,6.960757236,307.3986362,7.398636213,308.9554808,8.955480752,310.4993196,10.49931955,312.0806824,12.08068239,314.3445448,14.34454484,22,80,0.15176,186)
predict(317.0984232,10.13766596,318.0466367,10.64800045,320.1254568,11.16997602,322.0250302,11.52571068,323.7712713,11.69058893,325.9724701,11.62792521,22,80,0.15176,186)
predict(338.1262666,8.450332858,340.1940827,8.445262945,340.2314765,8.436740312,341.8235338,8.426615038,343.4966526,8.416144531,344.9068712,8.406307219,22,80,0.11548,141)
predict(346.5216907,8.395424158,348.5806873,8.386604662,348.6086564,8.3771799,350.1911463,8.367612478,351.8547873,8.358134786,353.2556874,8.348816201,22,80,0.11548,141)
predict(1307.66992,1.505811212,1308.78443,1.50497088,1308.154549,1.504130556,1308.975574,1.503291085,1309.510766,1.502452007,1309.097258,1.501613148,22.0001,79.9999,0.15348,93)
predict(1309.170527,1.500607229,1310.2842,1.499769804,1309.653482,1.498932386,1310.473669,1.498095819,1311.008025,1.497259643,1310.593682,1.496423685,22.0001,79.9999,0.15348,93)
predict(1310.665948,1.495421246,1311.778787,1.494586717,1311.147234,1.493752196,1311.966588,1.492918522,1312.500111,1.492085239,1312.084934,1.491252172,22.0001,79.9999,0.15348,93)
predict(1312.156201,1.4902532,1313.268208,1.489421558,1312.635824,1.488589923,1313.454347,1.487759132,1313.987039,1.486928731,1313.571032,1.486098546,22.0001,79.9999,0.15348,93)
predict(1582.048727,2.140190562,1570.230499,2.137825679,1570.67917,2.135438444,1571.383093,2.133053966,1571.214463,2.130671813,1571.690807,2.128292561,22,80,0.11548,140)
predict(1584.174142,2.125415797,1572.353566,2.123067209,1572.799866,2.120696421,1573.501421,2.118328371,1573.330426,2.115962631,1573.804406,2.113599773,22,80,0.11548,140)
predict(1586.284885,2.110742827,1574.461976,2.108410424,1574.905922,2.106055971,1575.605126,2.103704239,1575.43178,2.101354799,1575.903415,2.099008222,22,80,0.11548,140)
predict(1588.381056,2.096170961,1576.555831,2.093854632,1576.997438,2.091516404,1577.694306,2.089180877,1577.518628,2.086847628,1577.987932,2.084517222,22,80,0.11548,140)
predict(959.8195978,9.987734718,960.3264064,9.976577452,962.9213935,9.965436531,964.0335683,9.954305644,965.6476531,9.943188087,968.2610848,9.932084613,22,80,0.058074,20)
predict(969.7383702,9.918772339,970.2340988,9.907692373,972.818022,9.896628486,973.9191427,9.885574379,975.5221866,9.874533489,978.1245914,9.863506657,22,80,0.058074,20)
predict(979.5886565,9.85028634,980.0733817,9.839282913,982.6463174,9.828295416,983.7364603,9.817317615,985.3285395,9.806352948,987.9199937,9.795402256,22,80,0.058074,20)
predict(989.3709297,9.78227323,989.8447274,9.77134578,992.4067516,9.760434146,993.4859924,9.74953214,995.0671827,9.73864318,997.6477618,9.7277681,22,80,0.058074,20)
predict(2037.67282,-8.217941023,2035.269917,-8.208759537,2034.928782,-8.199591447,2033.137892,-8.190432907,2031.822189,-8.181285898,2031.541861,-8.172150344,22,80,0.058074,20)
predict(2013.357892,-8.04888483,2010.982151,-8.039893713,2010.668138,-8.030915573,2008.904345,-8.021945397,2007.615709,-8.01298591,2007.362411,-8.004037888,22,80,0.058074,20)
predict(1989.543155,-7.88330843,1987.194017,-7.874502275,1986.906568,-7.865708823,1985.169316,-7.856923172,1983.907188,-7.848147994,1983.680366,-7.839384046,22,80,0.058074,20)
predict(1966.218319,-7.721138154,1963.895236,-7.712513153,1963.633805,-7.703900594,1961.922548,-7.695295676,1960.686384,-7.686701015,1960.485492,-7.678117353,22,80,0.058074,20)

