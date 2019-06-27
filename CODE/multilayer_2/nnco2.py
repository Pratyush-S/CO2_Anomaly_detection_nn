import tensorflow as tf
import paho.mqtt.client as mqtt
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
def loadmodel():
    global model
    
    # Model reconstruction from JSON file
    json_file = open('mlp_arch_2019.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('mlp_weights_CO2.h5')
    model.compile(Adam(lr=0.01),'categorical_crossentropy',metrics=['accuracy'])

    print('model loaded')

loadmodel()

def on_connect(self, client, userdata, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    self.subscribe("co1")
    self.subscribe("co2")
    self.subscribe("co3")
    self.subscribe("co4")
    self.subscribe("co5")
    self.subscribe("co6")
    self.subscribe("vf")
    self.subscribe("CS")
    self.subscribe("SP")
    self.subscribe("Pas")

def on_message(client, userdata, msg):
    global k
    global ErP
    global C1, C2, C3, C4, C5, C6
    global vef
    global initial,final
    global err
    global cf
    global cs
    global SetPoint
    global Pas
    print ("Topic: ", msg.topic+'\nMessage: '+str(msg.payload))
    c = str(msg.topic)
    d = str(msg.payload)
    print(c)
    if (c == "co1"):
        C1 = float(d[2:-1])
        print("c1",C1)
        k= k+1
        print(k)
    if (c == "co2"):
        C2 = float(d[2:-1])
        print("c2",C2)
        k+= 1
        print(k)
    if (c == "co3"):
        C3 = float(d[2:-1])
        print("c3",C3)
        k= k+1
        print(k)
    if (c == "co4"):
        C4 = float(d[2:-1])
        print("c4",C4)
        k= k+1
        print(k)
    if (c == "co5"):
        C5 = float(d[2:-1])
        print("c5",C5)
        k= k+1
        print(k)
    if (c == "co6"):
        C6 = float(d[2:-1])
        print("c6",C6)
        k= k+1
        print(k)
    if (c == "vf"):
        vef = float(d[2:-1])
    if (c == "SP"):
        SetPoint = float(d[2:-1])
    if (c == "Pas"):
        Pas = float(d[2:-1])

    if(k == 6):
        print("wenthere")
        k=0
        cs=predict(C1, C2, C3, C4,C5,C6,Pas)
        print(cs)
        if(cs ==0):
            print("nnolthingtodo")
            client.publish("Stat",str(0))
        else:
            print("wenttocontroller")
            C = (C1 + C2 + C3 + C4 + C5 + C6)/6
            kp = -0.007
           # kd = -0.00016
            kd=0
            ki = -0.000009
            print("m")
            ErCR = SetPoint - C
            print("n")
            err.append(ErCR)
            print("o")
            P = kp * ErCR
            print("p",P)
            I = ki * sum(err)
            print("q",I)
            CS = P + I
            print("r")
            if(CS > 1):
                CS = 1
            if (CS < 0.2):
                CS = 0.2
            print("cs",CS)
            client.publish("Stat",str(1))
            client.publish("CS",str(CS))


 

def predict(co1,co2,co3,co4,co5,co6,pas):
    global model
    global X_test
    global y_pred
    global y_pred_class
    print("A")
    a=((co1-1143.799)/515.8405435735974 ) 
    b=((co2-1272.8486392837608)/481.23202212981795 ) 
    c=((co3-1195.5114295223411 )/503.3446724232689 ) 
    d=((co4-1277.787129553847)/462.80346960715343 ) 
    e=((co5-1222.2298587077155 )/503.71812479771967) 
    f=((co6-1256.8228244610748)/468.345682341992 )
    g= ((pas-116.5)/51.87755917738837)
    print("b")
    V_X=pd.DataFrame({'Z1':a,'Z2':b,'Z3':c,'Z4':d,'Z5':e,'Z6':f, 'Z7':g},index=[0])
    print("c")
    X_test = V_X.values
    print("D")
    print(X_test)
    print("E")
    y_pred = model.predict(X_test)
    print("F")
    print(X_test)
    y_pred_class = np.argmax(y_pred,axis=1)
    ans = y_pred_class[0]
    print(y_pred)
    print(y_pred_class)
    print(ans)
    return ans


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("10.114.64.66", 1883, 60)
client.loop_forever()
# Compile model


