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




if os.path.isfile('#mlp_weights_CO2_b.h5'):

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


#................................................................................................................
def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,temp,press,humid,pas):
    
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





print(',,,,,,,,,,,   0   ,,,,,,,,,')
predict(319.3273971,8.441205769,322.3333745,8.59062697,324.5778249,8.657322699,325.7317421,8.667427667,326.8045949,8.646426723,328.219232,8.608324553,22,80,0.04,140)
predict(344.9103602,8.479867846,347.9225554,8.470895694,350.1477759,8.461296781,351.2702025,8.451624193,352.3083874,8.44209157,353.6902002,8.432703232,22,80,0.04,140)
predict(361.6952029,8.363340653,364.6886839,8.354000024,366.8951289,8.344663523,367.9988189,8.335341908,369.0183233,8.32603518,370.3814929,8.316739717,22,80,0.04,140)
predict(1403.414654,-0.000114317,1404.685259,-0.00011419,1405.407014,-0.000114062,1405.294218,-0.000113935,1405.294199,-0.000113808,1405.081157,-0.00011368,22,80,0.11548,140)
predict(1403.414205,-0.000111193,1404.68481,-0.000111069,1405.406565,-0.000110945,1405.29377,-0.000110821,1405.293752,-0.000110697,1405.08071,-0.000110573,22,80,0.11548,140)
predict(1505.363289,0.112885865,1506.621994,0.112759818,1507.452118,0.112633903,1507.357432,0.11250811,1507.619859,0.112382465,1507.317744,0.112256949,22,80,0.11548,140)
predict(1506.024424,0.108289471,1507.282391,0.108168556,1508.111777,0.108047768,1508.016354,0.107927097,1508.278046,0.107806568,1507.975196,0.107686163,22,80,0.11548,140)
predict(1521.495086,0.000733124,1522.735779,0.000732305,1523.547909,0.000731488,1523.435247,0.000730671,1523.679719,0.000729855,1523.359667,0.00072904,22,80,0.11548,140)
print(',,,,,,,,,,   3   ,,,,,,,,,,')
predict(1576.019006,4.958078471,1577.935753,4.952542191,1579.603494,4.947011872,1580.510958,4.941487063,1581.304753,4.935968325,1581.978616,4.930455647,22,80,0.11548,140)
predict(1605.056815,4.756197769,1606.941138,4.750886914,1608.57649,4.745581776,1609.451597,4.740281923,1610.213071,4.734987894,1610.854647,4.729699679,22,80,0.11548,140)
predict(1655.25677,4.407190649,1657.085039,4.402269501,1658.664397,4.397353651,1659.483565,4.392442699,1660.189163,4.387537143,1660.774924,4.382636974,22,80,0.11548,140)
predict(1628.329972,5.890105262,1630.546824,5.883528408,1632.213751,5.876958321,1633.045038,5.870394703,1633.988052,5.863838533,1634.716981,5.857289465,22,80,0.11548,140)
predict(1645.757404,5.768943598,1647.954798,5.762502032,1649.602285,5.756067094,1650.414152,5.749638492,1651.337768,5.743217185,1652.047319,5.736802833,22,80,0.11548,140)
print(',,,,,,,   4    ,,,,,,,,,,,,,')
 
predict(1606.686473,-1.413331441,1607.730106,-1.411753324,1608.225122,-1.410176832,1607.885811,-1.408601891,1607.659535,-1.407028738,1607.22048,-1.405457288,22,80,0.11548,140)
predict(1599.764848,-1.365209868,1600.816209,-1.363685484,1601.318946,-1.362162668,1600.987348,-1.360641351,1600.768776,-1.359121761,1600.337417,-1.357603817,22,80,0.11548,140)
predict(1591.76927,-1.309621803,1592.829559,-1.308159488,1593.341215,-1.306698678,1593.018527,-1.305239306,1592.808854,-1.303781589,1592.386385,-1.302325452,22,80,0.11548,140)
predict(1585.35555,-1.265031369,1586.423,-1.263618844,1586.941811,-1.262207772,1586.62627,-1.260798089,1586.423736,-1.259390005,1586.008398,-1.257983447,22,80,0.11548,140)
predict(1585.35555,-1.265031369,1586.423,-1.263618844,1586.941811,-1.262207772,1586.62627,-1.260798089,1586.423736,-1.259390005,1586.008398,-1.257983447,22,80,0.11548,140)
predict(1574.356158,-1.18855973,1575.43589,-1.187232592,1575.966969,-1.18590682,1575.663686,-1.184582353,1575.473395,-1.183259388,1575.070287,-1.181937857,22,80,0.11548,140)
predict(1568.535325,-1.148091257,1569.621557,-1.146809306,1570.159129,-1.145528674,1569.862332,-1.144249303,1569.67852,-1.142971383,1569.281885,-1.141694848,22,80,0.11548,140)

