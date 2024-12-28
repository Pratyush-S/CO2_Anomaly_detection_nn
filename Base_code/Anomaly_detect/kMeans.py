'''#importing libraries
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


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
 99.6,
 -18.337950864380463,
 0.7688135236768454,
 3.964536516299539,
 3.504219275105078,
 3.313538986878003,
 6.786842562421079,
 2.438972222222222]

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
 56.82824973050747,
 228.03556365288256,
 210.59013071227005,
 134.61469646704273,
 163.46972110972752,
 206.7182081562485,
 150.23821049087152,
 1.675712586365308]
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
'''
dataset=dataset3

dataset =dataset.reset_index(drop=True)
col=dataset.columns.values
dataset=dataset.drop(col[0],axis=1)

#############################################################################################################################################################################


#############################################################################################################################################################################


all_dev=dataset[['dz1','dz2','dz3','dz4','dz5','dz6']].abs()

X_complete=all_dev


#############################################################################################################################################################################
'''
# Feature scaling according to training set data 
col=list(X_complete.columns.values)


for i in col:#[:]:
    print(i)
    avg=X_complete[str(i)].mean()
    sd=X_complete[str(i)].std()
    X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))
    print(avg)
    print(sd)
    print(i)
'''
#############################################################################################


    
#############################################################################################

  

    
print("Normalized Data\n", X_complete[:5], "\n")

# covert to array for processing
X_complete=X_complete.values



km = KMeans(n_clusters=3,algorithm = 'auto',random_state=seed) # You want cluster the passenger records into 2: Survived or Not survived
km.fit(X_complete)


y_km = km.fit_predict(X_complete)
X=X_complete



plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s',
    label='cluster 1'
)

plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='brown',
    marker='o',
    label='cluster 2'
)
plt.scatter(
    X[y_km == 2, 0], X[y_km == 2, 1],
    s=50, c='red',
    marker='o',
    label='cluster 3'
)


# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()


plt.subplot(211)
plt.plot(dataset['CO2_Zone_1'])
plt.subplot(212)
plt.plot(y_km)

                














