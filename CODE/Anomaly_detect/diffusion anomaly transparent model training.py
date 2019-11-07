from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np






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

dataset23 = pd.read_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\final training data\C3_P50_labeled.xlsx')

dataset=dataset20

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

dataset = dataset.reset_index(drop=True)
col=dataset.columns.values
dataset=dataset.drop(col[0],axis=1)
col3=dataset.columns.values



#creating datasets for input output for training and validation
X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]
print("Unnormalized Data", "\n", X_complete[:5], "\n")
print("Unnormalized Data", "\n", y_complete[:5], "\n")

# Feature scaling according to training set data 
col=list(X_complete.columns.values)


scaler = StandardScaler()
X_complete = scaler.fit_transform(X_complete)





class_total=dataset['class_0']*1+dataset['class_1']*2+dataset['class_2']*3+dataset['class_3']*4+dataset['class_4']*5

    

print(class_total.max())
print(class_total.min())


# covert to array for processing
X_complete=X_complete.values
#One hot encoding
#_complete= pd.get_dummies(y_complete).values
y_complete=y_complete.values
class_total=class_total.values

y_complete=class_total
# Creating a Train and a Test Dataset

X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=seed)


model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial')


fitted_model=model.fit(X_train, y_train)


print( fitted_model.intercept_ )

print( fitted_model.coef_ )



# use the model to make predictions with the test data
y_pred = model.predict(X_test)

y_pred = model.predict(X_complete)

plt.plot(y_pred)
plt.plot(y_complete)
# how did our model perform?

len(y_complete)
len(y_pred)

count=0

for i in range(len(y_pred)):
    print(i)
    if(y_complete[i]!=y_pred[i]):
        count=count+1
        
print(count*100/len(y_complete))    
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.4f}'.format(accuracy))


in_data=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
in_data['preds']=y_pred
in_data.to_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\temp file\predictions_23.xlsx')









