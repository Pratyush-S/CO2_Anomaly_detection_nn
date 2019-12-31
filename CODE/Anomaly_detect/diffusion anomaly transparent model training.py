from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math  



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

###############################################################################################################################

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


#Combine datasets into one single data file
frames=[dataset1, dataset2, dataset3, dataset4, dataset5, dataset6, dataset7, dataset8, dataset9, dataset10, dataset11, dataset12, dataset13, dataset14, dataset15, dataset16, dataset17, dataset18, dataset19, dataset20, dataset21, dataset22, dataset23, dataset24, dataset25, dataset26, dataset27, dataset28, dataset29, dataset30]
dataset = pd.concat(frames)

dataset=dataset10

dataset=process_raw(dataset)

train_linear_model(dataset)

load_model()

format_data_for_pred(dataset)

pred()
###############################################################################################################################
def process_raw(dataset):
    avg_val=(dataset['CO2_Zone_1']+dataset['CO2_Zone_2']+dataset['CO2_Zone_3']+dataset['CO2_Zone_4']+dataset['CO2_Zone_5']+dataset['CO2_Zone_6'])/6
    #avg_slope=(dataset['dz1']+dataset['dz2']+dataset['dz3']+dataset['dz4']+dataset['dz5']+dataset['dz6'])/6
    #assigning different values to each output class
    diffz1=pd.Series((avg_val-dataset['CO2_Zone_1']))
    diffz2=pd.DataFrame((avg_val-dataset['CO2_Zone_2']))
    diffz3=pd.DataFrame((avg_val-dataset['CO2_Zone_3']))
    diffz4=pd.DataFrame((avg_val-dataset['CO2_Zone_4']))
    diffz5=pd.DataFrame((avg_val-dataset['CO2_Zone_5']))
    diffz6=pd.DataFrame((avg_val-dataset['CO2_Zone_6']))
    
   
  
  
    #creating dataframe for deviation from mean
    frames2=[diffz1,diffz2,diffz3,diffz4,diffz5,diffz6]
    dataset_2=pd.concat(frames2, axis=1)
    dataset_2.columns=['dif1','dif2','dif3','dif4','dif5','dif6']
    dataset=pd.concat([dataset,dataset_2],axis=1)
#    dataset['humid_f']=dataset['humid_f']*10

    #  Shuffle Data
    #The frac keyword argument specifies the fraction of rows to return in the random sample
    #so frac=1 means return all rows (in random order)
    #https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
    
    dataset = dataset.reset_index(drop=True)
    col=dataset.columns.values
    dataset=dataset.drop(col[0],axis=1)
  
    
    return dataset

###############################################################################################################################
def x_y_split(dataset):   
    global avg_array
    global sd_array
    global X_complete
    global y_complete
    
    #creating datasets for input output for training and validation
    X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
    y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]
    #print("Unnormalized Data", "\n", X_complete[:5], "\n")
    #print("Unnormalized Data", "\n", y_complete[:5], "\n")
    
    # Feature scaling according to training set data 
    col=list(X_complete.columns.values)
    
    #Normalisation
    avg_array=[]
    sd_array=[]
    for i in col:
     #   print(i)
        avg=X_complete[str(i)].mean()
        sd=X_complete[str(i)].std()
       
        avg_array.append(avg)
        sd_array.append(sd)
        
        X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg)/(sd))
        #print(avg)
        #print(sd)
        #print(i)
        



    class_total=dataset['class_0']*1+dataset['class_1']*2+dataset['class_2']*3+dataset['class_3']*4+dataset['class_4']*5   

    # covert to array for processing
    
    #One hot encoding
    #_complete= pd.get_dummies(y_complete).values
    y_complete=class_total.values
    X_complete=X_complete.values
    
   
###############################################################################################################################

def train_linear_model(dataset):
    global X_complete
    global y_complete
    x_y_split(dataset)
    
    # Creating a Train and a Test Dataset
    X_train, X_test, y_train, y_test = train_test_split(X_complete, y_complete, test_size=0.3, random_state=seed)
    
    model = LogisticRegression(solver = 'lbfgs',multi_class='multinomial')
    fitted_model=model.fit(X_train, y_train)
    
    #saving model
    filename = 'linear_model.sav'
    pickle.dump(fitted_model, open(filename, 'wb'))
    
   # print( fitted_model.intercept_ )
   # print( fitted_model.coef_ )
    print("__________MODEL TRAINED__________")
    
    # load the model from disk
    #model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)

###############################################################################################################################
   
   
def format_data_for_pred(dataset):   
    global X_complete
    global y_complete
    #creating datasets for input output for training and validation
    X_complete=dataset.drop(['class_0','class_1','class_2','class_3','class_4','temp_f','press_f'],axis=1)
    y_complete=dataset[['class_0','class_1','class_2','class_3','class_4']]  

    # Feature scaling according to training set data 
    col=list(X_complete.columns.values)    
    #Normalisation
    j=0
    for i in col:
        print(j)       
        X_complete[str(i)]=X_complete[str(i)].apply(lambda X:(X-avg_array[j])/(sd_array[j]))
        j=j+1
        



    class_total=dataset['class_0']*1+dataset['class_1']*2+dataset['class_2']*3+dataset['class_3']*4+dataset['class_4']*5
    # covert to array for processing
    
    #One hot encoding
    #_complete= pd.get_dummies(y_complete).values
    y_complete=class_total.values
    X_complete=X_complete.values
  
###############################################################################################################################

def load_model():
    global linear_model
    filename = 'linear_model.sav'
    linear_model = pickle.load(open(filename, 'rb'))
    print("__________MODEL LOADED__________")
###############################################################################################################################
    
# use the model to make predictions with the test data

def pred():
    y_pred = linear_model.predict(X_complete)

    #plt.plot(y_pred)
    plt.plot(y_complete)
# how did our model perform?

    count=0

    for i in range(len(y_pred)):
        #print(i)
        if(y_complete[i]!=y_pred[i]):
            count=count+1
        
    print((len(y_complete)-count)*100/len(y_complete))

   
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)
print('Accuracy: {:.4f}'.format(accuracy))


in_data=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)
in_data['preds']=y_pred
in_data.to_excel(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\temp file\predictions_23.xlsx')





in_data=dataset.drop(['class_0','class_1','class_2','class_3','class_4'],axis=1)

a=pd.concat([in_data,class_total],axis=1)

column_index=['CO2_Zone_1', 'dz1', 'CO2_Zone_2', 'dz2', 'CO2_Zone_3', 'dz3',
       'CO2_Zone_4', 'dz4', 'CO2_Zone_5', 'dz5', 'CO2_Zone_6', 'dz6', 'temp_f',
       'press_f', 'humid_f', 'pass_f','label']

a.columns=column_index

a.to_excel(r'D:\PS!\Data_Management\Integrated prod\raw.xlsx')






