import pandas as pd
from keras.models import model_from_json
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt
import pickle
#in BYTES
total_size=40000#10485760#(10mb)
space_consumed=0#default
space_array=[]
#sample_dataset(x) to compresss x
column_index=['index', 'CO2_Zone_1', 'dz1', 'CO2_Zone_2', 'dz2', 'CO2_Zone_3', 'dz3',
       'CO2_Zone_4', 'dz4', 'CO2_Zone_5', 'dz5', 'CO2_Zone_6', 'dz6', 'temp_f',
       'press_f', 'humid_f', 'pass_f']
###############################################################################################################################
compression_array=[]
error_array=[]
cost_array=[]

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

################################################################################################################################


def process_list(a):
    global space_consumed
    
    a=pd.DataFrame(a)
    X_data=pd.DataFrame()
    col=a.columns

    #creating rate change columns and inserting them within the code
    for j in range(6):
        print(j)
        #one shifter vale 234
        aa=a[col[j]][1:].values.tolist()
        #adding repeated values at end
        aa.append(aa[-1])
        #original column
        aaa=a[col[j]].values.tolist()
        #difference between shifted and original to get the change btw subsequent values
        aaaa=[aa[i]-aaa[i] for i in range(len(aa))]
    
        #joining columns and their rates as a dataframe and joining them to a parent dataframe
        new=pd.DataFrame(aaaa)
        frames2=[X_data,a[col[j]],new]
        X_data=pd.concat(frames2, axis=1)

    #adding the TPH and passenger count columns    
    frames2=[X_data,a[col[6:]]]
    X_data=pd.concat(frames2, axis=1)
    X_data.columns=column_index[1:]
    
    #adding index field for compression
    X_data = X_data.reset_index(drop=False)

    #converting allto string for NN
    input_data=X_data.values.tolist()
    
    
    anomaly=[]
    severity=[]   
    
    for k in range(len(input_data)):    
            #skipping the fisrt field of index
            x,y=predict_2(input_data[k][1:])
            anomaly.append(x)
            severity.append(y)
    print("Predictions Done")    
            
            
    plt.subplot(311)
    plt.title('Dataset')
    plt.plot(dataset[['CO2_Zone_1', 'CO2_Zone_2', 'CO2_Zone_3','CO2_Zone_4', 'CO2_Zone_5', 'CO2_Zone_6']])
  
    plt.subplot(312)
    plt.title('Severity')
    plt.plot(severity)
   
    plt.subplot(313)
    plt.title('Anomaly pred')
    plt.plot(anomaly)
    
    
    prev=severity[0]
    subwindow_beginings=[]
    
#STORING INDEX WHERE WINDOW HAS TO BE BROKEN
    subwindow_beginings.append(0)
    for l in range(1,len(severity)-1):
        if severity[l]!=prev:       
            prev=severity[l]
            subwindow_beginings.append(l)
    subwindow_beginings.append(len(severity)-1)


    data_output=pd.DataFrame()
    #sending severity windows for compression
    i=0
    print('--------------------------------------------')
    while i<len(subwindow_beginings)-1:
    
        start=subwindow_beginings[i]
        print('Starting index:'+str(start))
        end=subwindow_beginings[i+1]
        print('ending index:'+str(end))
        s=severity[subwindow_beginings[i]]
        print('severity :'+str(s))
     
        batch_data=pd.DataFrame(input_data[start:end])
        
        batch_data.columns=column_index
        
        batch_data=batch_data.drop(['dz1','dz2','dz3','dz4','dz5','dz6'],axis=1)
        print('----------Starting batch compression----------')
    
        i=i+1
    
        temp=sample_dataset(batch_data,s)
        temp.columns
        data_output=pd.concat([data_output,temp],axis=0)
        
        space_consumed=data_output.memory_usage(index=True).sum()
        print('space consumed :'+str(space_consumed))
    return data_output
    





###############################################################################################################################

def loadmodel_1():
    global model_anomaly
    global model_severity    
    
    # Model reconstruction from JSON file

    json_file = open('diffusion_anomaly_detect.json', 'r')
    loaded_model_json = json_file.read()
    
    json_file.close()
    model_anomaly = model_from_json(loaded_model_json)
    model_anomaly.load_weights('diffusion_anomaly_detect.h5')
    model_anomaly.compile(Adam(lr=0.0001),'categorical_crossentropy',metrics=['accuracy'])
    print('Anomaly model loaded')
    
        # Model reconstruction from JSON file
    json_file = open('diffusion_severity_detect2.json', 'r')
    loaded_model_json = json_file.read()    
    json_file.close()
    model_severity = model_from_json(loaded_model_json)
    model_severity.load_weights('diffusion_severity_detect2.h5')
    model_severity.compile(Adam(lr=0.0001),'categorical_crossentropy',metrics=['accuracy'])
    print('Severity model loaded')

def loadmodel_2():
    global model_anomaly
    global model_severity    
    
    # Model reconstruction from pickel file

    filename_e = 'clf_entropy_model'
    infile = open(filename_e,'rb')
    model_anomaly = pickle.load(infile)
    infile.close()
    print('Anomaly model loaded')
    
        # Model reconstruction from JSON file
    json_file = open('diffusion_severity_detect2.json', 'r')
    loaded_model_json = json_file.read()    
    json_file.close()
    model_severity = model_from_json(loaded_model_json)
    model_severity.load_weights('diffusion_severity_detect2.h5')
    model_severity.compile(Adam(lr=0.0001),'categorical_crossentropy',metrics=['accuracy'])
    print('Severity model loaded')

###############################################################################################################################    
#def predict(co1,dz1,co2,dz2,co3,dz3,co4,dz4,co5,dz5,co6,dz6,T,Pcabin,H,P):
def predict_1(raw_data):

    #normalising
    for i in range(len(raw_data)):
        raw_data[i]=(raw_data[i]-avg_array[i])/(sd_array[i])
    
   
        
    input_1=np.array([raw_data]) 
    
    y_pred = model_anomaly.predict(input_1)
    y_pred_class = int(np.argmax(y_pred,axis=1))
    normalised_class=y_pred_class+1

    raw_data.append(normalised_class)
    
    input_2=np.array([raw_data])
    y_pred2 = model_severity.predict(input_2)
    y_pred_severity = int(np.argmax(y_pred2,axis=1))
    
    #print("Anomaly class :    "+str(y_pred_class))
   # print("Severity level:    "+str(y_pred_severity))
    
    #print("-----------------------------------------------------")
    
    return y_pred_class,y_pred_severity

def predict_2(raw_data):
     
    #normalising
    raw_data_s=[]
    for i in range(len(raw_data)):
        raw_data_s.append((raw_data[i]-avg_array[i])/(sd_array[i]))
        
    input_1=np.array([raw_data]) 
    
    y_pred = model_anomaly.predict(input_1)
    normalised_class=y_pred[0]

    raw_data_s.append(normalised_class)
    
    input_2=np.array([raw_data_s])
    y_pred2 = model_severity.predict(input_2)
    y_pred_severity = int(np.argmax(y_pred2,axis=1))
    
    #print("Anomaly class :    "+str(y_pred_class))
   # print("Severity level:    "+str(y_pred_severity))
    
    #print("-----------------------------------------------------")
    
    return y_pred[0],y_pred_severity

###############################################################################################################################

def get_A(s):
    global A
    #class 0
    if s==0:
        A=0.10
    elif s==1:
        #class 1
        A=0.5
    else:
        #class 2
        A=0.95

def get_B1():
    global B
    B=1
    space_array.append(space_consumed)

def get_B():
    global B
    #insert func for file size 
    if space_consumed==0:
        B=0.1
    else:
        B=(total_size-space_consumed)/total_size
    space_array.append(space_consumed)
    

def get_B3():
    global B
    #insert func for file size 
    
    B=(total_size-space_consumed)/total_size
    space_array.append(space_consumed)
    
    
################################################################################################################################
#NORAML FUNCTION
#get dataframe column along with index
def sample_dataset(dataset,s):
  
    #sets cost function coefficient
    get_A(s)
    get_B()
   
    col=dataset.columns.values
    sampled_array= pd.DataFrame()
    sampled_indices= pd.DataFrame()
    
    for j in col[1:]:
        print(j)
        current_column=dataset[[j,'index']]
    #get threshold
        
    #compress
        #for error optimisation using search
        
        err,column_to_array,com=opti_err_sample(current_column)   
        frame=[sampled_array,column_to_array]
        
        sampled_array=pd.concat(frame,axis=1)
        
#EXTRACTING INDICES OF THE SAMPLED DATA
    print('----------Extracting indices----------')
   
    
    
  
    col_index=sampled_array.columns
    i=0
    
    while i<len(col_index):
        i=i+2
        #print(col_index[i-1])
        sampled_indices=pd.concat([sampled_indices,sampled_array[col_index[i-1]]],axis=1)
            
    #return sampled_array
    return sampled_indices
    
 #save 

##################################################################################################################################### 
    
       
#Error/compression cost function optimisation
#search algorith for global minima
def opti_err_sample(current_column):
    #print('opti_err_sample')
    #Limits in which we impliment binary search
    error_ulimit=10
    error_llimit=0.0
    error=[0,0,0]
    compression=[0,0,0]
    #compression_array=[0] #to store compression ratios
    #error_array=[]          #to store error threshold
    #comp_rate=[0]           #to store the rate of change of compression ratio
            
    data_sampled=[0,0,0]
    #a = datetime.datetime.now()  #timer start
    buffer=0
    flag=0
    
    while flag==0:      #looping conditions for error and count
        #calculation of mid point
        error[1]=(error_ulimit+error_llimit)/2
        error[0]=error[1]*0.5
        error[2]=error[1]*1.5
        
        
        data_sampled[1]=sample_column(current_column,error[1])  #function for sample selection
        data_sampled[0]=sample_column(current_column,error[0])  #function for sample selection
        data_sampled[2]=sample_column(current_column,error[2])  #function for sample selection
        
        #equations for calculation of derived values
        compression[1]=(data_sampled[1].shape[0]/current_column.shape[0])*100
        compression[0]=(data_sampled[0].shape[0]/current_column.shape[0])*100
        compression[2]=(data_sampled[2].shape[0]/current_column.shape[0])*100
        
        
        #diff=compression-compression_array[-1]
        #print('Compression %: '+str("%.3f" % compression)+' %    error thresh:'+str("%.3f" % error_th)+"    diff:" +str("%.3f" %diff))

        
        #Max values for error threshold and copression ratio, used for normalisation
        y_max=100
        x_max=10
        
        #normalised error th and compression ratio
        comp2=[i/y_max for i in compression]
        error2=[i/x_max for i in error]
        
        #COST FUNCTION
        
        cost=(1-(A+B)/2)*np.array(comp2)+(A+B)/2*np.array(error2)               
        
        cost_array.append(cost[1])
        compression_array.append(comp2[1])
        error_array.append(error2[1])
        #cost=(1-A*B)*np.array(comp2)+(A*B)*np.array(error2)               
        
        #print("error: "+str(error))
        #print("compression:"+str(compression))
        #print("cost:"+str(cost))
        #print("-------------------------------------------")
        
        
        if cost[0]<cost[1]:
            error_ulimit=error[1]
            #print("left")
        elif cost[2]<cost[1]:
            error_llimit=error[1]
            #print("right")
        elif ((buffer-cost[1])/buffer)<0.001:
            error_th=error[1]
            flag=1
        buffer=cost[1]
        if compression[1]==compression[0] and compression[1]==compression[2]:
            flag=1
            error_th=error[1]

    error_th=error[1]
    data_sample=data_sampled[1]
    
    
    #print("--------------------Error_th="+str("%.3f"%error_th)+"--------------------")    
  
    return  "%.3f"%error_th, data_sample,compression[1]
   

    
 #FUNCTIONS
#Windowing operation for a column
def sample_column(current_column,error_th):
    data_sampled=current_column[0:0]

         #sampling
    samples=optimum_points(error_th,current_column)
         
         #a=samples.values.tolist()
         #storing
    frames=[data_sampled,samples]
    data_sampled=pd.concat(frames)
         #samples.to_excel(r'D:\PS!\Data_Management\temp1.xlsx')

   
    data_sampled=data_sampled.reset_index(drop=True)
    data_sampled.rename(columns = {'index':'index_'+str(current_column.columns[0])}, inplace = True) 
    
    #print(data_sampled)
    return data_sampled    
        
 
#Selection of optimum points in a given window         
def optimum_points(error_th,current_column):
    #print('optimum_points')
    
    #global error_total
    #data_window=create_data(i,window_len,j)
    error_array=[]
    
    data_window=list(current_column[current_column.columns[0]])
    #data_window=list(current_column[current_column.columns[0]][i:i+window_len])
    samples=current_column[0:0]
    error=0
    #length=i+window_len
    length=current_column.shape[0]
    
    x=0
    y=x+2
    
    y_limit=length
    
    #save(x)
    frames=[samples,current_column[x:x+1]]
    samples=pd.concat(frames)
    #print(samples)
    
    while(y<y_limit):   
    
        #print(y)
        #print(samples)
        if(x<y_limit-1): 
            #print(x)
            data_int=interpolate(data_window[x],data_window[y],x,y)
            diff=np.array(data_window[x:y+1])-np.array(data_int)
            #print(data_int)
            #print(data_window[x:y+1])
            
            error=sum(abs(diff))/len(diff)
            error_array.append(error)
            if(error<error_th):
                #print('in')
                if(y==length-1):
                    #print(123)
                    #save y
                    frames=[samples,current_column[y:y+1]]
                    samples=pd.concat(frames)                
                    x=length            
                    y=x+10
                y=y+1
                
            else:
                #save y-1
                #print('error crossed')
                y=y-1
                frames=[samples,current_column[y:y+1]]
                samples=pd.concat(frames)              
                x=y
                y=x+2
        else:
            #save x,x+1
            #print('at end')
            frames=[samples,current_column[x:x+1]]
            samples=pd.concat(frames)         
            x=x+1                 
            frames=[samples,current_column[x:x+1]]
            samples=pd.concat(frames)         


            x=length
            y=x+10
    #print(samples)
    return samples
            
            
    
#linear interpolation function    
def interpolate(y1,y2,x1,x2): 
    
        slope=(y2-y1)/(x2-x1)
        data_pred=[y1]
        
        for k in range(1,x2-x1):
            data_pred.append((k)*slope+y1)
            
        data_pred.append(y2)
        
        return data_pred
    
############################################################################################



dataset=dataset10

X_complete=dataset[['CO2_Zone_1','CO2_Zone_2','CO2_Zone_3','CO2_Zone_4','CO2_Zone_5','CO2_Zone_6','temp_f','press_f', 'humid_f', 'pass_f']]
a=X_complete.values.tolist()

#X_complete.columns
#FROM HERE

loadmodel_2()



space_consumed=0

total_size=300000#10485760#(10mb)
while total_size>10000:
   
    space_consumed=0#default
    space_array=[]
    plt.plot(space_array)
    
    b=np.transpose(a)
    data_1=process_list(a)
    
    space_array.append(space_consumed)
    plt.plot(space_array)
        
  
    
    y.append(space_array)
    total_size=total_size-5000
    
    
    
    
plt.plot(y[0],label="a")
plt.plot(y[1],label="b")
plt.plot(y[2],label="c")
plt.plot(y[3],label="d")

  
plt.legend(loc="upper left")


################################################################################    
    
################################################################################
