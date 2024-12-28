import pandas as pd
import time
import pickle

T= int(time.time())-14400 

original=pd.read_excel('original.xlsx')
stored1=pd.read_excel('compressed.xlsx')


######################################################################
#discarding garbage column
col_original=original.columns
original=original.drop(col_original[0],axis=1)
col_original=original.columns
#discarding garbage column
col_stored=stored1.columns
stored=stored1.drop(col_stored[0],axis=1)
col_stored=stored.columns
######################################################################
#zone1_original[ *write index here*]
zone1_original = pd.Series(original[col_original[1]]).tolist()
#zone1_stored=stored[[col_stored[0],col_stored[1]]]
#zone1_stored = zone1_stored[zone1_stored['CO2_Zone_1'].notnull()]

zone1_stored = pd.Series(stored[col_stored[0]])

zone1_stored.index = stored[col_stored[1]]  
zone1_stored=zone1_stored.dropna()


stored_index=stored[col_stored[1]].dropna()  
stored_index = stored_index.reset_index(drop=True)





temp=stored[col_stored[0]].dropna()  
temp = temp.reset_index(drop=True)



######################################################################

compressed=temp.tolist()

compressed_index=stored_index.tolist()




uncompressed=original[col_original[1]].tolist()

uncompressed[800:10800]

f=open('a.txt','a+')
k=0
f.write('[')
for i in uncompressed:
    k=k+1
    print(k)
    f.write(str(i)+',')
f.write(']')    
f.close()

f=open('a.txt','a+')



######################################################################
def get_original(index): 
    index=index-1
    value=zone1_original[index]
    
   # print(index)
    index=index+T+1  
    #print(index)

    return value,index

def get_stored(index):
    index=index-1
    value=zone1_stored[stored_index[index]]
    
      
    index=stored_index[index]+T 
    #print(index)
    return value,index


######################################################################
 
def functiontodatabase_o(c1,ts):
 
    lines = "C_final" + ",type=C_test" + " " + "C1=" + str(c1) + " "+str(ts)+"000000000"
    thefile = open(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\temp file\uncompressed.txt', 'a+')
    thefile.write(lines)
    thefile.write("\n")
    thefile.close()
    #flag = 0       
   
def functiontodatabase_s(c1,ts):
 
    lines = "C_final" + ",type=C_test" + " " + "C1=" + str(c1) + " "+str(ts)+"000000000"
    thefile = open(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\temp file\compressed.txt', 'a+')
    thefile.write(lines)
    thefile.write("\n")
    thefile.close()
    #flag = 0       

    
######################################################################
    
for i in range(1,2):
   # print(i)
    c1,ts=get_stored(i)
    functiontodatabase_s(c1,ts)

   
for i in range(1,2):
   # print(i)
    c1,ts=get_original(i)
    functiontodatabase_o(c1,ts)

'''
for i in range(1,len(zone1_stored)+1):
   # print(i)
    c1,ts=get_stored(i)
    functiontodatabase_s(c1,ts)

   
for i in range(1,len(zone1_original)+1):
   # print(i)
    c1,ts=get_original(i)
    functiontodatabase_o(c1,ts)

###################################################################### 
'''



import pickle

filename = 'x.sav'
pickle.dump(stored_index, open(filename, 'wb'))

filename = 'y.sav'
pickle.dump(zone1_original, open(filename, 'wb'))


filename = 'z.sav'
pickle.dump(zone1_stored, open(filename, 'wb'))




stored_index = pickle.load(open( 'x.sav', 'rb'))
zone1_original = pickle.load(open( 'y.sav', 'rb'))
zone1_stored = pickle.load(open( 'z.sav', 'rb'))



import numpy as geek 
  

  
# the array is saved in the file geekfile.npy  
geek.save('geekfile', original) 
  
print("the array is saved in the file geekfile.npy") 
  


b = geek.load('geekfile.npy') 





with open(r'D:\PS!\CO2_Anomaly_detection\CODE\Anomaly_detect\temp file\file.csv','rt')as f:
    data = csv.reader(f)


import csv
with open('file.csv','rt')as f:
  data = csv.reader(f)
  print(data)
  a=data
  



a=