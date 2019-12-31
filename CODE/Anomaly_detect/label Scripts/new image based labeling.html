import pandas as pd
import matplotlib.pyplot as plt

address="D:\PS!\dg models\Data_generation\diffusion added\raw_data\C2\2_10\C2_P10_D240_TS.xlsx"
dataset1 = pd.read_excel(r'D:\PS!\dg models\Data_generation\diffusion added\raw_data\C2\2_10\C2_P10_D240_TS.xlsx')
col=dataset1.columns

a_class = int(input("Enter the anomaly type:") )
while a_class>3 or a_class<0:
    print('in')
    a_class = int(input("Invalid input, Enter the anomaly type:") )

print("------------anomaly"+str(a_class)+"-----------")

val = int(input("Enter the number of class "+str(a_class)+" anomalies?")) 
print(val)


while val>4:
       val = int(input("Invalid input, Enter the number of class 2 anomalies?") )

if a_class>1
a=[[0]*3]*val
edit_flag=1

while edit_flag==1:
    for i in range(0,val):
        x=int(input("Enter start  times:")) 
        y=int(input("Enter end  times:")) 
        z=int(input("Zone:")) 
        
        a[i]=[x*60,y*60,z]
        
    edit_flag=2
    print(a)
    while edit_flag==2:
        edit_flag = input("Confirm timings?[y:n]")
        if edit_flag=="y":
            edit_flag=0            
        elif edit_flag=="n":
            edit_flag=1
        else:
            print("Invalid input")
            edit_flag=2
 
    
dataset1['class_0']=1
dataset1['class_1']=0
dataset1['class_2']=0
dataset1['class_3']=0
dataset1['class_4']=0

b=[0]*len(dataset1)   

for i in range(0,val):   
    print(i)
    
    dataset1['class_2'][a[i][0]-1:a[i][1]]=1
    
    for j in range(len(dataset1)):        
        if dataset1[col[2*a[i][2]-1]][j]<-0.7:
            b[j]=1  
            
dataset1['class_4']=b
    
  
 
dataset1['class_0']=dataset1['class_0']-dataset1['class_4']-dataset1['class_2']-dataset1['class_3']-dataset1['class_1']



plt.plot(dataset1[['CO2_Zone_1', 'CO2_Zone_2']])

plt.plot(dataset1[['dz1','dz2','dz3','dz4','dz5','dz6']])

plt.plot(dataset1[ 'class_0'])

plt.plot(dataset1[ 'class_1'])

plt.plot(dataset1[ 'class_2'])

plt.plot(dataset1['class_4'])

sum=dataset1['class_0']+dataset1['class_4']+dataset1['class_2']

plt.plot(sum)

dataset1.to_excel(r'D:\PS!\dg models\Data_generation\diffusion added\raw_data\final training data\labeled.xlsx')
