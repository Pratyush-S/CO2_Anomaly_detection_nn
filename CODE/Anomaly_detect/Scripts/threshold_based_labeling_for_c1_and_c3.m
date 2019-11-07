

x=10800;
y=60+x/60 ;

class_3=zeros(x,1);
class_4=zeros(x,1);
class_0=zeros(x,1);
class_1=zeros(x,1);
class_2=zeros(x,1);
humid_f=zeros(x,1);
press_f=zeros(x,1);
pass_f=zeros(x,1);
temp_f=zeros(x,1);


CO2_Zone_1(1)=[];
CO2_Zone_2(1)=[];
CO2_Zone_3(1)=[];
CO2_Zone_4(1)=[];
CO2_Zone_5(1)=[];
CO2_Zone_6(1)=[];


z1=cat(1,300,CO2_Zone_1);
z2=cat(1,300,CO2_Zone_2);
z3=cat(1,300,CO2_Zone_3);
z4=cat(1,300,CO2_Zone_4);
z5=cat(1,300,CO2_Zone_5);
z6=cat(1,300,CO2_Zone_6);

z1(x)=[];
z2(x)=[];
z3(x)=[];
z4(x)=[];
z5(x)=[];
z6(x)=[];




dz1=CO2_Zone_1-z1;
dz2=CO2_Zone_2-z2;
dz3=CO2_Zone_3-z3;
dz4=CO2_Zone_4-z4;
dz5=CO2_Zone_5-z5;
dz6=CO2_Zone_6-z6;

average_slope=(dz1+dz2+dz3+dz4+dz5+dz6)/6;
average_value=(CO2_Zone_1+CO2_Zone_2+CO2_Zone_3+CO2_Zone_4+CO2_Zone_5+CO2_Zone_6)/6;

k=0;
for i = 1:(x/60)
    for j = 1:60
        k=k+1;
        temp_f(k)=double(string(t_sim(i)));
        press_f(k)=double(string(p_sim(i)));
        humid_f(k)=h_sim(i);
        pass_f(k)=double(string(pas_cnt(i)));
        
    end
end




    

    
%final_array=cat(2,CO2_Zone_1,dz1,CO2_Zone_2,dz2,CO2_Zone_3,dz3,CO2_Zone_4,dz4,CO2_Zone_5,dz5,CO2_Zone_6,dz6,class_0,class_1,class_2,class_3,class_4);
final_array=cat(2,CO2_Zone_1,dz1,CO2_Zone_2,dz2,CO2_Zone_3,dz3,CO2_Zone_4,dz4,CO2_Zone_5,dz5,CO2_Zone_6,dz6,temp_f,press_f,humid_f,pass_f,class_0,class_1,class_2,class_3,class_4);

%final_table=array2table(final_array,'VariableNames',{'CO2_Zone_1','dz1','CO2_Zone_2','dz2','CO2_Zone_3','dz3','CO2_Zone_4','dz4','CO2_Zone_5','dz5','CO2_Zone_6','dz6','class_0','class_1','class_2','class_3','class_4'});
final_table=array2table(final_array,'VariableNames',{'CO2_Zone_1','dz1','CO2_Zone_2','dz2','CO2_Zone_3','dz3','CO2_Zone_4','dz4','CO2_Zone_5','dz5','CO2_Zone_6','dz6','temp_f','press_f','humid_f','pass_f','class_0','class_1','class_2','class_3','class_4'});

%file_name=strcat('TS_C1_P_set_3hr_pascnt_10.xlsx');
file_name=strcat('C1_P',num2str(pass_f(12)),'_D',num2str(y),'_TS.xlsx');

%%  
writetable(final_table,file_name)


%%
clear all
clc
