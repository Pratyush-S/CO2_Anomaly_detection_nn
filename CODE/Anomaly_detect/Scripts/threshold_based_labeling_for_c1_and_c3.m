%x=14400;
x=7200;
th_0=310;%%1
th_47=750;%%
th_93=1365;%%
th_140=1565;%%
th_186=1980;
th_20=490;
th_70=940;
th_160=1850;
th_170=1920;

hm_0=0.0012612;
hm_47=0.03943;
hm_93=0.10000;
hm_140=0.11548;
hm_186=0.15176;
hm_20=0.01760;
hm_70=0.058074;
hm_163=0.13323;
hm_160=0.13081;
hm_170=0.13887;



class_3=zeros(x,1);
class_4=zeros(x,1);
class_0=zeros(x,1);
class_1=zeros(x,1);
class_2=zeros(x,1);
humid_f=zeros(x,1);
press_f=zeros(x,1);
pass_f=zeros(x,1);
temp_f=zeros(x,1);

class_3=zeros(x,1);
class_4=zeros(x,1);
class_0=zeros(x,1);
class_1=zeros(x,1);
class_2=zeros(x,1);
humid_f=zeros(x,1);
press_f=zeros(x,1);
pass_f=zeros(x,1);
temp_f=zeros(x,1);




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
        temp_f(k)=t_sim(i);
        press_f(k)=p_sim(i);
        humid_f(k)=h_sim(i);
        pass_f(k)=pas_cnt(i);
        
    end
end


for i=1:x

if humid_f(i)>hm_160
   class_1(i)=1;
elseif average_value(i)>th_160
    if average_slope(i)<0
        class_4(i)=1;
    end
else 
    class_0(i)=1;
    

end
end

a=class_0+class_1+class_4;

    subplot(6,1,1)
    plot(average_value);
    title("average value");
    subplot(6,1,2)
    plot(average_slope);
    title("Average slope");
    subplot(6,1,3)
    plot(humid_f);
    title("humid");
    subplot(6,1,4)
    plot(class_0);
    title("class 0");
    subplot(6,1,5)
    plot(class_1);
    title("class_1");
    subplot(6,1,6)
    plot(class_4);
    title("class_4");
    

    
%final_array=cat(2,CO2_Zone_1,dz1,CO2_Zone_2,dz2,CO2_Zone_3,dz3,CO2_Zone_4,dz4,CO2_Zone_5,dz5,CO2_Zone_6,dz6,class_0,class_1,class_2,class_3,class_4);
final_array=cat(2,CO2_Zone_1,dz1,CO2_Zone_2,dz2,CO2_Zone_3,dz3,CO2_Zone_4,dz4,CO2_Zone_5,dz5,CO2_Zone_6,dz6,temp_f,press_f,humid_f,pass_f,class_0,class_1,class_2,class_3,class_4);

%final_table=array2table(final_array,'VariableNames',{'CO2_Zone_1','dz1','CO2_Zone_2','dz2','CO2_Zone_3','dz3','CO2_Zone_4','dz4','CO2_Zone_5','dz5','CO2_Zone_6','dz6','class_0','class_1','class_2','class_3','class_4'});
final_table=array2table(final_array,'VariableNames',{'CO2_Zone_1','dz1','CO2_Zone_2','dz2','CO2_Zone_3','dz3','CO2_Zone_4','dz4','CO2_Zone_5','dz5','CO2_Zone_6','dz6','temp_f','press_f','humid_f','pass_f','class_0','class_1','class_2','class_3','class_4'});

file_name=strcat('training_set_4hr_pascnt_160.xlsx');
%file_name=strcat('training_set_4hr_pascnt_',num2str(20),'.xlsx');
writetable(final_table,file_name);

