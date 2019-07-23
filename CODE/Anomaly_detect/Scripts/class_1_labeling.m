

CO2_Zone_1(1)=[];
CO2_Zone_2(1)=[];
CO2_Zone_3(1)=[];
CO2_Zone_4(1)=[];
CO2_Zone_5(1)=[];
CO2_Zone_6(1)=[];
tailNo(1)=[];
flightNo(1)=[];

k=0;
for i = 1:240
    for j = 1:60
        k=k+1;
        temp(k)=t_sim(i);
        press(k)=p_sim(i);
        humid(k)=h_sim(i);
        pass(k)=pas_cnt(i);
        
    end
end
humid_f= humid.';
temp_f=temp.';
press_f=press.';
pass_f=pass.';

final_array=cat(2,CO2_Zone_1,CO2_Zone_2,CO2_Zone_3,CO2_Zone_4,CO2_Zone_5,CO2_Zone_6,temp_f,press_f,humid_f,pass_f);

final_table=array2table(final_array,'VariableNames',{'CO2_Zone_1','CO2_Zone_2','CO2_Zone_3','CO2_Zone_4','CO2_Zone_5','CO2_Zone_6','temp_f','press_f','humid_f','pass_f'});

file_name=strcat('dataset_4hr_pascnt_',num2str(pass_f(4)),'.xlsx');
writetable(final_table,file_name);


clear all;

