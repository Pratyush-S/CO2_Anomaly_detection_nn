%CO2_Zone_1(1)=[];
%CO2_Zone_2(1)=[];
CO2_Zone_3(1)=[];
CO2_Zone_4(1)=[];
CO2_Zone_5(1)=[];
CO2_Zone_6(1)=[];
tailNo(1)=[];
flightNo(1)=[];

k=0;
for i = 1:30
    for j = 1:60
        k=k+1;
        temp(k)=t_sim(i);
        press(k)=t_sim(i);
        humid(k)=h_sim(i);
        pass(k)=pas_cnt(i);
        
    end
end
humid_f= humid.';
temp_f=temp.';
press_f=press.';
pass_f=pass.';
