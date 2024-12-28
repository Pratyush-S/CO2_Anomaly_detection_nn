subplot(3,1,1)
plot(CO2_Zone_1)
subplot(3,1,2)
plot(CO2_dt_mean)
subplot(3,1,3)
plot(x)


for i=2:10802
    if cric(i)=="x"
        if CO2_dt_mean(i)>=0.6
        x(i)=2;;
        elseif CO2_dt_mean(i)<0.6
        x(i)=1;
        end
    else
       x(i)=0;
    end
end

figure('Name','CO2 training dataset and expected control actions');
subplot(3,1,1)
plot(CO2_Zone_1)
title('Mean CO2 lvl across the Cabin')
subplot(3,1,2)
plot(CO2_dt_mean)
title('Rate of change of CO2 ')
subplot(3,1,3)
plot(x)
title('Criticality levels in the training dataset')

