x=zeros(10802,1);
y=zeros(10802,1);
for i=2:10802
    if class1(i)==0
        x(i)=CO2_Zone_6(i);
    elseif class1(i)==1
        y(i)=CO2_Zone_6(i);
    end
end

%figure('Name','CO2 training dataset and expected control actions');
subplot(3,1,1)
plot(CO2_Zone_6)
%title('Mean CO2 lvl across the Cabin')
subplot(3,1,2)
plot(x)
%title('Rate of change of CO2 ')
subplot(3,1,3)
plot(y)
%title('Criticality levels in the training dataset')

        