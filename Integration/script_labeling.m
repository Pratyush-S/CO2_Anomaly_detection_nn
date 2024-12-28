
class2_z1=zeros(3600,1);
class2_z2=zeros(3600,1);
class2_z3=zeros(3600,1);
class2_z4=zeros(3600,1);
class2_z5=zeros(3600,1);
class2_z6=zeros(3600,1);
flag1=0;
flag2=0;
flag3=0;
flag4=0;
flag5=0;
flag6=0;

for i=1:3599
   slopea(i)=a(i+1)-a(i);
   slopeb(i)=b(i+1)-b(i);
   slopec(i)=c(i+1)-c(i);
   sloped(i)=d(i+1)-d(i);
   slopee(i)=e(i+1)-e(i);
   slopef(i)=f(i+1)-f(i);
   
    if (slopea(i)>20)
        flag1=1;
    elseif (slopea(i)<-20)
        flag1=0;
    end
    
     if (slopeb(i)>20)
        flag2=1;
    elseif (slopeb(i)<-20)
        flag2=0;
     end
    
      if (slopec(i)>20)
        flag3=1;
    elseif (slopec(i)<-20)
        flag3=0;
      end
    
    if (sloped(i)>20)
        flag4=1;
    elseif (sloped(i)<-20)
        flag4=0;
    end
    
     if (slopee(i)>20)
        flag5=1;
    elseif (slopee(i)<-20)
        flag5=0;
     end
    
      if (slopef(i)>20)
        flag6=1;
    elseif (slopef(i)<-20)
        flag6=0;
      end
    
    class2_z1(i)=flag1;
    class2_z2(i)=flag2;
    class2_z3(i)=flag3;
    class2_z4(i)=flag4;
    class2_z5(i)=flag5;
    class2_z6(i)=flag6;
    class2(i)= class2_z1(i)+class2_z2(i)+class2_z3(i)+class2_z4(i)+class2_z5(i)+class2_z6(i);
    if(class2(i)>0)
        class2(i)=1;
    end
end




subplot(3,2,1);
plot(class2_z1)
subplot(3,2,2);
plot(class2_z2)
subplot(3,2,3);
plot(class2_z3)
subplot(3,2,4);
plot(class2_z4)
subplot(3,2,5);
plot(class2_z5)
subplot(3,2,6);
plot(class2)


