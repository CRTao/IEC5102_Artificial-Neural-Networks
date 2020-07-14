f1 = [0.1,0.1,0.1,0.1]; 
g1 = [0.1,0.1,0.1,0.1]; 

%% toolbox
toolbox_1D_Convolution = conv(f1,g1)

%% Self-programming
m=length(f1);
n=length(g1);
X=[f1,zeros(1,n)]; 
H=[g1,zeros(1,m)]; 
for i=1:n+m-1
Y(i)=0;
    for j=1:m
        if(i-j+1>0)
            Y(i)=Y(i)+X(j)*H(i-j+1);
        end
    end
end
Self_programming_1D_convolution=Y
stem(Y);
ylabel('Y[n]');
xlabel('n');
title('Self-programming 1D-Convolution');