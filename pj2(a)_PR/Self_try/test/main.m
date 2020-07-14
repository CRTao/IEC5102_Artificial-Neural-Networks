%% generate the helical data, each category has 100 examples
train_num=100;
train_circle_number=5000;
test_number=100;
i=(1:1:train_num)';
% the equation to generate the points of helical
alpha1=pi*(i-1)/25;
beta=0.4*((105-i)/104);
x0=0.5+beta.*sin(alpha1);
y0=0.5+beta.*cos(alpha1);
z0=zeros(train_num,1);
x1=0.5-beta.*sin(alpha1);
y1=0.5-beta.*cos(alpha1);
z1=ones(train_num,1);
%% It is provinced that the result of training is related to the order of +/- 
% we need to mix them.
k=rand(1,2*train_num);
[m,n]=sort(k);

train=[x0 y0 z0;x1,y1,z1];                     % the point of one helical line data, a matrix of 200*3
trian_label1=train(n(1:2*train_num),end)';     % the label for the training data a vector of 1*200
train_data1 =train(n(1:2*train_num),1:end-1)'; % the input of training data - matrix of 2*200

% change 1-D result to 2D, studytra_labe2 is a matrix of 200*2
for i=1:2*train_num
    switch trian_label1(i)
        case 0
            train_label2(i,:)=[1 0];
        case 1
            train_label2(i,:)=[0 1];
    end
end

train_label=train_label2'; %train_label - matrix of 2*200
         
plot(x0,y0,'r+');
hold on;
plot(x1,y1,'go');
%legend();

%% initial the structure of BP nuerual network
%network structure - 2 inputs 3 nueron and 2 outputs
innum=2;
midnum=5;
outnum=2;

[train_data,train_datas]=mapminmax(train_data1);

%?入?出取值?值?机初始化
%w1矩?表示每一行?一??含?神?元的?入?值
w1=rands(midnum,innum);  %rands函?用?初始化神?元的?值和?值是很合适的,w1?3*2的矩?
b1=rands(midnum,1);      %b1?3*1的矩?
%w2矩?表示每一列?一??出?神?元的?入?值
w2=rands(midnum,outnum); %w2?3*2的矩?
b2=rands(outnum,1);      %b2?2*1的矩?

%用?保存上一次的?值和?值，因?后面的更新方差是??的，要用到
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
w2_1=w2;w2_2=w2_1;
b2_1=b2;b2_2=b2_1;

%??率的?定
alpha=0.05;

%??10次就ok了，而不管??后的?果如何
for train_circle=1:train_circle_number  ;
    for i=1:2*train_num; %200????本
       %% ?入?的?出
        x=train_data(:,i);%取出第i??本，x(i)?2*1的列向量
        %% ?含?的?出
        for j=1:midnum;
            I(j)=train_data(:,i)'*w1(j,:)'+b1(j);  %I(j)?1*1的??
            Iout(j)=1/(1+exp(-I(j)));   %Iout(j)也?1*1的??
        end     %Iout?1*3的行向量   
        %% ?出?的?出
         yn=(Iout*w2)'+b2;   %yn?2*1的列向量，因此此?的?函??性的，所以可以一步到位，不必上面
        
        %% ?算?差
        e=train_label(:,i)-yn; %e?2*1的列向量，保存的是?差值
        
        %?算?值??率
        dw2=e*Iout; %dw2?2*3的矩?，每一行表示?出接?的?入?值?化率
        db2=e'; %e?1*2的行向量
        
        for j=1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);  %FI(j)?一??，FI?1*3的行向量
        end
        
        for k=1:1:innum
            for j=1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));    %dw1?2*3的矩?
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));   %db1?1*3的矩?
            end
        end
        
        %% ?值更新方程
        w1=w1_1+alpha*dw1'; %w1仍?3*2的矩?
        b1=b1_1+alpha*db1'; %b1仍?3*1的矩?
        w2=w2_1+alpha*dw2'; %w2仍?3*2的矩?
        b2=b2_1+alpha*db2'; %b2仍?2*1的矩?
        
        %% 保存上一次的?值和?值
        w1_2=w1_1;w1_1=w1;
        b1_2=b1_1;b1_1=b1;
        w2_2=w2_1;w2_1=w2;
        b2_2=b2_1;b2_1=b2;
    end
end


%% ?生?螺旋???据
%% ?生?螺旋?据,每?100??本?，共200??本
i=(1.5:1:test_number+0.5)';    %每?51??本

%?螺旋?据?的?生方程
alpha2=pi*(i-1)/25;
beta2=0.4*((105-i)/104);
m0=0.5+beta2.*sin(alpha2);
n0=0.5+beta2.*cos(alpha2);
s0=zeros(test_number,1);
m1=0.5-beta2.*sin(alpha2);
n1=0.5-beta2.*cos(alpha2);
s1=ones(test_number,1);

test=[m0 n0 s0;m1,n1,s1];    %1?螺旋??据?,3*102的矩?
test_label1=test(:,end)';    %???据??，1*102的行向量
test_data1=test(:,1:end-1)'; %???据?性，2*102的矩?

%把1?的?出?成2?的?出,train_labe2?200*2的矩?
for i=1:2*test_number
    switch test_label1(i)
        case 0
            test_label2(i,:)=[1 0];
        case 1
            test_label2(i,:)=[0 1];
    end
end

test_label=test_label2'; %test_label?2*102的矩?
         
%%  ?出???据?螺旋曲?
plot(m0,n0,'c+');
hold on;
plot(m1,n1,'yo');
legend('training data - helical line1','training data - helical line2','test data - helical line1','test data - helical line2');

test_data=mapminmax('apply',test_data1,train_datas);

% %% 用??到的模型????据本身?行??
% for i=1:102
%     for j=1:midnum
%         I(j)=train_data(:,i)'*w1(j,:)'+b1(j);
%         Iout(j)=1/(1+exp(-I(j)));%Iout?1*3的行向量
%     end
%     predict(:,i)=w2'*Iout'+b2;%predict?2*102的矩?
% end
% 
% test_data=mapminmax('apply',train_data1,train_datas);
% test_label=train_label;
% test_label1=trian_label1;

%% 用??到的模型???据
for i=1:2*test_number
    for j=1:midnum
        I(j)=test_data(:,i)'*w1(j,:)'+b1(j);
        Iout(j)=1/(1+exp(-I(j)));%Iout?1*3的行向量
    end
    predict(:,i)=w2'*Iout'+b2;%predict?2*102的矩?
end

%% ???果分析
for i=1:2*test_number
    output_pred(i)=find(predict(:,i)==max(predict(:,i)));    %out_pred?1*102的矩?
end

error=output_pred-test_label1-1;    %


%% ?算出每一?????的???和
k=zeros(1,2); %k=[0 0]
for i=1:2*test_number
    if error(i)~=0    %matlab中不能用if error(i)！=0 
        [b c]=max(test_label(:,i));
        switch c
            case 1
                k(1)=k(1)+1;
            case 2
                k(2)=k(2)+1;
        end
    end
end


%% 求出每一??体的??和
kk=zeros(1,2); %k=[0 0]
for i=1:2*test_number
    [b c]=max(test_label(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
    end
end


%% ?算每一?的正确率
accuracy=(kk-k)./kk