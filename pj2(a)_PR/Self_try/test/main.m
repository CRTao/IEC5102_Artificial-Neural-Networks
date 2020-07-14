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

%?�J?�X����?��?���l��
%w1�x?��ܨC�@��?�@??�t?��?����?�J?��
w1=rands(midnum,innum);  %rands��?��?��l�Ư�?����?�ȩM?�ȬO�ܦX�쪺,w1?3*2���x?
b1=rands(midnum,1);      %b1?3*1���x?
%w2�x?��ܨC�@�C?�@??�X?��?����?�J?��
w2=rands(midnum,outnum); %w2?3*2���x?
b2=rands(outnum,1);      %b2?2*1���x?

%��?�O�s�W�@����?�ȩM?�ȡA�]?�Z������s��t�O??���A�n�Ψ�
w1_1=w1;w1_2=w1_1;
b1_1=b1;b1_2=b1_1;
w2_1=w2;w2_2=w2_1;
b2_1=b2;b2_2=b2_1;

%??�v��?�w
alpha=0.05;

%??10���Nok�F�A�Ӥ���??�Z��?�G�p��
for train_circle=1:train_circle_number  ;
    for i=1:2*train_num; %200????��
       %% ?�J?��?�X
        x=train_data(:,i);%���X��i??���Ax(i)?2*1���C�V�q
        %% ?�t?��?�X
        for j=1:midnum;
            I(j)=train_data(:,i)'*w1(j,:)'+b1(j);  %I(j)?1*1��??
            Iout(j)=1/(1+exp(-I(j)));   %Iout(j)�]?1*1��??
        end     %Iout?1*3����V�q   
        %% ?�X?��?�X
         yn=(Iout*w2)'+b2;   %yn?2*1���C�V�q�A�]����?��?��??�ʪ��A�ҥH�i�H�@�B���A�����W��
        
        %% ?��?�t
        e=train_label(:,i)-yn; %e?2*1���C�V�q�A�O�s���O?�t��
        
        %?��?��??�v
        dw2=e*Iout; %dw2?2*3���x?�A�C�@����?�X��?��?�J?��?�Ʋv
        db2=e'; %e?1*2����V�q
        
        for j=1:midnum
            S=1/(1+exp(-I(j)));
            FI(j)=S*(1-S);  %FI(j)?�@??�AFI?1*3����V�q
        end
        
        for k=1:1:innum
            for j=1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));    %dw1?2*3���x?
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));   %db1?1*3���x?
            end
        end
        
        %% ?�ȧ�s��{
        w1=w1_1+alpha*dw1'; %w1��?3*2���x?
        b1=b1_1+alpha*db1'; %b1��?3*1���x?
        w2=w2_1+alpha*dw2'; %w2��?3*2���x?
        b2=b2_1+alpha*db2'; %b2��?2*1���x?
        
        %% �O�s�W�@����?�ȩM?��
        w1_2=w1_1;w1_1=w1;
        b1_2=b1_1;b1_1=b1;
        w2_2=w2_1;w2_1=w2;
        b2_2=b2_1;b2_1=b2;
    end
end


%% ?��?����???�u
%% ?��?����?�u,�C?100??��?�A�@200??��
i=(1.5:1:test_number+0.5)';    %�C?51??��

%?����?�u?��?�ͤ�{
alpha2=pi*(i-1)/25;
beta2=0.4*((105-i)/104);
m0=0.5+beta2.*sin(alpha2);
n0=0.5+beta2.*cos(alpha2);
s0=zeros(test_number,1);
m1=0.5-beta2.*sin(alpha2);
n1=0.5-beta2.*cos(alpha2);
s1=ones(test_number,1);

test=[m0 n0 s0;m1,n1,s1];    %1?����??�u?,3*102���x?
test_label1=test(:,end)';    %???�u??�A1*102����V�q
test_data1=test(:,1:end-1)'; %???�u?�ʡA2*102���x?

%��1?��?�X?��2?��?�X,train_labe2?200*2���x?
for i=1:2*test_number
    switch test_label1(i)
        case 0
            test_label2(i,:)=[1 0];
        case 1
            test_label2(i,:)=[0 1];
    end
end

test_label=test_label2'; %test_label?2*102���x?
         
%%  ?�X???�u?���ۦ�?
plot(m0,n0,'c+');
hold on;
plot(m1,n1,'yo');
legend('training data - helical line1','training data - helical line2','test data - helical line1','test data - helical line2');

test_data=mapminmax('apply',test_data1,train_datas);

% %% ��??�쪺�ҫ�????�u����?��??
% for i=1:102
%     for j=1:midnum
%         I(j)=train_data(:,i)'*w1(j,:)'+b1(j);
%         Iout(j)=1/(1+exp(-I(j)));%Iout?1*3����V�q
%     end
%     predict(:,i)=w2'*Iout'+b2;%predict?2*102���x?
% end
% 
% test_data=mapminmax('apply',train_data1,train_datas);
% test_label=train_label;
% test_label1=trian_label1;

%% ��??�쪺�ҫ�???�u
for i=1:2*test_number
    for j=1:midnum
        I(j)=test_data(:,i)'*w1(j,:)'+b1(j);
        Iout(j)=1/(1+exp(-I(j)));%Iout?1*3����V�q
    end
    predict(:,i)=w2'*Iout'+b2;%predict?2*102���x?
end

%% ???�G���R
for i=1:2*test_number
    output_pred(i)=find(predict(:,i)==max(predict(:,i)));    %out_pred?1*102���x?
end

error=output_pred-test_label1-1;    %


%% ?��X�C�@?????��???�M
k=zeros(1,2); %k=[0 0]
for i=1:2*test_number
    if error(i)~=0    %matlab�������if error(i)�I=0 
        [b c]=max(test_label(:,i));
        switch c
            case 1
                k(1)=k(1)+1;
            case 2
                k(2)=k(2)+1;
        end
    end
end


%% �D�X�C�@??�^��??�M
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


%% ?��C�@?�����̲v
accuracy=(kk-k)./kk