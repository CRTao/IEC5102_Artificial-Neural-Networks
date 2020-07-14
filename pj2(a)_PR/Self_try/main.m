%% Practice 1
%--------------------------------------------------------
P1_i = [0:1:96];
P1_th = pi./16 * P1_i;
P1_r = (104 - P1_i) / 104;
for i = 1:97
	P1_x1(i) = P1_r(i) * cos(P1_th(i));
    P1_y1(i) = P1_r(i) * sin(P1_th(i));
    P1_x2(i) = -P1_r(i) * cos(P1_th(i));
    P1_y2(i) = -P1_r(i) * sin(P1_th(i));
end
P1_o=[P1_x1' P1_y1' ones(1,97)'];
P1_x=[P1_x2' P1_y2' 2*ones(1,97)'];
Rad=[1:194];
Rad = Rad(randperm(length(Rad)));
data=[];
for i=1:97
    data=[data;P1_o(i,:);P1_x(i,:);];
end
for i=1:194
    Input(i,:)=data(Rad(i),:);
end


Figure_P1=figure;
scatter(P1_x1,P1_y1,'o');
hold on
scatter(P1_x2,P1_y2,'x');
axis([-8 8 -8 8])
grid on;
title('Two spiral problem');
%--------------------------------------------------------

P1_alpha=1.0;
P1_itmax=5000;
P1_errormax=100.0;
P1_errorlow=0.0;
P1_w=[0 0 0]';
P1_it=0;
P1_error=P1_errormax;


%% Practice 2
%--------------------------------------------------------
P2_N=250;
P2_theta1 = linspace(-180,180, P2_N)*pi/360;
P2_r = 8
P2_x1 = -5 + P2_r*sin(P2_theta1)+randn(1,P2_N);
P2_y1 = P2_r*cos(P2_theta1)+randn(1,P2_N);
P2_x2 = 5 + P2_r*sin(P2_theta1)+randn(1,P2_N);
P2_y2 = -P2_r*cos(P2_theta1)+randn(1,P2_N);
P2_o=[P2_x1' P2_y1' ones(1,250)'];
P2_x=[P2_x2' P2_y2' 2*ones(1,250)'];
data2=[];
for i=1:250
    data2=[data2;P2_o(i,:);P2_x(i,:);];
end

Figure_P2=figure;
hold on;
axis equal;
plot(P2_x1,P2_y1,'bo');
plot(P2_x2,P2_y2,'rs');
grid on;
title('Double-moon problem');



