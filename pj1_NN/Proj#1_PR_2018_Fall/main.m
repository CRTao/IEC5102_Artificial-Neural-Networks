%% Practice 1
%--------------------------------------------------------
P1_mean=3;
P1_variance=2; 
P1_x=[P1_mean-3*P1_variance:0.01:P1_mean+3*P1_variance];
P1_y = exp(-.5 * ((P1_x - P1_mean)/P1_variance) .^ 2) ./ (P1_variance * sqrt(2*pi)); 
%Figure_P1=figure;
plot(P1_x,P1_y,'.');
grid on;
title('Practice 1');
xlabel('Numbers');
ylabel('Gauss Distribution');

%% Practice 2
%--------------------------------------------------------
% mean =1, 2;
% sigma = 1, 1;
P2_mean=[1 2];
P2_cov=[1 0;0 1;];
P2_x1= [P2_mean(1)-3:0.05:P2_mean(1)+3];
P2_x2= [P2_mean(2)-3:0.05:P2_mean(2)+3];
[P2_X1,P2_X2]=meshgrid(P2_x1,P2_x2);
F=mvnpdf([P2_X1(:) P2_X2(:)],P2_mean,P2_cov);
F=reshape(F,length(P2_x2),length(P2_x1));
Figure_P2=figure;
surf(P2_x1,P2_x2,F);
grid on;
title('Practice 2');
xlabel('x1');
ylabel('x2');
zlabel('Gauss Distribution');

%% Practice 3
%--------------------------------------------------------
Figure_P3=figure;
P3_ran = randn(1,10000);
[P3_fun,P3_x] = hist(P3_ran,50);
bar (P3_x,P3_fun/trapz(P3_x,P3_fun));
grid on;
title('Practice 3');
xlabel('Numbers');
ylabel('Gauss Distribution');

%% Practice 4
%--------------------------------------------------------
P4_mu = [1 2];
P4_cov = [3 0;0 4];
P4_N = 10000;
rng default  % For reproducibility
P4_ran = mvnrnd(P4_mu,P4_cov,P4_N);
Figure_P41=figure;
plot(P4_ran(:,1),P4_ran(:,2),'.');
grid on;
title({'Practice 4-1';'2-d Gaussian random data'});
Figure_P42=figure;
P4_hist = hist3(P4_ran,[25 25]);
hist3(P4_ran,[25 25],'CdataMode','auto','FaceColor','interp');
grid on;
title({'Practice 4-2';'Histogram for 2-d Gaussian random data'});

%% Practice 5
%--------------------------------------------------------
Figure_P5=figure;
[P5_c,P5_handle] = contour(P4_hist);
clabel(P5_c, P5_handle);
grid on;
title({'Practice 5';'2-d Histogram Contour'});

%% Practice 6
%--------------------------------------------------------
P6_x=[-7:0.02:7];
P6_y=P6_x-1;
Figure_P6=figure;
plot(P6_x,P6_y,'.');
grid on;
title({'Practice 6';'x-y=1'});

%% Practice 7
%--------------------------------------------------------
P7_th = 0:pi/250:2*pi;
P7_r=1;
P7_x = P7_r * cos(P7_th);
P7_y = P7_r * sin(P7_th);
Figure_P7=figure;
plot(P7_x,P7_y,'.');
axis([-2 2 -2 2])
grid on;
title({'Practice 7';'x^2+y^2=1'});

%% Practice 8
%--------------------------------------------------------
P8_th = 0:pi/250:2*pi;
P8_x = 1 * cos(P8_th);
P8_y = 2 * sin(P8_th);
Figure_P8=figure;
plot(P8_x,P8_y,'.');
axis([-4 4 -4 4])
grid on;
title({'Practice 8';'x^2+y^2/4=1'});

%% Practice 9
%--------------------------------------------------------
P9_y=[-5:0.05:5];
P9_x=sqrt(1+P9_y.^2/4);
Figure_P9=figure;
plot(P9_x,P9_y,'.');
hold on;
plot(-P9_x,P9_y,'.');
axis([-4 4 -4 4])
grid on;
title({'Practice 9';'x^2-y^2/4=1'});

%% Practice 10
%--------------------------------------------------------
P10_x=linspace(0,100);
P10_y=P10_x.*2;
Figure_P10=figure;
plot(P10_x,P10_y,'.');
axis([0 100 0 200])
grid on;
title({'Practice 10';'2x-y=0'});

%% Practice 11
%--------------------------------------------------------
P11_th = linspace(0,2*pi);
P11_r=2;
P11_x = P11_r * cos(P11_th);
P11_y = P11_r * sin(P11_th);
Figure_P11=figure;
plot(P11_x,P11_y,'.');
axis([-4 4 -4 4])
grid on;
title({'Practice 11';'x^2+y^2=4'});

%% Practice 12
%--------------------------------------------------------
P12_th = linspace(0,2*pi);
P12_x = 2 * cos(P12_th);
P12_y = 1 * sin(P12_th);
Figure_P12=figure;
plot(P12_x,P12_y,'.');
axis([-4 4 -4 4])
grid on;
title({'Practice 12';'x^2/4+y^2=1'});

%% Practice 13
%--------------------------------------------------------
P13_x=linspace(-1,1);
P13_y = 1./P13_x;
Figure_P13=figure;
plot(P13_x,P13_y,'.');
axis([-1 1 -10 10])
grid on;
title({'Practice 13';'xy=1'});

%% Practice 14
%--------------------------------------------------------
P14_i = [0:1:96];
P14_th = pi./16 * P14_i;
P14_r = 6.5 * (104 - P14_i) / 104;
for i = 1:97
	P14_x1(i) = P14_r(i) * cos(P14_th(i));
    P14_y1(i) = P14_r(i) * sin(P14_th(i));
    P14_x2(i) = -P14_r(i) * cos(P14_th(i));
    P14_y2(i) = -P14_r(i) * sin(P14_th(i));
end
Figure_P14=figure;
scatter(P14_x1,P14_y1,'o');
hold on
scatter(P14_x2,P14_y2,'x');
axis([-8 8 -8 8])
grid on;
title('Practice 14');

%% Practice 15
%--------------------------------------------------------
P15_x=[0 0 1 1 0 1 1 0];
P15_y=[1 0 0 0 0 1 1 1];
P15_z=[1 0 0 1 1 0 1 0];
P15_c=[1 2 2 2 1 2 1 1];
Figure_P15=figure;
for i = 1:8
    if P15_c(i)==1
        scatter3(P15_x(i),P15_y(i),P15_z(i),'o','r');
    else
        scatter3(P15_x(i),P15_y(i),P15_z(i),'x','b');
    end
    hold on
end
%plane  x ¡V y - z + 0.5 =0
[P15_px P15_py] = meshgrid(-1:0.1:1);
P15_pz = (P15_px + -P15_py + 0.5);
surf(P15_px, P15_py, P15_pz,'EdgeColor','none','FaceAlpha','0.5');
hold on;
axis([0 1 0 1 0 1.5])
grid on;
title('Practice 15');

%% Practice 16
%--------------------------------------------------------
P16_uth = linspace(0,pi,150);
P16_dth = linspace(pi,2*pi,150);
P16_xdis = 1;
P16_ydis = -0.2;
P16_ux = 2*cos(P16_uth) + rand(1,150) - P16_xdis;
P16_uy = 2*sin(P16_uth) + rand(1,150) + P16_ydis;
P16_dx = 2*cos(P16_dth) + rand(1,150) + P16_xdis;
P16_dy = 2*sin(P16_dth) + rand(1,150) - P16_ydis;
Figure_P16=figure;
scatter(P16_ux,P16_uy,'r');
hold on
scatter(P16_dx,P16_dy,'b');
hold on;
axis([-6 6 -4 4])
grid on;
title('Practice 16');

%% Practice 17
%--------------------------------------------------------
P17_x=[0:0.1:30*pi];
P17_y1=sin(P17_x/1);
P17_y2=sin(P17_x/2);
P17_y3=sin(P17_x/3);
P17_y4=sin(P17_x/4);
P17_y5=sin(P17_x/5);
Figure_P17=figure;
set(gca,'XLim',[0 30*pi],'YLim',[-1.2 1.2]);
grid on;
curve1 = animatedline('Color','b');
curve2 = animatedline('Color','y');
curve3 = animatedline('Color','k');
curve4 = animatedline('Color','r');
curve5 = animatedline('Color','g');
title('Practice 17');
for i=1:length(P17_x)
    addpoints(curve1,P17_x(i),P17_y1(i));
    addpoints(curve2,P17_x(i),P17_y2(i));
    addpoints(curve3,P17_x(i),P17_y3(i));
    addpoints(curve4,P17_x(i),P17_y4(i));
    addpoints(curve5,P17_x(i),P17_y5(i));
    drawnow
end

%% Practice 18
us = randi([32768 65535]);
un = 100;
P18_output = [];
for i=1:un
    us = ((us * 1268752) + 37549)/65536;  
    P18_output = [P18_output round(mod(us,10000)/10000,4)];
end
P18_output
[mean(P18_output) std(P18_output) var(P18_output)]

%% Practice 19
gs = randi([32768 65535]);
gn = 100;
P19_output = [];
for i=1:gn
    gs = ((gs * 1268752) + 37549)/65536;
    temp = round(mod(gs,10000)/10000,4);
    P19_output = [P19_output 3+10*norminv(temp)];
end
P19_output
[mean(P19_output) std(P19_output) var(P19_output)]

%% Practice 20

Figure_P20=figure;
% load data
% http://yann.lecun.com/exdb/mnist
fp = fopen('t10k-images.idx3-ubyte', 'rb');
assert(fp ~= -1, ['Could not open ', 't10k-images.idx3-ubyte', '']);
 
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', 't10k-images.idx3-ubyte', '']);
 
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
 
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
 
fclose(fp);
 
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images)/255 ;

num=ceil(random('unif',1,length(images),10,15));
for i=1:10
    for j=1:15
              img(28*(i-1)+1:28*i,28*(j-1)+1:28*j)=reshape(images(:,num(i,j)),28,28);
    end
end
%imshow(uint8(256*reshape(Images(:,floor(random('unif',0,60000,1,1))),28,28)))
imshow(uint8(256*img))





