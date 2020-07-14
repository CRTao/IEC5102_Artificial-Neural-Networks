P3a_mean=[0 0];
P3a_cov=[1 0;0 1;];
P3b_mean=[14 0];
P3b_cov=[1 0;0 4;];
P3c_mean=[7 14];
P3c_cov=[4 0;0 1;];
P3d_mean=[7 7];
P3d_cov=[1 0;0 1;];
P3_Colors = {'r', 'b','g','m'};
data=[];
P3_N = 150;

rng default
P3a_ran = mvnrnd(P3a_mean,P3a_cov,P3_N);
P3b_ran = mvnrnd(P3b_mean,P3b_cov,P3_N);
P3c_ran = mvnrnd(P3c_mean,P3c_cov,P3_N);
P3d_ran = mvnrnd(P3d_mean,P3d_cov,P3_N);

training = cell(4,1);
training{1}=[P3a_ran(:,1) P3a_ran(:,2)];
training{2}=[P3b_ran(:,1) P3b_ran(:,2)];
training{3}=[P3c_ran(:,1) P3c_ran(:,2)]; 
training{4}=[P3d_ran(:,1) P3d_ran(:,2)]; 

sample_means = cell(length(training),1);
 
for i=1:length(training),
    sample_means{i} = mean(training{i});
end
xrange = [-7 20];
yrange = [-7 20];
inc = 0.1;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
 
image_size = size(x);
xy = [x(:) y(:)];
xy = [reshape(x, image_size(1)*image_size(2),1) reshape(y, image_size(1)*image_size(2),1)];
numxypairs = length(xy);
dd = [];
for i=1:length(training),
    disttemp = sum(abs(xy - repmat(sample_means{i}, [numxypairs 1])), 2);
    dd = [dd disttemp];
end
 
[m,idx] = min(dd, [], 2);
dmap = reshape(idx, image_size);
figure;
imagesc(xrange,yrange,dmap);
hold on;
set(gca,'ydir','normal');
mapping = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1; 0.95 0.95 0.6];
colormap(mapping);
xlabel('x');
ylabel('y');

imagesc(xrange,yrange,dmap);
hold on;
set(gca,'ydir','normal');

% plot the class training data.
hold on
plot(training{1}(:,1),training{1}(:,2), 'r.');
hold on
plot(training{2}(:,1),training{2}(:,2), 'go');
hold on
plot(training{3}(:,1),training{3}(:,2), 'b*');
hold on
plot(training{4}(:,1),training{4}(:,2), 'kx');
hold on
