function [Train_image,Train_label, Test_image, Test_label] = loadMNIST()

Train_image = loadMNISTImages('train-images.idx3-ubyte');
Train_label = loadMNISTLabels('train-labels.idx1-ubyte');
Test_image = loadMNISTImages('t10k-images.idx3-ubyte');
Test_label = loadMNISTLabels('t10k-labels.idx1-ubyte');

end
