%% Reset all
clc;
clear all;
close all;

%% Config Parameters
InputNodes = 3;
NeuronsHiddenLayer = [20 20];
OutputNodes = 2;
numOfLayers = 4;
NodesPerLayer = [InputNodes NeuronsHiddenLayer OutputNodes];

stop_point=0.01;
learningRate = 0.03;
Ep_max = 500000;
draw_Ep = 200;

learningRate_plus = 1.2;
learningRate_negative = 0.7;
deltas_start = 1;
deltas_min = 0.00001;
deltas_max = 50;

SamplesColors = {'k.', 'b.'};
SamplesSize = 12;
SamplesMark = {'.', '*'};
BoundaryColors = {'g.', 'y.'};

%% Read Data

P1_i = [0:1:96];
P1_th = pi./16 * P1_i;
P1_r = (104 - P1_i) / 104;
for i = 1:97
	P1_x1(i) = P1_r(i) * cos(P1_th(i));
    P1_y1(i) = P1_r(i) * sin(P1_th(i));
    P1_x2(i) = -P1_r(i) * cos(P1_th(i));
    P1_y2(i) = -P1_r(i) * sin(P1_th(i));
end
P1_o=[P1_x1' P1_y1' zeros(1,97)'];
P1_x=[P1_x2' P1_y2' ones(1,97)'];
Ran_sort=[1:194];
Ran_sort = Ran_sort(randperm(length(Ran_sort)));
data=[];
for i=1:97
    data=[data;P1_o(i,:);P1_x(i,:);];
end
for i=1:194
    Input(i,:)=data(Ran_sort(i),:);
end

Samples = Input(:, 1:length(Input(1,:))-1);
Samples_Class = Input(:, length(Input(1,:)));
Desicion_Class = -1*ones(size(Samples_Class));

%% Adding Nodes of 1 in Layer
Samples = [ones(length(Samples(:,1)),1) Samples];

%% Calculate DesicionOutputs
DesicionOutputs = zeros(length(Samples_Class), OutputNodes);
for i=1:length(Samples_Class)
    if (Samples_Class(i) == 1)
        DesicionOutputs(i,:) = [1 0];
    else
        DesicionOutputs(i,:) = [0 1];
    end
end

%% Initialize Random Wieghts Matrices
Weights = cell(1, numOfLayers); 
Delta_Weights = cell(1, numOfLayers);
ResilientDeltas = Delta_Weights;
for i = 1:length(Weights)-1
    Weights{i} = 2*rand(NodesPerLayer(i), NodesPerLayer(i+1))-1;
    Weights{i}(:,1) = 0;
    Delta_Weights{i} = zeros(NodesPerLayer(i), NodesPerLayer(i+1));
    ResilientDeltas{i} = deltas_start*ones(NodesPerLayer(i), NodesPerLayer(i+1));
end
Weights{end} = ones(NodesPerLayer(end), 1); 
Old_Delta_Weights_for_Momentum = Delta_Weights;
Old_Delta_Weights_for_Resilient = Delta_Weights;

ActingNodes = cell(1, numOfLayers);
for i = 1:length(ActingNodes)
    ActingNodes{i} = zeros(1, NodesPerLayer(i));
end
NodesBackPropagatedErrors = ActingNodes; 

MLP_end = 0;

%% Iterating all the Data
MSE = -1 * ones(1,Ep_max);
for Epoch = 1:Ep_max
    
    for Sample = 1:length(Samples(:,1))
        %% Backpropagation Training
        %Forward Pass
        ActingNodes{1} = Samples(Sample,:);
        for Layer = 2:numOfLayers
            ActingNodes{Layer} = ActingNodes{Layer-1}*Weights{Layer-1};
            ActingNodes{Layer} = 1./(1 + exp(-ActingNodes{Layer}));
            if (Layer ~= numOfLayers)
                ActingNodes{Layer}(1) = 1;
            end
        end
        
        % Backward Pass Errors Storage
        NodesBackPropagatedErrors{numOfLayers} =  DesicionOutputs(Sample,:)-ActingNodes{numOfLayers};
        for Layer = numOfLayers-1:-1:1
            gradient =  ActingNodes{Layer+1} .* (1 - ActingNodes{Layer+1});
            for node=1:length(NodesBackPropagatedErrors{Layer}) % For all the Nodes in current Layer
                NodesBackPropagatedErrors{Layer}(node) =  sum( NodesBackPropagatedErrors{Layer+1} .* gradient .* Weights{Layer}(node,:) );
            end
        end
        
        % Backward Pass Delta Weights Calculation (Before multiplying by learningRate)
        for Layer = numOfLayers:-1:2
            derivative = ActingNodes{Layer} .* (1 - ActingNodes{Layer}); 
            Delta_Weights{Layer-1} = Delta_Weights{Layer-1} + ActingNodes{Layer-1}' * (NodesBackPropagatedErrors{Layer} .* derivative);
        end
    end
    
    %% Apply resilient gradient descent or/and momentum to the delta_weights

    if (mod(Epoch,100)==0) %Reset Deltas
        for Layer = 1:numOfLayers
            ResilientDeltas{Layer} = learningRate*Delta_Weights{Layer};
        end
    end
    for Layer = 1:numOfLayers-1
        mult = Old_Delta_Weights_for_Resilient{Layer} .* Delta_Weights{Layer};
        ResilientDeltas{Layer}(mult > 0) = ResilientDeltas{Layer}(mult > 0) * learningRate_plus; % Sign didn't change
        ResilientDeltas{Layer}(mult < 0) = ResilientDeltas{Layer}(mult < 0) * learningRate_negative; % Sign changed
        ResilientDeltas{Layer} = max(deltas_min, ResilientDeltas{Layer});
        ResilientDeltas{Layer} = min(deltas_max, ResilientDeltas{Layer});

        Old_Delta_Weights_for_Resilient{Layer} = Delta_Weights{Layer};

        Delta_Weights{Layer} = sign(Delta_Weights{Layer}) .* ResilientDeltas{Layer};
    end


    %% Backward Pass Weights Update
    for Layer = 1:numOfLayers-1
        Weights{Layer} = Weights{Layer} + Delta_Weights{Layer};
    end
    
    for Layer = 1:length(Delta_Weights)
        Delta_Weights{Layer} = 0 * Delta_Weights{Layer};
    end
   
    %% Evaluation
    for Sample = 1:length(Samples(:,1))
        outputs = DoingNetwork(Samples(Sample,:), ActingNodes, Weights);
        bound = 1/2;
        if (outputs(1) >= bound && outputs(2) < bound)
            Desicion_Class(Sample) = 1;
        elseif (outputs(1) < bound && outputs(2) >= bound)
            Desicion_Class(Sample) = 0;
        else
            if (outputs(1) >= outputs(2))
                Desicion_Class(Sample) = 1;
            else
                Desicion_Class(Sample) = 0;
            end
        end
    end
    
    MSE(Epoch) = sum((Desicion_Class-Samples_Class).^2)/(length(Samples(:,1)));
    if (MSE(Epoch) <= stop_point)
        MLP_end = 1;
    end
        
    %% Visualization
    if (MLP_end || mod(Epoch,draw_Ep)==0)
        % Draw Decision Boundary
        % || mod(Epoch,draw_Ep)==0
        uni_Samples_Class = [0;1];
        cla;
        set(gcf,'Renderer', 'painters', 'Position', [10 10 900 600])
        hold on;
        margin = 0.05; step = 0.05;
        xlim([min(Samples(:,2))-margin max(Samples(:,2))+margin]);
        ylim([min(Samples(:,3))-margin max(Samples(:,3))+margin]);
        for x = min(Samples(:,2))-margin : step : max(Samples(:,2))+margin
            for y = min(Samples(:,3))-margin : step : max(Samples(:,3))+margin
                outputs = DoingNetwork([1 x y], ActingNodes, Weights);
                bound = 1/2;
                if (outputs(1) >= bound && outputs(2) < bound) %TODO: Not generic role for any number of output nodes
                    plot(6.5*x, 6.5*y, BoundaryColors{1}, 'markersize', 20);
                elseif (outputs(1) < bound && outputs(2) >= bound)
                    plot(6.5*x, 6.5*y, BoundaryColors{2}, 'markersize', 20);
                else
                    if (outputs(1) >= outputs(2))
                        plot(6.5*x, 6.5*y, BoundaryColors{1}, 'markersize', 20);
                    else
                        plot(6.5*x, 6.5*y, BoundaryColors{2}, 'markersize', 20);
                    end
                end
            end
        end

        for i = 1:length(uni_Samples_Class)
            points = Samples(Samples_Class==uni_Samples_Class(i), 2:end);
            plot(6.5*points(:,1), 6.5*points(:,2), SamplesColors{i},'marker', SamplesMark{i},'markersize', SamplesSize);
        end
        axis([-7 7 -7 7])
        axis equal
        title({'P_1';
               ['Decision at ',int2str(Epoch),' Epoch'];
               ['MSE= ',num2str(MSE(Epoch))];
               ['Structure ',mat2str(NodesPerLayer)];
               ['Learning Rate= ',num2str(learningRate)];
               ['Stop at MSE < ',num2str(stop_point)] 
               })

        pause(0.01);
    end
    display([int2str(Epoch) ' Epochs:''MSE = ' num2str(MSE(Epoch)) ' Stop at MSE < ' num2str(stop_point)]);
    if(MLP_end)
        break;
    end    
end
