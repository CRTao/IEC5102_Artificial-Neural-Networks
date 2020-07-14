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

stop_point=0.001;
learningRate = 0.03;
It_max = 500000;
draw_It = 10;

learningRate_plus = 1.2;
learningRate_negative = 0.7;
deltas_start = 1;
deltas_min = 0.00001;
deltas_max = 50;

SamplesColors = {'b.', 'r.'};
SamplesSize = 4;
SamplesMark = {'o', 's'};
BoundaryColors = {'g.', 'y.'};

%% Read Data
P2_N=250;
P2_theta1 = linspace(-180,180, P2_N)*pi/360;
P2_r = 8;
P2_x1 = -5 + P2_r*sin(P2_theta1)+randn(1,P2_N);
P2_y1 = P2_r*cos(P2_theta1)+randn(1,P2_N);
P2_x2 = 5 + P2_r*sin(P2_theta1)+randn(1,P2_N);
P2_y2 = -P2_r*cos(P2_theta1)+randn(1,P2_N);
P2_o=[(1/15)*P2_x1' (1/15)*P2_y1' zeros(1,250)'];
P2_x=[(1/15)*P2_x2' (1/15)*P2_y2' ones(1,250)'];
data2=[];
Ran_sort=[1:500];
Ran_sort = Ran_sort(randperm(length(Ran_sort)));
data2=[];
for i=1:250
    data2=[data2;P2_o(i,:);P2_x(i,:);];
end
for i=1:500
    Input(i,:)=data2(Ran_sort(i),:);
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
Old_Delta_Weights_for_Resilient = Delta_Weights;
ActingNodes = cell(1, numOfLayers);
for i = 1:length(ActingNodes)
    ActingNodes{i} = zeros(1, NodesPerLayer(i));
end
NodesBackPropagatedErrors = ActingNodes; 

MLP_end = 0;

%% Iterating all the Data
MSE = -1 * ones(1,It_max);
for Iteration = 1:It_max
    
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
    
    %% Apply resilient gradient descent to the delta_weights

    if (mod(Iteration,100)==0) %Reset Deltas
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
    
    MSE(Iteration) = sum((Desicion_Class-Samples_Class).^2)/(length(Samples(:,1)));
    if (MSE(Iteration) <= stop_point)
        MLP_end = 1;
    end
        
    %% Visualization
    if (MLP_end || mod(Iteration,draw_It)==0)
        % Draw Decision Boundary
        % || mod(Iteration,draw_It)==0
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
                    plot(15*x, 15*y, BoundaryColors{1}, 'markersize', 20);
                elseif (outputs(1) < bound && outputs(2) >= bound)
                    plot(15*x, 15*y, BoundaryColors{2}, 'markersize', 20);
                else
                    if (outputs(1) >= outputs(2))
                        plot(15*x, 15*y, BoundaryColors{1}, 'markersize', 20);
                    else
                        plot(15*x, 15*y, BoundaryColors{2}, 'markersize', 20);
                    end
                end
            end
        end

        for i = 1:length(uni_Samples_Class)
            points = Samples(Samples_Class==uni_Samples_Class(i), 2:end);
            plot(15*points(:,1), 15*points(:,2), SamplesColors{i},'marker', SamplesMark{i},'markersize', SamplesSize);
        end
        axis([-12 12 -12 12])
        axis equal
        title({'P2 Double Moons Problem';
               ['Decision at ',int2str(Iteration),' Iteration'];
               ['MSE= ',num2str(MSE(Iteration))];
               ['Structure ',mat2str(NodesPerLayer)];
               ['Learning Rate= ',num2str(learningRate)];
               ['Stop at MSE < ',num2str(stop_point)] 
               })

        pause(0.01);
    end
    display([int2str(Iteration) ' Iteration: MSE = ' num2str(MSE(Iteration))]);
    if(MLP_end)
        figure;
        MSE(MSE==-1) = [];
        plot([MSE(1:Iteration)]);
        ylim([-0.1 0.6]);
        title('Mean Square Error for P2');
        xlabel('Iteration');
        ylabel('MSE');
        grid on;
        break;
    end    
end
