function outputs = DoingNetwork(Sample, Nodes, Weights)
Nodes{1} = Sample;
n = length(Nodes);

for Layer = 2:n
    Nodes{Layer} = Nodes{Layer-1}*Weights{Layer-1};
    Nodes{Layer} = 1./(1 + exp(-Nodes{Layer}));
    if (Layer ~= n)
        Nodes{Layer}(1) = 1;
    end
end

outputs = Nodes{end};

end