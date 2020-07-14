f2 = 0.1*ones(4,4); 
g2 = 0.1*ones(4,4); 

%% toolbox
toolbox_2D_Convolution = conv2(f2,g2)

%% Self-programming
[f2_row,f2_col] = size(f2);
[g2_row,g2_col] = size(g2);
h = rot90(g2, 2);
center = floor((size(h)+1)/2);
Rep = zeros(f2_row + g2_row*2-2, f2_col + g2_col*2-2);
for x = g2_row : g2_row+f2_row-1
    for y = g2_col : g2_col+f2_row-1
        Rep(x,y) = f2(x-g2_row+1, y-g2_col+1);
    end
end
B = zeros(f2_row+g2_row-1,g2_col+f2_col-1);
for x = 1 : f2_row+g2_row-1
    for y = 1 : g2_col+f2_col-1
        for i = 1 : g2_row
            for j = 1 : g2_col
                B(x, y) = B(x, y) + (Rep(x+i-1, y+j-1) * h(i, j));
            end
        end
    end
end
Self_programming_1D_convolution=B
mesh(B);
title('Self-programming 2D-Convolution');