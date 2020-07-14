function [fig, display_array] = displayMNIST(data)

% display array col/row of image
display_rows = 10;
display_cols = 10;

% nbr of pixel from single image 
sample_width = round(sqrt(size(data, 2)));
sample_height = ( 784 / sample_width );

gap = 12;

% display array 
display_array = ones(display_rows * (sample_height + gap), ...
                       gap + display_cols * (sample_width + gap ));

curr_sam = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_sam > 100 
			break; 
        end
        display_array( ...
           (j - 1) * (sample_height + gap) + (1:sample_height), ...
           gap + (i - 1) * (sample_width + gap) + (1:sample_width)) = ...
        reshape( data(curr_sam, :), sample_height, sample_width) / ...
        max(abs(data(curr_sam, :) ));
		curr_sam = curr_sam + 1;
	end
	if curr_sam > 100 
		break; 
	end
end

% display
colormap(gray);
fig = imagesc(display_array);
axis image off
drawnow;

end
