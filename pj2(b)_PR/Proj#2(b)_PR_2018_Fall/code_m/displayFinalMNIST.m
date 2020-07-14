function displayFinalMNIST(test_array,predict_array,data)

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

% all final data
%
% d_v : desired class
% d_p : d_v location on figure
%
% t_v : correct predict class
% t_p : t_v location on figure
% w_v : wrong predict class
% w_p : w_v location on figure
%
tmpyp=[gap/2+sample_height:gap+sample_height:(gap+sample_height)*10];
tmpyp=[tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp]';
tmpyp=sort(tmpyp);
tmpxp=[gap:gap+sample_width:(gap+sample_width)*10];
tmpxp=[tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp]';
d_p = [tmpxp tmpyp];
d_v = test_array(1:100);

tmpyp=[gap/2+sample_height:gap+sample_height:(gap+sample_height)*10];
tmpyp=[tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp tmpyp]';
tmpyp=sort(tmpyp);
tmpxp=[gap/4+sample_width:gap+sample_width:(gap+sample_width)*10];
tmpxp=[tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp tmpxp]';
t_v = predict_array(1:100);
wrongidx = find(test_array(1:100)~=predict_array(1:100));
w_v = t_v(wrongidx);
actual_v = d_v(wrongidx);
wrong_xp = tmpxp(wrongidx);
wrong_yp = tmpyp(wrongidx);
t_v(wrongidx)=[];tmpxp(wrongidx)=[];tmpyp(wrongidx)=[];
t_p = [tmpxp tmpyp];
w_p = [wrong_xp wrong_yp];

% display
figure;
colormap(gray);
set(gcf,'Renderer', 'painters', 'Position', [10 10 900 700])
TI = insertText(display_array,d_p,d_v,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','yellow');
TCI = insertText(TI,t_p,t_v,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','green');
if ~isempty(w_v) 
    FI = insertText(TCI,w_p,w_v,'AnchorPoint','LeftBottom','FontSize',8,'BoxColor','red');
    imagesc(FI,[-1 1]),title('First 100 of Data Results');
else 
    imagesc(TCI,[-1 1]),title('First 100 of Data Results');
end 
axis image off
drawnow;

% Confusion Matrix
PrintConfusionMatrix(d_v,t_v,w_v,actual_v);

end
