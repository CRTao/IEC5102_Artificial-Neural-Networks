function PrintConfusionMatrix(d_v,t_v,wrong_v,actual_v)
    ActualClass = zeros(1,10);PredictClass = zeros(1,10);WrongClass = zeros(1,10);
    for i = 1:10
        ActualClass(i) = sum(d_v(:) == i-1);
        PredictClass(i) = sum(t_v(:) == i-1);
        WrongClass(i) = sum(wrong_v(:) == i-1);
    end
    BB = zeros(12,11);
    BB(1,2:11)=ActualClass;
    BB(2,3:11)=1:9;
    BB(4:12,1)=1:9;
    for i = 1:10
        BB(2+i,1+i)=PredictClass(i);
    end
    for i = 1:length(actual_v)
        BB(3+wrong_v(i),2+actual_v(i))=BB(3+wrong_v(i),2+actual_v(i))+1;
    end
   
    
    fig = figure('NumberTitle','off','Name','Matrix');
    % Create the axes
    ax = axes('Units','normal','Position',[.1 .1 .8 .8]);
    axis ij   % Origin at top left
    % Define the default values for text
    set(fig,'DefaultTextFontName','courier', ...  % So text lines up
               'DefaultTextHorizontalAlignment','left', ...
               'DefaultTextVerticalAlignment','bottom', ...
               'DefaultTextClipping','on')
    % Define the matrix
    CM=BB(3:12,2:11);
    % Determine the number of rows and columns
    [m,n] = size(CM);
    % Draw the first column
    axis([1 m 1 n])
    drawnow
    tmp = text(.5,.5,'t');
    ext = get(tmp,'Extent');
    hch = ext(4);  % Height of a single character.
    wch = ext(3); % Width of a single character.
    fw = 3;
    wc = fw*wch;   % Width of 8 digit column
    dwc = 2*wch;  % Distance between columns
    dx = wc+dwc;  % Step used for columns
    dy = 2*hch;  % Step used for columns
    x = 1.75;        % Location of first column
    delete(tmp)
    for i = 1:n   % Column
        y = 1;      
        for j = 1:m % Row
          y = y + abs(dy);  % Location of row
          t(j,i) = text(x,y,sprintf('%2d',CM(j,i)));
        end
        x = x+dx;   % Location of next column
    end
    for i = 1:n   % Column
        y = 0;      
        u(j,i) = text(x,y,sprintf('%2d',ActualClass(i)));
        x = x+dx;
    end
    title('Confusion Matrix')
    set(gca,'XTick',[],'YTick',[])
    ylabel('9   ----   Predict Class   ----   0'); xlabel('0   ----   Actual Class   ----   9');
    % Add a horizontal slider
    

end