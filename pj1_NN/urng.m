function output = grng(n,s)
output = [];
switch nargin
    case 2
        s = s;
        n = n;
    case 1
        s = randi([32768 65535]);
        n = n;
    otherwise
        s = randi([32768 65535]);
        n = 1;
end
for i=1:n
    s = ((s * 1268752) + 37549)/65536;
    temp = round(mod(s,10000)/10000,4);
    output = [output norminv(temp,0,1)];
end



