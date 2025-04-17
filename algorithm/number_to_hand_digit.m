
%--------------------------------------------------------------------------------
% number_to_hand_digit.m
%
% This demo is included in
% Triple equivalence for the emergence of biological intelligence
% Takuya Isomura
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-02-21
%

%--------------------------------------------------------------------------------

function output = number_to_hand_digit(input)

%fprintf(1,'read files\n');
M        = 60000;
M2       = 10000;
Nimg     = 28*28;
fid      = fopen('train-images-idx3-ubyte');
data     = fread(fid); data = reshape(data(17:28*28*M+16,:),[28*28 M]) / 255;
fclose(fid);
fid      = fopen('train-labels-idx1-ubyte');
label    = fread(fid); label = reshape(label(9:M+8,:),[1 M]);
fclose(fid);
fid      = fopen('t10k-images-idx3-ubyte');
data2    = fread(fid); data2 = reshape(data2(17:28*28*M2+16,:),[28*28 M2]) / 255;
fclose(fid);
fid      = fopen('t10k-labels-idx1-ubyte');
label2   = fread(fid); label2 = reshape(label2(9:M2+8,:),[1 M2]);
fclose(fid);

lab      = cell(10,1);
lab2     = cell(10,1);
for i = 1:10
 lab{i}  = find(label  == i-1);
 lab2{i} = find(label2 == i-1);
end

%--------------------------------------------------------------------------------

%fprintf(1,'create sequences\n');
N        = size(input,1);
T        = size(input,2);
input    = reshape(input, [N*T 1]);
output   = zeros(Nimg,N*T);

for i = 1:10
 rnd                    = randi([1 length(lab{i})],sum(input==i-1),1);
 output(:,input == i-1) = data(:,lab{i}(1,rnd));
end

input    = reshape(input, [N T]);
output   = reshape(permute(reshape(output, [28 28 N T]), [2 1 3 4]), [Nimg N T]);
clear data data2

end

%--------------------------------------------------------------------------------
