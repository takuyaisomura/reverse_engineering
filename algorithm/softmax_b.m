
%--------------------------------------------------------------------------------
% softmax_b.m
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

function y = softmax_b(x)

N = size(x,1)/2;
v = exp(x-max(x));
%fprintf(1,'%f\n',max(v))
%size(v)
%size([v(1:N,:)+v(N+1:N*2,:); v(1:N,:)+v(N+1:N*2,:)])
y = v ./ [v(1:N,:)+v(N+1:N*2,:); v(1:N,:)+v(N+1:N*2,:)];
%y = exp(x) ./ [exp(x(1:N))+exp(x(N+1:N*2)); exp(x(1:N))+exp(x(N+1:N*2))];

end

