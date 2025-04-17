
%--------------------------------------------------------------------------------
% softmax_a.m
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
% 2024-09-02
%

%--------------------------------------------------------------------------------

function y = softmax_a(x)

v = exp(x-max(x));
y = v ./ sum(v);

end

