
%--------------------------------------------------------------------------------
% binary_to_decimal.m
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

function y = binary_to_decimal(x)

y = 2.^(0:length(x)-1)*x;

end

