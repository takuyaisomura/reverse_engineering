
%--------------------------------------------------------------------------------
% logit.m
%
% This analysis is included in
% Experimental validation of the free-energy principle with in vitro neural networks
% Takuya Isomura, Kiyoshi Kotani, Yasuhiko Jimbo, Karl Friston
%
% Copyright (C) 2022 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2022-09-19
%

%--------------------------------------------------------------------------------

function y = logit(x)

y = log(x./(1-x));

end
