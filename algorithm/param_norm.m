
%--------------------------------------------------------------------------------
% param_norm.m
%
% This demo is included in
% Triple equivalence for the emergence of biological intelligence
% Takuya Isomura
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-02-21
%

%--------------------------------------------------------------------------------

function [qA,qlnA] = param_norm(qa,sim_type)

N     = size(qa,1)/2;
qAsum = qa(1:N,:)+qa(N+1:N*2,:);
qA    = qa ./ [qAsum; qAsum];
qA    = max(min(qA,1-10^-8),10^-8);
if (sim_type == 1)
  psi_sum = psi(max(10^-8, qa(1:N,:)+qa(N+1:N*2,:)));
  qlnA    = psi(max(10^-8, qa)) - [psi_sum; psi_sum];
elseif (sim_type == 2)
  qlnA    = log(qA);
end

end

%--------------------------------------------------------------------------------

