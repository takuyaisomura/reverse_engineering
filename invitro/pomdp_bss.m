
%--------------------------------------------------------------------------------
% pomdp_bss.m
%
% This analysis is included in
% Experimental validation of the free-energy principle with in vitro neural networks
% Takuya Isomura, Kiyoshi Kotani, Yasuhiko Jimbo, Karl Friston
%
% Copyright (C) 2022 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2022-09-16
%

%--------------------------------------------------------------------------------

function [qs qA qlnA qa] = pomdp_bss(N,M,T,o,qa_init,D,sim_type)

v    = zeros(T,N,2);
qs   = zeros(T,N,2);
qA   = zeros(T+1,M,2,N,2);
qlnA = zeros(T+1,M,2,N,2);
qa   = zeros(T+1,M,2,N,2);
lnD  = log(D);

qa(1,:,:,:,:)   = qa_init;
qA(1,:,1,:,1)   = qa(1,:,1,:,1) ./ (qa(1,:,1,:,1) + qa(1,:,2,:,1));
qA(1,:,1,:,2)   = qa(1,:,1,:,2) ./ (qa(1,:,1,:,2) + qa(1,:,2,:,2));
qA(1,:,2,:,1)   = 1 - qA(1,:,1,:,1);
qA(1,:,2,:,2)   = 1 - qA(1,:,1,:,2);
if (sim_type == 1)
 psi_sum1        = psi(max(10^-6, qa(1,:,1,:,1) + qa(1,:,2,:,1)));
 psi_sum2        = psi(max(10^-6, qa(1,:,1,:,2) + qa(1,:,2,:,2)));
 qlnA(1,:,1,:,1) = psi(max(10^-6, qa(1,:,1,:,1))) - psi_sum1;
 qlnA(1,:,1,:,2) = psi(max(10^-6, qa(1,:,1,:,2))) - psi_sum2;
 qlnA(1,:,2,:,1) = psi(max(10^-6, qa(1,:,2,:,1))) - psi_sum1;
 qlnA(1,:,2,:,2) = psi(max(10^-6, qa(1,:,2,:,2))) - psi_sum2;
elseif (sim_type == 2)
 log_sum1        = log(max(10^-6, qa(1,:,1,:,1) + qa(1,:,2,:,1)));
 log_sum2        = log(max(10^-6, qa(1,:,1,:,2) + qa(1,:,2,:,2)));
 qlnA(1,:,1,:,1) = log(max(10^-6, qa(1,:,1,:,1))) - log_sum1;
 qlnA(1,:,1,:,2) = log(max(10^-6, qa(1,:,1,:,2))) - log_sum2;
 qlnA(1,:,2,:,1) = log(max(10^-6, qa(1,:,2,:,1))) - log_sum1;
 qlnA(1,:,2,:,2) = log(max(10^-6, qa(1,:,2,:,2))) - log_sum2;
end

for t = 1:T
 
 % inference
 v(t,:,1)  = exp(reshape(qlnA(t,:,1,:,1),[M N])'*o(t,:,1)' + reshape(qlnA(t,:,2,:,1),[M N])'*o(t,:,2)' + lnD(:,1))';
 v(t,:,2)  = exp(reshape(qlnA(t,:,1,:,2),[M N])'*o(t,:,1)' + reshape(qlnA(t,:,2,:,2),[M N])'*o(t,:,2)' + lnD(:,2))';
 qs(t,:,1) = v(t,:,1) ./ (v(t,:,1) + v(t,:,2));
 qs(t,:,2) = 1 - qs(t,:,1);
 
 % learning
 qa(t+1,:,1,:,1)   = qa(t,:,1,:,1) + reshape(o(t,:,1)' * qs(t,:,1),[1 M 1 N 1]);
 qa(t+1,:,2,:,1)   = qa(t,:,2,:,1) + reshape(o(t,:,2)' * qs(t,:,1),[1 M 1 N 1]);
 qa(t+1,:,1,:,2)   = qa(t,:,1,:,2) + reshape(o(t,:,1)' * qs(t,:,2),[1 M 1 N 1]);
 qa(t+1,:,2,:,2)   = qa(t,:,2,:,2) + reshape(o(t,:,2)' * qs(t,:,2),[1 M 1 N 1]);
 qA(t+1,:,1,:,1)   = qa(t+1,:,1,:,1) ./ (qa(t+1,:,1,:,1) + qa(t+1,:,2,:,1));
 qA(t+1,:,1,:,2)   = qa(t+1,:,1,:,2) ./ (qa(t+1,:,1,:,2) + qa(t+1,:,2,:,2));
 qA(t+1,:,2,:,1)   = 1 - qA(t+1,:,1,:,1);
 qA(t+1,:,2,:,2)   = 1 - qA(t+1,:,1,:,2);
 if (sim_type == 1)
  psi_sum1          = psi(max(10^-6, qa(t+1,:,1,:,1) + qa(t+1,:,2,:,1)));
  psi_sum2          = psi(max(10^-6, qa(t+1,:,1,:,2) + qa(t+1,:,2,:,2)));
  qlnA(t+1,:,1,:,1) = psi(max(10^-6, qa(t+1,:,1,:,1))) - psi_sum1;
  qlnA(t+1,:,1,:,2) = psi(max(10^-6, qa(t+1,:,1,:,2))) - psi_sum2;
  qlnA(t+1,:,2,:,1) = psi(max(10^-6, qa(t+1,:,2,:,1))) - psi_sum1;
  qlnA(t+1,:,2,:,2) = psi(max(10^-6, qa(t+1,:,2,:,2))) - psi_sum2;
 elseif (sim_type == 2)
  log_sum1          = log(max(10^-6, qa(t+1,:,1,:,1) + qa(t+1,:,2,:,1)));
  log_sum2          = log(max(10^-6, qa(t+1,:,1,:,2) + qa(t+1,:,2,:,2)));
  qlnA(t+1,:,1,:,1) = log(max(10^-6, qa(t+1,:,1,:,1))) - log_sum1;
  qlnA(t+1,:,1,:,2) = log(max(10^-6, qa(t+1,:,1,:,2))) - log_sum2;
  qlnA(t+1,:,2,:,1) = log(max(10^-6, qa(t+1,:,2,:,1))) - log_sum1;
  qlnA(t+1,:,2,:,2) = log(max(10^-6, qa(t+1,:,2,:,2))) - log_sum2;
 end
end

qA   = qA(1:T,:,:,:,:);
qlnA = qlnA(1:T,:,:,:,:);
qa   = qa(1:T,:,:,:,:);

