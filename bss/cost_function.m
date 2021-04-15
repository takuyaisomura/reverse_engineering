
%--------------------------------------------------------------------------------
% This demo is included in
% Reverse engineering neural networks to characterise their cost functions
% Takuya Isomura, Karl Friston
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-05-08
%--------------------------------------------------------------------------------

function L = cost_function(N,M,T,o,qa_init,alpha,beta,sim_type)

L    = 0;
v    = zeros(T,N,2);
qs   = zeros(T,N,2);
qA   = zeros(1,M,2,N,2);
qlnA = zeros(1,M,2,N,2);
qa   = zeros(1,M,2,N,2);

beta = reshape(beta,[1 M 1 N 2]);
qa(1,:,:,:,:)   = qa_init;
qA(1,:,1,:,1)   = qa(1,:,1,:,1) ./ (qa(1,:,1,:,1) + qa(1,:,2,:,1)) + beta(1,:,1,:,1);
qA(1,:,1,:,2)   = qa(1,:,1,:,2) ./ (qa(1,:,1,:,2) + qa(1,:,2,:,2)) + beta(1,:,1,:,2);
qA(1,:,2,:,1)   = 1 - qA(1,:,1,:,1);
qA(1,:,2,:,2)   = 1 - qA(1,:,1,:,2);
if (sim_type == 1)
  psi_sum1        = psi(max(10^-6, qa(1,:,1,:,1) + qa(1,:,2,:,1)));
  psi_sum2        = psi(max(10^-6, qa(1,:,1,:,2) + qa(1,:,2,:,2)));
  qlnA(1,:,1,:,1) = psi(max(10^-6, qa(1,:,1,:,1) + (qa(1,:,1,:,1) + qa(1,:,2,:,1)) .* beta(1,:,1,:,1))) - psi_sum1;
  qlnA(1,:,1,:,2) = psi(max(10^-6, qa(1,:,1,:,2) - (qa(1,:,1,:,1) + qa(1,:,2,:,1)) .* beta(1,:,1,:,1))) - psi_sum2;
  qlnA(1,:,2,:,1) = psi(max(10^-6, qa(1,:,2,:,1) + (qa(1,:,1,:,2) + qa(1,:,2,:,2)) .* beta(1,:,1,:,2))) - psi_sum1;
  qlnA(1,:,2,:,2) = psi(max(10^-6, qa(1,:,2,:,2) - (qa(1,:,1,:,2) + qa(1,:,2,:,2)) .* beta(1,:,1,:,2))) - psi_sum2;
elseif (sim_type == 2)
  log_sum1        = log(max(10^-6, qa(1,:,1,:,1) + qa(1,:,2,:,1)));
  log_sum2        = log(max(10^-6, qa(1,:,1,:,2) + qa(1,:,2,:,2)));
  qlnA(1,:,1,:,1) = log(max(10^-6, qa(1,:,1,:,1) + (qa(1,:,1,:,1) + qa(1,:,2,:,1)) .* beta(1,:,1,:,1))) - log_sum1;
  qlnA(1,:,1,:,2) = log(max(10^-6, qa(1,:,1,:,2) - (qa(1,:,1,:,1) + qa(1,:,2,:,1)) .* beta(1,:,1,:,1))) - log_sum2;
  qlnA(1,:,2,:,1) = log(max(10^-6, qa(1,:,2,:,1) + (qa(1,:,1,:,2) + qa(1,:,2,:,2)) .* beta(1,:,1,:,2))) - log_sum1;
  qlnA(1,:,2,:,2) = log(max(10^-6, qa(1,:,2,:,2) - (qa(1,:,1,:,2) + qa(1,:,2,:,2)) .* beta(1,:,1,:,2))) - log_sum2;
end

phi = zeros(N,2);
for t = 1:T
  % inference
  phi(:,1)  = alpha(:,1) + reshape(sum(atanh(2*qA(1,:,1,:,1)-1) .* beta(1,:,1,:,1)),[2 1]);
  phi(:,2)  = alpha(:,2) + reshape(sum(atanh(2*qA(1,:,1,:,2)-1) .* beta(1,:,1,:,2)),[2 1]);
  v(t,:,1)  = exp(reshape(qlnA(1,:,1,:,1),[M N])'*o(t,:,1)' + reshape(qlnA(1,:,2,:,1),[M N])'*o(t,:,2)' + phi(:,1));
  v(t,:,2)  = exp(reshape(qlnA(1,:,1,:,2),[M N])'*o(t,:,1)' + reshape(qlnA(1,:,2,:,2),[M N])'*o(t,:,2)' + phi(:,2));
  qs(t,:,1) = v(t,:,1) ./ (v(t,:,1) + v(t,:,2));
  qs(t,:,2) = 1 - qs(t,:,1);
  L         = L + qs(t,:,1) * log(qs(t,:,1)./v(t,:,1)+10^-24)' + qs(t,:,2) * log(qs(t,:,2)./v(t,:,2)+10^-24)';
end
