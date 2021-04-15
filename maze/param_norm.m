
%--------------------------------------------------------------------------------
% This demo is included in
% Canonical neural networks perform active inference
% Takuya Isomura et al
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-09-02
%--------------------------------------------------------------------------------

function [qA,qlnA] = param_norm(qa,sim_type)

qA(:,:,1,1)   = qa(:,:,1,1) ./ (qa(:,:,1,1) + qa(:,:,2,1));
qA(:,:,1,2)   = qa(:,:,1,2) ./ (qa(:,:,1,2) + qa(:,:,2,2));
qA(:,:,1,1)   = max(min(qA(:,:,1,1),1-10^-6),10^-6);
qA(:,:,1,2)   = max(min(qA(:,:,1,2),1-10^-6),10^-6);
qA(:,:,2,1)   = 1 - qA(:,:,1,1);
qA(:,:,2,2)   = 1 - qA(:,:,1,2);
if (sim_type == 1)
  psi_sum1      = psi(max(10^-6, qa(:,:,1,1) + qa(:,:,2,1)));
  psi_sum2      = psi(max(10^-6, qa(:,:,1,2) + qa(:,:,2,2)));
  qlnA(:,:,1,1) = psi(max(10^-6, qa(:,:,1,1))) - psi_sum1;
  qlnA(:,:,1,2) = psi(max(10^-6, qa(:,:,1,2))) - psi_sum2;
  qlnA(:,:,2,1) = psi(max(10^-6, qa(:,:,2,1))) - psi_sum1;
  qlnA(:,:,2,2) = psi(max(10^-6, qa(:,:,2,2))) - psi_sum2;
elseif (sim_type == 2)
  log_sum1      = log(max(10^-6, qa(:,:,1,1) + qa(:,:,2,1)));
  log_sum2      = log(max(10^-6, qa(:,:,1,2) + qa(:,:,2,2)));
  qlnA(:,:,1,1) = log(max(10^-6, qa(:,:,1,1))) - log_sum1;
  qlnA(:,:,1,2) = log(max(10^-6, qa(:,:,1,2))) - log_sum2;
  qlnA(:,:,2,1) = log(max(10^-6, qa(:,:,2,1))) - log_sum1;
  qlnA(:,:,2,2) = log(max(10^-6, qa(:,:,2,2))) - log_sum2;
end

end

%--------------------------------------------------------------------------------

