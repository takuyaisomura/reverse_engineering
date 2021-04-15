
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

function F = mdp_vfe(Q,sim_type,range)

Ns          = length(Q.s(:,1));
No          = length(Q.o(:,1));
Nd          = length(Q.d(:,1));
t           = Q.t;
G           = Q.Gamma;
F           = 0;
%[qA,qlnA]   = param_norm(Q.qa,sim_type);
%[qBi,qlnBi] = param_norm(Q.qbi,sim_type);
[qCi,qlnCi] = param_norm(Q.qci,sim_type);
%lnD         = log(max(10^-6, Q.D));
lnE         = log(max(10^-6, Q.E));

for i = 1:ceil((t-1)/range)
  if (i == 1), tt  = (i-1)*range+2:min(i*range,t-1);
  else,        tt  = (i-1)*range+1:min(i*range,t-1); end
  T1    = length(tt);
  Ftemp =         sum(sum(sum( Q.qs(:,1,tt) .* (log(max(10^-6, Q.qs(:,1,tt))) - Q.vs(:,1,tt)) )));
  Ftemp = Ftemp + sum(sum(sum( Q.qs(:,2,tt) .* (log(max(10^-6, Q.qs(:,2,tt))) - Q.vs(:,2,tt)) )));
  vd1   = (1-2*G(i)) * (qlnCi(:,:,1,1)'*reshape(Q.qs(:,1,tt-1),[Ns,T1]) + qlnCi(:,:,2,1)'*reshape(Q.qs(:,2,tt-1),[Ns,T1])) + lnE(:,1)*ones(1,length(tt));
  vd2   = (1-2*G(i)) * (qlnCi(:,:,1,2)'*reshape(Q.qs(:,1,tt-1),[Ns,T1]) + qlnCi(:,:,2,2)'*reshape(Q.qs(:,2,tt-1),[Ns,T1])) + lnE(:,2)*ones(1,length(tt));
  Ftemp = Ftemp + sum(sum(sum( Q.qd(:,1,tt) .* (log(max(10^-6, Q.qd(:,1,tt))) - reshape(vd1,[Nd,1,T1])) )));
  Ftemp = Ftemp + sum(sum(sum( Q.qd(:,2,tt) .* (log(max(10^-6, Q.qd(:,2,tt))) - reshape(vd2,[Nd,1,T1])) )));
  F     = F + Ftemp;
end

end

%--------------------------------------------------------------------------------

