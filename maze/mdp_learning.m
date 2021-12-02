
%--------------------------------------------------------------------------------
% mdp_learning.m
%
% This demo is included in
% Canonical neural networks perform active inference
% Takuya Isomura, Hideaki Shimazaki, Karl J. Friston
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-09-02
%

%--------------------------------------------------------------------------------

function Q = mdp_learning(Q0,sim_type,range)

Q   = Q0;
Q.t = 1;
Ns  = length(Q0.s(:,1));
No  = length(Q0.o(:,1));
Nd  = length(Q0.d(:,1));
t   = Q0.t;

Q.qa(:,:,1,1)   = Q0.qa(:,:,1,1)  + Q0.o(:,2:t)     * reshape(Q0.qs(:,1,2:t),[Ns t-1])';
Q.qa(:,:,1,2)   = Q0.qa(:,:,1,2)  + Q0.o(:,2:t)     * reshape(Q0.qs(:,2,2:t),[Ns t-1])';
Q.qa(:,:,2,1)   = Q0.qa(:,:,2,1)  + (1-Q0.o(:,2:t)) * reshape(Q0.qs(:,1,2:t),[Ns t-1])';
Q.qa(:,:,2,2)   = Q0.qa(:,:,2,2)  + (1-Q0.o(:,2:t)) * reshape(Q0.qs(:,2,2:t),[Ns t-1])';
Q.qbi(:,:,1,1)  = Q0.qbi(:,:,1,1) + reshape(Q0.qs(:,1,1:t-1),[Ns t-1]) * reshape(Q0.qs(:,1,2:t),[Ns t-1])';
Q.qbi(:,:,1,2)  = Q0.qbi(:,:,1,2) + reshape(Q0.qs(:,1,1:t-1),[Ns t-1]) * reshape(Q0.qs(:,2,2:t),[Ns t-1])';
Q.qbi(:,:,2,1)  = Q0.qbi(:,:,2,1) + reshape(Q0.qs(:,2,1:t-1),[Ns t-1]) * reshape(Q0.qs(:,1,2:t),[Ns t-1])';
Q.qbi(:,:,2,2)  = Q0.qbi(:,:,2,2) + reshape(Q0.qs(:,2,1:t-1),[Ns t-1]) * reshape(Q0.qs(:,2,2:t),[Ns t-1])';

qs1 = reshape(Q0.qs(:,1,1:t-1),[Ns t-1]);
qs2 = reshape(Q0.qs(:,2,1:t-1),[Ns t-1]);
d1  = Q0.d(:,2:t);
d2  = 1-Q0.d(:,2:t);
for i = 1:ceil((t-1)/range)
  tt  = (i-1)*range+1:min(i*range,t-1);
  G               = Q.Gamma(i);
  Q.qci(:,:,1,1)  = max(Q.qci(:,:,1,1) + qs1(:,tt) * (1-2*G)*d1(:,tt)', 1);
  Q.qci(:,:,1,2)  = max(Q.qci(:,:,1,2) + qs1(:,tt) * (1-2*G)*d2(:,tt)', 1);
  Q.qci(:,:,2,1)  = max(Q.qci(:,:,2,1) + qs2(:,tt) * (1-2*G)*d1(:,tt)', 1);
  Q.qci(:,:,2,2)  = max(Q.qci(:,:,2,2) + qs2(:,tt) * (1-2*G)*d2(:,tt)', 1);
end

end

%--------------------------------------------------------------------------------

