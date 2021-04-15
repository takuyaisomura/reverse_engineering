
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

function Gamma = mdp_risk(Q,sim_type,range,del_th,risk)

Ns  = length(Q.s(:,1));
No  = length(Q.o(:,1));
Nd  = length(Q.d(:,1));
T   = length(Q.o(1,:));
t   = Q.t;
Gamma = zeros(ceil((T-1)/range),1);

for i = 1:ceil((t-1)/range)
  tt  = (i-1)*range+1:min(i*range,t-1);
  del = 1 - mean(Q.G(tt)) * 2;
  del = del * range;
  if (del <= del_th(1))     G = risk(1);
  elseif (del <= del_th(2)) G = risk(2);
  elseif (del <= del_th(3)) G = risk(3);
  else                      G = risk(4);
  end
  Gamma(i) = G;
end

end

%--------------------------------------------------------------------------------

