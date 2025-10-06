
%--------------------------------------------------------------------------------
% phi_estimate.m
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-10-28
%--------------------------------------------------------------------------------

function phi1 = phi_estimate(phi0,o,x,y,d,pos,isL,gain_y)

No          = size(phi0.o,1);
Nx          = size(phi0.x,1);
Ny          = size(phi0.y,1);
T           = size(phi0.o,2);
phi1        = phi0;
phi1.o      = o;
phi1.x      = x;
phi1.y      = y;
phi1.d      = d;
phi1.pos    = pos;
phi1.W1     = logit(phi0.HebbW1./phi0.HomeW1);
phi1.W0     = logit(phi0.HebbW0./phi0.HomeW0);
phi1.K1     = logit(phi0.HebbK1./phi0.HomeK1);
phi1.K0     = logit(phi0.HebbK0./phi0.HomeK0);
phi1.V1     = logit(phi0.HebbV1./phi0.HomeV1);
phi1.V0     = logit(phi0.HebbV0./phi0.HomeV0);
x1m         = [phi0.x(:,end) x(:,1:T-1)];

Gamma       = 0;
phi1.HebbW1 = phi0.HebbW1 + x*o';
phi1.HebbW0 = phi0.HebbW0 + (1-x)*o';
phi1.HebbK1 = phi0.HebbK1 + x*x1m';
phi1.HebbK0 = phi0.HebbK0 + (1-x)*x1m';
phi1.HebbV1 = max(phi0.HebbV1 + (1-2*Gamma)*d*x1m', 1);
phi1.HebbV0 = max(phi0.HebbV0 + (1-2*Gamma)*(1-d)*x1m', 1);
phi1.HomeW1 = phi0.HomeW1 + x*ones(No,T)';
phi1.HomeW0 = phi0.HomeW0 + (1-x)*ones(No,T)';
phi1.HomeK1 = phi0.HomeK1 + x*ones(Nx,T)';
phi1.HomeK0 = phi0.HomeK0 + (1-x)*ones(Nx,T)';
phi1.HomeV1 = phi0.HomeV1 + d*ones(Nx,T)';
phi1.HomeV0 = phi0.HomeV0 + (1-d)*ones(Nx,T)';

if isL == 0, return, end
logsigW     = log([sig(phi1.W1) 1-sig(phi1.W1); sig(phi1.W0) 1-sig(phi1.W0)]+10^-8);
logsigK     = log([sig(phi1.K1) 1-sig(phi1.K1); sig(phi1.K0) 1-sig(phi1.K0)]+10^-8);
logsigV     = log([sig(phi1.V1) 1-sig(phi1.V1); sig(phi1.V0) 1-sig(phi1.V0)]+10^-8);
phi1.L      = sum(sum([x; 1-x] .* (log([x; 1-x]+10^-8) - logsigW*[o; 1-o] - logsigK*[x1m; 1-x1m] - [phi1.phix1; phi1.phix0]))) ...
            + sum(sum([y; 1-y] .* (log([y; 1-y]+10^-8) - logsigV*[x1m; 1-x1m] - [phi1.phiy1; phi1.phiy0])));

end

%--------------------------------------------------------------------------------
