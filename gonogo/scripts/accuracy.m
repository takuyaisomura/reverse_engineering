
%--------------------------------------------------------------------------------
% accuracy.m
%
% Copyright (C) 2025 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2025-05-12
%--------------------------------------------------------------------------------

function rate = accuracy(x,y)

Nx      = size(x,1);
Nsample = size(x,2);
qy      = zeros(1,Nsample);
if min(y) == 0, y = 2*y-1; end
for i = 1:Nsample
 x_      = x(:,[1:i-1,i+1:Nsample]);
 x_      = [x_; ones(1,Nsample-1)];
 y_      = y(:,[1:i-1,i+1:Nsample]);
 W       = (y_*x_')/(x_*x_'+eye(Nx+1)*10^-6);
 qy(i)   = W * [x(:,i); 1];
end
z       = (qy > 0)*2 - 1;
match   = (y == z)*1;
rate    = sum(match)/Nsample;

end

%--------------------------------------------------------------------------------
