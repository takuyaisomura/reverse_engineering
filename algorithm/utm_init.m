
%--------------------------------------------------------------------------------
% utm_init.m
%
% This demo is included in
% Triple equivalence for the emergence of biological intelligence
% Takuya Isomura
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-09-02
%

%--------------------------------------------------------------------------------

function [A,Blist,C,C2,qa,qb,qc,qc2] = utm_init(No,Ns,Nd,Nd2,NC,NC2)

Ncontext = NC2;

% define likelihood mapping
A            = zeros(No*2,Ns*2); % likelihood mapping
A(1:No,1:Ns) = eye(No,Ns)*0.5*0.99 + 0.5;
A            = [A(1:No,1:Ns) 1-A(1:No,1:Ns); 1-A(1:No,1:Ns) A(1:No,1:Ns)];
% define list of transition matrices
Blist = cell(Ncontext,1);
for i = 1:Ncontext
 B        = ones(Ns*2,Ns*2*2)*0.5;
 order    = randperm(Ns);
 for j = 1:Ns, B(order(rem(j,Ns)+1),order(j)) = 0.995; end
 order    = randperm(Ns);
 for j = 1:Ns, B(order(rem(j,Ns)+1),order(j)+Ns) = 0.995; end
 B        = [B(1:Ns,1:Ns*2) 1-B(1:Ns,1:Ns*2); 1-B(1:Ns,1:Ns*2) B(1:Ns,1:Ns*2)];
 Blist{i} = B;
end
C      = zeros(1,NC);
% optimal policy matrix
C2     = zeros(Nd2,NC2);
for i = 1:NC2, C2(:,i) = reshape(Blist{i}(1:Ns,1:Ns*2),[Nd2 1]); end

qa       = A * 10000 + 100; % + 20*(2*rand(No*2,Ns*2)-1);
qb       = ones(Ns*2,Ns*2*2)*1000;
qc       = [ones(Nd,NC)*0.005;ones(Nd,NC)*0.995];
qc2      = ones(Nd2*2,NC2);

end

