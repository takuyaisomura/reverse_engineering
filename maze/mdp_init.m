
%--------------------------------------------------------------------------------
% mdp_init.m
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

function Q = mdp_init(qa0,qbi0,qci0,D,E,T)

Q.t   = 1;
Q.s   = zeros(length(D(:,1)),T);       % hidden states
Q.o   = zeros(length(qa0(:,1,1,1)),T); % observations
Q.d   = zeros(length(E(:,1)),T);       % decisions
Q.u   = zeros(4,T);                    % action
Q.G   = zeros(1,T);                    % risk
Q.F   = 0;
Q.Gamma = 0;

Q.qs  = zeros(length(D(:,1)),2,T);
Q.qd  = zeros(length(E(:,1)),2,T);
Q.qd_ = zeros(length(E(:,1)),T);
Q.vs  = zeros(length(D(:,1)),2,T);
Q.vd  = zeros(length(E(:,1)),2,T);
Q.qs(:,:,1) = D;
Q.qd(:,:,1) = E;
Q.vs(:,:,1) = D;
Q.vd(:,:,1) = E;

Q.qa  = qa0;
Q.qbi = qbi0;
Q.qci = qci0;
Q.D   = D;
Q.E   = E;

end

%--------------------------------------------------------------------------------

