
%--------------------------------------------------------------------------------
% maze.m
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
% initialisation

clear
nx       = 99;    % length of maze
ny       = 19;    % width of maze
nv       = 11;    % size of field of vision

T        = 20000; % maximum duration
No       = nv*nv; % dimensionality of observations
Ns       = No;    % dimensionality of hidden states
Nd       = 4^4;   % dimensionality of decisions
rview    = (nv-1)/2;
sim_type = 2;
Ntrial   = 100;
Nintvl   = 2000;
Nsample  = 30;
pos      = zeros(2,T,'single');
pos(:,1) = [(ny+1)/2; 2];

qa0      = ones(No,Ns,2,2,'single') * 10000000;
qbi0     = ones(Ns,Ns,2,2,'single') * 10000000;
qci0     = ones(Ns,Nd,2,2,'single') * 3000;
qa0(:,:,1,1) = qa0(:,:,1,1) + eye(No,Ns) * 990000000;
qa0(:,:,2,2) = qa0(:,:,2,2) + eye(No,Ns) * 990000000;
D        = ones(Ns,2,'single')*0.5/Nd;
E        = ones(Nd,2,'single')*0.5/Nd;

Eright   = 0.25;  % prior of selecting rightward motion
% In the paper 'Canonical neural networks perform active inference',
% Eright = 0.15 corresponds to the E_right = 0.0023 condition (black)
% Eright = 0.25 corresponds to the E_right = 0.0039 condition (blue)
% Eright = 0.35 corresponds to the E_right = 0.0055 condition (cyan)
% Please refer to Figs. 4 and 5 in the paper.

E(:,1)   = kron([0.25 0.25 0.5-Eright Eright]',ones(Nd/4,1))/(Nd/4);
E(:,2)   = 1 - E(:,1);
Q(1)     = mdp_init(qa0,qbi0,qci0,D,E,T);
Qtrue    = Q(1);

qCilist  = zeros(Ns*Nd*2,Ntrial*Nsample,'single');
Flist    = zeros(1,Ntrial*Nsample,'single');
Eest     = zeros(Nd,Ntrial*Nsample,'single');

time     = ones(Ntrial,Nsample) * T;
seed     = 0;
rng(seed+1000000);

Maplist  = cell(20,1);
for i = 1:20, Maplist{i,1} = create_maze(nx, ny); end

%--------------------------------------------------------------------------------

fig         = figure();

for h2 = 1:Nsample

Q(1)        = Qtrue;
if (h2 <= 20)
  Map = Maplist{h2,1};
else
  Map = Maplist{h2-10,1};
  Q(1).E(:,1) = mean(Eest(:,1+(0:10-1)*Ntrial)');
  Q(1).E(:,2) = 1 - Q(1).E(:,1);
end
if (h2 == 21)
  csvwrite(['nn_mdp_Eest_E',num2str(Eright),'.csv'], mean(Eest(:,1+(0:10-1)*Ntrial)')'/sum(mean(Eest(:,1+(0:10-1)*Ntrial)')))
end

for h = 1:Ntrial

[qA,qlnA]   = param_norm(Q(h).qa,sim_type);
[qBi,qlnBi] = param_norm(Q(h).qbi,sim_type);
[qCi,qlnCi] = param_norm(Q(h).qci,sim_type);
lnD         = log(max(10^-6, Q(h).D));
lnE         = log(max(10^-6, Q(h).E));
qCilist(:,(h2-1)*Ntrial+h) = reshape(qCi(:,:,1,:),[Ns*Nd*2,1]);

for t = 2:T
  % generative process
  i   = pos(1,t-1);
  j   = pos(2,t-1);
  rnd = (1:4) * Q(h).u(:,t-1);
  if (rnd == 1 && Map(i-1,j) == 0), i = i - 2; end
  if (rnd == 2 && Map(i+1,j) == 0), i = i + 2; end
  if (rnd == 3 && Map(i,j-1) == 0), j = j - 2; end
  if (rnd == 4 && Map(i,j+1) == 0), j = j + 2; end
  pos(1,t)  = i;
  pos(2,t)  = min(j,nx);
  omat      = ones(nv,nv);
  omat_temp = Map(max(i-rview,1):min(i+rview,ny),max(j-rview,1):min(j+rview,nx));
  omat(max(i-rview,1)-(i-rview)+1:min(i+rview,ny)-(i+rview)+nv,max(j-rview,1)-(j-rview)+1:min(j+rview,nx)-(j+rview)+nv) = omat_temp;
  Q(h).s(:,t) = reshape(omat,[Ns 1]); % states
  Q(h).o(:,t) = Q(h).s(:,t);          % observation
  
  % inference
  Q(h).vs(:,1,t) = qlnA(:,:,1,1)'*Q(h).o(:,t) + qlnA(:,:,2,1)'*(1-Q(h).o(:,t)) + qlnBi(:,:,1,1)'*Q(h).qs(:,1,t-1) + qlnBi(:,:,2,1)'*Q(h).qs(:,2,t-1) + lnD(:,1);
  Q(h).vs(:,2,t) = qlnA(:,:,1,2)'*Q(h).o(:,t) + qlnA(:,:,2,2)'*(1-Q(h).o(:,t)) + qlnBi(:,:,1,2)'*Q(h).qs(:,1,t-1) + qlnBi(:,:,2,2)'*Q(h).qs(:,2,t-1) + lnD(:,2);
  Q(h).vd(:,1,t) = qlnCi(:,:,1,1)'*Q(h).qs(:,1,t-1) + qlnCi(:,:,2,1)'*Q(h).qs(:,2,t-1) + lnE(:,1);
  Q(h).vd(:,2,t) = qlnCi(:,:,1,2)'*Q(h).qs(:,1,t-1) + qlnCi(:,:,2,2)'*Q(h).qs(:,2,t-1) + lnE(:,2);
  Q(h).qs(:,1,t) = 1 ./ ( 1 + exp(-(Q(h).vs(:,1,t)-Q(h).vs(:,2,t))) );
  Q(h).qs(:,2,t) = 1 - Q(h).qs(:,1,t);
  Q(h).qd(:,1,t) = 1 ./ ( 1 + exp(-(Q(h).vd(:,1,t)-Q(h).vd(:,2,t))) );
  Q(h).qd(:,2,t) = 1 - Q(h).qd(:,1,t);
  
  % decision
  Q(h).qd_(:,t)  = (Q(h).qd(:,1,t)./(Q(h).qd(:,2,t)+10^-6)) / sum(Q(h).qd(:,1,t)./(Q(h).qd(:,2,t)+10^-6));
  Q(h).d(:,t) = mnrnd(1,Q(h).qd_(:,t));                  % decision
  Q(h).u(:,t) = kron(eye(4),ones(1,Nd/4)) * Q(h).d(:,t); % action
  
  % risk
  if (pos(2,t) > pos(2,t-1))     Q(h).G(t) = 0;
  elseif (pos(2,t) < pos(2,t-1)) Q(h).G(t) = 1;
  else                           Q(h).G(t) = 0.5;
  end
  
  % judge whether the agent reaches the goal
  if (pos(:,t) == [(ny+1)/2; nx]), break, end
  
  % figure output
  if (rem(t,Nintvl) == 0)
    qsmat = reshape(Q(h).qs(:,1,t),[nv nv]);
    qdmat = qd_matrix(Q(h).qd_(:,t),nv);
    figure_output(Map, pos, t, T, h, omat, qsmat, min(qdmat*32,1), time(:,h2), kron(eye(4),ones(1,Nd/4)) * Q(h).qd_(:,t), Nintvl);
    drawnow
    pause(0.01)
  end
end
time(h,h2) = t;
Q(h).t     = t;

% compute risk
Q(h).Gamma = mdp_risk(Q(h),sim_type,200,[-10,0,10],[0.55,0.55,0.45,0]);

% learning
Q(h+1) = mdp_learning(Q(h),sim_type,200);

% compute variational free energy
Q(h).F = mdp_vfe(Q(h),sim_type,200);
Flist(:,(h2-1)*Ntrial+h) = Q(h).F;

% estimate E based exclusively on decisions
Eest(:,(h2-1)*Ntrial+h) = mean(reshape(Q(h).d(:,3:T),[Nd T-2])');

%--------------------------------------------------------------------------------

qsmat = reshape(Q(h).qs(:,1,t),[nv nv]);
qdmat = qd_matrix(Q(h).qd_(:,t),nv);
figure_output(Map, pos, t, T, h, omat, qsmat, min(qdmat*32,1), time(:,h2), kron(eye(4),ones(1,Nd/4)) * Q(h).qd_(:,t), Nintvl);
drawnow

csvwrite(['nn_mdp_time_E',num2str(Eright),'.csv'], time)
csvwrite(['nn_mdp_Flist_E',num2str(Eright),'.csv'], reshape(Flist,[Ntrial,Nsample]))

if (h2 == 1)
  if (h == 1 || h == Ntrial)
    set(fig, 'PaperPosition', [0,2,20,26])
    set(fig, 'PaperPosition', [0,16,20,12])
    if (h == 1),      print(fig, ['nn_mdp_figure_1_E',num2str(Eright),'.pdf'], '-dpdf'); end
    if (h == Ntrial), print(fig, ['nn_mdp_figure_100_E',num2str(Eright),'.pdf'], '-dpdf'); end
  end
  if (h == 1)
    csvwrite(['nn_mdp_pos_pre_E',num2str(Eright),'.csv'], pos(:,1:time(h,h2))')
  end
  if (h == Ntrial)
    csvwrite(['nn_mdp_pos_post_E',num2str(Eright),'.csv'], pos(:,1:time(h,h2))')
    img = ones(11*10,11*10,3);
    img(:,:,1) = kron(1-reshape(Q(h).o(:,t-1),[11 11]),ones(10,10));
    img(:,:,2) = img(:,:,1);
    img(:,:,3) = img(:,:,1);
    imwrite(img, ['nn_mdp_o_E',num2str(Eright),'.png'])
    img = ones(11*10,11*10,3);
    img(:,:,1) = kron(1-reshape(Q(h).qs(:,1,t-1),[11 11]),ones(10,10));
    img(:,:,2) = img(:,:,1);
    img(:,:,3) = img(:,:,1);
    imwrite(img, ['nn_mdp_qs_E',num2str(Eright),'.png'])
    qdmat = qd_matrix(Q(h).qd_(:,t-1),nv);
    img = ones(11*10,11*10,3);
    img(:,:,1) = kron(1-min(qdmat*32,1),ones(10,10));
    img(:,:,2) = img(:,:,1);
    imwrite(img, ['nn_mdp_qd_E',num2str(Eright),'.png'])
    img0 = figure_output(Map, pos, t, T, h, omat, qsmat, min(qdmat*32,1), time(:,h2), kron(eye(4),ones(1,Nd/4)) * Q(h).qd_(:,t), T);
    img = ones(ny*10,nx*10,3);
    img(:,:,1) = kron(img0(:,:,1),ones(10,10));
    img(:,:,2) = kron(img0(:,:,2),ones(10,10));
    img(:,:,3) = kron(img0(:,:,3),ones(10,10));
    imwrite(img, ['nn_mdp_maze_E',num2str(Eright),'.png'])
  end
end

end

end

%--------------------------------------------------------------------------------

