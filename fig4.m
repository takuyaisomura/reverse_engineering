
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

clear
N        = 2;     % number of hidden states
M        = 32;    % number of observations
T        = 10000; % time
sim_type = 2;     % 1:MDP, 2:neural network
seed     = 0;
rng(1000000+seed) % set seed for reproducibility

%--------------------------------------------------------------------------------
% define generative process

A = zeros(M,2,2,2); % if N != 2, change the dimentionality of A
for i = 1:M/2,   A(i,1,:,:) = [1 3/4; 1/4 0]; end
for i = M/2+1:M, A(i,1,:,:) = [1 1/4; 3/4 0]; end
A(:,2,:,:) = 1 - A(:,1,:,:);

s  = zeros(T,N,2); % hidden states
o  = zeros(T,M,2); % observations
s0 = zeros(T,N,2); % hidden states
o0 = zeros(T,M,2); % observations
for t = 1:T
  s0(t,:,1) = randi([0 1],N,1);
  s0(t,:,2) = 1 - s0(t,:,1);
  o0(t,:,1) = (rand(M,1) < A(:,1,2-s0(t,1,1),2-s0(t,2,1))) * 1;
  o0(t,:,2) = 1 - o0(t,:,1);
end

%--------------------------------------------------------------------------------
% initial connection strengths (prior beliefs of parameters)

eps     = 0.01;           % bias for prior of parameters
amp     = 100;            % amplitude for prior of parameters
qa_init = zeros(M,2,N,2); % parameter prior or initial synaptic strengths
for i = 1:M/2,   qa_init(i,1,:,:) = [0.5+eps 0.5-eps; 0.5 0.5]; end
for i = M/2+1:M, qa_init(i,1,:,:) = [0.5 0.5; 0.5+eps 0.5-eps]; end
qa_init(:,2,:,:) = 1 - qa_init(:,1,:,:);
qa_init = qa_init * amp;

%--------------------------------------------------------------------------------
%--------------------------------------------------------------------------------
% estimation

fprintf(1,'----------------------------------------\n');
fprintf('estimation of alpha\n');
exp_phi1 = zeros(13,13);
exp_phi2 = zeros(13,13);
for h1 = 1:13
  for h2 = 1:13
    for t = 1:T
      s(t,:,1) = randi([0 1],N,1);
      s(t,:,2) = 1 - s(t,:,1);
      o(t,:,1) = (rand(M,1) < A(:,1,2-s(t,1,1),2-s(t,2,1))) * 1;
      o(t,:,2) = 1 - o(t,:,1);
    end
    alpha           = log([h1*0.05+0.15 1-h1*0.05-0.15; h2*0.05+0.15 1-h2*0.05-0.15]); % constants
    beta            = zeros(M,N,2);                            % constants
    [qs qA qlnA qa] = mdp_bss(N,M,T,o,qa_init,alpha,beta,sim_type);
    exp_phi1(h1,h2) = mean(qs(1:T,1,1));
    exp_phi2(h1,h2) = mean(qs(1:T,2,1));
    fprintf('exp_alpha = (%.3f, %.3f), exp_phi1 = %.3f, exp_phi2 = %.3f\n', h1*0.05+0.15, h2*0.05+0.15, exp_phi1(h1,h2), exp_phi2(h1,h2));
  end
end
csvwrite(['mdp_bss_exp_phi1.csv'],[1:13; exp_phi1])
csvwrite(['mdp_bss_exp_phi2.csv'],[1:13; exp_phi2])
drawnow
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

fprintf('estimation of cost function\n');
alpha_est = log([exp_phi1(7,7) 1-exp_phi1(7,7); exp_phi2(7,7) 1-exp_phi2(7,7)]);
beta      = zeros(M,N,2);
L = zeros(25,25);
for ii = 1:25
 for jj = 1:25
  for i = 1:M/2,   qa_init(i,1,:,:) = [0.5+(ii-1)*0.02 0.5-(ii-1)*0.02; 0.5+(jj-1)*0.02 0.5-(jj-1)*0.02]; end
  for i = M/2+1:M, qa_init(i,1,:,:) = [0.5+(jj-1)*0.02 0.5-(jj-1)*0.02; 0.5+(ii-1)*0.02 0.5-(ii-1)*0.02]; end
  qa_init(:,2,:,:) = 1 - qa_init(:,1,:,:);
  qa_init  = qa_init * amp;
  L(ii,jj) = cost_function(N,M,T,o0,qa_init,alpha_est,beta,sim_type);
  fprintf(1,'i=%d,j=%d,L=%.0f\n',ii,jj,L(ii,jj));
 end
end
csvwrite(['cost_function_est.csv'],[1:25; L])
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

fprintf('prediction of synaptic trajectory\n');
% new environment for test
A = zeros(M,2,2,2); % if N != 2, change the dimentionality of A
for i = 1:M
  rnd = rand();
  A(i,1,:,:) = [1 rnd; 1-rnd 0];
end
A(:,2,:,:) = 1 - A(:,1,:,:);

s = zeros(T,N,2); % hidden states
o = zeros(T,M,2); % observations
for t = 1:T
  s(t,:,1) = randi([0 1],N,1);
  s(t,:,2) = 1 - s(t,:,1);
  o(t,:,1) = (rand(M,1) < A(:,1,2-s(t,1,1),2-s(t,2,1))) * 1;
  o(t,:,2) = 1 - o(t,:,1);
end

% initial connection strengths (prior beliefs of parameters)
qa_init = zeros(M,2,N,2); % parameter prior or initial synaptic strengths
for i = 1:M/2
  rnd = rand() * 4 + 1;
  qa_init(i,1,:,:) = [0.5+eps*rnd 0.5-eps*rnd; 0.5 0.5];
end
for i = M/2+1:M
  rnd = rand() * 4 + 1;
  qa_init(i,1,:,:) = [0.5 0.5; 0.5+eps*rnd 0.5-eps*rnd];
end
qa_init(:,2,:,:) = 1 - qa_init(:,1,:,:);
qa_init = qa_init * amp;
qa_init2 = zeros(M,2,N,2); % parameter prior or initial synaptic strengths
for i = 1:M/2
  rnd = rand() * 4 + 1;
  qa_init2(i,1,:,:) = [0.5+eps*rnd 0.5-eps*rnd; 0.5 0.5];
end
for i = M/2+1:M
  rnd = rand() * 4 + 1;
  qa_init2(i,1,:,:) = [0.5 0.5; 0.5+eps*rnd 0.5-eps*rnd];
end
qa_init2(:,2,:,:) = 1 - qa_init2(:,1,:,:);
qa_init2 = qa_init2 * amp;

alpha     = log([0.5 0.5; 0.5 0.5]);
alpha_est = log([exp_phi1(7,7) 1-exp_phi1(7,7); exp_phi2(7,7) 1-exp_phi2(7,7)]);
beta      = zeros(M,N,2);
[qs qA qlnA qa]     = mdp_bss(N,M,T,o,qa_init, alpha,    beta,sim_type);
[qs2 qA2 qlnA2 qa2] = mdp_bss(N,M,T,o,qa_init2,alpha_est,beta,sim_type);
qA  = reshape(mean(reshape(qA, [10 T/10*M*2*N*2])),[T/10 M 2 N 2]);
qA2 = reshape(mean(reshape(qA2,[10 T/10*M*2*N*2])),[T/10 M 2 N 2]);

subplot(1,2,1), plot(1:T/10,qA(:,1,1,1,1),1:T/10,qA2(:,1,1,1,1))
subplot(1,2,2), plot(reshape(qA(T/10,:,1,:,:),[M*N*2 1]),reshape(qA2(T/10,:,1,:,:),[M*N*2 1]),'+')
drawnow

csvwrite(['synapse_true.csv'],[1:128; reshape(qA(:,:,1,:,:), [1000 M*N*2])])
csvwrite(['synapse_pred.csv'],[1:128; reshape(qA2(:,:,1,:,:),[1000 M*N*2])])
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------
