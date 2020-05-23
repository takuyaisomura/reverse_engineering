
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

function fig3(seed)

%clear
%seed = 0;
N        = 2;     % number of hidden states
M        = 32;    % number of observations
T        = 10000; % time
sim_type = 2;     % 1:MDP, 2:neural network
rng(1000000+seed)

%--------------------------------------------------------------------------------
% define generative process

A = zeros(M,2,2,2); % When N != 2, change the dimensions of A accordingly
for i = 1:M/2,   A(i,1,:,:) = [1 3/4; 1/4 0]; end
for i = M/2+1:M, A(i,1,:,:) = [1 1/4; 3/4 0]; end
A(:,2,:,:) = 1 - A(:,1,:,:);

s = zeros(T,N,2); % hidden states
o = zeros(T,M,2); % observations
for t = 1:T
  s(t,:,1) = randi([0 1],N,1);
  s(t,:,2) = 1 - s(t,:,1);
  o(t,:,1) = (rand(M,1) < A(:,1,2-s(t,1,1),2-s(t,2,1))) * 1;
  o(t,:,2) = 1 - o(t,:,1);
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
% simulation

fprintf(1,'----------------------------------------\n');
fprintf('alpha dependency\n');
corr_s_qs1 = zeros(19,2);
corr_s_qs2 = zeros(19,2);
for h = 1:19
  alpha           = log([h*0.05 1-h*0.05; h*0.05 1-h*0.05]); % constants
  beta            = zeros(M,N,2);                            % constants
  [qs qA qlnA qa] = mdp_bss(N,M,T,o,qa_init,alpha,beta,sim_type);
  corr_s_qs1(h,:) = corr(s(T/2:T,:,1),qs(T/2:T,1,1));
  corr_s_qs2(h,:) = corr(s(T/2:T,:,1),qs(T/2:T,2,1));
  fprintf('%d/%d, alpha_intensity = %.3f, corr = %.3f\n', h, 19, h*0.05, corr_s_qs1(h,1));
end
csvwrite(['mdp_bss_alpha_corr_s_qs1_',num2str(seed),'.csv'],[1:2; corr_s_qs1])
csvwrite(['mdp_bss_alpha_corr_s_qs2_',num2str(seed),'.csv'],[1:2; corr_s_qs2])
subplot(2,2,1), plot((1:19)*0.05,abs(corr_s_qs1(:,1)),'-r',(1:19)*0.05,abs(corr_s_qs1(:,2)),'-b')
subplot(2,2,2), plot((1:19)*0.05,abs(corr_s_qs2(:,1)),'-r',(1:19)*0.05,abs(corr_s_qs2(:,2)),'-b')
drawnow

%--------------------------------------------------------------------------------

fprintf('beta dependency\n');
corr_s_qs1 = zeros(17,2);
corr_s_qs2 = zeros(17,2);
for h = 1:17
  alpha           = log([0.5 0.5; 0.5 0.5]);    % constants
  beta            = randn(M,N,2) * (h-1)*0.005; % constants
  [qs qA qlnA qa] = mdp_bss(N,M,T,o,qa_init,alpha,beta,sim_type);
  corr_s_qs1(h,:) = corr(s(T/2:T,:,1),qs(T/2:T,1,1));
  corr_s_qs2(h,:) = corr(s(T/2:T,:,1),qs(T/2:T,2,1));
  fprintf('%d/%d, beta_intensity = %.3f, corr = %.3f\n', h, 17, (h-1)*0.005, corr_s_qs1(h,1));
end
csvwrite(['mdp_bss_beta_corr_s_qs1_',num2str(seed),'.csv'],[1:2; corr_s_qs1])
csvwrite(['mdp_bss_beta_corr_s_qs2_',num2str(seed),'.csv'],[1:2; corr_s_qs2])
subplot(2,2,3), plot(((1:17)-1)*0.005,abs(corr_s_qs1(:,1)),'r'), hold on
subplot(2,2,3), plot(((1:17)-1)*0.005,abs(corr_s_qs1(:,2)),'b'), hold off
subplot(2,2,4), plot(((1:17)-1)*0.005,abs(corr_s_qs2(:,1)),'r'), hold on
subplot(2,2,4), plot(((1:17)-1)*0.005,abs(corr_s_qs2(:,2)),'b'), hold off
drawnow
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

