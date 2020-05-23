
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
fprintf('trajectory of Bayes optimal encoder\n');
alpha   = log([0.5 0.5; 0.5 0.5]); % constants
beta    = zeros(M,N,2);            % constants
[qs qA qlnA qa] = mdp_bss(N,M,T,o,qa_init,alpha,beta,sim_type);

fprintf('corr(s1,qs1) = %.3f, corr(s1,qs2) = %.3f\n', corr(s(T/2:T,1,1),qs(T/2:T,1,1)), corr(s(T/2:T,1,1),qs(T/2:T,2,1)));
fprintf('corr(s2,qs1) = %.3f, corr(s2,qs2) = %.3f\n', corr(s(T/2:T,2,1),qs(T/2:T,1,1)), corr(s(T/2:T,2,1),qs(T/2:T,2,1)));

s1qs1 = zeros(100,2);
s2qs2 = zeros(100,2);
for t = 1:100
  set        = T/100*(t-1)+1:T/100*t;
  s1qs1(t,1) = corr(s(set,1,1),qs(set,1,1)) * std(qs(set,1,1)) / std(s(set,1,1));
  s1qs1(t,2) = corr(s(set,1,1),qs(set,1,2)) * std(qs(set,1,2)) / std(s(set,1,2));
  s2qs2(t,1) = corr(s(set,2,1),qs(set,2,1)) * std(qs(set,2,1)) / std(s(set,2,1));
  s2qs2(t,2) = corr(s(set,2,1),qs(set,2,2)) * std(qs(set,2,2)) / std(s(set,2,2));
end

omega  = ((1:5)'-0.5)*pi/T;
coef11 = (sin(omega*find(s(:,1,1)==1)')*sin(omega*find(s(:,1,1)==1)')')^(-1)*sin(omega*find(s(:,1,1)==1)')*(qs(s(:,1,1)==1,1,1)-0.5);
coef12 = (sin(omega*find(s(:,1,1)==0)')*sin(omega*find(s(:,1,1)==0)')')^(-1)*sin(omega*find(s(:,1,1)==0)')*(qs(s(:,1,1)==0,1,1)-0.5);
coef21 = (sin(omega*find(s(:,2,1)==1)')*sin(omega*find(s(:,2,1)==1)')')^(-1)*sin(omega*find(s(:,2,1)==1)')*(qs(s(:,2,1)==1,2,1)-0.5);
coef22 = (sin(omega*find(s(:,2,1)==0)')*sin(omega*find(s(:,2,1)==0)')')^(-1)*sin(omega*find(s(:,2,1)==0)')*(qs(s(:,2,1)==0,2,1)-0.5);

curve = ones(T,2,2)*0.5;
for k = 1:5
  curve(:,1,1) = curve(:,1,1) + coef11(k)*sin(omega(k)*(1:T))';
  curve(:,1,2) = curve(:,1,2) + coef12(k)*sin(omega(k)*(1:T))';
  curve(:,2,1) = curve(:,2,1) + coef21(k)*sin(omega(k)*(1:T))';
  curve(:,2,2) = curve(:,2,2) + coef22(k)*sin(omega(k)*(1:T))';
end

csvwrite(['mdp_bss_s1_qs1_',num2str(seed),'.csv'],[1:4; s(:,1,1) qs(:,1,1) curve(:,1,1) curve(:,1,2)])
csvwrite(['mdp_bss_s2_qs2_',num2str(seed),'.csv'],[1:4; s(:,2,1) qs(:,2,1) curve(:,2,1) curve(:,2,2)])

subplot(2,1,1), plot(find(s(:,1,1)==1),qs(s(:,1,1)==1,1,1),'r.',find(s(:,1,1)==0),qs(s(:,1,1)==0,1,1),'b.'), hold on
subplot(2,1,1), plot(1:T,curve(:,1,1),'r.',1:T,curve(:,1,2),'b.'), hold off
axis([0 T 0 1])
subplot(2,1,2), plot(find(s(:,2,1)==1),qs(s(:,2,1)==1,2,1),'r.',find(s(:,2,1)==0),qs(s(:,2,1)==0,2,1),'b.'), hold on
subplot(2,1,2), plot(1:T,curve(:,2,1),'r.',1:T,curve(:,2,2),'b.'), hold off
axis([0 T 0 1])
drawnow
fprintf(1,'----------------------------------------\n\n');

%--------------------------------------------------------------------------------

traj_qA      = zeros(T,2);
traj_qA(:,1) = mean(reshape(qA(:,1:M/2,1,1,1),[T M/2])');
traj_qA(:,2) = mean(reshape(qA(:,M/2+1:M,1,2,1),[T M/2])');
csvwrite(['traj_qA_',num2str(seed),'.csv'],[1:2; traj_qA])

qA  = reshape(mean(reshape(qA, [10 T/10*M*2*N*2])),[T/10 M 2 N 2]);
csvwrite(['synapse_',num2str(seed),'.csv'],[1:128; reshape(qA(:,:,1,:,:), [1000 M*N*2])])

L = zeros(25,25);
for ii = 1:25
 for jj = 1:25
  for i = 1:M/2,   qa_init(i,1,:,:) = [0.5+(ii-1)*0.02 0.5-(ii-1)*0.02; 0.5+(jj-1)*0.02 0.5-(jj-1)*0.02]; end
  for i = M/2+1:M, qa_init(i,1,:,:) = [0.5+(jj-1)*0.02 0.5-(jj-1)*0.02; 0.5+(ii-1)*0.02 0.5-(ii-1)*0.02]; end
  qa_init(:,2,:,:) = 1 - qa_init(:,1,:,:);
  qa_init  = qa_init * amp;
  L(ii,jj) = cost_function(N,M,T,o,qa_init,alpha,beta,sim_type);
  fprintf(1,'i=%d,j=%d,L=%.0f\n',ii,jj,L(ii,jj));
 end
end
csvwrite(['cost_function_',num2str(seed),'.csv'],[1:25; L])

%--------------------------------------------------------------------------------
