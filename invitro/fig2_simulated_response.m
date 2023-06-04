
%--------------------------------------------------------------------------------
% fig2_simulated_response.m
%
% This analysis is included in
% Experimental validation of the free-energy principle with in vitro neural networks
% Takuya Isomura, Kiyoshi Kotani, Yasuhiko Jimbo, Karl Friston
%
% Copyright (C) 2022 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2022-09-16
%

%--------------------------------------------------------------------------------

clear
N           = 2;             % number of hidden states
M           = 32;            % number of observations
T           = 100*256;       % total time
sim_type    = 1;             % 1:POMDP, 2:neural network
prob_on     = [0.2 0.5 0.8]; % state prior (D_1)
qa_eps      = 0.01;          % bias for prior of parameters
qa_amp_min  = 200;           % amplitude for prior of parameters
qa_amp_max  = 400;           % amplitude for prior of parameters

seed        = 0;
rng(1000000+seed)

load('response_data_ctrl.mat')

for condition = 1:3
for h = 1:100
%--------------------------------------------------------------------------------
% load generative process

s = zeros(T,N,2); % hidden states
o = zeros(T,M,2); % observations
rand_sample  = randi([1 length(data_ctrl)]);
rand_permute = randi([1 2]);
if rand_permute == 1
 s(:,:,1) = data_ctrl{rand_sample}.s;     % read hidden states from data file
 o(:,:,1) = data_ctrl{rand_sample}.o;     % read observations from data file
else
 s(:,:,1) = 1 - data_ctrl{rand_sample}.s; % read hidden states from data file
 o(:,:,1) = 1 - data_ctrl{rand_sample}.o; % read observations from data file
end
s(:,:,2) = 1 - s(:,:,1);
o(:,:,2) = 1 - o(:,:,1);

%--------------------------------------------------------------------------------
% initial connection strengths (prior beliefs about parameters)

qa_init = zeros(M,2,N,2); % parameter prior or initial synaptic strengths
for i = 1:M/2,   qa_init(i,1,:,:) = [0.5+qa_eps*2 0.5-qa_eps*2; 0.5+qa_eps 0.5-qa_eps]; end
for i = M/2+1:M, qa_init(i,1,:,:) = [0.5+qa_eps 0.5-qa_eps; 0.5+qa_eps*2 0.5-qa_eps*2]; end
qa_init(:,2,:,:) = 1 - qa_init(:,1,:,:);
for i = 1:M
 qa_init(i,:,1,:) = qa_init(i,:,1,:) * (qa_amp_min + (qa_amp_max-qa_amp_min) * rand());
 qa_init(i,:,2,:) = qa_init(i,:,2,:) * (qa_amp_min + (qa_amp_max-qa_amp_min) * rand());
end

%--------------------------------------------------------------------------------
% simulation

fprintf(1,'----------------------------------------\n');
fprintf('trajectory of ideal Bayesain encoder, D_1 = %.2f, h = %d\n', prob_on(condition), h);
D   = [prob_on(condition) 1-prob_on(condition); prob_on(condition) 1-prob_on(condition)]; % constants
[qs qA qlnA qa] = pomdp_bss(N,M,T,o,qa_init,D,sim_type);

tt = T/2+1:T;
fprintf('corr(s1,qs1)=%.3f, corr(s1,qs2)=%.3f, ', corr(s(tt,1,1),qs(tt,1,1)), corr(s(tt,1,1),qs(tt,2,1)));
fprintf('corr(s2,qs1)=%.3f, corr(s2,qs2)=%.3f\n', corr(s(tt,2,1),qs(tt,1,1)), corr(s(tt,2,1),qs(tt,2,1)));

qs_mean = zeros(100,2,2);
for i = 1:100
 tt   = 256*(i-1)+1:256*i;
 s_i  = s(tt,:,:);
 qs_i = qs(tt,:,:);
 qs_mean(i,1,1) = mean(qs_i(s_i(:,1,1)==1,1,1));
 qs_mean(i,1,2) = mean(qs_i(s_i(:,1,1)==0,1,1));
 qs_mean(i,2,1) = mean(qs_i(s_i(:,2,1)==1,2,1));
 qs_mean(i,2,2) = mean(qs_i(s_i(:,2,1)==0,2,1));
end

subplot(2,1,1), plot(1:100,qs_mean(:,1,1),'r-',1:100,qs_mean(:,1,2),'b-'), axis([0 100 0 1])
subplot(2,1,2), plot(1:100,qs_mean(:,2,1),'r-',1:100,qs_mean(:,2,2),'b-'), axis([0 100 0 1])
drawnow

csvwrite(['pomdp_bss_qs_',num2str(floor(prob_on(condition)*100)),'_',num2str(h),'.csv'],[1 0 1 0; qs_mean(:,1,1) qs_mean(:,1,2) qs_mean(:,2,1) qs_mean(:,2,2)])

%--------------------------------------------------------------------------------
end
end

return

