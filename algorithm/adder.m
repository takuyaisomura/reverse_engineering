
%--------------------------------------------------------------------------------
% adder.m
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
% 2024-08-04
%

%--------------------------------------------------------------------------------
% initialisation

clear
T        = 2048;                   % duration of a session
Nnum     = 16;                     % number of bits for representing input numbers
Nct      = 16;                     % number of bits for counting time steps
Nimg     = 28*28;                  % size of hand digit image
No       = Nnum*Nimg + Nct*16 + 1; % dimensionality of sensory inputs = 12800
Ns       = Nnum + Nct;             % dimensionality of hidden states = 32
Nd       = 1;                      % dimensionality of actions
Ng       = 2;                      % dimensionality of risks
NC       = Nct;                    % dimensionality of memory matrix = 16
sim_type = 2;                      % 1:MDP, 2:neural network
T1       = Nct*1;                  % interval for learning
Nsession = 100;                    % number of sessions

seed     = 0;
rng(seed+1000000);

qs       = zeros(Ns*2,T);          % hidden state posterior
qd       = zeros(Nd*2,T);          % action posterior
qg       = zeros(Ng*2,T);          % risk posterior
psis     = zeros(Ns,T);            % basis
psid     = zeros(Nct,T);           % basis
psig     = zeros(4,T);             % basis
G        = [1 0 0 0; 0 1 1 1];     % risk matrix determined by genes

qlnA_o   = zeros(Ns*2,T);
qClist   = zeros(2,NC,T);          % memory posterior
sum_est  = zeros(T/T1,1);
err_sum  = zeros(Nsession,T/T1);
err_B    = zeros(Nsession,T/T1);
err_C    = zeros(Nsession,T/T1);
Glist    = [0 1];
for i = 2:8, Glist = [Glist Glist; zeros(1,size(Glist,2)) ones(1,size(Glist,2))]; end

%--------------------------------------------------------------------------------
% simulation

for h = 1:Nsession
 fprintf(1,'h=%d\n',h)
 [o,s,A,B,Clist,sum_true,qa0,qb0,qc0] = adder_init(T); % define input sequences and initial parameters
 qa = qa0;
 qb = qb0*(rand()*3+1);
 qc = qc0;
 
 for t = 1:T
  if rem(t,T1) == 1
   [qA,qlnA] = param_norm(qa,sim_type);
   [qB,qlnB] = param_norm(qb,sim_type);
   qlnA_o(:,t:t+T1-1) = qlnA([1:No-1,No+1:No*2-1],:)'*[o(1:No-1,t:t+T1-1);1-o(1:No-1,t:t+T1-1)];
  end
  [qClist(:,:,t),~] = param_norm(qc,sim_type);
  if rem(t,T/4)==0, qc = qc0; end
  [qC,qlnC] = param_norm(qc,sim_type);
  
  % inference at step1
  if t == 1
   qs(:,t)   = softmax_b(qlnA_o(:,t));
   qd(:,t)   = qC(:,1);
   psis(:,t) = qs(1:Ns,t);
   psid(:,t) = circshift(eye(Nct),[1 0])*qs(Nnum+1:Ns,t);
   err_B(t)  = sum(sum((qB-B).^2))/sum(sum((B).^2));
   err_C(t)  = sum((qC(1,:)'-Clist(:,t)).^2)/NC;
   continue
  end
  
  % sensory input
  % o(1:No-1) is determined by the environment, while o(No,t) is determined by the previous mental action
  o(No,t)     = (rand() < qd(1,t-1)) * 1;
  qlnA_o(:,t) = qlnA_o(:,t) + qlnA([No,No*2],:)'*[o(No,t);1-o(No,t)];
  
  % inference
  qs(:,t)   = softmax_b(qlnA_o(:,t) + qlnB*[psis(:,t-1);1-psis(:,t-1)]); % inference of hidden states (state update)
  qd(:,t)   = softmax_b(qlnC*psid(:,t-1));                               % inference of mental actions (memory read)
  % compute basis functions
  psis(:,t) = qs(1:Ns,t);
  psid(:,t) = circshift(eye(Nct),[1 0])*qs(Nnum+1:Ns,t);
  psig(:,t) = [qs(1:Nnum,t)'*qs(Nnum+1:Ns,t); qs(1:Nnum,t-1)'*qs(Nnum+1:Ns,t-1); qd(1,t-1); qg(2,t-1)]-0.5;
  % compute risks
  qg(:,t)   = softmax_b(50*[G;-G] * psig(:,t));
  
  % learning
  if rem(t,T1) == 0
   qa = qa + [o(:,t-T1+1:t);1-o(:,t-T1+1:t)]*qs(:,t-T1+1:t)';
   qb = qb + qs(:,max(t-T1,1)+1:t)*qs(:,max(t-T1,1):t-1)';
  end
  qc = qc + 2*(1-2*qg(2,t))*(1-2*qg(1,t))*qd(:,t)*psid(:,t-1)'; % memory write
  qc = min(qc - min(qc) + 10^-8,1);                             % restrict the values within the range of 0 and 1
  
  if rem(t,T1) == 0
   sum_est(t/T1)   = binary_to_decimal(qClist(1,:,t)');
   err_sum(h,t/T1) = abs(sum_true(t/T1) - sum_est(t/T1))/mean(sum_true);
   err_B(h,t/T1)   = sum(sum((qB-B).^2))/sum(sum((B).^2));
   err_C(h,t/T1)   = sum((reshape(qClist(1,:,t),[NC,1])-Clist(:,t)).^2)/NC;
  end
 end
 subplot(5,1,1), image(reshape(o(1:Nnum*Nimg,t),28,[])*300), colormap(gray), title('sensory inputs o')
 subplot(5,1,2), image(reshape(qs(1:Ns,1:t),[Ns,t])*300), axis([0 T 0.5 Ns+0.5]), title('state posterior qs')
 subplot(5,1,3), image(reshape(qClist(1,:,1:t),[NC,t])*300), axis([0 T 0.5 NC+0.5]), title('memory posterior qC')
 subplot(5,3,[10 13]), plot(1:T/T1,sum_est,'r-',1:T/T1,sum_true,'k-',1:T/T1,sum_est,'r--','LineWidth',2)
 axis([0 T/T1 0 4*10^4]), title('true and estimated total numbers')
 subplot(5,3,[11 14]), fill([1:T/T1 T/T1:-1:1],[prctile(err_sum(1:h,:),25,1) flip(prctile(err_sum(1:h,:),75,1))],[0.8,0.8,1],'LineStyle','none'), hold on
 plot(1:T/T1,prctile(err_sum(1:h,:),50,1),'b-','LineWidth',2), axis([0 T/T1 0 1]), title('total-number estimation error'), hold off
 subplot(5,3,[12 15]), fill([1:T/T1 T/T1:-1:1],[prctile(err_B(1:h,:),25,1) flip(prctile(err_B(1:h,:),75,1))],[0.8,0.8,1],'LineStyle','none'), hold on
 fill([1:T/T1 T/T1:-1:1],[prctile(err_C(1:h,:),25,1) flip(prctile(err_C(1:h,:),75,1))],[0.8,1,0.8],'LineStyle','none')
 plot(1:T/T1,median(err_B(1:h,:),1),'b-',1:T/T1,median(err_C(1:h,:),1),'g-','LineWidth',2), axis([0 T/T1 0 1]), title('parameter estimation errors'), hold off
 drawnow, pause(0.1)
 if h == 1
  % data and figure output
  csvwrite('output_true_and_estimated_sum.csv',[1:T/T1; sum_true'; sum_est']);
  img = reshape(o(1:Nnum*Nimg,t),28,[]); img = kron(img,ones(10,10)); img(:,:,2) = img(:,:,1); img(:,:,3) = img(:,:,1)*0.7+0.3;
  imwrite(img, ['fig_sensory_inputs_o.png'])
  img = reshape(qs(1:Ns,1:t),[Ns,t]); img = kron(flip(img,1),ones(16,4)); img(:,:,2) = img(:,:,1); img(:,:,3) = img(:,:,1)*0.7+0.3;
  imwrite(img, ['fig_state_posterior_qs.png'])
  img = reshape(qClist(1,:,1:t),[NC,t]); img = kron(flip(img,1),ones(32,4)); img(:,:,2) = img(:,:,1); img(:,:,3) = img(:,:,1)*0.7+0.3;
  imwrite(img, ['fig_memory_posterior_qC.png'])
 end
end

% data output
csvwrite(['output_err_sum.csv'], [1:T/T1; err_sum])
csvwrite(['output_err_B.csv'], [1:T/T1; err_B])
csvwrite(['output_err_C.csv'], [1:T/T1; err_C])

%--------------------------------------------------------------------------------
