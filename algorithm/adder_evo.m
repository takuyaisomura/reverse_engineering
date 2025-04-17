
%--------------------------------------------------------------------------------
% adder_evo.m
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
Nsample  = 100;                    % number of samples
Nsession = 40;                     % number of sessions

seed     = 0;
rng(seed+1000000);

Xicode   = [0 1];
for i = 2:8, Xicode = [Xicode Xicode; zeros(1,size(Xicode,2)) ones(1,size(Xicode,2))]; end
Pxi      = zeros(2^8,Nsession);
Xilist   = zeros(Nsample,Nsession);
Pxi(:,1) = ones(256,1)/256;
err_xi   = zeros(Nsample,Nsession);
err_xi2  = zeros(Nsample,Nsession);
err_Bevo = zeros(Nsample,Nsession);
err_Cevo = zeros(Nsample,Nsession);
Mut      = zeros(2^8,2^8);
Mut_eps  = 0.01;
for i = 1:2^8
 Mut(i,i) = 1-Mut_eps*8;
 for j = 1:8, Mut(bitxor(i-1,2^(j-1))+1,i) = Mut_eps; end
end
fig      = figure();

%--------------------------------------------------------------------------------
% simulation

for h2 = 1:Nsession
 if h2 >= 2
  Pxi(:,h2) = Pxi(:,h2-1);
  Eye    = eye(2^8);
  Err_xi = err_xi(:,h2-1)'*Eye(:,Xilist(:,h2-1))'./sum(Eye(:,Xilist(:,h2-1))');
  Pxi(~isnan(Err_xi),h2) = exp(-Err_xi(~isnan(Err_xi))/50)' .* Pxi(~isnan(Err_xi),h2-1);
  Pxi(~isnan(Err_xi),h2) = Pxi(~isnan(Err_xi),h2) / sum(Pxi(~isnan(Err_xi),h2)) * sum(Pxi(~isnan(Err_xi),h2-1));
  Pxi(:,h2) = Mut * Pxi(:,h2);
  Pxi(:,h2) = Pxi(:,h2)/sum(Pxi(:,h2));
 end
 for h = 1:Nsample, Xilist(h,h2) = (1:2^8)*mnrnd(1,Pxi(:,h2))'; end
 
 parfor h = 1:Nsample
  [o,s,A,B,Clist,sum_true,qa0,qb0,qc0] = adder_init(T); % define input sequences and initial parameters
  qa = qa0;
  qb = qb0*(rand()*3+1);
  qc = qc0;
  G  = reshape(Xicode(:,Xilist(h,h2)), [2 4]); % risk matrix determined by genes
  qClist   = zeros(2,NC,T);      % memory posterior
  qs       = zeros(Ns*2,T);      % hidden state posterior
  qd       = zeros(Nd*2,T);      % action posterior
  qg       = zeros(Ng*2,T);      % risk posterior
  psis     = zeros(Ns,T);        % basis
  psid     = zeros(Nct,T);       % basis
  psig     = zeros(4,T);         % basis
  qlnA_o   = zeros(Ns*2,T);
  sum_est  = zeros(T/T1,1);
  err_sum  = zeros(Nsample,T/T1);
  err_B    = zeros(Nsample,T/T1);
  err_C    = zeros(Nsample,T/T1);
  
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
%  subplot(5,1,1), image(reshape(o(1:Nnum*Nimg,t),28,[])*300), colormap(gray), title('sensory inputs o')
%  subplot(5,1,2), image(reshape(qs(1:Ns,1:t),[Ns,t])*300), axis([0 T 0.5 Ns+0.5]), title('state posterior qs')
%  subplot(5,1,3), image(reshape(qClist(1,:,1:t),[NC,t])*300), axis([0 T 0.5 NC+0.5]), title('memory posterior qC')
%  subplot(5,3,[10 13]), plot(1:T/T1,sum_est,'r-',1:T/T1,sum_true,'k-',1:T/T1,sum_est,'r--','LineWidth',2)
%  axis([0 T/T1 0 4*10^4]), title('true and estimated total numbers')
%  subplot(5,3,[11 14]), plot(1:T/T1,err_sum(h,:),'b-','LineWidth',2), axis([0 T/T1 0 1]), title('total-number estimation error')
  % subplot(5,3,[12 15]), plot(1:T/T1,err_B(h,:),'b-',1:T/T1,err_C(h,:),'g-','LineWidth',2), axis([0 T/T1 0 1]), title('parameter estimation errors'), hold off
%  subplot(5,3,[12 15]), plot(1:2^8,Pxi(:,h2),'b-','LineWidth',2), axis([0 2^8 0 max(Pxi(:,h2))]), title('gene distribution'), hold off
%  drawnow
  
  err_xi(h,h2)   = sum(err_C(h,:)) + 1000*mean(mean(Clist-reshape(qClist(1,:,:),[NC,T])))^2;
  err_xi2(h,h2)  = sum(err_sum(h,:));
  err_Bevo(h,h2) = mean(err_B(h,T/T1/4*3+1:T/T1));
  err_Cevo(h,h2) = mean(err_C(h,T/T1/4*3+1:T/T1));
  fprintf(1,'gen=%d, h=%d, gene=(', h2, h), fprintf(1,'%d,',Xicode(:,Xilist(h,h2))), fprintf(1,'), err_sum=%f\n', err_xi(h,h2))
 end
 
 % data output
 csvwrite(['output_err_xi.csv'], [1:Nsession; err_xi])
 csvwrite(['output_err_xi2.csv'], [1:Nsession; err_xi2])
 csvwrite(['output_Xilist.csv'], [1:Nsession; Xilist])
 csvwrite(['output_Pxi.csv'], [1:Nsession; Pxi])
 csvwrite(['output_err_Bevo.csv'], [1:Nsession; err_Bevo])
 csvwrite(['output_err_Cevo.csv'], [1:Nsession; err_Cevo])
 
 img = reshape(log10(max(Pxi,10^-3))+3,[2^8,Nsession]);
 img = kron(img,ones(2,12));
 img(:,:,3) = img(:,:,1);
 img(:,:,2) = max(img(:,:,3)-1,0);
 img(:,:,1) = max(img(:,:,3)-2,0);
 img = min(img,1);
 imwrite(img, ['fig_output_Pxi.png'])
 
 %--------------------------------------------------------------------------------
 
 if h2 == 6
  rng(seed+1100000);
  [o,s,A,B,Clist,sum_true,qa0,qb0,qc0] = adder_init(T); % define input sequences and initial parameters
  xi_id = zeros(6,1);
  for h3 = 1:6
   qc        = qc0;
   [~,h]     = min(err_xi2(:,h3));
   xi_id(h3) = Xilist(h,h3);
   G         = reshape(Xicode(:,Xilist(h,h3)), [2 4]);
   fprintf(1,'h2 = %d, xi_id = %d, xi = %d%d%d%d%d%d%d%d\n', h3, Xilist(h,h3), Xicode(:,Xilist(h,h3)))
   
   qs       = zeros(Ns*2,T);      % hidden state posterior
   qd       = zeros(Nd*2,T);      % action posterior
   qg       = zeros(Ng*2,T);      % risk posterior
   psis     = zeros(Ns,T);        % basis
   psid     = zeros(Nct,T);       % basis
   psig     = zeros(4,T);         % basis
   
   for t = 1:48
    [qC,qlnC] = param_norm(qc,sim_type);
    
    % inference at step1
    if t == 1
     qs(:,t)   = [s(:,t); 1-s(:,t)];
     qd(:,t)   = qC(:,1);
     psis(:,t) = qs(1:Ns,t);
     psid(:,t) = circshift(eye(Nct),[1 0])*qs(Nnum+1:Ns,t);
     continue
    end
    
    % inference
    qs(:,t)   = [s(:,t); 1-s(:,t)];
    qd(:,t)   = softmax_b(qlnC*psid(:,t-1));                               % inference of mental actions (memory read)
    % compute basis functions
    psis(:,t) = qs(1:Ns,t);
    psid(:,t) = circshift(eye(Nct),[1 0])*qs(Nnum+1:Ns,t);
    psig(:,t) = [qs(1:Nnum,t)'*qs(Nnum+1:Ns,t); qs(1:Nnum,t-1)'*qs(Nnum+1:Ns,t-1); qd(1,t-1); qg(2,t-1)]-0.5;
    % compute risks
    qg(:,t)   = softmax_b(50*[G;-G] * psig(:,t));
    
    % learning
    qc = qc + 2*(1-2*qg(2,t))*(1-2*qg(1,t))*qd(:,t)*psid(:,t-1)'; % memory write
    qc = min(qc - min(qc) + 10^-8,1);
    if t == 32, [qC_old,~] = param_norm(qc,sim_type); end
   end
   [qC,qlnC] = param_norm(qc,sim_type);
   subplot(3,6,h3+0), image(s(1:16,48)'*300), title(['xi-id = ',num2str(xi_id(h3))])
   subplot(3,6,h3+6), image(qC_old(1,:)*300), title(['xi-id = ',num2str(xi_id(h3))])
   subplot(3,6,h3+6*2), image(qC(1,:)*300), title(['xi-id = ',num2str(xi_id(h3))])
  end
  print(fig, ['results_writing_rules.pdf'], '-dpdf','-bestfit');
 end
 
 %--------------------------------------------------------------------------------
 
 subplot(1,2,1), plot(1:2^8,Pxi(:,h2),'b-','LineWidth',2), axis([0 2^8 0 max(Pxi(:,h2))]), title('gene distribution'), hold off
 if h2 >= 6
  subplot(1,2,2), plot(1:Nsession,Pxi(xi_id(1),:),1:Nsession,Pxi(xi_id(2),:),1:Nsession,Pxi(xi_id(3),:),1:Nsession,Pxi(xi_id(4),:),1:Nsession,Pxi(xi_id(5),:),1:Nsession,Pxi(xi_id(6),:))
 end
 drawnow
 
end

%--------------------------------------------------------------------------------
