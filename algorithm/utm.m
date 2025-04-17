
%--------------------------------------------------------------------------------
% utm.m
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
% initialisation

clear
T        = 100000;       % duration
Ncontext = 10;           % number of external turing machines
Nsample  = 20;           % number of samples
No   = 10;               % dimensionality of sensory inputs
Ns   = 10;               % dimensionality of hidden states
Nd   = 1;                % dimensionality of actions
Ng   = 1;                % dimensionality of risks
NC   = Ns;               % dimensionality of memory matrix
Nd2  = Ns^2*2;           % dimensionality of actions for utm
Ng2  = Ns^2*2;           % dimensionality of risks for utm
NC2  = Ncontext;         % dimensionality of memory matrix for utm
Nswitch  = 100;
sim_type = 2;

seed     = 0;
rng(seed+1000000);

o    = zeros(No,T);      % observations
s    = zeros(Ns,T);      % hidden states
d    = zeros(1,T);       % actions
g    = zeros(2,T);       % risks
qs   = zeros(Ns*2,T);    % state posterior
qd   = zeros(Nd*2,T);    % action posterior
qd2  = zeros(Nd2*2,T);   % action posterior
qg2  = zeros(Ng2*2,T);   % risk posterior
qs2  = zeros(No*2,T);    % predicted states
psis = zeros(Ns*2*2,T);  % basis functions for states
psid = zeros(NC2,T);     % basis functions for actions
psig = zeros(Nd2*2,T);   % basis functions for risks

err_s_list        = zeros(100,Nsample);
err_C_list        = zeros(100,Nsample);
err_psid_list     = zeros(100,Nsample);
err_B_list        = zeros(100,NC2*Nsample);

%--------------------------------------------------------------------------------
% simulations

for sample = 1:Nsample
 fprintf(1,'sample=%d\n', sample)
 err_s        = zeros(100,1);
 err_C        = zeros(100,1);
 err_psid     = zeros(100,1);
 err_B        = zeros(100,NC2);
 Ctemp        = zeros(Ns,T);
 qCtemp       = zeros(Ns,T);
 
 % set generative process
 [A,Blist,C,C2,qa,qb,qc_init,qc2_init] = utm_init(No,Ns,Nd,Nd2,NC,NC2);
 context_id   = zeros(Nswitch,1);
 context_time = [1;(T/Nswitch+1:T/Nswitch:T)'+randi([-T/Nswitch*0.4,T/Nswitch*0.4],Nswitch-1,1)];
 context      = zeros(T,1);
 for i = 1:Nswitch/NC2, context_id(NC2*(i-1)+1:NC2*i) = randperm(NC2); end
 for i = 2:Nswitch, context(context_time(i-1):context_time(i)-1) = context_id(i-1); end
 context(context_time(i):T) = context_id(i);
 
 % run generative process
 s(:,1)  = mnrnd(1,ones(Ns,1)/Ns);
 o(:,1)  = mnrnd(1,s(:,1)*0.99+0.01/No);
 d(:,1)  = zeros(Nd,1);
 for t = 2:T
  % generative process (external turing machines)
  B          = Blist{context(t)};    % switch transition matrix
  s(:,t)     = mnrnd(1,softmax_a(ln(B(1:Ns,:))*[s(:,t-1)*d(:,t-1);s(:,t-1)*(1-d(:,t-1));(1-s(:,t-1))*d(:,t-1);(1-s(:,t-1))*(1-d(:,t-1))]));
%  s(:,t)     = mnrnd(1,softmax_a(ln(B(1:Ns,:))*[s(:,t-1)*d(:,t-1);s(:,t-1)*(1-d(:,t-1));1-s(:,t-1)*d(:,t-1);1-s(:,t-1)*(1-d(:,t-1))]));
  o(:,t)     = mnrnd(1,s(:,t)*0.99+0.01/No);
  hid        = (1:Ns)*s(:,t-1);  % reading header position id
  d(:,t)     = C(:,hid);         % memory reading
%  C(:,hid)   = 1 - C(:,hid);     % memory writing (sign flip)
  C(:,hid)   = (rem(t,3)==0)*1;  % memory writing
  Ctemp(:,t) = C;
 end
 
 % set initial posterior
 [qA,qlnA] = param_norm(qa,sim_type); % likelihood mapping
 qc  = qc_init;
 qc2 = qc2_init;
 qC2 = softmax_b(qc2);
 
 % inference at t=1
 t = 1;
 [qC,qlnC]   = param_norm(qc,sim_type);  % memory
 qlnC2       = log(qC2);                 % memory for utm
 qs(:,t)     = softmax_b(qlnA'*[o(:,t);1-o(:,t)]);
 qd(:,t)     = softmax_b(qlnC*qs(1:Ns,t));
% psis(:,t)   = [qs(1:Ns,t)*qd(1,t);qs(1:Ns,t)*(1-qd(1,t));(1-qs(1:Ns,t))*qd(1,t);(1-qs(1:Ns,t))*(1-qd(1,t))];
 psis(:,t)   = [qs(1:Ns,t)*qd(1,t);qs(1:Ns,t)*(1-qd(1,t));(1-qs(1:Ns,t))*qd(1,t);(1-qs(1:Ns,t))*(1-qd(1,t))];
 psid(:,t)   = softmax_a(rand(NC2,1));
 qd2(:,t)    = softmax_b(qlnC2*psid(:,t));
 
 % inference during t=2:T
 for t = 2:T
  [qC,qlnC]   = param_norm(qc,sim_type);  % memory
  qlnC2       = log(qC2);                 % memory for utm
  
  % set transition matrix
  qB11       = reshape(qd2(1:Nd2,t-1),[Ns Ns*2])*0.5+0.5;
  qB         = [qB11 1-qB11; 1-qB11 qB11];
  qlnB       = ln(qB);
  % inference
  qs(:,t)    = softmax_b(qlnA'*[o(:,t);1-o(:,t)]*100 + qlnB*psis(:,t-1)); % inference of hidden states
  qd(:,t)    = softmax_b(qlnC*qs(1:Ns,t-1)); % inference of mental actions
  qd2(:,t)   = softmax_b(qlnC2*psid(:,t-1)); % inference of mental actions for utm
  
  % compute basis functions
%  psis(:,t)  = [qs(1:Ns,t)*qd(1,t);qs(1:Ns,t)*(1-qd(1,t));1-qs(1:Ns,t)*qd(1,t);1-qs(1:Ns,t)*(1-qd(1,t))];
  psis(:,t)  = [qs(1:Ns,t)*qd(1,t);qs(1:Ns,t)*(1-qd(1,t));(1-qs(1:Ns,t))*qd(1,t);(1-qs(1:Ns,t))*(1-qd(1,t))];
  psid(:,t)  = softmax_a(ln(psid(:,t-1)) + qlnC2(1:Nd2,:)'*kron(psis(1:Ns*2,t-1),qs(1:Ns,t)));
  psig(:,t)  = reshape([qs(1:Ns,t)*psis(1:Ns*2,t-1)' qs(Ns+1:Ns*2,t)*psis(1:Ns*2,t-1)'],[Nd2*2 1]);
  
  % compute risk
  qg2(:,t)    = psig(:,t)*0.1;
  
  % memory update
%  qc  = qc  + (1-2*qd(:,t))*qs(1:Ns,t-1)';
  qc  = qc  + 100*[rem(t,3)==0;rem(t,3)~=0]*qs(1:Ns,t-1)';
  qc  = softmax_b(100*qc);
  
  % memory update
  qc2 = qc2 + (qg2(:,t).*qd2(:,t))*psid(:,t-1)';
%  qC2 = softmax_b(qc2);
  [qC2,~] = param_norm(qc2,sim_type);  % memory
  
  % prediction
  qs2(1:Ns,t+1)      = softmax_a(qlnB(1:Ns,:)*psis(:,t));
  qs2(Ns+1:Ns*2,t+1) = 1-qs2(1:Ns,t+1);
  
  qCtemp(:,t) = qC(1,:);
  
  if rem(t,T/100) == 0
   Mcontext = eye(Ncontext);
   Corr_psi = corr(Mcontext(:,context)',psid');
   Corr_psi = (Corr_psi == max(Corr_psi')'*ones(1,NC2))*1;
   err_s(t/(T/100))    = mean(sum((qs2(1:Ns,t-1000+(1:1000))-s(:,t-1000+(1:1000))).^2));
   err_C(t/(T/100))    = mean(mean((qCtemp(:,t-1000+(1:1000))-Ctemp(:,t-1000+(1:1000))).^2));
   err_psid(t/(T/100)) = mean(sum((Corr_psi*psid(:,t-1000+(1:1000))-Mcontext(:,context(t-1000+(1:1000)))).^2));
   qBlist = Corr_psi*(qC2(1:Nd2,:)'*0.5+0.5);
   for i = 1:NC2, err_B(t/(T/100),i)  = sum(sum((qBlist(i,:)-reshape(Blist{i}(1:Ns,1:Ns*2),[1 Ns*Ns*2])).^2)); end
%   err_B(t/(T/1000))    = sum(sum((qB-Blist{context(t)}).^2));
  end
  
  if rem(t,T/Nswitch) == T/Nswitch/2
   fprintf(1, "t=%d, %.2f ", t, min(ln(psid(:,t))));
   fprintf(1, "%.2f ", psid(:,t));
   fprintf(1, "| %f, %f\n", mean(sum((qs2(1:Ns,t-T/Nswitch/2+1:t)-s(1:Ns,t-T/Nswitch/2+1:t)).^2))/mean(sum((s(1:Ns,:)).^2)), mean(mean((qC(1,:)-C).^2)));
   
   subplot(2,2,1), plot(err_s)
   subplot(2,2,2), plot(err_C)
   subplot(2,2,3), plot(err_psid) %, image(psid*300), %image(corr(psid',Mcontext(:,context)')*300)
   subplot(2,2,4), plot(err_B)
   drawnow
  end
 end
 % learning
% qa       = qa + [o(:,t-Nct+1:t);1-o(:,t-Nct+1:t)]*qs(:,t-Nct+1:t)';
% qb       = qb + qs(:,t-Nct+2:t)*qs(:,t-Nct+1:t-1)';
% err_A(session) = sum(sum((qA-A).^2));
% err_B(session) = sum(sum((qB-B).^2));
% err_d(session) = mean(mean((qd2(1:Nd2,:) - C2(:,context_id(session))).^2));
 
 if sample == 1
  col = [1 0 0; 1 0.5 0; 1 1 0; 0 1 0; 0 1 0.5; 0 1 1; 0 0.5 1; 0 0 1; 0.5 0 1; 1 0 1];
  img1 = zeros(250,2000,3);
  img2 = zeros(250,2000,3);
  for i = 1:2000
   img1(:,i,1) = col(context(5*(i-1)+1),1)*0.4+0.6;
   img1(:,i,2) = col(context(5*(i-1)+1),2)*0.4+0.6;
   img1(:,i,3) = col(context(5*(i-1)+1),3)*0.4+0.6;
   img2(:,i,1) = col(context(T-10000+5*(i-1)+1),1)*0.4+0.6;
   img2(:,i,2) = col(context(T-10000+5*(i-1)+1),2)*0.4+0.6;
   img2(:,i,3) = col(context(T-10000+5*(i-1)+1),3)*0.4+0.6;
   for j = 1:10
    img1(25*(j-1)+1:25*j,i,1) = img1(25*(j-1)+1:25*j,i,1) * (1-mean(psid(j,5*(i-1)+(1:5))));
    img1(25*(j-1)+1:25*j,i,2) = img1(25*(j-1)+1:25*j,i,2) * (1-mean(psid(j,5*(i-1)+(1:5))));
    img1(25*(j-1)+1:25*j,i,3) = img1(25*(j-1)+1:25*j,i,3) * (1-mean(psid(j,5*(i-1)+(1:5))));
    img2(25*(j-1)+1:25*j,i,1) = img2(25*(j-1)+1:25*j,i,1) * (1-mean(psid(j,T-10000+5*(i-1)+(1:5))));
    img2(25*(j-1)+1:25*j,i,2) = img2(25*(j-1)+1:25*j,i,2) * (1-mean(psid(j,T-10000+5*(i-1)+(1:5))));
    img2(25*(j-1)+1:25*j,i,3) = img2(25*(j-1)+1:25*j,i,3) * (1-mean(psid(j,T-10000+5*(i-1)+(1:5))));
   end
  end
  imwrite(img1, ['fig_header_pre.png'])
  imwrite(img2, ['fig_header_post.png'])
  csvwrite(['output_err_B.csv'], [1:Ncontext; err_B])
 end
 
 err_s_list(:,sample)                           = err_s;
 err_C_list(:,sample)                           = err_C;
 err_psid_list(:,sample)                        = err_psid;
 err_B_list(:,Ncontext*(sample-1)+(1:Ncontext)) = err_B;
 csvwrite(['output_err_s_list.csv'], [1:Nsample; err_s_list])
 csvwrite(['output_err_C_list.csv'], [1:Nsample; err_C_list])
 csvwrite(['output_err_psid_list.csv'], [1:Nsample; err_psid_list])
 csvwrite(['output_err_B_list.csv'], [1:Nsample*Ncontext; err_B_list]) 
% subplot(2,2,1), image(qB*300)
% drawnow
end

%plot(err_d)

return

%--------------------------------------------------------------------------------

function y = ln(x)
y = log(x + 10^-8);
end

%--------------------------------------------------------------------------------
