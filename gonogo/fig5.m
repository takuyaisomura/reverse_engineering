
%--------------------------------------------------------------------------------
% fig5.m
%
% Copyright (C) 2025 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2025-06-28
%--------------------------------------------------------------------------------
% initialization

% read files
clear
tic
No         = 4;
Nx         = 2;
Ny         = 1;
Nsample_l  = 30;
Nsample_nl = 15;
Nsample    = Nsample_l + Nsample_nl;
Ttrial     = 150;                                    % duration of a single trial
Tinit      = 20;                                     % number of initial trials
Tlate      = 61;                                     % 61-100 trials are the post-training period
Tend       = 100;                                    % number of trials
dstdir     = 'output/';
addpath(genpath('scripts/'));

pos_max    = 2;                                      % maximum position
betax      = 1;                                      % gain for middle layer activity
betay      = 3;                                      % gain for output layer activity

seed       = 0;
rng(seed+1000000);

load([dstdir,'/data_autonomous_states.mat']);        % read files
load([dstdir,'/data_predicted_states.mat']);         % read files

% restrict position between 0 and pos_max
for num = 1:Nsample
 for t = 1:Tend, phi{num,t}.pos  = max(min(phi{num,t}.pos,pos_max),0); end
 for t = 1:Tend, phip{num,t}.pos = max(min(phip{num,t}.pos,pos_max),0); end
end
gamma  = func(@(a)a.g_max, phi(:,1));                % empirical subjective risk
qgamma = func(@(a)a.gamma1, phip(:,1));              % predicted subjective risk

%--------------------------------------------------------------------------------
% open loop condition

n_open = [1:7,9:12,16:17,20:24,26:27];
Nopen  = length(n_open);
Topen  = 15;
phip1  = phip(1:Nsample_l,:);

isgo_op           = randi([0,1],Nsample_l,Tend+Topen);
isgo_op(:,1:Tend) = func(@(a)a.isgo, phi(1:Nsample_l,1:Tend));
for num = 1:Nopen, isgo_op(n_open(num),:) = func(@(a)a.isgo, phi(n_open(num),1:Tend+Topen)); end

for num = 1:Nsample_l
% param = [learning insensitivity of W, learning insensitivity of K, learning insensitivity of K, gain for x, gain for y, subjective risk]
 param                 = [0.5, 0.5, 0.5, betax, betay, gamma(num)];                                    % set hyper parameters
 phip1{num,Tend}       = phi_weight_correct(phi{num,Tend},param);                                      % phi in the last training trial with reduced plasticity insensitivity
 phip1{num,Tend}.phiy1 = phip1{num,Tend}.phiy1 - 20;                                                   % modify phiy to model open-loop condition
 phip1{num,Tend}.phiy0 = phip1{num,Tend}.phiy0 + 20;                                                   % modify phiy to model open-loop condition
 for t = Tend+1:Tend+Topen, phip1{num,t} = phi_predict(phip1{num,t-1},isgo_op(num,t),0,0,pos_max); end % multi-trial prediction
 phip1{num,Tend}       = phip{num,Tend};
end

t_range   = 50;
op        = open_loop_analysis(phi(n_open,:),Tlate,Tend,Topen,t_range);
opp       = open_loop_analysis(phip1,Tlate,Tend,Topen,t_range);

isgo      = func(@(a)a.isgo, phi(:,1:Tend));
pos       = func(@(a)a.pos(:,125), phi(:,1:Tend));
posp      = func(@(a)a.pos(:,125), phip(:,1:Tend));
tt        = Tlate:Tend;
pos1      = [sum(pos(:,tt).*isgo(:,tt),2)./sum(isgo(:,tt),2),sum(pos(:,tt).*(1-isgo(:,tt)),2)./sum(1-isgo(:,tt),2)];
posp1     = [sum(posp(:,tt).*isgo(:,tt),2)./sum(isgo(:,tt),2),sum(posp(:,tt).*(1-isgo(:,tt)),2)./sum(1-isgo(:,tt),2)];
dpos1     = pos1(:,1)-pos1(:,2);
dposp1    = posp1(:,1)-posp1(:,2);

dx2       = mean(op.x2(:,Tend+1:Tend+Topen),2,'omitnan')-mean(op.x2(:,tt),2,'omitnan');
dxp2      = mean(opp.x2(n_open,Tend+1:Tend+Topen),2,'omitnan')-mean(opp.x2(n_open,tt),2,'omitnan');

id0       = find(dx2<prctile(dx2,2/3*100));
id1       = find(dx2>=prctile(dx2,2/3*100));
idp0      = find(dxp2<prctile(dxp2,2/3*100));
idp1      = find(dxp2>=prctile(dxp2,2/3*100));
op.dpos0  = dpos1(n_open(id0));
op.dpos1  = dpos1(n_open(id1));
opp.dpos0 = dposp1(n_open(idp0));
opp.dpos1 = dposp1(n_open(idp1));

save([dstdir,'fig5_op.mat'],'-struct','op','-V6') % save file
save([dstdir,'fig5_opp.mat'],'-struct','opp','-V6') % save file

fprintf(1,'x2 late    v.s. x2 oepn p_value = %f\n', ranksum(reshape(op.x2(:,86:100),[],1),reshape(op.x2(:,101:115),[],1)))
fprintf(1,'x2 open(2) v.s. x2 late p_value = %f\n', ranksum(op.x2(:,102),reshape(op.x2(:,86:100),[],1)))
fprintf(1,'x2 open(2) v.s. x2 open p_value = %f\n', ranksum(op.x2(:,102),reshape(op.x2(:,111:115),[],1)))
fprintf(1,'xp2 late    v.s. xp2 oepn p_value = %f\n', ranksum(reshape(opp.x2(:,86:100),[],1),reshape(opp.x2(:,101:115),[],1)))
fprintf(1,'xp2 open(2) v.s. xp2 late p_value = %f\n', ranksum(opp.x2(:,102),reshape(opp.x2(:,86:100),[],1)))
fprintf(1,'xp2 open(2) v.s. xp2 open p_value = %f\n', ranksum(opp.x2(:,102),reshape(opp.x2(:,111:115),[],1)))
fprintf(1,'Empirical moving distance with small x2 v.s. large x2 p_value = %f\n', ranksum(op.dpos0,op.dpos1))
fprintf(1,'Predicted moving distance with small x2 v.s. large x2 p_value = %f\n', ranksum(opp.dpos0,opp.dpos1))

fig               = figure();
fig.Position(3:4) = [800 800];
subplot(4,6,1), plotdist(1:Ttrial,op.x1_late_nogo,'red',2), plotdist(1:Ttrial,op.x1_late_go,'blue',2), axis([0 Ttrial 0 1]), title('x_1 late')
subplot(4,6,2), plotdist(1:Ttrial,op.x1_open_nogo,'red',2), plotdist(1:Ttrial,op.x1_open_go,'blue',2), axis([0 Ttrial 0 1]), title('x_1 open')
subplot(4,6,3), plotdist(1:Ttrial,op.x2_late_nogo,'red',2), plotdist(1:Ttrial,op.x2_late_go,'blue',2), axis([0 Ttrial 0 1]), title('x_2 late')
subplot(4,6,4), plotdist(1:Ttrial,op.x2_open_nogo,'red',2), plotdist(1:Ttrial,op.x2_open_go,'blue',2), axis([0 Ttrial 0 1]), title('x_2 open')
subplot(4,6,5), plotdist(-14:15,op.x2(:,100-14:100+15),'blue',2), axis([-Topen Topen 0 0.8]), title('x_2 in go trials')
subplot(4,6,7), plotdist(1:Ttrial,opp.x1_late_nogo,'red',2), plotdist(1:Ttrial,opp.x1_late_go,'blue',2), axis([0 Ttrial 0 1]), title('xp_1 late')
subplot(4,6,8), plotdist(1:Ttrial,opp.x1_open_nogo,'red',2), plotdist(1:Ttrial,opp.x1_open_go,'blue',2), axis([0 Ttrial 0 1]), title('xp_1 open')
subplot(4,6,9), plotdist(1:Ttrial,opp.x2_late_nogo,'red',2), plotdist(1:Ttrial,opp.x2_late_go,'blue',2), axis([0 Ttrial 0 1]), title('xp_2 late')
subplot(4,6,10), plotdist(1:Ttrial,opp.x2_open_nogo,'red',2), plotdist(1:Ttrial,opp.x2_open_go,'blue',2), axis([0 Ttrial 0 1]), title('xp_2 open')
subplot(4,6,11), plotdist(-14:15,opp.x2(:,100-14:100+15),'blue',2), axis([-Topen Topen 0 0.4]), title('xp_2 in go trials')

subplot(4,6,6), plot(id0*0.02,op.dpos0,'bo',1-id1*0.02,op.dpos1,'ro')
subplot(4,6,12), plot(idp0*0.02,opp.dpos0,'bo',1-idp1*0.02,opp.dpos1,'ro')
fprintf(1,'mean(op.dpos0)=%f, mean(op.dpos1)=%f\n', mean(op.dpos0), mean(op.dpos1))
fprintf(1,'mean(opp.dpos0)=%f, mean(opp.dpos1)=%f\n', mean(opp.dpos0), mean(opp.dpos1))

%--------------------------------------------------------------------------------
% reversal learning

n_rev = [28:30];
Nrev  = length(n_rev);
Trev  = 58;
isrev = 1;
phip2 = phip(1:Nsample_l,:);

isgo_rev           = randi([0,1],Nsample_l,Tend+Trev);
isgo_rev(:,1:Tend) = func(@(a)a.isgo, phi(1:Nsample_l,1:Tend));
for num = 1:Nrev, isgo_rev(n_rev(num),:) = func(@(a)a.isgo, phi(n_rev(num),1:Tend+Trev)); end

for num = 1:Nsample_l
 param           = [0.02, 0.5, 0.5, betax, betay, gamma(num)];                                                         % set hyper parameters
 phip2{num,Tend} = phi_weight_correct(phip{num,Tend},param);                                                           % phip in the last training trial with reduced plasticity insensitivity
 for t = Tend+1:Tend+Trev, phip2{num,t} = phi_predict_rev(phip2{num,t-1},isgo_rev(num,t),0,isrev,pos_max); end         % multi-trial prediction
 for t = Tend+1:Tend+Trev, phip2{num,t}.isscc = func(@(a)(1-a.isgo)*a.ispass + a.isgo*(1-a.ispass), phip2(num,t)); end % restrict position between 0 and pos_max
end

rev           = reversal_learning_analysis(phi(n_rev,:),Tlate,Tend,Trev);
revp          = reversal_learning_analysis(phip2(1:Nsample_l,:),Tlate,Tend,Trev);
rev.scc       = reshape([func(@(a)a.isscc,phi(n_rev,:)),zeros(Nrev,2)*NaN],Nrev*5,[]);
revp.scc      = reshape([func(@(a)a.isscc,phip2(1:Nsample_l,:)),zeros(Nsample_l,2)*NaN],Nsample_l*5,[]);
rev.scc_mean  = mean(rev.scc,'omitnan');
rev.scc_std   = std(rev.scc,'omitnan');
revp.scc_mean = mean(revp.scc,'omitnan');
revp.scc_std  = std(revp.scc,'omitnan');

save([dstdir,'fig5_rev.mat'],'-struct','rev','-V6') % save file
save([dstdir,'fig5_revp.mat'],'-struct','revp','-V6') % save file

subplot(4,5,11), plotdist(1:Ttrial,rev.x1_late_nogo,'red',2), plotdist(1:Ttrial,rev.x1_late_go,'blue',2), title('x_1 late')
subplot(4,5,12), plotdist(1:Ttrial,rev.x1_rev_nogo,'red',2), plotdist(1:Ttrial,rev.x1_rev_go,'blue',2), title('x_1 rev')
subplot(4,5,13), plotdist(1:Ttrial,rev.x2_late_nogo,'red',2), plotdist(1:Ttrial,rev.x2_late_go,'blue',2), title('x_2 late')
subplot(4,5,14), plotdist(1:Ttrial,rev.x2_rev_nogo,'red',2), plotdist(1:Ttrial,rev.x2_rev_go,'blue',2), title('x_2 rev')
subplot(4,5,15), plotdist(5/2:5:160,rev.scc,'blue',2), axis([0,Tend+60,0,1]),title('actual success rate')
subplot(4,5,16), plotdist(1:Ttrial,revp.x1_late_nogo,'red',2), plotdist(1:Ttrial,revp.x1_late_go,'blue',2), title('xp_1 late')
subplot(4,5,17), plotdist(1:Ttrial,revp.x1_rev_nogo,'red',2), plotdist(1:Ttrial,revp.x1_rev_go,'blue',2), title('xp_1 rev')
subplot(4,5,18), plotdist(1:Ttrial,revp.x2_late_nogo,'red',2), plotdist(1:Ttrial,revp.x2_late_go,'blue',2), title('xp_2 late')
subplot(4,5,19), plotdist(1:Ttrial,revp.x2_rev_nogo,'red',2), plotdist(1:Ttrial,revp.x2_rev_go,'blue',2), title('xp_2 rev')
subplot(4,5,20), plotdist(5/2:5:160,revp.scc,'blue',2), axis([0,Tend+60,0,1]),title('predicted success rate')

print(fig,[dstdir,'fig5.png'],'-dpng')

return

%--------------------------------------------------------------------------------
