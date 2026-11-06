
%--------------------------------------------------------------------------------
% fig4.m
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2026-10-10
%--------------------------------------------------------------------------------
% initialization

% read files
clear
tic
Nsample_l  = 30;                                     % number of learners
Nsample_nl = 15;                                     % number of non-learners
Nsample    = Nsample_l + Nsample_nl;                 % number of all samples
Ttrial     = 150;                                    % duration of a single trial
Tinit      = 20;                                     % number of initial trials
Tlate      = 61;                                     % 61-100 trials are the late training phase
Tend       = 100;                                    % number of trials
dstdir     = 'output/';
addpath(genpath('scripts/'));

pos_max    = 2;                                      % maximum position
betax      = 1;                                      % gain for middle layer activity
betay      = 3;                                      % gain for output layer activity

seed       = 0;
rng(seed+1000000);

load([dstdir,'/data_empirical_states.mat']);         % read files
load([dstdir,'/fig1_data.mat']);                     % read files

% restrict position between 0 and pos_max
for num = 1:Nsample
 for t = 1:Tend, phi{num,t}.pos      = max(min(phi{num,t}.pos,pos_max),0); end
 for t = 1:Tend, phi_init{num,t}.pos = max(min(phi_init{num,t}.pos,pos_max),0); end
end

%--------------------------------------------------------------------------------
% pretrained initialization

phip                 = cell(Nsample,Tend);
[phip(:,1:Tinit),ql] = phi_pretrained_init(phi,phi_init,Nsample_l,Tinit,betax,betay); % pretrained initialization
% This function predicts initial states based on entire period of training dataset and initial 20-trial adaptation period of test data using leave-one-out cross validation

% prediction of subjective risk (gamma)
gamma                = func(@(a)a.g_max, phi(:,1));                                   % empirical subjective risk
qgamma               = func(@(a)a.gamma1, phip(:,1));                                 % predicted subjective risk

%--------------------------------------------------------------------------------
% prediction with pretraining

% predict test data
dst     = cell(Nsample,1);
for num = 1:Nsample
 for t = Tinit+1:Tend, phip{num,t} = phi_predict(phip{num,t-1},phi{num,t}.isgo,1,0,pos_max); end % multi-trial prediction
 dst{num} = phi_report(phi(num,1:Tend)',phip(num,1:Tend)',Ttrial,Tinit,Tlate,Tend,1);            % summarize results
end
scc     = mean(func(@(a)a.isscc, phi(:,Tlate:Tend)),2);
sccp    = mean(func(@(a)a.isscc, phip(:,Tlate:Tend)),2);
err_pos = func(@(dst)dst.errn_pos, dst);
fprintf(1,'RevEng with PTI: scc_actl=%.2f, scc_pred=%.2f, err_scc=%.2f, err_pos=%.2f\n',...
        mean(scc,'all'), mean(sccp,'all'), mean(abs(scc-sccp)), mean(err_pos(:,Tlate:Tend),'all'))
fprintf(1,'scc_pred = %f, scc_corr = %f\n', mean(abs(scc-sccp)), corr(scc,sccp))

%--------------------------------------------------------------------------------
% figure and file outputs

fig = figure(); fig.Position(3:4) = [800 600];
figure_output(phi(:,1:Tend),phip(:,1:Tend),dst,0,0,Ttrial,Tinit,Tlate,Tend,Nsample_l,Nsample_nl,1,pos_max,qgamma,ql,[dstdir,'fig4_data.mat']);
drawnow
print(fig,[dstdir,'fig4.png'],'-dpng')

save([dstdir,'data_predicted_states.mat'],'phip','-v7.3') % save file

resp   = ensemble_response_analysis(phip,Ttrial,Tinit,Tlate,Tend,Nsample_l,[]);
save([dstdir,'fig4_data2.mat'],'-struct','resp','-V6') % save file

file_output([dstdir,'fig4_data3.mat'],phi,phip,dst,0,0,Ttrial,Tinit,Tlate,Tend,Nsample_l,[],qgamma);

X  = func(@(a)a.x(1,:), phi(:,Tlate:80));
Xp = func(@(a)a.x(1,:), phip(:,Tlate:80));
P  = func(@(a)a.pos, phi(:,Tlate:80));
Pp = func(@(a)a.pos, phip(:,Tlate:80));
save([dstdir,'fig4_data4.mat'],'X','Xp','P','Pp','-v6') % save file

%--------------------------------------------------------------------------------

VIDEO_ON = 0;
if VIDEO_ON
 for i = 1
  figure_landscape(phi(i,1:Tend),phip(i,1:Tend),Tinit,Tlate,Tend,['video/free_energy_W_',num2str(i),'_e.mp4'],['video/free_energy_W_',num2str(i),'_p.mp4'],0)
 end
 i = 1;
 figure_landscape(phi(i,1:Tend),phip(i,1:Tend),Tinit,Tlate,Tend,['video/wideview_free_energy_W_',num2str(i),'_e.mp4'],['video/wideview_free_energy_W_',num2str(i),'_p.mp4'],1)
end

%% --------------------------------------------------------------------------------
%% comparison of conditions

%--------------------------------------------------------------------------------
% single-layer perceptron baseline

% define data and target
x  = func(@(a)[reshape([a.W1,a.W0,a.K1,a.K0,a.V1',a.V0',a.phix1,a.phix0,[a.phiy1;a.phiy0]],1,[]),1],phi_init(:,Tinit))'; % data
y  = [scc'; [ones(1,Nsample_l),zeros(1,Nsample_nl)]; gamma']; % target
y_ = logitapprox(y);

% compute posterior expectation
qy = zeros(3,Nsample);
for num = 1:Nsample
 id = [1:num-1,num+1:Nsample];
 qy(:,num) = sig( (y_(:,id)*x(:,id)')/(x(:,id)*x(:,id)'+eye(size(x,1))*10^-6) * x(:,num) );
end
scc1  = y(1,:)';
sccp1 = qy(1,:)';
fprintf(1,'Perceptron: scc_pred = %f, scc_corr = %f, p_value = %f\n', mean(abs(scc1-sccp1)), corr(scc1,sccp1), signrank(abs(scc-sccp),abs(scc1-sccp1)))

%--------------------------------------------------------------------------------
% prediction based on reverse engineering without pretrained initialization

phip2 = cell(Nsample,Tend);
for num = 1:Nsample
 param = [1.5, 1.5, 1.0, betax, betay, 1];                                                         % set hyper parameters
 for t = 1:Tinit, phip2{num,t} = phi_weight_correct(phi_init{num,t},param); end                    % initialization
 for t = Tinit+1:Tend, phip2{num,t} = phi_predict(phip2{num,t-1},phi{num,t}.isgo,1,0,pos_max); end % multi-trial prediction
end

scc2  = mean(func(@(a)a.isscc, phi(:,Tlate:Tend)),2);
sccp2 = mean(func(@(a)a.isscc, phip2(:,Tlate:Tend)),2);
fprintf(1,'RevEng w/o PTI: scc_pred = %f, scc_corr = %f, p_value = %f\n', mean(abs(scc2-sccp2)), corr(scc2,sccp2), signrank(abs(scc-sccp),abs(scc2-sccp2)))

%--------------------------------------------------------------------------------
% prediction based on an unbiased canonical neural network

No            = 4;
Nx            = 2;
Ny            = 1;
phi_th.o      = phi_init{1,1}.o;
phi_th.x      = phi_init{1,1}.x;
phi_th.y      = phi_init{1,1}.y;
phi_th.d      = phi_init{1,1}.d;
phi_th.pos    = phi_init{1,1}.pos;
phi_th.HomeW1 = ones(Nx,No)*1000;
phi_th.HomeW0 = ones(Nx,No)*1000;
phi_th.HebbW1 = [0.9,0.5,0.5,0.5; 0.5,0.9,0.5,0.5].*phi_th.HomeW1;
phi_th.HebbW0 = [0.5,0.5,0.5,0.5; 0.5,0.5,0.5,0.5].*phi_th.HomeW0;
phi_th.HomeK1 = ones(Nx,Nx)*1000;
phi_th.HomeK0 = ones(Nx,Nx)*1000;
phi_th.HebbK1 = [0.9,0.5; 0.5,0.9].*phi_th.HomeK1;
phi_th.HebbK0 = [0.1,0.5; 0.5,0.1].*phi_th.HomeK0;
phi_th.HomeV1 = ones(Ny,Nx)*1000;
phi_th.HomeV0 = ones(Ny,Nx)*1000;
phi_th.HebbV1 = [0.7,0.5].*phi_th.HomeV1;
phi_th.HebbV0 = [0.3,0.5].*phi_th.HomeV0;
phi_th.W1     = logit(phi_th.HebbW1./phi_th.HomeW1);
phi_th.W0     = logit(phi_th.HebbW0./phi_th.HomeW0);
phi_th.K1     = logit(phi_th.HebbK1./phi_th.HomeK1);
phi_th.K0     = logit(phi_th.HebbK0./phi_th.HomeK0);
phi_th.V1     = logit(phi_th.HebbV1./phi_th.HomeV1);
phi_th.V0     = logit(phi_th.HebbV0./phi_th.HomeV0);
phi_th.W      = phi_th.W1 - phi_th.W0;
phi_th.K      = phi_th.K1 - phi_th.K0;
phi_th.V      = phi_th.V1 - phi_th.V0;
phi_th.phix1  = log([0.5;0.5]);
phi_th.phix0  = log([0.5;0.5]);
phi_th.phiy1  = log(0.5);
phi_th.phiy0  = log(0.5);
phi_th.betax  = betax;
phi_th.betay  = betay;
phi_th.gamma1 = 1;

phip3 = cell(Nsample,Tend);
for num = 1:Nsample
 for t = 1:Tinit, phip3{num,t} = phi_th; end                                                       % initialization
 for t = Tinit+1:Tend, phip3{num,t} = phi_predict(phip3{num,t-1},phi{num,t}.isgo,1,0,pos_max); end % multi-trial prediction
end
scc3  = mean(func(@(a)a.isscc, phi(:,Tlate:Tend)),2);
sccp3 = mean(func(@(a)a.isscc, phip3(:,Tlate:Tend)),2);
fprintf(1,'Unbiased CanonNN: scc_pred = %f, scc_corr = %f, p_value = %f\n', mean(abs(scc3-sccp3)), corr(scc3,sccp3), signrank(abs(scc-sccp),abs(scc3-sccp3)))

%--------------------------------------------------------------------------------
% figure and file outputs

save([dstdir,'fig4_data5.mat'],'gamma','qgamma','scc','sccp','scc1','sccp1','scc2','sccp2','scc3','sccp3','-V6') % save file

figure()
subplot(2,2,1), plot([0.2,1],[0.2,1],'k--',scc,sccp,'bo'), axis([0.2,1,0.2,1]), title('RevEng with PTI')
subplot(2,2,2), plot([0.2,1],[0.2,1],'k--',scc1,sccp1,'bo'), axis([0.2,1,0.2,1]), title('Perceptron')
subplot(2,2,3), plot([0.2,1],[0.2,1],'k--',scc2,sccp2,'bo'), axis([0.2,1,0.2,1]), title('RevEng w/o PTI')
subplot(2,2,4), plot([0.2,1],[0.2,1],'k--',scc3,sccp3,'bo'), axis([0.2,1,0.2,1]), title('Unbiased CanonNN')

return

%--------------------------------------------------------------------------------
