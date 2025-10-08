
%--------------------------------------------------------------------------------
% fig3.m
%
% Copyright (C) 2025 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2025-06-18
%--------------------------------------------------------------------------------
% initialization

clear
tic
Tinit      = 20;                     % end of initial trials
Tlate      = 61;                     % start of late trials 
Tend       = 100;                    % number of trials
Ttrial     = 150;                    % duration of 1 trial (15 s)
T          = Ttrial*Tend;            % entire duration
Nsample    = 45;                     % number of samples
dstdir     = 'output/';              % export directory
addpath(genpath('scripts/'));        % import directory

%--------------------------------------------------------------------------------
% initialization

load('preprocessed_data0.mat');      % read files (data without deconvolution filter)
load([dstdir,'/fig1_data.mat']);     % read files

for num = 1:Nsample
 T1                 = data{num}.T;
 stim               = max(reshape(data{num}.stim,Ttrial,[]));
 data{num}.stim     = kron(stim,[zeros(1,125),1,zeros(1,24)])*0.5 + 0.5;
 data{num}.flow     = reshape(data{num}.flow,150,[]); data{num}.flow(1:25,:) = 0;
 data{num}.action   = reshape(data{num}.action,150,[]); data{num}.action(1:25,:) = 0;
 data{num}.flow     = max(min(reshape(data{num}.flow,1,[]),1),0);
 data{num}.action   = max(min(reshape(data{num}.action,1,[]),1),0);
 data{num}.position = max(reshape(data{num}.position,1,[]),0);
 data{num}.s        = [data{num}.go.*data{num}.blue; data{num}.go.*data{num}.red; data{num}.nogo.*data{num}.red; data{num}.nogo.*data{num}.blue; data{num}.white];
 data{num}.o        = [data{num}.blue; data{num}.red; data{num}.flow; data{num}.stim];
end

data           = [data(learner_orig==1); data(learner_orig==0)];
learner        = [learner_orig(learner_orig==1); learner_orig(learner_orig==0)];
Nsample_l      = sum(learner);
Nsample_nl     = Nsample - Nsample_l;

fprintf(1,'number of neurons = %f +/- %f (from %d samples)\n', mean(func(@(a)size(a.activity,1),data)), std(func(@(a)size(a.activity,1),data)), size(data,1));

%--------------------------------------------------------------------------------
% reverse engineering

No             = 4;                   % input-layer dimensionality
Nx             = 2;                   % middle-layer dimensionality
Ny             = 1;                   % output-layer dimensionality
Ttraining      = 100;                 % number of trials used for training
Ttraining_init = 20;                  % number of trials used for training
Nrepeat        = 1000;                % number of iteration
y_gain         = 1;
phi            = cell(Nsample,Tend);  % set of internal states
results        = cell(Nsample,1);     %
phi_init       = cell(Nsample,Tend);  % set of internal states made of initial data
results_init   = cell(Nsample,1);     %
energy         = zeros(1,Nrepeat/10); %

% perform reverse engineering using multiple cpu cores
NEW_FILE = 0;
if NEW_FILE
 parfor num = 1:Nsample
  [phi(num,:),results{num}]           = reverse_engineering(data{num},Nx,Ttrial,Tinit,Tend,Ttraining,Nrepeat,y_gain);
  [phi_init(num,:),results_init{num}] = reverse_engineering(data{num},Nx,Ttrial,Tinit,Tend,Ttraining_init,Nrepeat,y_gain);
  energy(num,:)                       = results{num}.energy;
 end
 fprintf(1,'reverse engineering was completed (%d h %d m)\n', floor(toc/3600), rem(floor(toc/60),60))
 save([dstdir,'data_autonomous_states.mat'],'phi','results','phi_init','results_init','-v7.3') % save file
else
 load('output/data_autonomous_states.mat'); % read files (data with deconvolution filter applied)
 for num = 1:Nsample, energy(num,:) = results{num}.energy; end
end

%--------------------------------------------------------------------------------
% open loop

Topenloop = cell(Nsample,1);
for num = 1:27
 t        = data{num}.T / Ttrial;
 pos1max  = max(reshape(data{num}.position,Ttrial,[]));
 pos15max = max([pos1max(1:t-14);pos1max(2:t-13);pos1max(3:t-12);pos1max(4:t-11);pos1max(5:t-10);...
                 pos1max(6:t-9);pos1max(7:t-8);pos1max(8:t-7);pos1max(9:t-6);pos1max(10:t-5);...
                 pos1max(11:t-4);pos1max(12:t-3);pos1max(13:t-2);pos1max(14:t-1);pos1max(15:t)]);
 id = find(pos15max == 0 & 1:t-14 > 100);
 if isempty(id), continue, end
 id1 = id(1);
 for id2 = id1:t, if pos1max(id2) > 0, id2 = id2 - 1; break, end, end
 Topenloop(num,1:3) = {id1, id2, id2 - id1 + 1};
 
 tt1            = 1:data{num}.T;
 r              = cast(data{num}.activity(:,tt1),'double');             % neural activity (df/f)
 r              = (r-mean(r')')./std(r')';                              % normalization
 d              = cast(max(min(data{num}.action(:,tt1),1),0),'double'); % action
 y              = d;                                                    % output neural activity
 o              = cast(max(min(data{num}.o(:,tt1),1),0),'double');      % sensory inputs
 pos            = cast(data{num}.position(:,tt1),'double');             % position
 s              = cast(data{num}.s(:,tt1),'double');
 for t = 1:15
  tt1                   = Ttrial*(Topenloop{num,1}-1+t-1)+(1:Ttrial);
  phi{num,100+t}        = phi_estimate(phi{num,100+t-1},o(:,tt1),sig(phi{num,1}.M1*[r(:,tt1);ones(1,Ttrial)]),y(:,tt1),d(:,tt1),pos(:,tt1),1,1);
  phi{num,100+t}.g      = cast(o(4,tt1),'double');
  phi{num,100+t}.g_max  = 1;
  phi{num,100+t}.isgo   = (phi{num,100+t}.o(1,26)==1)*1;
  phi{num,100+t}.ispass = (phi{num,100+t}.pos(125)>=1)*1;
  phi{num,100+t}.isscc  = phi{num,100+t}.isgo*phi{num,100+t}.ispass + (1-phi{num,100+t}.isgo)*(1-phi{num,100+t}.ispass);
 end
end

%--------------------------------------------------------------------------------
% reversal learning

Treversal = cell(Nsample,1);
for num = 28:30
 if num == 28, id1 = 102; end % fish35, reversal is from trial 102 (adaptation period = 20)
 if num == 29, id1 = 122; end % fish36, reversal is from trial 142 (adaptation period = 40)
 if num == 30, id1 = 102; end % fish37, reversal is from trial 102 (adaptation period = 20)
 id2                = data{num}.T / Ttrial;
 Treversal(num,1:3) = {id1, id2, id2 - id1 + 1};
 
 tt1            = 1:data{num}.T;
 r              = cast(data{num}.activity(:,tt1),'double');             % neural activity (df/f)
 r              = (r-mean(r')')./std(r')';                              % normalization
 d              = cast(max(min(data{num}.action(:,tt1),1),0),'double'); % action
 y              = d;                                                    % output neural activity
 o              = cast(max(min(data{num}.o(:,tt1),1),0),'double');      % sensory inputs
 pos            = cast(data{num}.position(:,tt1),'double');             % position
 s              = cast(data{num}.s(:,tt1),'double');
 for t = 1:58
  tt1                   = Ttrial*(Treversal{num,1}-1+t-1)+(1:Ttrial);
  phi{num,100+t}        = phi_estimate(phi{num,100+t-1},o(:,tt1),sig(phi{num,1}.M1*[r(:,tt1);ones(1,Ttrial)]),y(:,tt1),d(:,tt1),pos(:,tt1),1,1);
  phi{num,100+t}.g      = cast(o(4,tt1),'double');
  phi{num,100+t}.g_max  = 1;
  phi{num,100+t}.isgo   = (phi{num,100+t}.o(1,26)==1)*1;
  phi{num,100+t}.ispass = (phi{num,100+t}.pos(125)>=1)*1;
  phi{num,100+t}.isscc  = phi{num,100+t}.isgo*phi{num,100+t}.ispass + (1-phi{num,100+t}.isgo)*(1-phi{num,100+t}.ispass);
 end
end

%--------------------------------------------------------------------------------
% ensemble response analysis

if NEW_FILE
 save([dstdir,'data_autonomous_states.mat'],'phi','results','phi_init','results_init','-v7.3') % save file
end

resp = ensemble_response_analysis(phi(:,1:Tend),Ttrial,Tinit,Tlate,Tend,Nsample_l,score_post);

n_bin     = 5;
resp.corr_x1o1 = zeros(Nsample,Tend/n_bin);
resp.corr_x2o1 = zeros(Nsample,Tend/n_bin);
resp.corr_x1o2 = zeros(Nsample,Tend/n_bin);
resp.corr_x2o2 = zeros(Nsample,Tend/n_bin);
resp.corr_x1o3 = zeros(Nsample,Tend/n_bin);
resp.corr_x2o3 = zeros(Nsample,Tend/n_bin);
for num = 1:Nsample
 for t = 1:Tend/n_bin
  resp.corr_x1o1(num,t) = corr(func(@(a)a.x(1,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(1,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
  resp.corr_x2o1(num,t) = corr(func(@(a)a.x(2,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(1,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
  resp.corr_x1o2(num,t) = corr(func(@(a)a.x(1,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(2,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
  resp.corr_x2o2(num,t) = corr(func(@(a)a.x(2,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(2,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
  resp.corr_x1o3(num,t) = corr(func(@(a)a.x(1,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(3,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
  resp.corr_x2o3(num,t) = corr(func(@(a)a.x(2,:),phi(num,n_bin*(t-1)+1:n_bin*t))',func(@(a)a.o(3,:),phi(num,n_bin*(t-1)+1:n_bin*t))'+randn(n_bin*Ttrial,1)*10^-6);
 end
end

%--------------------------------------------------------------------------------
% distance from optimal generative model
A1 = [1,0.5,0.5,0.5; 0.5,1,0,0.5]';
A0 = [0.5,0.5,0,0.5; 0.5,0.5,0,0.5]';
B1 = [1,0.5; 0.5,1];
B0 = [0,0.5; 0.5,0];
C1 = [1,0];
C0 = [0,1];
resp.err_A     = func(@(a)sum(([sig(a.W1)',sig(a.W0)';1-sig(a.W1)',1-sig(a.W0)']-[A1,A0;1-A1,1-A0]).^2,'all'), phi(:,1:Tend));
resp.err_B     = func(@(a)sum(([sig(a.K1),sig(a.K0);1-sig(a.K1),1-sig(a.K0)]-[B1,B0;1-B1,1-B0]).^2,'all'), phi(:,1:Tend));
resp.err_C     = func(@(a)sum(([sig(a.V1),sig(a.V0);1-sig(a.V1),1-sig(a.V0)]-[C1,C0;1-C1,1-C0]).^2,'all'), phi(:,1:Tend));
resp.err_model = (resp.err_A+resp.err_B+resp.err_C)/(sum([A1,A0;1-A1,1-A0].^2,'all')+sum([B1,B0;1-B1,1-B0].^2,'all')+sum([C1,C0;1-C1,1-C0].^2,'all'));

%--------------------------------------------------------------------------------
% compute free energy

resp.F    = zeros(Nsample,Tend);
for num = 1:Nsample, resp.F(num,:) = compute_free_energy(phi(num,1:Tend)); end

save([dstdir,'fig3_data.mat'],'-struct','resp','-V6') % save file

%--------------------------------------------------------------------------------
% figure output

fig               = figure();
fig.Position(3:4) = [1200 600];

% ensemble activity
subplot(3,6,1), plotdist(1:Ttrial,resp.x1_l_post_go,'blue',2), plotdist(1:Ttrial,resp.x1_l_post_nogo,'red',2), axis([0 Ttrial 0 1]), title('ensemble activity x_1')
subplot(3,6,2), plotdist(1:Ttrial,resp.x2_l_post_go,'blue',2), plotdist(1:Ttrial,resp.x2_l_post_nogo,'red',2), axis([0 Ttrial 0 1]), title('ensemble activity x_2')
subplot(3,6,3), plotdist(1:Ttrial,resp.y_l_post_go,'blue',2), plotdist(1:Ttrial,resp.y_l_post_nogo,'red',2), axis([0 Ttrial 0 1]), title('output activity y')

subplot(3,6,4), plotdist(n_bin/2:n_bin:Tend,resp.corr_x1o1(1:Nsample_l,:),'blue',2), plotdist(n_bin/2:n_bin:Tend,resp.corr_x2o1(1:Nsample_l,:),'red',2), title('corr x_{1,2} v.s. danger'), axis([0 Tend -0.5 1])
subplot(3,6,5), plotdist(n_bin/2:n_bin:Tend,resp.corr_x1o2(1:Nsample_l,:),'blue',2), plotdist(n_bin/2:n_bin:Tend,resp.corr_x2o2(1:Nsample_l,:),'red',2), title('corr x_{1,2} v.s. safety'), axis([0 Tend -0.5 1])
subplot(3,6,6), plotdist(n_bin/2:n_bin:Tend,resp.corr_x1o3(1:Nsample_l,:),'blue',2), plotdist(n_bin/2:n_bin:Tend,resp.corr_x2o3(1:Nsample_l,:),'red',2), title('corr x_{1,2} v.s. optic-flow'), axis([0 Tend -0.2 0.6])
fprintf(1,'p(corr(x1,o1))=%f, p(corr(x2,o1))=%f\n', signrank(resp.corr_x1o1(1:Nsample_l,1),resp.corr_x1o1(1:Nsample_l,Tend/n_bin)), signrank(resp.corr_x2o1(1:Nsample_l,1),resp.corr_x2o1(1:Nsample_l,Tend/n_bin)))
fprintf(1,'p(corr(x1,o2))=%f, p(corr(x2,o2))=%f\n', signrank(resp.corr_x1o2(1:Nsample_l,1),resp.corr_x1o2(1:Nsample_l,Tend/n_bin)), signrank(resp.corr_x2o2(1:Nsample_l,1),resp.corr_x2o2(1:Nsample_l,Tend/n_bin)))
fprintf(1,'p(corr(x1,o3))=%f, p(corr(x2,o3))=%f\n', signrank(resp.corr_x1o3(1:Nsample_l,1),resp.corr_x1o3(1:Nsample_l,Tend/n_bin)), signrank(resp.corr_x2o3(1:Nsample_l,1),resp.corr_x2o3(1:Nsample_l,Tend/n_bin)))

% effective synaptic connectivity
subplot(3,8,9), plotdist(1:Tend,resp.W11(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W11(1:Nsample_l,:),'green',2), title('W_{11}')
subplot(3,8,10), plotdist(1:Tend,resp.W12(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W12(1:Nsample_l,:),'green',2), title('W_{12}')
subplot(3,8,11), plotdist(1:Tend,resp.W13(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W13(1:Nsample_l,:),'green',2), title('W_{13}')
subplot(3,8,12), plotdist(1:Tend,resp.W14(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W14(1:Nsample_l,:),'green',2), title('W_{14}')
subplot(3,8,13), plotdist(1:Tend,resp.W21(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W21(1:Nsample_l,:),'green',2), title('W_{21}')
subplot(3,8,14), plotdist(1:Tend,resp.W22(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W22(1:Nsample_l,:),'green',2), title('W_{22}')
subplot(3,8,15), plotdist(1:Tend,resp.W23(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W23(1:Nsample_l,:),'green',2), title('W_{23}')
subplot(3,8,16), plotdist(1:Tend,resp.W24(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.W24(1:Nsample_l,:),'green',2), title('W_{24}')
subplot(3,5,11), plotdist(1:Tend,resp.V11(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.V11(1:Nsample_l,:),'green',2), title('V_{11}')
subplot(3,5,12), plotdist(1:Tend,resp.V12(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.V12(1:Nsample_l,:),'green',2), title('V_{12}')

% distance from optimal generative model
subplot(3,5,13), plotdist(1:Tend,resp.err_model(Nsample_l+1:Nsample,:),'black',2), plotdist(1:Tend,resp.err_model(1:Nsample_l,:),'blue',2), title('model est err')

% representational dimensionality
subplot(3,5,14), plotdist(1:Tend,resp.dim2(Nsample_l+1:end,:),'black',2), plotdist(1:Tend,resp.dim2(1:Nsample_l,:),'blue',2), title('dimensionality')

% free energy (change from trial Tinit+1)
subplot(3,5,15), plotdist(1:Tend,resp.F(Nsample_l+1:end,:)-resp.F(Nsample_l+1:end,Tinit+1),'black',2), plotdist(1:Tend,resp.F(1:Nsample_l,:)-resp.F(1:Nsample_l,Tinit+1),'blue',2), title('free energy')

print(fig,[dstdir,'fig3.png'],'-dpng')
return

%--------------------------------------------------------------------------------
