
%--------------------------------------------------------------------------------
% fig1.m
%
% Copyright (C) 2025 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2025-06-18
%--------------------------------------------------------------------------------
% initialization

clear
Tinit      = 20;                     % end of initial trials
Tlate      = 61;                     % start of late trials 
Tend       = 100;                    % number of trials
Ttrial     = 150;                    % duration of 1 trial (15 s)
T          = Ttrial*Tend;            % entire duration
Nsample    = 45;                     % number of samples
dstdir     = 'output/';              % export directory
addpath(genpath('scripts/'));        % import directory

%--------------------------------------------------------------------------------
% neural activity analysis

load('preprocessed_data.mat'); % read files (data with deconvolution filter applied)

data_var   = zeros(Nsample,1);
for num = 1:Nsample, data_var(num) = mean(var(data{num}.activity')); end
datalist   = find(data_var<10^3);
dst        = cell(length(datalist),1);
for num = 1:length(datalist), dst{num} = data{datalist(num)}; end
data       = dst;
Nsample    = length(datalist);
clear dst

for num = 1:Nsample
 data{num}.flow     = max(min(smooth(data{num}.flow_orig,11)',1),0);
 data{num}.action   = max(min(smooth(data{num}.action_orig,11)',1),0);
 data{num}.position = max(reshape(data{num}.position,1,[]),0);
end

%--------------------------------------------------------------------------------

for num = 1:Nsample
 data{num}.s = [data{num}.go.*data{num}.blue; data{num}.go.*data{num}.red; data{num}.nogo.*data{num}.red; data{num}.nogo.*data{num}.blue;...
                data{num}.white; data{num}.flow; data{num}.stim; data{num}.action; data{num}.position];
end

% examples of blue, red, flow coding activity
num               = 6;
tt                = 1:Tend*Ttrial;
tt1               = Ttrial*(Tlate-1)+1:Ttrial*Tend;
fig               = figure();
fig.Position(3:4) = [800 400];
[~,sort_goblue]   = sort(corr(data{num}.activity(:,tt)',data{num}.s(1,tt)'),'descend');
[~,sort_gored]    = sort(corr(data{num}.activity(:,tt)',data{num}.s(2,tt)'),'descend');
[~,sort_nogored]  = sort(corr(data{num}.activity(:,tt)',data{num}.s(3,tt)'),'descend');
[~,sort_nogoblue] = sort(corr(data{num}.activity(:,tt)',data{num}.s(4,tt)'),'descend');
[~,sort_interval] = sort(corr(data{num}.activity(:,tt)',data{num}.s(5,tt)'),'descend');
subplot(2,4,5), area(1:Ttrial*(Tend-Tlate+1),data{num}.s(1,tt1),'LineStyle','none','FaceColor',[0.75,0.75,1]), hold on
plot(1:Ttrial*(Tend-Tlate+1),data{num}.activity(sort_goblue(1),tt1),'b-','LineWidth',1), axis([0 Ttrial*(Tend-Tlate+1) 0 1]), hold off
subplot(2,4,6), area(1:Ttrial*(Tend-Tlate+1),data{num}.s(2,tt1),'LineStyle','none','FaceColor',[1,0.75,0.75]), hold on
plot(1:Ttrial*(Tend-Tlate+1),data{num}.activity(sort_gored(1),Ttrial*(Tlate-1)+1:Ttrial*Tend),'r-','LineWidth',1), axis([0 Ttrial*(Tend-Tlate+1) 0 1]), hold off
subplot(2,4,7), area(1:Ttrial*(Tend-Tlate+1),data{num}.s(3,tt1),'LineStyle','none','FaceColor',[0.75,0.75,0.75]), hold on
plot(1:Ttrial*(Tend-Tlate+1),data{num}.activity(sort_nogored(1),tt1),'r-','LineWidth',1), axis([0 Ttrial*(Tend-Tlate+1) 0 1]), hold off

% regression analysis
reg = regression_analysis(data,Ttrial,Tend);

subplot(2,4,8), pie(flip(sum(reg.tot_var,'double'))), title({'external information','encoded in neural activity'})
drawnow

activity_post_example          = data{num}.activity(:,Ttrial*(Tlate-1)+1:Ttrial*Tend);
activity_post_goblue_example   = [data{num}.activity(sort_goblue(1),tt1); data{num}.s(1,tt1)];
activity_post_gored_example    = [data{num}.activity(sort_gored(1),tt1); data{num}.s(2,tt1)];
activity_post_nogored_example  = [data{num}.activity(sort_nogored(1),tt1); data{num}.s(3,tt1)];
activity_post_nogoblue_example = [data{num}.activity(sort_nogoblue(1),tt1); data{num}.s(4,tt1)];
activity_post_interval_example = [data{num}.activity(sort_interval(1),tt1); data{num}.s(5,tt1)];
tot_var_values                 = sum(reg.tot_var,'double');
tot_var_labels                 = {'GoBlue','GoRed','NogoRed','NogoBlue','Interval','Flow','Stim','Action','Position','Others'};
var_qs1                        = func(@(a)[mean(a(1,1:Tinit)),mean(a(1,Tinit+1:Tlate-1)),mean(a(1,Tlate:Tend))], reg.qs_var);
var_qs2                        = func(@(a)[mean(a(2,1:Tinit)),mean(a(2,Tinit+1:Tlate-1)),mean(a(2,Tlate:Tend))], reg.qs_var);
var_qs3                        = func(@(a)[mean(a(3,1:Tinit)),mean(a(3,Tinit+1:Tlate-1)),mean(a(3,Tlate:Tend))], reg.qs_var);
var_qs4                        = func(@(a)[mean(a(4,1:Tinit)),mean(a(4,Tinit+1:Tlate-1)),mean(a(4,Tlate:Tend))], reg.qs_var);
var_blue                       = var_qs1 + var_qs4;
var_red                        = var_qs2 + var_qs3;

fprintf(1,'%d neurons from %d samples are used for Fig. 1. neural activity data analysis\n', sum(func(@(a)size(a,1), reg.Mat)), size(reg.Mat,1));

%--------------------------------------------------------------------------------
% initialization

Nsample    = 45;
load('preprocessed_data0.mat'); % read files (data without deconvolution filter)

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

fprintf(1,'number of neurons = %f +/- %f (from %d samples)\n', mean(func(@(a)size(a.activity,1),data)), std(func(@(a)size(a.activity,1),data)), size(data,1));

%--------------------------------------------------------------------------------
% behavior analysis

num                   = 8;
bhv                   = behavior_analysis(data,Ttrial,Tinit,Tlate,Tend);
pos_pre_go_example    = bhv.pos_pre_go{num};
pos_pre_nogo_example  = bhv.pos_pre_nogo{num};
pos_post_go_example   = bhv.pos_post_go{num};
pos_post_nogo_example = bhv.pos_post_nogo{num};

% sort the data in order of learners and non-learners
learner_orig          = (bhv.score_post(:,1)>=0.5) .* (bhv.score_post(:,2)>=0.5) .* (bhv.score_post(:,1)+bhv.score_post(:,2)>=1.2);
score_pre             = [bhv.score_pre(learner_orig==1,:); bhv.score_pre(learner_orig==0,:)];
score_post            = [bhv.score_post(learner_orig==1,:); bhv.score_post(learner_orig==0,:)];
data                  = [data(learner_orig==1); data(learner_orig==0)];
learner               = [learner_orig(learner_orig==1); learner_orig(learner_orig==0)];
Nsample_l             = sum(learner);
Nsample_nl            = Nsample - Nsample_l;

subplot(2,4,1), plot([0,100],[1,1],'k--',1:100,bhv.pos_pre_go{num}(:,26:125),'b-',1:100,bhv.pos_pre_nogo{num}(:,26:125),'r-','LineWidth',2), axis([0 100 0 5]), title('position pre')
subplot(2,4,2), plot([0,100],[1,1],'k--',1:100,bhv.pos_post_go{num}(:,26:125),'b-',1:100,bhv.pos_post_nogo{num}(:,26:125),'r-','LineWidth',2), axis([0 100 0 5]), title('position post')
subplot(2,4,3), scatter(score_pre(1:Nsample_l,1),score_pre(1:Nsample_l,2),40,[0,1,0],'filled'), hold on
scatter(score_pre(Nsample_l+1:end,1),score_pre(Nsample_l+1:end,2),40,[0,0,0],'filled'), axis([0 1 0 1]), title('score pre'), hold off
subplot(2,4,4), scatter(score_post(1:Nsample_l,1),score_post(1:Nsample_l,2),40,[0,1,0],'filled'), hold on
scatter(score_post(Nsample_l+1:end,1),score_post(Nsample_l+1:end,2),40,[0,0,0],'filled'), axis([0 1 0 1]), title('score post'), hold off
drawnow

save([dstdir,'fig1_data.mat'], 'pos_pre_go_example', 'pos_pre_nogo_example', 'pos_post_go_example', 'pos_post_nogo_example', 'score_pre', 'score_post', 'learner', 'learner_orig',...
     'activity_post_example', 'activity_post_goblue_example', 'activity_post_gored_example', 'activity_post_nogored_example', 'activity_post_nogoblue_example', 'activity_post_interval_example',...
     'tot_var_values', 'tot_var_labels', 'var_blue', 'var_red', '-V6')

print(fig,[dstdir,'fig1.png'],'-dpng')

%--------------------------------------------------------------------------------
