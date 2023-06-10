
%--------------------------------------------------------------------------------
% fig3.m
%
% This analysis is included in
% Experimental validation of the free-energy principle with in vitro neural networks
% Takuya Isomura, Kiyoshi Kotani, Yasuhiko Jimbo, Karl Friston
%
% Copyright (C) 2022 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2022-09-19
%

%--------------------------------------------------------------------------------
% initialisation

clear
No            = 32;                                                         % number of sensory stimuli
Nx            = 2;                                                          % number of neuronal ensembles
T             = 256*100;                                                    % total time
Nsession      = 100;                                                        % number of sessions
Ninit         = 10;                                                         % number of initial sessions
lambda        = 3000;                                                       % prior strength (insensitivity to plasticity)
gain          = 2;                                                          % relative strength of initial Hebb and Home for prediction
SHOWVIDEO     = 1;                                                          % show animations and save them as videos
dirname       = 'results/';                                                 % directry for output files

% load empirical neuronal response data
load('response_data_ctrl.mat')                                              % neuronal response data under control condition (n = 30)
load('response_data_bic.mat')                                               % neuronal response data treated with bicuculline (n = 6)
load('response_data_dzp.mat')                                               % neuronal response data treated with bicuculline (n = 7)
load('response_data_mix0.mat')                                              % neuronal response data under 0% mix condition (n = 4)
load('response_data_mix50.mat')                                             % neuronal response data under 50% mix condition (n = 4)

% compute average baseline excitability in each condition
[baseline_ctrl, baseline_bic, baseline_dzp, baseline_mix0, baseline_mix50] = compute_baseline_excitability(data_ctrl, data_bic, data_dzp, data_mix0, data_mix50);

% create figure windows
fig1 = figure();
if SHOWVIDEO == 1, fig2 = figure('Color','white'); fig2.Position(3:4) = [600 600]; end

%--------------------------------------------------------------------------------
% run analysis

for datatype = {'ctrl' 'bic' 'dzp' 'mix0' 'mix50'}

% define data, baseline, sample_num_ (representative sample), and fig_pos (figure position) for each condition
datatype = datatype{1};
fprintf(1,'%s condition\n',datatype)
if strcmp(datatype,'ctrl'), data = data_ctrl; baseline = baseline_ctrl; sample_num_ = 19; fig_pos = 1; end
if strcmp(datatype,'bic'), data = data_bic; baseline = baseline_bic; sample_num_ = 1; fig_pos = 2; end
if strcmp(datatype,'dzp'), data = data_dzp; baseline = baseline_dzp; sample_num_ = 5; fig_pos = 3; end
if strcmp(datatype,'mix0'), data = data_mix0; baseline = baseline_mix0; sample_num_ = 1; fig_pos = 4; end
if strcmp(datatype,'mix50'), data = data_mix50; baseline = baseline_mix50; sample_num_ = 1; fig_pos = 5; end

% create lists
err_x_xp_list        = zeros(length(data),100);                             % error between observed and predicted responses
err_W_Wp_list        = zeros(length(data),100);                             % error between estimated and predicted synaptic weights
err_W_qA_list        = zeros(length(data)*4,100);                           % save error between empirical and ideal posteriors
L_list               = zeros(length(data),100);                             % values of cost function
x1_list              = zeros(length(data),100,1);                           % response of source 1-preferring ensemble
x2_list              = zeros(length(data),100,2);                           % response of source 2-preferring ensemble
x_post               = zeros(2,length(data)*256*10);                        % observed responses in session 91-100
xp_post              = zeros(2,length(data)*256*10);                        % predicted responses in session 91-100
qA1_list             = zeros(length(data),32,4);                            % empirical posterior belief about A1
plasticity_amount    = zeros(length(data),Nx*No);                           % amount of plasticity for Suppl Fig 1b
deviation_of_W       = zeros(length(data),Nx*No);                           % deviation of synaptic weights for Suppl Fig 1c

% ideal Bayesian posterior belief about A matrix
qA11id = kron([0.875 0.625; 0.625 0.875],ones(16,1));
qA10id = kron([0.125 0.375; 0.375 0.125],ones(16,1));
if strcmp(datatype,'mix0')
 qA11id = kron([1 0.5; 0.5 1],ones(16,1));
 qA10id = kron([0 0.5; 0.5 0],ones(16,1));
elseif strcmp(datatype,'mix50')
 qA11id = kron([0.75 0.75; 0.75 0.75],ones(16,1));
 qA10id = kron([0.25 0.25; 0.25 0.25],ones(16,1));
end

%--------------------------------------------------------------------------------
% run estimation and prediction

for num = [1:length(data) sample_num_]

% read sources, stimuli, and responses
s      = data{num}.s';                                                      % hidden sources
o      = data{num}.o';                                                      % sensory stimuli
r      = data{num}.r';                                                      % neuronal responses

% categorise into sources 1- and 2-preferring ensembles
[r_,r_s_11,r_s_10,r_s_01,r_s_00,~,~,~,~] = conditional_expectations(r',s'); % compute conditional expectations
g1     = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)>0.5));        % find source 1-preferring ensemble
g2     = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)<-0.5));       % find source 2-preferring ensemble
if length(g1) == 1, g1 = [g1 g1]; end
if length(g2) == 1, g2 = [g2 g2]; end

% compute ensemble responses
x      = [mean(r(g1,:)); mean(r(g2,:))];                                    % compute ensemble responses
x_     = [mean(r_(:,g1)'); mean(r_(:,g2)')];                                % compute mean response intensity in each session
x_s_11 = [mean(r_s_11(:,g1)'); mean(r_s_11(:,g2)')];                        % compute conditional ensemble responses given s = (1,1)
x_s_10 = [mean(r_s_10(:,g1)'); mean(r_s_10(:,g2)')];                        % compute conditional ensemble responses given s = (1,0)
x_s_01 = [mean(r_s_01(:,g1)'); mean(r_s_01(:,g2)')];                        % compute conditional ensemble responses given s = (0,1)
x_s_00 = [mean(r_s_00(:,g1)'); mean(r_s_00(:,g2)')];                        % compute conditional ensemble responses given s = (0,0)

% normalise ensemble responses
x(:,s(1,:)==1&s(2,:)==1) = x(:,s(1,:)==1&s(2,:)==1) - x_s_11(:,1);          % subtract initial values to remove the stimulus-specific components
x(:,s(1,:)==1&s(2,:)==0) = x(:,s(1,:)==1&s(2,:)==0) - x_s_10(:,1);          % subtract initial values to remove the stimulus-specific components
x(:,s(1,:)==0&s(2,:)==1) = x(:,s(1,:)==0&s(2,:)==1) - x_s_01(:,1);          % subtract initial values to remove the stimulus-specific components
x(:,s(1,:)==0&s(2,:)==0) = x(:,s(1,:)==0&s(2,:)==0) - x_s_00(:,1);          % subtract initial values to remove the stimulus-specific components
x      = x - kron(x_-x_(:,1),ones(1,256));                                  % subtract mean response (trend) in each session
x      = diag(std(x'))^-1*(x-mean(x')');                                    % normalise to zero mean and unit variance
x      = 1/2 + x/2 + (x_(:,1)-baseline(num))/baseline(num)/4;               % add baseline excitability for each sample (relative value)
x      = max(min(x,1),0);                                                   % normalise in the range between 0 and 1

% ensemble responses for each session
x1_list(num,:,1) = mean(reshape(x(1,s(1,:)==1),[],100));
x1_list(num,:,2) = mean(reshape(x(1,s(1,:)==0),[],100));
x2_list(num,:,1) = mean(reshape(x(2,s(2,:)==1),[],100));
x2_list(num,:,2) = mean(reshape(x(2,s(2,:)==0),[],100));

% estimate firing threshold factor based on initial empirical data
phi1             = log(mean(x(:,1:256*Ninit)'))';                           % firing threshold factor phi1 = log(<x>)
phi0             = log(1-mean(x(:,1:256*Ninit)'))';                         % firing threshold factor phi0 = log(1-<x>)

%--------------------------------------------------------------------------------
% estimation of effective synaptic connectivity

W1        = zeros(Nx,No,Nsession);                                          % estimated synaptic weights
W0        = zeros(Nx,No,Nsession);                                          % estimated synaptic weights
W         = zeros(Nx,No,Nsession);                                          % estimated synaptic weights (W = W1 - W0)
L         = zeros(Nsession,1);                                              % cost function (= variational free energy)
Hebb1     = ones(Nx,No)*lambda/2;                                           % matrix for Hebbian product
Hebb0     = ones(Nx,No)*lambda/2;                                           % matrix for Hebbian product
Home1     = ones(Nx,No)*lambda;                                             % matrix for homeostatic term
Home0     = ones(Nx,No)*lambda;                                             % matrix for homeostatic term
for t = 1:Nsession
 tt        = 256*(t-1)+1:256*t;                                             % select data in session t
 W1(:,:,t) = logit(Hebb1./Home1);                                           % update synaptic weights
 W0(:,:,t) = logit(Hebb0./Home0);                                           % update synaptic weights
 W(:,:,t)  = W1(:,:,t) - W0(:,:,t);                                         % update synaptic weights
 h1        = log(1-sig(W1(:,:,t)))*ones(No,1) + phi1;                       % compute firing threshold
 h0        = log(1-sig(W0(:,:,t)))*ones(No,1) + phi0;                       % compute firing threshold
 Hebb1     = Hebb1 + x(:,tt)*o(:,tt)';                                      % compute synaptic plasticity
 Hebb0     = Hebb0 + (1-x(:,tt))*o(:,tt)';                                  % compute synaptic plasticity
 Home1     = Home1 + x(:,tt)*ones(No,256)';                                 % compute synaptic plasticity
 Home0     = Home0 + (1-x(:,tt))*ones(No,256)';                             % compute synaptic plasticity
 L(t)      = sum(sum([x(:,tt);1-x(:,tt)] .* (log([x(:,tt);1-x(:,tt)]+10^-6) - [W1(:,:,t);W0(:,:,t)]*o(:,tt) - [h1;h0]))); % compute cost function
end

%--------------------------------------------------------------------------------
% prediction of neuronal responses and effective synaptic connectivity

xp         = zeros(Nx,T);                                                   % predicted neuronal responses
W1p        = zeros(Nx,No,Nsession);                                         % predicted synaptic weights
W0p        = zeros(Nx,No,Nsession);                                         % predicted synaptic weights
Wp         = zeros(Nx,No,Nsession);                                         % predicted synaptic weights (Wp = W1p - W0p)
Hebb1      = ones(Nx,No)*lambda*gain/2;                                     % matrix for Hebbian product
Hebb0      = ones(Nx,No)*lambda*gain/2;                                     % matrix for Hebbian product
Home1      = ones(Nx,No)*lambda*gain;                                       % matrix for homeostatic term
Home0      = ones(Nx,No)*lambda*gain;                                       % matrix for homeostatic term
for t = 1:Ninit
 tt         = 256*(t-1)+1:256*t;                                            % select data in session t
 W1p(:,:,t) = W1(:,:,t);                                                    % update synaptic weights
 W0p(:,:,t) = W0(:,:,t);                                                    % update synaptic weights
 Wp(:,:,t)  = W1p(:,:,t) - W0p(:,:,t);                                      % update synaptic weights
 h1         = log(1-sig(W1p(:,:,t)))*ones(No,1) + phi1;                     % compute firing threshold
 h0         = log(1-sig(W0p(:,:,t)))*ones(No,1) + phi0;                     % compute firing threshold
 xp(:,tt)   = sig(Wp(:,:,t)*o(:,tt) + h1 - h0);                             % compute predicted neuronal responses
 Hebb1      = Hebb1 + x(:,tt)*o(:,tt)'*gain;                                % compute synaptic plasticity
 Hebb0      = Hebb0 + (1-x(:,tt))*o(:,tt)'*gain;                            % compute synaptic plasticity
 Home1      = Home1 + x(:,tt)*ones(No,256)'*gain;                           % compute synaptic plasticity
 Home0      = Home0 + (1-x(:,tt))*ones(No,256)'*gain;                       % compute synaptic plasticity
end
for t = Ninit+1:Nsession
 tt         = 256*(t-1)+1:256*t;                                            % select data in session t
 W1p(:,:,t) = logit(Hebb1./Home1);                                          % update synaptic weights
 W0p(:,:,t) = logit(Hebb0./Home0);                                          % update synaptic weights
 Wp(:,:,t)  = W1p(:,:,t) - W0p(:,:,t);                                      % update synaptic weights
 h1         = log(1-sig(W1p(:,:,t)))*ones(No,1) + phi1;                     % compute firing threshold
 h0         = log(1-sig(W0p(:,:,t)))*ones(No,1) + phi0;                     % compute firing threshold
 xp(:,tt)   = sig(Wp(:,:,t)*o(:,tt) + h1 - h0);                             % compute predicted neuronal responses
 Hebb1      = Hebb1 + xp(:,tt)*o(:,tt)';                                    % compute synaptic plasticity
 Hebb0      = Hebb0 + (1-xp(:,tt))*o(:,tt)';                                % compute synaptic plasticity
 Home1      = Home1 + xp(:,tt)*ones(No,256)';                               % compute synaptic plasticity
 Home0      = Home0 + (1-xp(:,tt))*ones(No,256)';                           % compute synaptic plasticity
end

%--------------------------------------------------------------------------------
% compute prediction errors

for t = 1:Nsession
 tt                   = 256*(t-1)+1:256*t;                                  % select data in session t
 err_x_xp_list(num,t) = mean(mean((x(:,tt)-xp(:,tt)).^2));                  % error between observed and predicted responses
 W_hat                = sig([W1(:,:,t)' W0(:,:,t)'])';                      % W_hat = sig([W1' W0'])'
 Wp_hat               = sig([W1p(:,:,t)' W0p(:,:,t)'])';                    % Wp_hat = sig([W1p' W0p'])'
 err_W_qA_list(4*(num-1)+(1:4),t) = mean((W_hat' - [qA11id qA10id]).^2)/mean(mean([qA11id qA10id].^2)); % error between empirical and ideal posteriors
 err_W_Wp_list(num,t) = sum(sum((W_hat - Wp_hat).^2))/sum(sum(W_hat.^2));   % error between estimated and predicted synaptic weights
end
L_list(num,:)     = L;                                                      % cost function

tt                                 = 256*(91-1)+1:256*100;                  % select data in session 91-100
x_post(:,2560*(num-1)+1:2560*num)  = x(:,tt);                               % observed responses in session 91-100
xp_post(:,2560*(num-1)+1:2560*num) = xp(:,tt);                              % predicted responses in session 91-100

qA11              = sig(W1(:,:,100))';                                      % compute empirical posterior belief about A11
qA10              = sig(W0(:,:,100))';                                      % compute empirical posterior belief about A10
qA1               = [qA11(:,1).*qA11(:,2) qA11(:,1).*qA10(:,2) qA10(:,1).*qA11(:,2) qA10(:,1).*qA10(:,2)]; % compute empirical posterior belief about A1
qA1_list(num,:,:) = qA1;

% compute plasticity
for t = 2:Nsession, plasticity_amount(num,:) = plasticity_amount(num,:) + [abs(W(1,:,t)-W(1,:,t-1)),abs(W(2,:,t)-W(2,:,t-1))]; end
deviation_of_W(num,:) = [W(1,:,100)-W(1,:,1), W(2,:,100)-W(2,:,1)];

fprintf(1,'%d, err_x=%.3f, err_W=%.3f\n', num, err_x_xp_list(num,100), err_W_Wp_list(num,100))

end

%--------------------------------------------------------------------------------
% output files and figures

% compute relationship between observed responses (x_post) and predicted responses (xp_post) during session 91-100
x_xp_mean = zeros(25,1);
x_xp_stdv = zeros(25,1);
for i = 1:25
 x_xp_mean(i) = mean([xp_post(1,ceil(x_post(1,:)*24.999+0.001)' == i,:) xp_post(2,ceil(x_post(2,:)*24.999+0.001)' == i,:)]); % compute mean
 x_xp_stdv(i) = std([xp_post(1,ceil(x_post(1,:)*24.999+0.001)' == i,:) xp_post(2,ceil(x_post(2,:)*24.999+0.001)' == i,:)]);  % compute standard deviation
end

% file output
err_W_Wp_list(:,1) = 0;
csvwrite([dirname,'err_x_xp_',datatype,'.csv'],[1:Nsession; err_x_xp_list]) % save error between observed and predicted responses
csvwrite([dirname,'err_W_Wp_',datatype,'.csv'],[1:Nsession; err_W_Wp_list]) % save error between estimated and predicted synaptic weights
csvwrite([dirname,'err_W_qA_',datatype,'.csv'],[1:Nsession; err_W_qA_list]) % save error between empirical and ideal posteriors
csvwrite([dirname,'L_',datatype,'.csv'],[1:Nsession; L_list])               % save values of cost function
csvwrite([dirname,'x_xp_',datatype,'.csv'],[1:T; s; x; xp])                 % save hidden sources and observed and predicted responses
csvwrite([dirname,'W_Wp_',datatype,'.csv'],[1:32; W(:,:,100); Wp(:,:,100)]) % save estimated and predicted synaptic weights at session 100
csvwrite([dirname,'x_xp_post_',datatype,'.csv'],[1:25; 0.02:0.04:0.98; x_xp_mean'; x_xp_stdv']) % save relationship between observed and predicted responses
csvwrite([dirname,'trajectory_W_',datatype,'.csv'],[1 1:32 1:32; (1:100)' reshape(W(1,:,:),[32 100])' reshape(W(2,:,:),[32 100])'])    % estimated synaptic weights
csvwrite([dirname,'trajectory_Wp_',datatype,'.csv'],[1 1:32 1:32; (1:100)' reshape(Wp(1,:,:),[32 100])' reshape(Wp(2,:,:),[32 100])']) % predicted synaptic weights
csvwrite([dirname,'plasticity_amount_',datatype,'.csv'],[1:32 1:32; plasticity_amount])         % amount of plasticity
csvwrite([dirname,'deviation_of_W_',datatype,'.csv'],[1:32 1:32; deviation_of_W])               % deviation of synapitc weights

% visualise empirical posterior belief about A matrix
img        = zeros(800,400,3);
img(:,:,1) = kron(max(min(qA1*4,1),0),ones(25,100));
img(:,:,2) = kron(max(min(qA1*4-1,1),0),ones(25,100));
imwrite(img, [dirname,'qA1_',datatype,'.png'])
csvwrite([dirname,'qA1_',datatype,'.csv'],qA1)

% visualise empirical posterior belief about A matrix (average over each condition)
img        = zeros(800,400,3);
img(:,:,1) = kron(max(min(reshape(mean(qA1_list)-0.1,[32 4])*20/3,1),0),ones(25,100));
img(:,:,2) = kron(max(min(reshape(mean(qA1_list)-0.1,[32 4])*20/3-1,1),0),ones(25,100));
imwrite(img, [dirname,'qA1_avg_',datatype,'.png'])

% figure output
fig3_figure_output(fig1, datatype, fig_pos, x1_list, x_xp_mean, x_xp_stdv, err_x_xp_list, W, Wp, err_W_Wp_list, err_W_qA_list, L_list)

% video output
if SHOWVIDEO == 1, fig3_video_output(fig2, dirname, datatype, Ninit, o, W, Wp, phi1, phi0), end

end

%--------------------------------------------------------------------------------
