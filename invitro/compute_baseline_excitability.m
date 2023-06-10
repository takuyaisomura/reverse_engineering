
%--------------------------------------------------------------------------------
% compute_baseline_excitability.m
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

function [baseline_ctrl, baseline_bic, baseline_dzp, baseline_mix0, baseline_mix50] = compute_baseline_excitability(data_ctrl, data_bic, data_dzp, data_mix0, data_mix50)

mean_resp1 = [];
mean_resp2 = [];
mean_resp3 = [];
mean_resp4 = [];

for datatype = {'ctrl' 'bic' 'dzp' 'mix0' 'mix50'}

datatype = datatype{1};
if strcmp(datatype,'ctrl'), data = data_ctrl; end
if strcmp(datatype,'bic'), data = data_bic; end
if strcmp(datatype,'dzp'), data = data_dzp; end
if strcmp(datatype,'mix0'), data = data_mix0; end
if strcmp(datatype,'mix50'), data = data_mix50; end

for num = 1:length(data)

% read sources and responses
s      = data{num}.s';                                                      % hidden sources
r      = data{num}.r';                                                      % neuronal responses

% compute ensemble averages
[r_,~,r_s_10,r_s_01,~,~,~,~,~] = conditional_expectations(r',s');           % compute conditional expectations
g1     = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)>0.5));        % find source 1-preferring ensemble
g2     = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)<-0.5));       % find source 2-preferring ensemble
if length(g1) == 1, g1 = [g1 g1]; end
if length(g2) == 1, g2 = [g2 g2]; end

x_     = [mean(r_(:,g1)'); mean(r_(:,g2)')];                                % mean response intensity in each session
if strcmp(datatype,'ctrl') && num <= 23,     mean_resp1 = [mean_resp1; mean(x_(1,1:10)) mean(x_(2,1:10))];
elseif strcmp(datatype,'ctrl') && num >= 24, mean_resp2 = [mean_resp2; mean(x_(1,1:10)) mean(x_(2,1:10))];
elseif strcmp(datatype,'bic'),               mean_resp2 = [mean_resp2; mean(x_(1,1:10)) mean(x_(2,1:10))];
elseif strcmp(datatype,'dzp'),               mean_resp2 = [mean_resp2; mean(x_(1,1:10)) mean(x_(2,1:10))];
elseif strcmp(datatype,'mix0'),              mean_resp3 = [mean_resp3; mean(x_(1,1:10)) mean(x_(2,1:10))];
elseif strcmp(datatype,'mix50'),             mean_resp4 = [mean_resp4; mean(x_(1,1:10)) mean(x_(2,1:10))]; end
% We separated the control group in two two groups: one obtained in previous work (Isomura et al., 2015)
% and one newly obtained for this work because of different noise and excitability levels in these groups.

end

end

% compute average baseline excitability in each condition
baseline_ctrl  = [ones(23,1)*mean(mean(mean_resp1)); ones(7,1)*mean(mean(mean_resp2))];
baseline_bic   = ones(length(data_bic),1)*mean(mean(mean_resp2));
baseline_dzp   = ones(length(data_dzp),1)*mean(mean(mean_resp2));
baseline_mix0  = ones(length(data_mix0),1)*mean(mean(mean_resp3));
baseline_mix50 = ones(length(data_mix50),1)*mean(mean(mean_resp4));

end

%--------------------------------------------------------------------------------
