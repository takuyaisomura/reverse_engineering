
%--------------------------------------------------------------------------------
% fig2_empirical_response.m
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
load('response_data_ctrl.mat')
load('response_data_bic.mat')
load('response_data_dzp.mat')
load('response_data_apv.mat')

for condition = 1:2
 if condition == 1, N = length(data_ctrl); end
 if condition == 2, N = length(data_apv); end
 r_ = zeros(100,N*64);
 r_s_00 = zeros(100,N*64); r_s_10 = zeros(100,N*64); r_s_01 = zeros(100,N*64); r_s_11 = zeros(100,N*64);
 for h = 1:N
  if condition == 1, s = data_ctrl{h}.s; r = data_ctrl{h}.r; end
  if condition == 2, s = data_apv{h}.s; r = data_apv{h}.r; end
  ii = 64*(h-1)+(1:64);
  [r_(:,ii),r_s_11(:,ii),r_s_10(:,ii),r_s_01(:,ii),r_s_00(:,ii),~,~,~,~] = conditional_expectations(r,s);
 end
 
 g1 = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)>0.5));
 g2 = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)<-0.5));
 g0 = find((mean(r_)>1)&(min(r_)>0.1)&(abs(mean(r_s_10-r_s_01))<=0.5));
 
 if condition == 1
  csvwrite(['resp_ctrl_g1_r_s.csv'],[1:100 1:100 1:100 1:100; r_s_11(:,g1)' r_s_10(:,g1)'  r_s_01(:,g1)'  r_s_00(:,g1)'])
  csvwrite(['resp_ctrl_g2_r_s.csv'],[1:100 1:100 1:100 1:100; r_s_11(:,g2)' r_s_10(:,g2)'  r_s_01(:,g2)'  r_s_00(:,g2)'])
 elseif condition == 2
  csvwrite(['resp_apv_g1_r_s.csv'],[1:100 1:100 1:100 1:100; r_s_11(:,g1)' r_s_10(:,g1)'  r_s_01(:,g1)'  r_s_00(:,g1)'])
  csvwrite(['resp_apv_g2_r_s.csv'],[1:100 1:100 1:100 1:100; r_s_11(:,g2)' r_s_10(:,g2)'  r_s_01(:,g2)'  r_s_00(:,g2)'])
 end
end

%--------------------------------------------------------------------------------

for condition = 1:3
 if condition == 1, N = length(data_ctrl); end
 if condition == 2, N = length(data_bic); end
 if condition == 3, N = length(data_dzp); end
 r_ = zeros(100,N*64);
 r_s_00 = zeros(100,N*64); r_s_10 = zeros(100,N*64); r_s_01 = zeros(100,N*64); r_s_11 = zeros(100,N*64);
 r_s1_0 = zeros(100,N*64); r_s1_1 = zeros(100,N*64); r_s2_0 = zeros(100,N*64); r_s2_1 = zeros(100,N*64);
 for h = 1:N
  if condition == 1, s = data_ctrl{h}.s; r = data_ctrl{h}.r; end
  if condition == 2, s = data_bic{h}.s; r = data_bic{h}.r; end
  if condition == 3, s = data_dzp{h}.s; r = data_dzp{h}.r; end
  ii = 64*(h-1)+(1:64);
  [r_(:,ii),r_s_11(:,ii),r_s_10(:,ii),r_s_01(:,ii),r_s_00(:,ii),r_s1_1(:,ii),r_s1_0(:,ii),r_s2_1(:,ii),r_s2_0(:,ii)] = conditional_expectations(r,s);
 end
 
 g1 = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)>0.5));
 g2 = find((mean(r_)>1)&(min(r_)>0.1)&(mean(r_s_10-r_s_01)<-0.5));
 g0 = find((mean(r_)>1)&(min(r_)>0.1)&(abs(mean(r_s_10-r_s_01))<=0.5));
 
 if condition == 1
  csvwrite(['resp_ctrl_g1_r_s1.csv'],[1:100 1:100; r_s1_1(:,g1)' r_s1_0(:,g1)'])
  csvwrite(['resp_ctrl_g2_r_s2.csv'],[1:100 1:100; r_s2_1(:,g2)' r_s2_0(:,g2)'])
  csvwrite(['resp_ctrl_new_r_.csv'],[1:100; r_(:,[g1(g1>23*64) g2(g2>23*64)])'])
 elseif condition == 2
  csvwrite(['resp_bic_g1_r_s1.csv'],[1:100 1:100; r_s1_1(:,g1)' r_s1_0(:,g1)'])
  csvwrite(['resp_bic_g2_r_s2.csv'],[1:100 1:100; r_s2_1(:,g2)' r_s2_0(:,g2)'])
  csvwrite(['resp_bic_r_.csv'],[1:100; r_(:,[g1 g2])'])
 elseif condition == 3
  csvwrite(['resp_dzp_g1_r_s1.csv'],[1:100 1:100; r_s1_1(:,g1)' r_s1_0(:,g1)'])
  csvwrite(['resp_dzp_g2_r_s2.csv'],[1:100 1:100; r_s2_1(:,g2)' r_s2_0(:,g2)'])
  csvwrite(['resp_dzp_r_.csv'],[1:100; r_(:,[g1 g2])'])
 end
end

return

%--------------------------------------------------------------------------------

