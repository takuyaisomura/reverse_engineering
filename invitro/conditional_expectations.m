
%--------------------------------------------------------------------------------
% conditional_expectations.m
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

function [r_,r_s_11,r_s_10,r_s_01,r_s_00,r_s1_1,r_s1_0,r_s2_1,r_s2_0] = conditional_expectations(r,s)

N      = length(r(1,:));
r_     = zeros(100,N);
r_s_00 = zeros(100,N); r_s_10 = zeros(100,N); r_s_01 = zeros(100,N); r_s_11 = zeros(100,N);
r_s1_0 = zeros(100,N); r_s1_1 = zeros(100,N); r_s2_0 = zeros(100,N); r_s2_1 = zeros(100,N);
for i = 1:N
 r_(:,i)     = mean(reshape(r(:,i),[],100))';
 r_s_11(:,i) = mean(reshape(r(s(:,1).*s(:,2)==1,i),[],100))';
 r_s_10(:,i) = mean(reshape(r(s(:,1).*(1-s(:,2))==1,i),[],100))';
 r_s_01(:,i) = mean(reshape(r((1-s(:,1)).*s(:,2)==1,i),[],100))';
 r_s_00(:,i) = mean(reshape(r((1-s(:,1)).*(1-s(:,2))==1,i),[],100))';
 r_s1_1(:,i) = mean(reshape(r(s(:,1)==1,i),[],100))';
 r_s1_0(:,i) = mean(reshape(r(s(:,1)==0,i),[],100))';
 r_s2_1(:,i) = mean(reshape(r(s(:,2)==1,i),[],100))';
 r_s2_0(:,i) = mean(reshape(r(s(:,2)==0,i),[],100))';
end

end

