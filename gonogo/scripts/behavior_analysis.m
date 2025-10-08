
%--------------------------------------------------------------------------------
% behavior_analysis.m
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-10-28
%--------------------------------------------------------------------------------

function bhv = behavior_analysis(data,Ttrial,Tinit,Tlate,Tend)

% initialization
Nsample           = size(data,1);
bhv.pos_pre_go    = cell(Nsample,1);
bhv.pos_pre_nogo  = cell(Nsample,1);
bhv.pos_post_go   = cell(Nsample,1);
bhv.pos_post_nogo = cell(Nsample,1);
bhv.score_pre     = zeros(Nsample,2);
bhv.score_post    = zeros(Nsample,2);

for num = 1:Nsample
 pos                    = data{num}.position;
 go                     = find(data{num}.go(26:Ttrial:Ttrial*Tinit) == 1);
 nogo                   = find(data{num}.nogo(26:Ttrial:Ttrial*Tinit) == 1);
 bhv.pos_pre_go{num}    = pos((go'-1)*Ttrial+(1:Ttrial));
 bhv.pos_pre_nogo{num}  = pos((nogo'-1)*Ttrial+(1:Ttrial));
 bhv.score_pre(num,:)   = [mean(pos((go-1)*Ttrial+125)>1), mean(pos((nogo-1)*Ttrial+125)<1)];
 
 go                     = (Tlate-1)+find(data{num}.go(Ttrial*(Tlate-1)+26:Ttrial:Ttrial*Tend) == 1);
 nogo                   = (Tlate-1)+find(data{num}.nogo(Ttrial*(Tlate-1)+26:Ttrial:Ttrial*Tend) == 1);
 bhv.pos_post_go{num}   = pos((go'-1)*Ttrial+(1:Ttrial));
 bhv.pos_post_nogo{num} = pos((nogo'-1)*Ttrial+(1:Ttrial));
 bhv.score_post(num,:)  = [mean(pos((go-1)*Ttrial+125)>1), mean(pos((nogo-1)*Ttrial+125)<1)];
end

end

%--------------------------------------------------------------------------------
