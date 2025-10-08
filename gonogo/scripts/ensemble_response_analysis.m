
%--------------------------------------------------------------------------------
% ensemble_response_analysis.m
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-10-28
%--------------------------------------------------------------------------------

function resp = ensemble_response_analysis(phi,Ttrial,Tinit,Tlate,Tend,Nsample_l,score_post)

Nsample             = size(phi,1);
resp.x_l_pre_go     = cell(2,1);
resp.x_l_pre_nogo   = cell(2,1);
resp.x_l_post_go    = cell(2,1);
resp.x_l_post_nogo  = cell(2,1);
resp.x_nl_pre_go    = cell(2,1);
resp.x_nl_pre_nogo  = cell(2,1);
resp.x_nl_post_go   = cell(2,1);
resp.x_nl_post_nogo = cell(2,1);
resp.y_l_pre_go     = [];
resp.y_l_pre_nogo   = [];
resp.y_l_post_go    = [];
resp.y_l_post_nogo  = [];
resp.y_nl_pre_go    = [];
resp.y_nl_pre_nogo  = [];
resp.y_nl_post_go   = [];
resp.y_nl_post_nogo = [];
resp.d_l_pre_go     = [];
resp.d_l_pre_nogo   = [];
resp.d_l_post_go    = [];
resp.d_l_post_nogo  = [];
resp.d_nl_pre_go    = [];
resp.d_nl_pre_nogo  = [];
resp.d_nl_post_go   = [];
resp.d_nl_post_nogo = [];
resp.pos_l_pre_go     = [];
resp.pos_l_pre_nogo   = [];
resp.pos_l_post_go    = [];
resp.pos_l_post_nogo  = [];
resp.pos_nl_pre_go    = [];
resp.pos_nl_pre_nogo  = [];
resp.pos_nl_post_go   = [];
resp.pos_nl_post_nogo = [];
resp.x1_l_go          = [];
resp.x2_l_go          = [];
resp.x1_l_nogo        = [];
resp.x2_l_nogo        = [];
resp.x1_nl_go         = [];
resp.x2_nl_go         = [];
resp.x1_nl_nogo       = [];
resp.x2_nl_nogo       = [];
resp.d_l_go           = [];
resp.d_l_nogo         = [];
resp.d_nl_go          = [];
resp.d_nl_nogo        = [];

for num = 1:Nsample_l
 if isempty(phi{num}), continue, end
 tr_id = find(cell2mat(cellfun(@(phi) phi.isgo, phi(num,:), 'UniformOutput', false)) == 1);
 for i = tr_id
  resp.x1_l_go = [resp.x1_l_go; phi{num,i}.x(1,:)];
  resp.x2_l_go = [resp.x2_l_go; phi{num,i}.x(2,:)];
  resp.d_l_go  = [resp.d_l_go; phi{num,i}.d];
 end
 for i = tr_id(tr_id<=Tinit)
  resp.x_l_pre_go{1} = [resp.x_l_pre_go{1}; phi{num,i}.x(1,:)];
  resp.x_l_pre_go{2} = [resp.x_l_pre_go{2}; phi{num,i}.x(2,:)];
  resp.y_l_pre_go    = [resp.y_l_pre_go; phi{num,i}.y];
  resp.d_l_pre_go    = [resp.d_l_pre_go; phi{num,i}.d];
  resp.pos_l_pre_go  = [resp.pos_l_pre_go; phi{num,i}.pos];
 end
 for i = tr_id(tr_id>=Tlate)
  resp.x_l_post_go{1} = [resp.x_l_post_go{1}; phi{num,i}.x(1,:)];
  resp.x_l_post_go{2} = [resp.x_l_post_go{2}; phi{num,i}.x(2,:)];
  resp.y_l_post_go    = [resp.y_l_post_go; phi{num,i}.y];
  resp.d_l_post_go    = [resp.d_l_post_go; phi{num,i}.d];
  resp.pos_l_post_go  = [resp.pos_l_post_go; phi{num,i}.pos];
 end
 
 tr_id = find(cell2mat(cellfun(@(phi) phi.isgo, phi(num,:), 'UniformOutput', false)) == 0);
 for i = tr_id
  resp.x1_l_nogo = [resp.x1_l_nogo; phi{num,i}.x(1,:)];
  resp.x2_l_nogo = [resp.x2_l_nogo; phi{num,i}.x(2,:)];
  resp.d_l_nogo  = [resp.d_l_nogo; phi{num,i}.d];
 end
 for i = tr_id(tr_id<=Tinit)
  resp.x_l_pre_nogo{1} = [resp.x_l_pre_nogo{1}; phi{num,i}.x(1,:)];
  resp.x_l_pre_nogo{2} = [resp.x_l_pre_nogo{2}; phi{num,i}.x(2,:)];
  resp.y_l_pre_nogo    = [resp.y_l_pre_nogo; phi{num,i}.y];
  resp.d_l_pre_nogo    = [resp.d_l_pre_nogo; phi{num,i}.d];
  resp.pos_l_pre_nogo  = [resp.pos_l_pre_nogo; phi{num,i}.pos];
 end
 for i = tr_id(tr_id>=Tlate)
  resp.x_l_post_nogo{1} = [resp.x_l_post_nogo{1}; phi{num,i}.x(1,:)];
  resp.x_l_post_nogo{2} = [resp.x_l_post_nogo{2}; phi{num,i}.x(2,:)];
  resp.y_l_post_nogo    = [resp.y_l_post_nogo; phi{num,i}.y];
  resp.d_l_post_nogo    = [resp.d_l_post_nogo; phi{num,i}.d];
  resp.pos_l_post_nogo  = [resp.pos_l_post_nogo; phi{num,i}.pos];
 end
end

for num = Nsample_l+1:Nsample
 if isempty(phi{num}), continue, end
 tr_id = find(cell2mat(cellfun(@(phi) phi.isgo, phi(num,:), 'UniformOutput', false)) == 1);
 for i = tr_id
  resp.x1_nl_go = [resp.x1_nl_go; phi{num,i}.x(1,:)];
  resp.x2_nl_go = [resp.x2_nl_go; phi{num,i}.x(2,:)];
  resp.d_nl_go  = [resp.d_nl_go; phi{num,i}.d];
 end
 for i = tr_id(tr_id<=Tinit)
  resp.x_nl_pre_go{1} = [resp.x_nl_pre_go{1}; phi{num,i}.x(1,:)];
  resp.x_nl_pre_go{2} = [resp.x_nl_pre_go{2}; phi{num,i}.x(2,:)];
  resp.y_nl_pre_go    = [resp.y_nl_pre_go; phi{num,i}.y];
  resp.d_nl_pre_go    = [resp.d_nl_pre_go; phi{num,i}.d];
  resp.pos_nl_pre_go  = [resp.pos_nl_pre_go; phi{num,i}.pos];
 end
 for i = tr_id(tr_id>=Tlate)
  resp.x_nl_post_go{1} = [resp.x_nl_post_go{1}; phi{num,i}.x(1,:)];
  resp.x_nl_post_go{2} = [resp.x_nl_post_go{2}; phi{num,i}.x(2,:)];
  resp.y_nl_post_go    = [resp.y_nl_post_go; phi{num,i}.y];
  resp.d_nl_post_go    = [resp.d_nl_post_go; phi{num,i}.d];
  resp.pos_nl_post_go  = [resp.pos_nl_post_go; phi{num,i}.pos];
 end
 
 tr_id = find(cell2mat(cellfun(@(phi) phi.isgo, phi(num,:), 'UniformOutput', false)) == 0);
 for i = tr_id
  resp.x1_nl_nogo = [resp.x1_nl_nogo; phi{num,i}.x(1,:)];
  resp.x2_nl_nogo = [resp.x2_nl_nogo; phi{num,i}.x(2,:)];
  resp.d_nl_nogo  = [resp.d_nl_nogo; phi{num,i}.d];
 end
 for i = tr_id(tr_id<=Tinit)
  resp.x_nl_pre_nogo{1} = [resp.x_nl_pre_nogo{1}; phi{num,i}.x(1,:)];
  resp.x_nl_pre_nogo{2} = [resp.x_nl_pre_nogo{2}; phi{num,i}.x(2,:)];
  resp.y_nl_pre_nogo    = [resp.y_nl_pre_nogo; phi{num,i}.y];
  resp.d_nl_pre_nogo    = [resp.d_nl_pre_nogo; phi{num,i}.d];
  resp.pos_nl_pre_nogo  = [resp.pos_nl_pre_nogo; phi{num,i}.pos];
 end
 for i = tr_id(tr_id>=Tlate)
  resp.x_nl_post_nogo{1} = [resp.x_nl_post_nogo{1}; phi{num,i}.x(1,:)];
  resp.x_nl_post_nogo{2} = [resp.x_nl_post_nogo{2}; phi{num,i}.x(2,:)];
  resp.y_nl_post_nogo    = [resp.y_nl_post_nogo; phi{num,i}.y];
  resp.d_nl_post_nogo    = [resp.d_nl_post_nogo; phi{num,i}.d];
  resp.pos_nl_post_nogo  = [resp.pos_nl_post_nogo; phi{num,i}.pos];
 end
end

resp.x1_l_post_go = resp.x_l_post_go{1}; resp.x1_l_post_nogo = resp.x_l_post_nogo{1}; resp.x1_nl_post_go = resp.x_nl_post_go{1}; resp.x1_nl_post_nogo = resp.x_nl_post_nogo{1};
resp.x2_l_post_go = resp.x_l_post_go{2}; resp.x2_l_post_nogo = resp.x_l_post_nogo{2}; resp.x2_nl_post_go = resp.x_nl_post_go{2}; resp.x2_nl_post_nogo = resp.x_nl_post_nogo{2};
resp = rmfield(resp, {'x_l_post_go', 'x_l_post_nogo', 'x_nl_post_go', 'x_nl_post_nogo'});
resp = rmfield(resp, {'x_l_pre_go', 'x_l_pre_nogo', 'x_nl_pre_go', 'x_nl_pre_nogo'});

%--------------------------------------------------------------------------------

resp.err_x_go      = cell2mat(cellfun(@(phi) mean((phi.x(1,:)-[zeros(1,25),phi.o(1,26)*ones(1,100),zeros(1,25)]).^2), phi, 'UniformOutput', false));
resp.err_x_nogo    = cell2mat(cellfun(@(phi) mean((phi.x(2,:)-[zeros(1,25),phi.o(2,26)*ones(1,100),zeros(1,25)]).^2), phi, 'UniformOutput', false));
resp.err_x_go_nogo = cell2mat(cellfun(@(phi) mean(mean((phi.x-[zeros(2,25),phi.o(1:2,26)*ones(1,100),zeros(2,25)]).^2)), phi, 'UniformOutput', false));

%--------------------------------------------------------------------------------

W = cell(2,4);
K = cell(2,2);
V = cell(1,2);
A1 = cell(2,4);
B1 = cell(2,2);
C1 = cell(1,2);
A0 = cell(2,4);
B0 = cell(2,2);
C0 = cell(1,2);
for num = 1:Nsample
 if isempty(phi{num}), continue, end
 for t = 1:Tend
  for i = 1:8, W{i}(num,t) = phi{num,t}.W1(i) - phi{num,t}.W0(i); end
  for i = 1:4, K{i}(num,t) = phi{num,t}.K1(i) - phi{num,t}.K0(i); end
  for i = 1:2, V{i}(num,t) = phi{num,t}.V1(i) - phi{num,t}.V0(i); end
  for i = 1:8, A1{i}(num,t) = sig(phi{num,t}.W1(i)); end
  for i = 1:4, B1{i}(num,t) = sig(phi{num,t}.K1(i)); end
  for i = 1:2, C1{i}(num,t) = sig(phi{num,t}.V1(i)); end
  for i = 1:8, A0{i}(num,t) = sig(phi{num,t}.W0(i)); end
  for i = 1:4, B0{i}(num,t) = sig(phi{num,t}.K0(i)); end
  for i = 1:2, C0{i}(num,t) = sig(phi{num,t}.V0(i)); end
 end
end
resp.W11 = W{1,1}; resp.W12 = W{1,2}; resp.W13 = W{1,3}; resp.W14 = W{1,4};
resp.W21 = W{2,1}; resp.W22 = W{2,2}; resp.W23 = W{2,3}; resp.W24 = W{2,4};
resp.K11 = K{1,1}; resp.K12 = K{1,2};
resp.K21 = K{2,1}; resp.K22 = K{2,2};
resp.V11 = V{1,1}; resp.V12 = V{1,2};
resp.A111 = A1{1,1}; resp.A112 = A1{1,2}; resp.A113 = A1{1,3}; resp.A114 = A1{1,4};
resp.A121 = A1{2,1}; resp.A122 = A1{2,2}; resp.A123 = A1{2,3}; resp.A124 = A1{2,4};
resp.B111 = B1{1,1}; resp.B112 = B1{1,2};
resp.B121 = B1{2,1}; resp.B122 = B1{2,2};
resp.C111 = C1{1,1}; resp.C112 = C1{1,2};
resp.A011 = A0{1,1}; resp.A012 = A0{1,2}; resp.A013 = A0{1,3}; resp.A014 = A0{1,4};
resp.A021 = A0{2,1}; resp.A022 = A0{2,2}; resp.A023 = A0{2,3}; resp.A024 = A0{2,4};
resp.B011 = B0{1,1}; resp.B012 = B0{1,2};
resp.B021 = B0{2,1}; resp.B022 = B0{2,2};
resp.C011 = C0{1,1}; resp.C012 = C0{1,2};

resp.phix = cell2mat(cellfun(@(phi) [phi.phix1',phi.phix0'], phi(:,1), 'UniformOutput', false));
resp.phiy = cell2mat(cellfun(@(phi) [phi.phiy1',phi.phiy0'], phi(:,1), 'UniformOutput', false));
resp.isgo = cell2mat(cellfun(@(phi) phi.isgo, phi, 'UniformOutput', false));
resp.isgo_post              = resp.isgo;
resp.isgo_post(:,1:Tlate-1) = -1;
resp.scc_post      = score_post;
resp.pos_post      = zeros(Nsample,3);
resp.pos           = zeros(Nsample,3);
for i = 1:Nsample
 resp.pos(i,1)      = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo(i,:)==1), 'UniformOutput', false)));
 resp.pos(i,2)      = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo(i,:)==0), 'UniformOutput', false)));
 resp.pos(i,3)      = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo(i,:)>=0), 'UniformOutput', false)));
 resp.pos_post(i,1) = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo_post(i,:)==1), 'UniformOutput', false)));
 resp.pos_post(i,2) = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo_post(i,:)==0), 'UniformOutput', false)));
 resp.pos_post(i,3) = mean(cell2mat(cellfun(@(phi) phi.pos(125), phi(i,resp.isgo_post(i,:)>=0), 'UniformOutput', false)));
end

%--------------------------------------------------------------------------------

if Nsample > Nsample_l
 resp.p_W = zeros(2,4);
 resp.p_K = zeros(2,2);
 resp.p_V = zeros(1,2);
 for i = 1:8, resp.p_W(i) = ranksum(W{i}(1:Nsample_l,end),W{i}(Nsample_l+1:end,end)); end
 for i = 1:4, resp.p_K(i) = ranksum(K{i}(1:Nsample_l,end),K{i}(Nsample_l+1:end,end)); end
 for i = 1:2, resp.p_V(i) = ranksum(V{i}(1:Nsample_l,end),V{i}(Nsample_l+1:end,end)); end
else
 resp.p_W = [];
 resp.p_K = [];
 resp.p_V = [];
end

%--------------------------------------------------------------------------------
% representational dimensionality

resp.dim1 = zeros(Nsample, Tend);
resp.dim2 = zeros(Nsample, Tend);
for i = 1:Nsample
 x = func(@(a)a.x, phi(i,1:Tend));
 for t = 1:Tend
  [~,~,L]        = pca(x(:,1:Ttrial*t)');
  P              = L/sum(L);
  resp.dim1(i,t) = exp(sum(-P.*log(P)));
  resp.dim2(i,t) = 1/sum(P.^2);
 end
end

% risk
resp.risk = func(@(a)a.g_max, phi(:,1));

% accuracy
feature          = zeros(6,Nsample);
feature(1,:)     = resp.dim2(:,Tend)';                            % representational dimensionality
feature(2,:)     = resp.W13(:,Tend)'-resp.W23(:,Tend)';           % W13 - W23
feature(3,:)     = resp.V11(:,Tend)'-resp.V12(:,Tend)';           % V11 - V12
feature(4,:)     = resp.K12(:,Tend)'+resp.K21(:,Tend)';           % K12 + K21
feature(5,:)     = resp.risk';                                    % subjective risk
feature(6,:)     = resp.pos_post(:,1)'/10-resp.pos_post(:,2)'/10; % behavioral difference
islearner        = [ones(1,Nsample_l),zeros(1,Nsample-Nsample_l)];
resp.accuracy    = zeros(6,1);
resp.accuracy(1) = accuracy(feature(1,:),islearner);
resp.accuracy(2) = accuracy(feature(2,:),islearner);
resp.accuracy(3) = accuracy(feature(3,:),islearner);
resp.accuracy(4) = accuracy(feature(4,:),islearner);
resp.accuracy(5) = accuracy(feature(5,:),islearner);
resp.accuracy(6) = accuracy(feature(6,:),islearner);

%--------------------------------------------------------------------------------

end
