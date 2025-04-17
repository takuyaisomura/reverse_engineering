
%--------------------------------------------------------------------------------
% adder_init.m
%
% This demo is included in
% Triple equivalence for the emergence of biological intelligence
% Takuya Isomura
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-08-04
%

%--------------------------------------------------------------------------------

function [o,s,A,B,Clist,sum_true,qa0,qb0,qc0] = adder_init(T)

Nnum = 16;                     % number of bits for representing input numbers
Nct  = 16;                     % number of bits for counting time steps
Nimg = 28*28;                  % size of hand digit image
No   = Nnum*Nimg + Nct*16 + 1; % dimensionality of sensory inputs
Ns   = Nnum + Nct;             % dimensionality of hidden states
NC   = Nct;                    % dimensionality of memory matrix
sim_type = 2;

% define mappings
pre_s  = [ones(1,10000) zeros(1,10000)];
pre_o  = (number_to_hand_digit(pre_s') > 0.5) * 1;
pre_o(:,10001:20000) = pre_o(:,10001:20000)/mean(mean(pre_o(:,10001:20000)))*mean(mean(pre_o(:,1:10000)));
pre_qa = [pre_o; 1-pre_o]*[pre_s; 1-pre_s]'/5000 - 1;
pre_qa = [kron(eye(Nnum),pre_qa(1:Nimg,1)), kron(eye(Nnum),pre_qa(1:Nimg,2));
          kron(eye(Nnum),pre_qa(Nimg+1:Nimg*2,1)), kron(eye(Nnum),pre_qa(Nimg+1:Nimg*2,2))] + 1;
qa0    = [pre_qa(1:Nnum*Nimg,1:Nnum), ones(Nnum*Nimg,Nct), pre_qa(1:Nnum*Nimg,Nnum+1:Nnum*2), ones(Nnum*Nimg,Nct);
          ones(Nct*16,Nnum), ones(Nct*16,Nct)+kron(eye(Nct),ones(16,1)), ones(Nct*16,Nnum), ones(Nct*16,Nct)-kron(eye(Nct),ones(16,1));
          ones(1,Ns*2);
          pre_qa(Nnum*Nimg+1:Nnum*Nimg*2,1:Nnum), ones(Nnum*Nimg,Nct), pre_qa(Nnum*Nimg+1:Nnum*Nimg*2,Nnum+1:Nnum*2), ones(Nnum*Nimg,Nct);
          ones(Nct*16,Nnum), ones(Nct*16,Nct)-kron(eye(Nct),ones(16,1)), ones(Nct*16,Nnum), ones(Nct*16,Nct)+kron(eye(Nct),ones(16,1));
          ones(1,Ns*2)];
qa0    = qa0 * 500000 + 1000;
[A,~]  = param_norm(qa0,sim_type); % likelihood mapping

qb0    = kron([1 -1; -1 1],eye(Ns))*100 + 101;
qc0    = [zeros(1,NC); ones(1,NC)] + 10^-8;

% run generative process
s        = zeros(Ns,T);   % hidden states
sum_true = zeros(T/Nct,1);
for t = 1:T
 if rem(t,NC) == 1
  if rem(t,T/4) == 1, s(:,t) = [zeros(16,1); (rem(t+(Nct:-1:1)'-1,Nct)==0)*1];
  else,               s(:,t) = [randi([0 1],10,1); zeros(6,1); (rem(t+(Nct:-1:1)'-1,Nct)==0)*1]; end
  sum_true((t-1)/Nct+1) = binary_to_decimal(s(1:Nnum,t));
  if t >= 2, sum_true((t-1)/Nct+1) = sum_true((t-1)/Nct+1) + sum_true((t-1)/Nct); end
 else
  s(:,t) = [s(1:Nnum,t-1); (rem(t+(Nct:-1:1)'-1,Nct)==0)*1];
 end
end
o     = [(reshape(number_to_hand_digit(s(1:Nnum,:)),[Nnum*Nimg T]) > 0.5) * 1; kron(s(Nnum+1:Ns,:),ones(16,1)); ones(1,T)]; % observations
for i = 1:3, sum_true(T/Nct*i/4+1:T/Nct) = sum_true(T/Nct*i/4+1:T/Nct) - sum_true(T/Nct*i/4); end

b     = [s;1-s]*[s(:,[1,1:T-1]);1-s(:,[1,1:T-1])]' + 10^-8;
[B,~] = param_norm(b,sim_type); % transition matrix (Bayes optimal B)
Clist = zeros(NC,T);
for t = 1:T, Clist(:,t) = flip(str2num(dec2bin(sum_true(floor((t-1)/Nct)+1),NC)')); end

end

%--------------------------------------------------------------------------------
