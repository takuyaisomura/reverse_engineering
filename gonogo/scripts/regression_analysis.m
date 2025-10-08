
%--------------------------------------------------------------------------------
% regression_analysis.m
%
% Copyright (C) 2024 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2024-10-28
%--------------------------------------------------------------------------------

function reg = regression_analysis(data,Ttrial,Tend)

T       = Ttrial*Tend;
Nsample = size(data,1);
Ns      = size(data{1}.s,1);
tt1     = 1:T;
n_cos   = 5;
time    = cos(pi*(1:n_cos)'*(0:T-1)/T)*sqrt(2);
time    = time(1) - time;
tot_var = zeros(Nsample,Ns+1);
for num = 1:Nsample
 % ridge regression
 s    = [data{num}.s(:,tt1); kron(time,ones(Ns,1)).*kron(ones(n_cos,1),data{num}.s(:,tt1)-1/2)];
 r    = data{num}.activity(:,tt1);
 Mat  = (r*s'/T)/(s*s'/T+eye(size(s,1))*10^-4);
 qr   = Mat*s;
 err1 = sum(sum((r - qr).^2))/sum(sum(r.^2));
 reg.Mat{num,1}  = Mat;
 reg.err1{num,1} = err1;
 
 % amount of external information
 temp              = zeros(1,Ns);
 reg.qs_var{num,1} = zeros(Ns,Tend);
 for i = 1:Ns
  reg.qs_var{num,1}(i,:) = mean(reshape(mean((Mat(:,Ns*(0:n_cos)+i)*s(Ns*(0:n_cos)+i,:)).^2), [Ttrial Tend]));
  temp(i) = mean(reg.qs_var{num,1}(i,:));
 end
 reg.var{num,1}    = temp;
 tot_var(num,:)    = [temp/sum(temp)*(1-err1), err1];
 reg.qs_var{num,1} = reg.qs_var{num,1}/sum(temp)*(1-err1);
end
reg.tot_var = tot_var;

end

%--------------------------------------------------------------------------------
