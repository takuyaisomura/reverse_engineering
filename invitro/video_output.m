
%--------------------------------------------------------------------------------
% video_output.m
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

function [] = video_output(fig2, dirname, datatype, Ninit, o, W, Wp, phi1, phi0)

No       = length(o(:,1));   % number of sensory stimuli
Nsession = length(W(1,1,:)); % number of sessions

% create free energy landscape
Nd  = 100;
L_W = zeros(Nd,Nd);
tt  = 1:256;
for i = 1:Nd
 for j = 1:Nd
  W1sim    = (kron(2*[(i-1)/Nd (j-1)/Nd; (j-1)/Nd (i-1)/Nd],ones(1,16)));
  W0sim    = (kron(-2*[(i-1)/Nd (j-1)/Nd; (j-1)/Nd (i-1)/Nd],ones(1,16)));
  h1sim    = log(1-sig(W1sim))*ones(No,1) + phi1;
  h0sim    = log(1-sig(W0sim))*ones(No,1) + phi0;
  xsim     = sig((W1sim-W0sim)*o(:,tt) + (h1sim-h0sim)*ones(1,256));
  L_W(i,j) = sum(sum([xsim;1-xsim] .* (log([xsim;1-xsim]+10^-6) - [W1sim;W0sim]*o(:,tt) - [h1sim;h0sim]*ones(1,256))));
 end
end

% create free energy landscape image
L_W        = (L_W-min(min(L_W)))./(max(max(L_W))-min(min(L_W)));
img        = zeros(Nd,Nd,3);
img(:,:,1) = max(min(((L_W)*3-1)/2,1),0);
img(:,:,2) = max(min((L_W)*3-0,1),0);
img(:,:,3) = max(min(((L_W)*3-1)/2,1),0);
img(20:20:100,:,:) = 0.5;
img(:,20:20:100,:) = 0.5;

if strcmp(datatype,'mix0'), lwd = 6;
else,                       lwd = 3; end

%--------------------------------------------------------------------------------
% plot empirically estimated trajectory

% prepare trajectory
W_s          = zeros(32*2,100);
W_s(1:32,:)  = reshape(W(1,1:32,:),[32 100])*2;
W_s(33:64,:) = reshape(W(2,1:32,:),[32 100])*2;
W_s          = max(min(W_s,3.96),0.001)*100/4;

% open video
clf(fig2)
vid = VideoWriter([dirname,'free_energy_gradient_W_',datatype,'.mp4'],'MPEG-4');
open(vid);

% run animation
figure(fig2)
for t = 1:Nsession
 surf(L_W,img,'EdgeColor','none'), hold on
 view(135,45)
 ax = gca; ax.XTickLabel = []; ax.YTickLabel = []; ax.ZTickLabel = []; ax.LineWidth = 2;
 axis square
 plot3(W_s(1:16,1:t)'+1,W_s(33:48,1:t)'+1,L_W((ceil(W_s(1:16,1:t)')-1)*100+ceil(W_s(33:48,1:t)'))+0.01,'r-','LineWidth',lwd)
 plot3(W_s(17:32,1:t)'+1,W_s(49:64,1:t)'+1,L_W((ceil(W_s(17:32,1:t)')-1)*100+ceil(W_s(49:64,1:t)'))+0.01,'b-','LineWidth',lwd), hold off
 drawnow
 pause(0.06)
 frame = getframe(gcf);
 writeVideo(vid,frame);
end

% close video
close(vid);
print(fig2,[dirname,'free_energy_gradient_W_',datatype,'.png'],'-dpng')

%--------------------------------------------------------------------------------
% plot theoretically predicted trajectory

% prepare trajectory
W_s          = zeros(32*2,100);
W_s(1:32,:)  = reshape(Wp(1,1:32,:),[32 100])*2;
W_s(33:64,:) = reshape(Wp(2,1:32,:),[32 100])*2;
W_s          = max(min(W_s,3.96),0.001)*100/4;

% open video
clf(fig2)
vid = VideoWriter([dirname,'free_energy_gradient_Wp_',datatype,'.mp4'],'MPEG-4');
open(vid);

% run animation
figure(fig2)
for t = 1:Nsession
 surf(L_W,img,'EdgeColor','none'), hold on
 view(135,45)
 ax = gca; ax.XTickLabel = []; ax.YTickLabel = []; ax.ZTickLabel = []; ax.LineWidth = 2;
 axis square
 if t <= Ninit
  plot3(W_s(1:16,1:t)'+1,W_s(33:48,1:t)'+1,L_W((ceil(W_s(1:16,1:t)')-1)*100+ceil(W_s(33:48,1:t)'))+0.01,'r-','LineWidth',lwd)
  plot3(W_s(17:32,1:t)'+1,W_s(49:64,1:t)'+1,L_W((ceil(W_s(17:32,1:t)')-1)*100+ceil(W_s(49:64,1:t)'))+0.01,'b-','LineWidth',lwd), hold off
 else
  plot3(W_s(1:16,1:Ninit)'+1,W_s(33:48,1:Ninit)'+1,L_W((ceil(W_s(1:16,1:Ninit)')-1)*100+ceil(W_s(33:48,1:Ninit)'))+0.01,'r-','LineWidth',lwd)
  plot3(W_s(17:32,1:Ninit)'+1,W_s(49:64,1:Ninit)'+1,L_W((ceil(W_s(17:32,1:Ninit)')-1)*100+ceil(W_s(49:64,1:Ninit)'))+0.01,'b-','LineWidth',lwd)
  plot3(W_s(1:16,Ninit:t)'+1,W_s(33:48,Ninit:t)'+1,L_W((ceil(W_s(1:16,Ninit:t)')-1)*100+ceil(W_s(33:48,Ninit:t)'))+0.01,'-','Color',[1 0.75 0],'LineWidth',lwd)
  plot3(W_s(17:32,Ninit:t)'+1,W_s(49:64,Ninit:t)'+1,L_W((ceil(W_s(17:32,Ninit:t)')-1)*100+ceil(W_s(49:64,Ninit:t)'))+0.01,'-','Color',[0 0.75 1],'LineWidth',lwd), hold off
 end
 drawnow
 pause(0.06)
 frame = getframe(gcf);
 writeVideo(vid,frame);
end

% close video
close(vid);
print(fig2,[dirname,'free_energy_gradient_Wp_',datatype,'.png'],'-dpng')

end

%--------------------------------------------------------------------------------
