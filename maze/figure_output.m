
%--------------------------------------------------------------------------------
% figure_output.m
%
% This demo is included in
% Canonical neural networks perform active inference
% Takuya Isomura, Hideaki Shimazaki, Karl J. Friston
%
% The MATLAB scripts are available at
% https://github.com/takuyaisomura/reverse_engineering
%
% Copyright (C) 2020 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2020-09-02
%

%--------------------------------------------------------------------------------

function img = figure_output(Map, pos, t, T, h, omat, qsmat, qdmat, time, qd, K)

nx    = length(Map(1,:));
ny    = length(Map(:,1));
nv    = length(omat(1,:));
img   = zeros(ny,nx,3,'single');
img2  = zeros(nv,nv,3,'single');
img3  = zeros(nv,nv,3,'single');
img4  = zeros(nv,nv,3,'single');

img(:,:,1) = 1 - Map;
img(:,:,2) = 1 - Map;
img(:,:,3) = 1 - Map;
K = min(K,t);
for k = 1:K-1, img(pos(1,t-k),pos(2,t-k),:) = [0.8 0.8 1]; end
for k = 1:K-1, img(floor((pos(1,t-k)+pos(1,t-k+1))/2),floor((pos(2,t-k)+pos(2,t-k+1))/2),:) = [0.8 0.8 1]; end
img(pos(1,t),pos(2,t),:) = [0 0 1];

img2(:,:,1) = 1 - omat;
img2(:,:,2) = 1 - omat;
img2(:,:,3) = 1 - omat;

img3(:,:,1) = 1 - qsmat;
img3(:,:,2) = 1 - qsmat;
img3(:,:,3) = 1 - qsmat;

img4(:,:,1) = 1 - qdmat;
img4(:,:,2) = 1 - qdmat;
img4(:,:,3) = 1;

subplot(2,5,1:5)
image(img)
percent   = num2str(round(sum(time(1:h-1)<T)/(h-1)*100));
mean_step = num2str(round(mean(time(time<T))));
title(['h = ',num2str(h), ', t = ', num2str(t), ', ', percent, '%, ', mean_step])

subplot(2,5,6)
image(img2)
title('o')

subplot(2,5,7)
image(img3)
title('Qs')

subplot(2,5,8)
image(img4)
title(['Qd, <', num2str(round(qd(3),3)), ', >', num2str(round(qd(4),3))])

subplot(2,5,9)
plot(1:t,pos(1,1:t)+pos(2,1:t),'b-')
axis([0 T 0 nx+ny])
title('Position')

subplot(2,5,10)
plot(1:length(time),time,'bo')
axis([1 length(time) 0 max(time)])
title('Duration')

end

