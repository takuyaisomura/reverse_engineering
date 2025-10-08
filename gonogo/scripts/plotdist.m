
%--------------------------------------------------------------------------------
% plotdist.m
%
% Copyright (C) 2025 Takuya Isomura
% (RIKEN Center for Brain Science)
%
% 2025-06-20
%--------------------------------------------------------------------------------

function plotdist(x,data,col,wid)

y = mean(data,'omitnan');
z = std(data,'omitnan');
patch([x,flip(x)],[y-z,flip(y+z)],col,'FaceAlpha',.2,'EdgeColor','none'), hold on
plot(x,y,[col,'-'],'LineWidth',wid), hold off

end
