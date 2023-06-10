
%--------------------------------------------------------------------------------
% fig3_figure_output.m
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

function [] = fig3_figure_output(fig, datatype, fig_pos, x1_list, x_xp_mean, x_xp_stdv, err_x_xp_list, W, Wp, err_W_Wp_list, err_W_qA_list, L_list)

figure(fig)
subplot(6,5,fig_pos+0)
patch([1:100 flip(1:100)],[mean(x1_list(:,:,2))+std(x1_list(:,:,2)) flip(mean(x1_list(:,:,2))-std(x1_list(:,:,2)))],'blue','FaceAlpha',.2,'EdgeColor','none'), hold on
plot(1:100,mean(x1_list(:,:,2)),'b-')
patch([1:100 flip(1:100)],[mean(x1_list(:,:,1))+std(x1_list(:,:,1)) flip(mean(x1_list(:,:,1))-std(x1_list(:,:,1)))],'red','FaceAlpha',.2,'EdgeColor','none')
plot(1:100,mean(x1_list(:,:,1)),'r-'), hold off
axis([0 100 0 1]), title([datatype,' x'])

subplot(6,5,fig_pos+5)
patch([0.02:0.04:0.98 flip(0.02:0.04:0.98)],[x_xp_mean+x_xp_stdv; flip(x_xp_mean-x_xp_stdv)]','blue','FaceAlpha',.2,'EdgeColor','none'), hold on
plot(0.02:0.04:0.98,x_xp_mean,'b-'), hold off
axis([0 1 0 1]), title('x vs xp 91-100')

subplot(6,5,fig_pos+10)
patch([1:100 flip(1:100)],[mean(err_x_xp_list)+std(err_x_xp_list) flip(mean(err_x_xp_list)-std(err_x_xp_list))],'blue','FaceAlpha',.2,'EdgeColor','none'), hold on
plot(1:100,mean(err_x_xp_list),'b-'), hold off
axis([0 100 0 0.5]), title('error x vs x^P')

subplot(6,5,fig_pos+15)
plot([-1 4],[-1 4],'k--'), hold on
plot(reshape(W(:,:,100),[2*32 1]),reshape(Wp(:,:,100),[2*32 1]),'b+'), hold off
axis([-1 4 -1 4]), title('W vs Wp at100')

subplot(6,5,fig_pos+20)
patch([1:100 flip(1:100)],[mean(err_W_Wp_list)+std(err_W_Wp_list) flip(mean(err_W_Wp_list)-std(err_W_Wp_list))],'red','FaceAlpha',.2,'EdgeColor','none'), hold on
plot(1:100,mean(err_W_Wp_list),'r-')
patch([1:100 flip(1:100)],[mean(err_W_qA_list)+std(err_W_qA_list) flip(mean(err_W_qA_list)-std(err_W_qA_list))],'blue','FaceAlpha',.2,'EdgeColor','none')
plot(1:100,mean(err_W_qA_list),'b-'), hold off
axis([0 100 0 0.3]), title('sig(W) vs qA^{Id},sig(W^P)')

subplot(6,5,fig_pos+25)
patch([1:100 flip(1:100)],[mean(L_list)+std(L_list) flip(mean(L_list)-std(L_list))],'blue','FaceAlpha',.2,'EdgeColor','none'), hold on
plot(1:100,mean(L_list),'b-'), hold off
axis([0 100 10000 12000]), title('F')
drawnow

end

%--------------------------------------------------------------------------------
