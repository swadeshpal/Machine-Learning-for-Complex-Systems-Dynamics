clear all

load t_stable_r.txt
load t_unstable_r.txt
load t_stable_us.txt
load t_unstable_us.txt
load u_train.txt
load original_t_r.txt

tiledlayout(1,2,'TileSpacing','Compact','Padding','tight');
nexttile



for n=1:101
    r1(n) = 1.78780710697174;
    r2(n) = 2.60280919075012;
    y(n) = (n-1)*1.5/100;
end

hold on

plot(u_train(1:83),original_t_r(1:83),'-b','LineWidth',1.5, 'DisplayName', 'EINN Prediction')
hold on
plot(t_stable_us(44:end),t_stable_r(44:end),'-b','LineWidth',1.5,'HandleVisibility','off')
plot(t_unstable_us,t_unstable_r,'-b','LineWidth',1.5,'HandleVisibility','off')

% r = 1.7869;
% p = 2;
% h = 0.5;
% Nu = 201;
% % u = linspace(0, 1.2, Nu);
% % for n=1:Nu
% %     v3(n) = r*u(n)^p/(u(n)^p+h^p);
% % end
% u1p = 0.4395;
% v1p = r*u1p^p/(u1p^p+h^p);
% u2p = 0.06485;
% v2p = r*u2p^p/(u2p^p+h^p);
r1p = 1.7869;
u1p = 0.06485;
r2p = 1.7869;
u2p = 0.4395;
r3p = 2.6049;
u3p = 0.1025;
r4p = 2.6049;
u4p = 1.1415;

hl = legend('show');
set(hl, 'Interpreter','latex', 'Location','northwest')
% set(h1,'Location','northwest')

plot(y,r1,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(y,r2,'--k','LineWidth',1.2,'HandleVisibility','off') 

ylim([0 3.5])
xlim([0 1.5])
box on;
text(1.4,3.3,'(a)','FontSize',12,'Color','k')
text(1.1,1.8,'$r_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(0.6,2.62,'$r_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
% scatter(u1p,r1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u2p,r2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u3p,r3p,20,"filled","o","MarkerEdgeColor","b","MarkerFaceColor",'m','HandleVisibility','off')
% scatter(u4p,r4p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
text(0.42,1.67,'$p_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k')
text(0.09,2.75,'$p_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k')

ax = gca;
ax.YTick=[0 1 2 3];
ax.XTick=[0 0.5 1 1.5];

ylabel('$r$','Interpreter', 'Latex','FontSize',18,'Color','k')
xlabel('$u_{*}$','Interpreter', 'Latex','FontSize',18,'Color','k')


nexttile


plot(original_t_r(1:83),u_train(1:83),'-b','LineWidth',1.5, 'DisplayName', 'Stable')
hold on
plot(t_stable_r(44:end),t_stable_us(44:end),'-b','LineWidth',1.5,'HandleVisibility','off')
plot(t_unstable_r,t_unstable_us,':r','LineWidth',1.5, 'DisplayName', 'Unstable')



hl = legend('show');
set(hl, 'Interpreter','latex', 'Location','northwest')
% set(h1,'Location','northwest')

plot(r1,y,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(r2,y,'--k','LineWidth',1.2,'HandleVisibility','off') 

xlim([0 3.5])
ylim([0 1.5])
box on;
text(3.3,1.4,'(b)','FontSize',12,'Color','k')
text(1.75,0.9,'$r_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(2.565,0.9,'$r_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
scatter(r1p,u1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(r2p,u2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(r3p,u3p,20,"filled","o","MarkerEdgeColor","b","MarkerFaceColor",'m','HandleVisibility','off')
scatter(r4p,u4p,20,"filled","o","MarkerEdgeColor","b","MarkerFaceColor",'m','HandleVisibility','off')
text(1.63,0.45,'$p_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k')
text(2.66,0.11,'$p_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k')

ax = gca;
ax.XTick=[0 1 2 3];
ax.YTick=[0 0.5 1 1.5];

xlabel('$r$','Interpreter', 'Latex','FontSize',18,'Color','k')
ylabel('$u_{*}$','Interpreter', 'Latex','FontSize',18,'Color','k')


set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 4.2])
print -dpng Fig_B1.png -r500
print -depsc Fig_B1.eps -r500

