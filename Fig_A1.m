clear all

alpha = 0.1;
beta = 2;
h = 0.5;
p = 2;
Nu = 201;
sz = 1.2;
u = linspace(0, 1.2, Nu);
% hl = legend('show');
% set(hl, 'Interpreter','latex')

for n=1:Nu
    v1(n) = beta*u(n)-alpha;
end

subplot(1,2,1)
% tiledlayout(1,2,'TileSpacing','Compact','Padding','tight');
% nexttile
hold on

plot(u,v1,':k','LineWidth',sz, 'DisplayName', '$v = \beta u - \alpha$')
hold on

% r = 1;
% for n=1:Nu
%     v2(n) = r*u(n)^p/(u(n)^p+h^p);
% end
% plot(u,v2,'-b','LineWidth',2)

r = 1.7869;
for n=1:Nu
    v3(n) = r*u(n)^p/(u(n)^p+h^p);
end
u1p = 0.4395;
v1p = r*u1p^p/(u1p^p+h^p);
u2p = 0.06485;
v2p = r*u2p^p/(u2p^p+h^p);

plot(u,v3,'-b','LineWidth',sz, 'DisplayName', '$v = r_{1}g(u)$')
scatter(u1p,v1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u2p,v2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')


r = 2.6049;
for n=1:Nu
    v4(n) = r*u(n)^p/(u(n)^p+h^p);
end
u3p = 0.1025;
v3p = r*u3p^p/(u3p^p+h^p);
u4p = 1.1415;
v4p = r*u4p^p/(u4p^p+h^p);

plot(u,v4,'-m','LineWidth',sz, 'DisplayName', '$v = r_{2}g(u)$')
scatter(u3p,v3p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
scatter(u4p,v4p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')

text(0.42,0.9,'$p_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k')
xlim([0 1.2])
ylim([0 2.5])
hl = legend('show');
set(hl, 'Interpreter','latex', 'Location','northwest')

text(1.115,2.43,'(a)','FontSize',12,'Color','k')
box on
xlabel('$u$','Interpreter', 'Latex','FontSize',18,'Color','k')
ylabel('$v$','Interpreter', 'Latex','FontSize',18,'Color','k')

rectangle('Position',[0.04 0 0.12 0.25],'EdgeColor',[0 0.5 0] ,'LineWidth',1.5)
% rectangle('Position',[0.35 0.5 0.25 0.6],'EdgeColor',[0 0.5 0] ,'LineWidth',1.5)

% return
% p1 = [0.02 0.25];
% p2 = [0.03 1.25];
% plot([ p1(1) p2(1)], [p1(2) p2(2)], '--k','HandleVisibility','off');
% 
% p1 = [0.16 0];
% p2 = [0.4 1.25];
% plot([ p1(1) p2(1)], [p1(2) p2(2)], '--k','HandleVisibility','off');
% 
% % return
p1 = [0.04 0.25];
p2 = [0.65 1.02];
plot([ p1(1) p2(1)], [p1(2) p2(2)], '--k','HandleVisibility','off');

p1 = [0.16 0];
p2 = [0.65 0.06];
plot([ p1(1) p2(1)], [p1(2) p2(2)], '--k','HandleVisibility','off');


ax3 = axes('Position',[.31 .15 .135 .305]);
plot(u,v1,':k','LineWidth',sz,'HandleVisibility','off')
hold on
plot(u,v3,'-b','LineWidth',sz,'HandleVisibility','off')
plot(u,v4,'-m','LineWidth',sz, 'HandleVisibility','off')
scatter(u2p,v2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u3p,v3p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
text(0.097,0.135,'$p_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k')

xlim([0.04 0.16])
ylim([0 0.25])
box on;
set(gca,'xaxisLocation','top')
set(gca,'yaxisLocation','right')
ax3 = gca;
ax3.XTick=[0.05 0.1 0.15];
ax3.YTick=[0 0.1 0.2];

subplot(1,2,2)

alpha = 0.1;
beta = 2;
h = 0.5;
Nr = 7001;
R = linspace(0, 3.5, Nr);

Eq_Pts = zeros(Nr,3);
for n=1:Nr
    r = R(n);
    Rts = roots([-beta, alpha + r, - beta*h^2, alpha*h^2]);
    for m=1:length(Rts)
        if imag(Rts(m))== 0 
            if Rts(m)>0
                Eq_Pts(n,m) = Rts(m);
            end
        end
    end
end

for n=1:101
    r1(n) = 1.78780710697174;
    r2(n) = 2.60280919075012;
    y(n) = (n-1)*1.5/100;
end

hold on
plot(R(1:5209),Eq_Pts(1:5209,3),'-b','LineWidth',1.5, 'DisplayName', 'Stable')
plot(R(3576:5209),Eq_Pts(3576:5209,2),':r','LineWidth',1.5, 'DisplayName', 'Unstable')
plot(R(3576:end),Eq_Pts(3576:end,1),'-b','LineWidth',1.5,'HandleVisibility','off')

h2 = legend('show');
set(h2, 'Interpreter','latex', 'Location','northwest')

plot(r1,y,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(r2,y,'--k','LineWidth',1.2,'HandleVisibility','off') 



xlim([0 3.5])
ylim([0 1.5])
box on;
text(3.25,1.46,'(b)','FontSize',12,'Color','k')
text(1.75,0.91,'$r_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(2.565,0.91,'$r_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
r1p = 1.7869;
u1p = 0.06485;
r2p = 1.7869;
u2p = 0.4395;
r3p = 2.6049;
u3p = 0.1025;
r4p = 2.6049;
u4p = 1.1415;
scatter(r1p,u1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(r2p,u2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(r3p,u3p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
scatter(r4p,u4p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
text(1.63,0.45,'$p_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k')
text(2.66,0.11,'$p_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k')
ax = gca;
ax.XTick=[0 1 2 3];
ax.YTick=[0 0.5 1 1.5];

xlabel('$r$','Interpreter', 'Latex','FontSize',18,'Color','k')
ylabel('$u_{*}$','Interpreter', 'Latex','FontSize',18,'Color','k')



set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 4.2])
print -dpng Fig_A1.png -r500
print -depsc Fig_A1.eps -r500

