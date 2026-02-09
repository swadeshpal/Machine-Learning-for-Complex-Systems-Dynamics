clear all

load A_1S_2nd_beta_eq.txt
load A_1S_2nd_u_train.txt


alpha = 0.1;
Nu = 201;
sz = 1.2;
u = linspace(0, 1.0, Nu);


for n=1:Nu
    v1(n) = u(n)*(1-u(n));
end

beta1 = 0.17852;
beta2 = 0.2602568;

tiledlayout(1,2,'TileSpacing','Compact','Padding','tight');
nexttile


for n=1:101
    x_beta1(n) = beta1;
    x_beta2(n) = beta2;
    y(n) = (n-1)*1/100;
end

hold on

plot(A_1S_2nd_beta_eq(1:230), A_1S_2nd_u_train(1:230),'-b','LineWidth',2,'HandleVisibility','off')
plot(A_1S_2nd_beta_eq(230:956), A_1S_2nd_u_train(230:956),':m','LineWidth',2, 'DisplayName', 'Unstable')
plot(A_1S_2nd_beta_eq(956:end), A_1S_2nd_u_train(956:end),'-b','LineWidth',2, 'DisplayName', 'Stable')

h2 = legend('show');
set(h2, 'Interpreter','latex', 'Location','northeast')

plot(x_beta1,y,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(x_beta2,y,'--k','LineWidth',1.2,'HandleVisibility','off') 

u1p = beta1;
v1p = 0.114;
u2p = beta1;
v2p = 0.772346;

scatter(u1p,v1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u2p,v2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')

u3p = beta2;
v3p = 0.04382;
u4p = beta2;
v4p = 0.4775;


scatter(u3p,v3p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
scatter(u4p,v4p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')


text(beta1-0.01,0.91,'$\beta_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(beta2-0.01,0.91,'$\beta_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')

box on

xlim([0 0.6])
ylim([0 1.0])

text(0.56,0.08,'(a)','FontSize',12,'Color','k')

ax = gca;
ax.XTick=[0 0.2 0.4 0.6];
ax.YTick=[0 0.2 0.4 0.6 0.8 1];

xlabel('$\beta$','Interpreter', 'Latex','FontSize',18,'Color','k')
ylabel('$u_{*}$','Interpreter', 'Latex','FontSize',18,'Color','k')

nexttile

for n=1:Nu
    v3(n) = beta1*u(n)^2/(u(n)^2+alpha^2);
end
u1p = 0.113;
v1p = 0.1;
u2p = 0.7728;
v2p = 0.17558;


hold on

plot(u,v1,':k','LineWidth',sz, 'DisplayName', '$v = u(1 - u)$')
hold on

plot(u,v3,'-b','LineWidth',sz, 'DisplayName', '$v = \beta_{1}q(u)$')
scatter(u1p,v1p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')
scatter(u2p,v2p,20,"filled","o","MarkerEdgeColor","k","MarkerFaceColor",'r','HandleVisibility','off')



for n=1:Nu
    v4(n) = beta2*u(n)^2/(u(n)^2+alpha^2);
end

u3p = 0.478;
v3p = 0.2493;
u4p = 0.04373;
v4p = 0.041813;


plot(u,v4,'-m','LineWidth',sz, 'DisplayName', '$v = \beta_{2}q(u)$')
scatter(u3p,v3p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')
scatter(u4p,v4p,20,"filled","o","MarkerEdgeColor","r","MarkerFaceColor",'b','HandleVisibility','off')

xlim([0 1])
ylim([0 0.3])
hl = legend('show');
set(hl, 'Interpreter','latex', 'Location','northwest')

text(0.92,0.015,'(b)','FontSize',12,'Color','k')
box on
xlabel('$u$','Interpreter', 'Latex','FontSize',18,'Color','k')
ylabel('$v$','Interpreter', 'Latex','FontSize',18,'Color','k')


ax = gca;
ax.XTick=[0 0.2 0.4 0.6 0.8 1];
ax.YTick=[0 0.1 0.2 0.3];



set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 4.2])
print -dpng Fig_D1.png -r500
print -depsc Fig_D1.eps -r500

