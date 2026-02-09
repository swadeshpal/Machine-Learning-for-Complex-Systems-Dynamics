clear all

load A_2S_1st_a2_eq.txt
load A_2S_1st_u_train.txt


%% Parameter set
a1 = 0.25;
% a2 = 2.89;
h1 = 1;
k1 = 0.35;
eps = 0.01;
% b1 = 0.11;
b2 = 1;
k2 = 5;
% % % %% Temporal functions
% % % f = @(us,vs) a1+a2*vs^2/(vs^2+h1^2)-k1*us;
% % % g = @(us,vs) (b1+b2*us-k2*vs)/eps;

%% Equilibrium points
Na21 = 81;
Na22 = 10001;
Na23 = 101;

p_a2 = linspace(2.2,2.8,Na21);
p_a2(Na21+1:Na21+Na22) = linspace(2.8,3,Na22);
p_a2(Na21+Na22+1:Na21+Na22+Na23) = linspace(3,3.5,Na23);

% return
Na2 = length(p_a2);
b1 = 0.11;
for j=1:Na2
    a2 = p_a2(j);
    P0 = k1*k2;
    P1 = -a1*b2-a2*b2-b1*k1;
    P2 = h1^2*k1*k2;
    P3 = -a1*h1^2*b2-b1*h1^2*k1;
    pol=[P0 P1 P2 P3];
    rt1=roots(pol);
    rt=rt1(imag(rt1)==0);
    % [M1,nn]=size(rt);
    v1=rt;
    for n=1:length(v1)
        vs(n,j) = v1(n);
        us(n,j)=(k2*vs(n,j)-b1)/b2;
    end
end

Sp1 = 2303;
Sp2 = 9303;
US1 = us(1,1:Sp1);
US1(Sp1+1:Sp2) = us(3,Sp1+1:Sp2);
VS1 = vs(1,1:Sp1);
VS1(Sp1+1:Sp2) = vs(3,Sp1+1:Sp2);

US2(Sp1+1:Sp2) = us(2,Sp1+1:Sp2);
VS2(Sp1+1:Sp2) = vs(2,Sp1+1:Sp2);


US3(Sp1+1:Na2) = us(1,Sp1+1:Na2);
VS3(Sp1+1:Na2) = vs(1,Sp1+1:Na2);


for n=1:101
    r1(n) = 2.8444;
    r2(n) = 2.98444;
    y(n) = (n-1)*8/100;
end


tiledlayout(1,2,'TileSpacing','Compact','Padding','tight');
nexttile
% subplot(1,2,1)
hold on


plot(p_a2(1:Sp2), US1,'-b','LineWidth',2,'HandleVisibility','off')
plot(p_a2(Sp1+1:Sp2), US2(Sp1+1:Sp2),':m','LineWidth',2, 'DisplayName', 'Unstable')
plot(p_a2(Sp1+1:Na2),US3(Sp1+1:Na2),'-b','LineWidth',2, 'DisplayName', 'Stable')

box on
xlabel('$$a_2$$','Interpreter', 'Latex','FontSize',16,'FontWeight','bold','Color','k')
ylabel('$$u_*$$','Interpreter', 'Latex','FontSize',16,'FontWeight','bold','Color','k')



h2 = legend('show');
set(h2, 'Interpreter','latex', 'Location','northwest')

plot(r1,y,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(r2,y,'--k','LineWidth',1.2,'HandleVisibility','off') 

ax = gca;
ax.XTick=[2 2.5 3 3.5];
ax.YTick=[0 2 4 6 8];
ylim([0 8])
xlim([2.2 3.5])

text(3.42,0.5,'(a)','FontSize',12,'Color','k')
text(2.83,6.5,'$r_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(2.97,6.5,'$r_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')

nexttile
% subplot(1,2,2)
hold on

plot(A_2S_1st_a2_eq(1:302), A_2S_1st_u_train(1:302),'-b','LineWidth',2,'HandleVisibility','off')
plot(A_2S_1st_a2_eq(302:668), A_2S_1st_u_train(302:668),':m','LineWidth',2, 'DisplayName', 'Unstable')
plot(A_2S_1st_a2_eq(668:end), A_2S_1st_u_train(668:end),'-b','LineWidth',2, 'DisplayName', 'Stable')


box on
xlabel('$$a_2$$','Interpreter', 'Latex','FontSize',16,'FontWeight','bold','Color','k')
ylabel('$$u_*$$','Interpreter', 'Latex','FontSize',16,'FontWeight','bold','Color','k')



h2 = legend('show');
set(h2, 'Interpreter','latex', 'Location','northwest')

plot(r1,y,'--k','LineWidth',1.2,'HandleVisibility','off')
plot(r2,y,'--k','LineWidth',1.2,'HandleVisibility','off') 

ax = gca;
ax.XTick=[2 2.5 3 3.5];
ax.YTick=[0 2 4 6 8];
ylim([0 8])
xlim([2.2 3.5])

text(3.42,0.5,'(b)','FontSize',12,'Color','k')
text(2.83,6.5,'$r_{1}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')
text(2.97,6.5,'$r_{2}$','Interpreter', 'Latex','FontSize',12,'Color','k','BackgroundColor','w')


set(gcf,'PaperUnits','inches','PaperPosition',[0 0 10 4.2])
print -dpng Fig_C1.png -r500
print -depsc Fig_C1.eps -r500

